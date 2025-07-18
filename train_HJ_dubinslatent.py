import argparse
import os
from pathlib import Path
import torch
import gymnasium as gym  # We use Gymnasium API for vector envs
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

# PyHJ components for DDPG training
from PyHJ.data import Collector, VectorReplayBuffer, Batch
from PyHJ.trainer import offpolicy_trainer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing
# Load your DINO-WM via plan.load_model
from plan import load_model

# import sys, os
# from pathlib import Path
# # 1) add dino_wm folder first
# ROOT = os.path.dirname(__file__)
# sys.path.insert(0, ROOT)
# # 2) add dreamerv3-torch *after* it
# dreamer = Path("/storage1/sibai/.../latent-safety") / "dreamerv3-torch"
# sys.path.insert(1, str(dreamer))   # note: insert at index=1, not 0
# import tools                         # now this pulls from dreamerv3-torch/tools.py


# Underlying Dubins Gym env (classic Gym)
from env.dubins.dubins import DubinsEnv
from gymnasium.spaces import Box

import yaml

# import shimmy
# # if GymV22CompatibilityV0 isn’t defined in this shimmy version,
# # alias it to GymV26CompatibilityV0
# if not hasattr(shimmy, "GymV22CompatibilityV0") and hasattr(shimmy, "GymV26CompatibilityV0"):
#     shimmy.GymV22CompatibilityV0 = shimmy.GymV26CompatibilityV0
    

def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

def get_args_and_merge_config():
    # 1) Top‐level parser for the flags you always need
    parser = argparse.ArgumentParser("DDPG HJ on DINO latent Dubins")
    parser.add_argument(
        "--dino_ckpt_dir", type=str,
        default="/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs/dubins",
        help="Where to find the DINO-WM checkpoints"
    )
    parser.add_argument(
        "--config", type=str, default="train_HJ_configs.yaml",
        help="Path to your flat YAML of hyperparameters"
    )
    
    parser.add_argument(
    "--with_proprio", action="store_true",
    help="Flag to include proprioceptive information in latent encoding"
    )
    
    parser.add_argument(
    "--dino_encoder", type=str, default="dino",
    help="Which encoder to use: dino, r3m, vc1, etc."
    )

    
    args, remaining = parser.parse_known_args()

    # 2) Load all keys & values from the YAML (no `defaults:` wrapper needed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)  # e.g. {'batch_size':16, 'critic-lr':1e-3, ...}

    # 3) Dynamically build a second parser so each key adopts the right type
    cfg_parser = argparse.ArgumentParser()
    for key, val in sorted(cfg.items()):
        arg_t = args_type(val)
        cfg_parser.add_argument(f"--{key}", type=arg_t, default=arg_t(val))
    cfg_args = cfg_parser.parse_args(remaining)

    # 4) Merge everything back into the top‐level args namespace
    for key, val in vars(cfg_args).items():
        setattr(args, key.replace("-", "_"), val)

    return args


class LatentDubinsEnv(gym.Env):
    """
    Wraps the classic Gym-based DubinsEnv into a Gymnasium-compatible Env.
    Encodes observations into DINO-WM latent space and uses info['h'] as reward.
    """
    def __init__(self, ckpt_dir: str, device: str, with_proprio: bool):
        super().__init__()
        # underlying Gym env
        self.env = DubinsEnv()
        self.device = torch.device(device)
        # Paths for loading DINO-WM
        ckpt_dir = Path(ckpt_dir)
        hydra_cfg = ckpt_dir / 'hydra.yaml'
        snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
        # load train config and model weights
        train_cfg = OmegaConf.load(str(hydra_cfg))
        num_action_repeat = train_cfg.num_action_repeat
        self.wm = load_model(snapshot, train_cfg, num_action_repeat, device=self.device)
        self.wm.eval()
        # probe a reset to set spaces
        reset_out = self.env.reset()
        # Gym reset returns obs; if obs is tuple unpack
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        self.with_proprio = with_proprio
        print("using proprio:", self.with_proprio)
        z = self._encode(obs)
        print(f"Example latent state z shape: {z.shape}")
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=z.shape, dtype=np.float32)
        self.action_space = self.env.action_space


    def reset(self):
        """
        Reset underlying Gym env and encode obs to latent.
        Returns: (obs_latent, info_dict)
        """
        reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        z = self._encode(obs)
        return z, {}

    def step(self, action):
        """
        Step in Gym env: returns (obs_latent, reward, terminated, truncated, info).
        Classic Gym returns (obs, reward, done, info).
        We map done->terminated and truncated=False.
        Reward is taken from info['h'].
        """
        obs_out, _, done, info = self.env.step(action)
        terminated = done
        truncated = False
        # extract obs if tuple
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        # override reward with safety metric
        h_s = info.get('h', 0.0) * 3 ##I multiplied by 3 to make HJ easier to learn
        z_next = self._encode(obs)
        return z_next, h_s, terminated, truncated, info

    def _encode(self, obs):
        """
        Encode raw obs via DINO-WM into a flat latent vector.
        Supports obs as dict or tuple (visual, proprio).
        """
        # unpack obs
        if isinstance(obs, dict):
            visual = obs['visual']
            proprio = obs['proprio']
        elif isinstance(obs, (tuple, list)) and len(obs) == 2:
            visual, proprio = obs
        else:
            raise ValueError(f"Unexpected obs type: {type(obs)}")
        
        with torch.no_grad():
            # prepare tensors
            visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32)  # (C, H, W)
            visual_np /= 255.0  # normalize to [0, 1]
            vis_t = torch.from_numpy(visual_np).unsqueeze(0)  # -> (1, C, H, W)
            vis_t = vis_t.unsqueeze(1)  # Add time dimension (1, 1, C, H, W)
            vis_t = vis_t.to(self.device)

            prop_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)
            prop_t = prop_t.unsqueeze(1)  # Add singleton dimension (1, 1, D_prop)
            
            data = {'visual': vis_t, 'proprio': prop_t}
            '''
            lat = self.wm.encode_obs(data)
                input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
                output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
            '''
            lat = self.wm.encode_obs(data)
            
            
            # flatten visual patches and concat proprio
            if (self.with_proprio):
                
                z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches, E_dim) -> (1, N_patches*E_dim)
                z_prop = lat['proprio']  # (1, D_prop)
                
                # flatten visual patches and concatenate proprio
                z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim) torch.Size([1, 75264])
                z_prop = lat['proprio']  # (1, 1, D_prop) ([1, 1, 10])
                z_prop=z_prop.squeeze(0)
                
                # Concatenate both visual and proprio embeddings
                z = torch.cat([z_vis, z_prop], dim=-1)#torch.Size([1, 75274])
                
                # print(z_prop.shape)torch.Size([1, 10])
                # print(z_prop.cpu().numpy())# when using dummyreapeatencoder[[2.2927914 1.471216  2.6900113 2.2927914 1.471216  2.6900113 2.2927914 1.471216  2.6900113 0.       ]]
                
                return z.squeeze(0).cpu().numpy()                       #dino torch.size(75274,)
            
            else:
                z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches, E_dim) -> (1, N_patches*E_dim)
                # z_prop = lat['proprio']  # (1, D_prop)
                
                # z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim) torch.Size([1, 75264])
          
                # print(z_vis.squeeze(0).cpu().numpy().shape) #dino torch.size(75264,)
                # print(z_vis.squeeze(0).cpu().numpy()[:-6])
                
                return z_vis.squeeze(0).cpu().numpy()
                

import os
# point Matplotlib to /tmp (or any other writable dir)
os.environ['MPLCONFIGDIR'] = '/storage1/fs1/sibai/Active/ihab/tmp'
# make sure it exists
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# now it’s safe to import pyplot
import matplotlib.pyplot as plt
import wandb

# … all your existing imports …

from PyHJ.data import Batch

# def compute_hj_value(x: float, y: float, theta: float,
#                      policy, helper_env, args) -> float:
#     """
#     Compute the Hamilton–Jacobi filter value in *latent* space:
#       1) Reset to (x,y,theta), get raw_obs and h_s from helper_env.step
#       2) Encode raw_obs -> z
#       3) Q = critic(z, actor_old(z))
#       4) return min(Q, h_s)
#     """
#     # 1) Reset underlying latent env (step returns (z, h_s, terminated, ..., info))
#     #    but Gym reset only gives obs, so we peek into info via a dummy step:
#     #    first reset:
#     reset_out = helper_env.env.reset(state=[x, y, theta])
#     # just extract h_s via a zero‐action step (or compute directly):
#     z0, h_s, _, _, _ = helper_env.step(np.zeros(helper_env.action_space.shape))
#     # 2) z0 is already the encoded latent for this state
#     # wrap into a batch of size 1
#     batch = Batch(obs=z0[None], info=Batch())
#     # 3) get Q(s,π_old(s))
#     a_old = policy(batch, model="actor_old").act
#     q_val = policy.critic(batch.obs, a_old).cpu().item()
#     # 4) HJ filter
#     return q_val
def compute_hj_value(x, y, theta, policy, helper_env, args):
    # set precise state without advancing dynamics
    obs_dict, _ = helper_env.env.reset(state=[x, y, theta])
    z0 = helper_env._encode(obs_dict)
    batch = Batch(obs=z0[None], info=Batch())
    a_old = policy(batch, model="actor_old").act
    q_val = policy.critic(batch.obs, a_old).cpu().item()
    return q_val
    #removed below, was wrong
    # # critic value V̂(s) = Q(s, π_old(s))
    # z_t = torch.as_tensor(z0, dtype=torch.float32, device=args.device).unsqueeze(0)
    # a_old = policy.actor_old(z_t)
    # q_val = policy.critic(batch.obs, a_old).cpu().item()
    # return q_val            # ← critic prediction only
# def evaluate_V(state):
#     tmp_obs = np.array(state)#.reshape(1,-1)
#     tmp_batch = Batch(obs = tmp_obs, info = Batch())
#     tmp = policy.critic(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
# return tmp.cpu().detach().numpy().flatten()

def plot_hj(policy, helper_env, thetas, args):
    """
    Plot the Hamilton–Jacobi safety filter in latent space:
    - Rows: different θ slices
    - Col1: binary (min(Q,h_s) > 0)
    - Col2: continuous min(Q,h_s)
    """
    xs = np.linspace(args.x_min, args.x_max, args.nx)
    ys = np.linspace(args.y_min, args.y_max, args.ny)
    fig1, axes1 = plt.subplots(len(thetas), 1, figsize=(3, 3*len(thetas)))
    fig2, axes2 = plt.subplots(len(thetas), 1, figsize=(3, 3*len(thetas)))

    for i, theta in enumerate(thetas):
        vals = np.zeros((args.nx, args.ny), dtype=np.float32)
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                vals[ix, iy] = compute_hj_value(x, y, theta, policy, helper_env, args)

        # Binary safe/unsafe
        axes1[i].imshow(
            (vals.T > 0),
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower"
        )
        axes1[i].set_title(f"θ={theta:.2f} (safe mask)")

        # Continuous value
        im = axes2[i].imshow(
            vals.T,
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower"
        )
        axes2[i].set_title(f"θ={theta:.2f} (HJ value)")
        fig2.colorbar(im, ax=axes2[i])

    fig1.tight_layout()
    fig2.tight_layout()
    return fig1, fig2

def main():
    # 1) parse args + merge YAML
    args = get_args_and_merge_config()
    # cast to the right types
    args.critic_lr         = float(args.critic_lr)
    args.actor_lr          = float(args.actor_lr)
    args.tau               = float(args.tau)
    args.gamma_pyhj        = float(args.gamma_pyhj)
    args.exploration_noise = float(args.exploration_noise)
    args.update_per_step   = float(args.update_per_step)
    args.step_per_epoch    = int(args.step_per_epoch)
    args.step_per_collect  = int(args.step_per_collect)
    args.test_num          = int(args.test_num)
    args.training_num      = int(args.training_num)
    args.total_episodes    = int(args.total_episodes)
    args.batch_size_pyhj   = int(args.batch_size_pyhj)
    args.buffer_size       = int(args.buffer_size)
    args.dino_ckpt_dir = os.path.join(args.dino_ckpt_dir, args.dino_encoder)
    
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)            # if you use CUDA
    torch.backends.cudnn.deterministic = True        # ▸ slower, deterministic
    torch.backends.cudnn.benchmark     = False
    
    # 2) init W&B + TB writer + logger
    import wandb
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    wandb.init(project=f"ddpg-hj-latent-dubins", name=f"ddpg-{args.dino_encoder}-{timestamp}" ,config=vars(args))
    writer    = SummaryWriter(log_dir=f"runs/ddpg_hj_latent/{args.dino_encoder}-{timestamp}/logs")
    wb_logger = WandbLogger()
    wb_logger.load(writer)    # must load the TB writer
    logger    = wb_logger     # use W&B for offpolicy_trainer

    # 3) make your latent envs
    train_envs = DummyVectorEnv(
        [lambda: LatentDubinsEnv(args.dino_ckpt_dir, args.device, args.with_proprio)
         for _ in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: LatentDubinsEnv(args.dino_ckpt_dir, args.device, args.with_proprio)
         for _ in range(args.test_num)]
    )

    # 4) extract shapes & max_action
    state_space  = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    state_shape  = state_space.shape
    action_shape = action_space.shape or action_space.n
    max_action   = torch.tensor(action_space.high,
                                device=args.device,
                                dtype=torch.float32)

    # 5) build critic + actor
    critic_net = Net(state_shape, action_shape,
                     hidden_sizes=args.critic_net,
                     activation=getattr(torch.nn, args.critic_activation),
                     concat=True, device=args.device)
    critic      = Critic(critic_net, device=args.device).to(args.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(), lr=args.critic_lr,
        weight_decay=args.weight_decay_pyhj
    )

    actor_net   = Net(state_shape,
                      hidden_sizes=args.control_net,
                      activation=getattr(torch.nn, args.actor_activation),
                      device=args.device)
    actor       = Actor(actor_net, action_shape,
                        max_action=max_action,
                        device=args.device).to(args.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)

    # 6) assemble your avoid‐DDPG policy
    policy = avoid_DDPGPolicy_annealing(
        critic=critic, critic_optim=critic_optim,
        tau=args.tau, gamma=args.gamma_pyhj,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=action_space,
        actor=actor, actor_optim=actor_optim,
        actor_gradient_steps=args.actor_gradient_steps,
    )

    # 7) hook into policy.learn to capture losses
    orig_learn = policy.learn
    policy.last_actor_loss  = 0.0
    policy.last_critic_loss = 0.0
    def learn_and_record(batch, **kw):
        metrics = orig_learn(batch, **kw)
        policy.last_actor_loss  = metrics["loss/actor"]
        policy.last_critic_loss = metrics["loss/critic"]
        return metrics
    policy.learn = learn_and_record

    # 8) define train_fn to log those to W&B
    def train_fn(epoch: int, step_idx: int):
        wandb.log({
            "loss/actor":  policy.last_actor_loss,
            "loss/critic": policy.last_critic_loss,
        })

    # 9) collectors
    buffer          = VectorReplayBuffer(args.buffer_size, args.training_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    print("collecting some data first")
    train_collector.collect(10000)
    print("done collecting some data first")
    test_collector  = Collector(policy, test_envs)

    # 10) choose headings & helper env
    thetas     = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    helper_env = LatentDubinsEnv(args.dino_ckpt_dir, args.device, args.with_proprio)

    # 11) training loop 
    log_path = Path(f"runs/ddpg_hj_latent/{args.dino_encoder}-{timestamp}")
    for epoch in range(1, args.total_episodes + 1):
        print(f"\n=== Epoch {epoch}/{args.total_episodes} ===")

        # a) one epoch of offpolicy_trainer
        stats = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=None,#test_collector,
            max_epoch=1,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size_pyhj,
            update_per_step=args.update_per_step,
            stop_fn=lambda r: False,
            train_fn=train_fn,        # log losses each epoch
            save_best_fn=None,
            logger=logger,
        )

        # b) log remaining numeric stats
        numeric = {}
        for k,v in stats.items():
            if isinstance(v,(int,float)): numeric[f"train/{k}"]=v
            elif isinstance(v,np.generic): numeric[f"train/{k}"]=float(v)
        wandb.log(numeric, step=epoch)

        # c) save policy checkpoint
        ckpt_dir = log_path / f"epoch_id_{epoch}"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        torch.save(policy.state_dict(), ckpt_dir/"policy.pth")

        # d) plot latent‐space HJ filter & log
        fig1, fig2 = plot_hj(policy, helper_env, thetas, args)
        wandb.log({
            "HJ_latent/binary":     wandb.Image(fig1),
            "HJ_latent/continuous": wandb.Image(fig2),
        })
        plt.close(fig1); plt.close(fig2)

    print("Training complete.")

if __name__ == "__main__":
    main()



# for using proprio: python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --with_proprio --dino_encoder dino
#for not using proprio: python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder dino