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

# Underlying Dubins Gym env (classic Gym)
from env.dubins.dubins import DubinsEnv
from gymnasium.spaces import Box

import yaml

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
        default="/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs/dubins/fully trained(prop repeated 3 times)",
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
        choices=["dino", "r3m", "resnet", "vc1", "scratch", "dino_cls"],
        help="Which encoder to use: dino, r3m, resnet, vc1, scratch, dino_cls"
    )

    # Finetuning arguments
    parser.add_argument(
        "--with_finetune", action="store_true",
        help="Enable finetuning of the encoder"
    )
    
    parser.add_argument(
        "--finetune_lr", type=float, default=5e-6,
        help="Learning rate for finetuning the encoder"
    )
    
    parser.add_argument(
        "--finetune_layers", type=int, default=2,
        help="Number of layers to finetune from the end of the encoder"
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


def setup_encoder_finetuning(wm, args):
    """
    Set up encoder finetuning by freezing/unfreezing appropriate layers
    and creating optimizer for encoder parameters.
    """
    if not args.with_finetune:
        return None
    
    # First, freeze all encoder parameters
    for param in wm.encoder.parameters():
        param.requires_grad = False
    
    # Identify which layers to unfreeze based on encoder type
    encoder_layers = []
    
    if args.dino_encoder == "dino":
        # For DINO, typically finetune the last transformer blocks
        if hasattr(wm.encoder, 'blocks'):
            encoder_layers = list(wm.encoder.blocks)[-args.finetune_layers:]
        elif hasattr(wm.encoder, 'transformer'):
            if hasattr(wm.encoder.transformer, 'layers'):
                encoder_layers = list(wm.encoder.transformer.layers)[-args.finetune_layers:]
    
    elif args.dino_encoder == "r3m":
        # For R3M, finetune the last convolutional or transformer layers
        if hasattr(wm.encoder, 'convnet'):
            # If it's a CNN-based R3M
            conv_layers = []
            for name, module in wm.encoder.convnet.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                    conv_layers.append(module)
            encoder_layers = conv_layers[-args.finetune_layers:]
        elif hasattr(wm.encoder, 'transformer'):
            encoder_layers = list(wm.encoder.transformer.layers)[-args.finetune_layers:]
    
    elif args.dino_encoder == "resnet":
        # For ResNet, finetune the last residual blocks
        if hasattr(wm.encoder, 'layer4'):
            encoder_layers = [wm.encoder.layer4]
            if args.finetune_layers > 1 and hasattr(wm.encoder, 'layer3'):
                encoder_layers.insert(0, wm.encoder.layer3)
        elif hasattr(wm.encoder, 'layers'):
            encoder_layers = list(wm.encoder.layers)[-args.finetune_layers:]
    
    elif args.dino_encoder == "vc1":
        # For VC1, finetune the last transformer blocks
        if hasattr(wm.encoder, 'blocks'):
            encoder_layers = list(wm.encoder.blocks)[-args.finetune_layers:]
        elif hasattr(wm.encoder, 'transformer'):
            if hasattr(wm.encoder.transformer, 'layers'):
                encoder_layers = list(wm.encoder.transformer.layers)[-args.finetune_layers:]
    
    elif args.dino_encoder == "scratch":
        # For scratch model, finetune the last layers
        if hasattr(wm.encoder, 'layers'):
            encoder_layers = list(wm.encoder.layers)[-args.finetune_layers:]
        elif hasattr(wm.encoder, 'blocks'):
            encoder_layers = list(wm.encoder.blocks)[-args.finetune_layers:]
    
    elif args.dino_encoder == "dino_cls":
        # For DINO with classification head, finetune last blocks + classifier
        if hasattr(wm.encoder, 'blocks'):
            encoder_layers = list(wm.encoder.blocks)[-args.finetune_layers:]
        if hasattr(wm.encoder, 'head'):
            encoder_layers.append(wm.encoder.head)
    
    # Unfreeze the selected layers
    trainable_params = []
    for layer in encoder_layers:
        for param in layer.parameters():
            param.requires_grad = True
            trainable_params.append(param)
    
    # Create optimizer for encoder parameters
    if trainable_params:
        encoder_optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=args.finetune_lr,
            weight_decay=args.weight_decay_pyhj if hasattr(args, 'weight_decay_pyhj') else 1e-4
        )
        
        print(f"Encoder finetuning enabled:")
        print(f"  - Encoder: {args.dino_encoder}")
        print(f"  - Learning rate: {args.finetune_lr}")
        print(f"  - Layers to finetune: {args.finetune_layers}")
        num_trainable_params = sum(p.numel() for p in trainable_params)
        print(f"  - Trainable parameters: {num_trainable_params}")

        
        return encoder_optimizer
    else:
        print("Warning: No encoder layers found for finetuning")
        return None


def load_shared_world_model(ckpt_dir: str, device: str):
    """Load a single world model to be shared across all environments"""
    ckpt_dir = Path(ckpt_dir)
    hydra_cfg = ckpt_dir / 'hydra.yaml'
    snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
    
    train_cfg = OmegaConf.load(str(hydra_cfg))
    num_action_repeat = train_cfg.num_action_repeat
    
    wm = load_model(snapshot, train_cfg, num_action_repeat, device=device)
    wm.eval()
    print(f"Loaded shared world model from {ckpt_dir}")
    return wm


class LatentDubinsEnv(gym.Env):
    """
    Wraps the classic Gym-based DubinsEnv into a Gymnasium-compatible Env.
    Encodes observations into DINO-WM latent space and uses info['h'] as reward.
    """
    def __init__(self, shared_wm=None, ckpt_dir: str = None, device: str = None, with_proprio: bool = False):
        super().__init__()
        # underlying Gym env
        self.env = DubinsEnv()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.with_proprio = with_proprio
        
        # Use shared world model if provided, otherwise load new one
        if shared_wm is not None:
            self.wm = shared_wm
            print("Using shared world model")
        else:
            # Load world model (only if not shared)
            if ckpt_dir is None:
                raise ValueError("Either shared_wm or ckpt_dir must be provided")
            ckpt_dir = Path(ckpt_dir)
            hydra_cfg = ckpt_dir / 'hydra.yaml'
            snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
            # load train config and model weights
            train_cfg = OmegaConf.load(str(hydra_cfg))
            num_action_repeat = train_cfg.num_action_repeat
            self.wm = load_model(snapshot, train_cfg, num_action_repeat, device=self.device)
            self.wm.eval()
            print(f"Loaded new world model from {ckpt_dir}")
        
        # probe a reset to set spaces
        reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        
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
        
        # Use gradients if encoder is being finetuned
        context = torch.no_grad() if not any(p.requires_grad for p in self.wm.encoder.parameters()) else torch.enable_grad()
        
        with context:
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
                z_prop = z_prop.squeeze(0)
                
                # Concatenate both visual and proprio embeddings
                z = torch.cat([z_vis, z_prop], dim=-1)  # torch.Size([1, 75274])
                
                # Always convert to numpy, regardless of gradient mode
                z = z.squeeze(0).detach().cpu().numpy()  # torch.size(75274,)
                return z
            
            else:
                z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim) torch.Size([1, 75264])
                # Always convert to numpy, regardless of gradient mode
                z_vis = z_vis.squeeze(0)
                return z_vis


# Set up matplotlib config
import os
os.environ['MPLCONFIGDIR'] = '/storage1/fs1/sibai/Active/ihab/tmp'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib.pyplot as plt
import wandb

def compute_hj_value(x, y, theta, policy, helper_env, args):
    """
    Compute the Hamilton–Jacobi filter value in *latent* space:
      1) Reset to (x,y,theta), get raw_obs from helper_env
      2) Encode raw_obs -> z
      3) Q = critic(z, actor_old(z))
      4) return Q (we removed the min(Q, h_s) part)
    """
    # Set precise state without advancing dynamics
    obs_dict, _ = helper_env.env.reset(state=[x, y, theta])
    z0 = helper_env._encode(obs_dict)
    batch = Batch(obs=z0[None], info=Batch())
    a_old = policy(batch, model="actor_old").act
    q_val = policy.critic(batch.obs, a_old).cpu().item()
    return q_val

def plot_hj(policy, helper_env, thetas, args):
    """
    Plot the Hamilton–Jacobi safety filter in latent space:
    - Rows: different θ slices
    - Col1: binary (Q > 0)
    - Col2: continuous Q values
    """
    xs = np.linspace(args.x_min, args.x_max, args.nx)
    ys = np.linspace(args.y_min, args.y_max, args.ny)
    
    if len(thetas) == 1:
        fig1, axes1 = plt.subplots(1, 1, figsize=(6, 6))
        fig2, axes2 = plt.subplots(1, 1, figsize=(6, 6))
        axes1 = [axes1]
        axes2 = [axes2]
    else:
        fig1, axes1 = plt.subplots(len(thetas), 1, figsize=(6, 6*len(thetas)))
        fig2, axes2 = plt.subplots(len(thetas), 1, figsize=(6, 6*len(thetas)))

    for i, theta in enumerate(thetas):
        vals = np.zeros((args.nx, args.ny), dtype=np.float32)
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                vals[ix, iy] = compute_hj_value(x, y, theta, policy, helper_env, args)

        # Binary safe/unsafe
        axes1[i].imshow(
            (vals.T > 0),
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower",
            cmap='RdYlBu'
        )
        axes1[i].set_title(f"θ={theta:.2f} (safe mask)")
        axes1[i].set_xlabel("x")
        axes1[i].set_ylabel("y")

        # Continuous value
        im = axes2[i].imshow(
            vals.T,
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower",
            cmap='viridis'
        )
        axes2[i].set_title(f"θ={theta:.2f} (HJ value)")
        axes2[i].set_xlabel("x")
        axes2[i].set_ylabel("y")
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
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 2) Load shared world model once
    shared_wm = load_shared_world_model(args.dino_ckpt_dir, args.device)
    
    # 3) Setup encoder finetuning
    encoder_optimizer = setup_encoder_finetuning(shared_wm, args)
    
    # 4) init W&B + TB writer + logger
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    wandb.init(
        project=f"ddpg-hj-latent-dubins", 
        name=f"ddpg-{args.dino_encoder}-{timestamp}",
        config=vars(args)
    )
    writer = SummaryWriter(log_dir=f"runs/ddpg_hj_latent/{args.dino_encoder}-{timestamp}/logs")
    wb_logger = WandbLogger()
    wb_logger.load(writer)
    logger = wb_logger

    # 5) Create environments with shared world model
    train_envs = DummyVectorEnv([
        lambda: LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio)
        for _ in range(args.training_num)
    ])
    
    test_envs = DummyVectorEnv([
        lambda: LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio)
        for _ in range(args.test_num)
    ])

    # 6) extract shapes & max_action
    state_space  = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    state_shape  = state_space.shape
    action_shape = action_space.shape or action_space.n
    max_action   = torch.tensor(action_space.high,
                                device=args.device,
                                dtype=torch.float32)

    # 7) build critic + actor
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

    # 8) assemble your avoid‐DDPG policy
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

    # 9) hook into policy.learn to capture losses and update encoder
    orig_learn = policy.learn
    policy.last_actor_loss  = 0.0
    policy.last_critic_loss = 0.0
    policy.last_encoder_loss = 0.0
    
    def learn_and_record(batch, **kw):
        metrics = orig_learn(batch, **kw)
        policy.last_actor_loss  = metrics["loss/actor"]
        policy.last_critic_loss = metrics["loss/critic"]
        
        # Update encoder if finetuning is enabled
        if encoder_optimizer is not None:
            # Compute encoder loss (use critic loss as proxy for encoder performance)
            encoder_loss = policy.last_critic_loss
            
            # Backpropagate through encoder
            encoder_optimizer.zero_grad()
            encoder_loss.backward(retain_graph=True)
            encoder_optimizer.step()
            
            policy.last_encoder_loss = float(encoder_loss)
        
        return metrics
    
    policy.learn = learn_and_record

    # 10) define train_fn to log those to W&B
    def train_fn(epoch: int, step_idx: int):
        log_dict = {
            "loss/actor":  policy.last_actor_loss,
            "loss/critic": policy.last_critic_loss,
        }
        
        if encoder_optimizer is not None:
            log_dict["loss/encoder"] = policy.last_encoder_loss
            log_dict["finetune/learning_rate"] = encoder_optimizer.param_groups[0]['lr']
        
        wandb.log(log_dict)

    # 11) collectors
    buffer          = VectorReplayBuffer(args.buffer_size, args.training_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    print("collecting some data first")
    train_collector.collect(10)
    print("done collecting some data first")
    test_collector  = Collector(policy, test_envs)

    # 12) choose headings & helper env (also uses shared model)
    thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    helper_env = LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio)

    # 13) training loop 
    log_path = Path(f"runs/ddpg_hj_latent/{args.dino_encoder}-{timestamp}")
    for epoch in range(1, args.total_episodes + 1):
        print(f"\n=== Epoch {epoch}/{args.total_episodes} ===")

        # Set encoder to training mode if finetuning
        if encoder_optimizer is not None:
            shared_wm.encoder.train()
        else:
            shared_wm.encoder.eval()

        # a) one epoch of offpolicy_trainer
        stats = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=None,  # test_collector,
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
        for k, v in stats.items():
            if isinstance(v, (int, float)): 
                numeric[f"train/{k}"] = v
            elif isinstance(v, np.generic): 
                numeric[f"train/{k}"] = float(v)
        wandb.log(numeric, step=epoch)

        # c) save policy checkpoint (including encoder state if finetuning)
        ckpt_dir = log_path / f"epoch_id_{epoch}"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        
        # Save policy
        torch.save(policy.state_dict(), ckpt_dir/"policy.pth")
        
        # Save encoder state if finetuning
        if encoder_optimizer is not None:
            torch.save({
                'encoder_state_dict': shared_wm.encoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            }, ckpt_dir/"encoder_finetune.pth")

        # d) plot latent‐space HJ filter & log
        try:
            # Set encoder to eval mode for plotting
            shared_wm.encoder.eval()
            
            fig1, fig2 = plot_hj(policy, helper_env, thetas, args)
            wandb.log({
                "HJ_latent/binary":     wandb.Image(fig1),
                "HJ_latent/continuous": wandb.Image(fig2),
            }, step=epoch)
            plt.close(fig1)
            plt.close(fig2)
        except Exception as e:
            print(f"Error plotting HJ values: {e}")

    print("Training complete.")

if __name__ == "__main__":
    main()
    
    
    
# Usage examples:
# For fine-tuning with proprioception:
# python "train_HJ_dubinslatent(can_fine_tune_PVR)2.py" --with_proprio --with_finetune --dino_encoder dino --finetune_lr 1e-5 --finetune_layers 3

# For fine-tuning without proprioception:
# python "train_HJ_dubinslatent(can_fine_tune_PVR)2.py" --with_finetune --dino_encoder r3m --finetune_lr 5e-6 --finetune_layers 2

# For progressive unfreezing:
# python "train_HJ_dubinslatent(can_fine_tune_PVR)2.py" --with_finetune --progressive_unfreezing --unfreeze_schedule "25,50,75" --dino_encoder vc1

# For comparison without fine-tuning:
# python "train_HJ_dubinslatent(can_fine_tune_PVR)2.py" --dino_encoder dino