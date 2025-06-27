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
def parse_args():
    parser = argparse.ArgumentParser("DDPG HJ on DINO latent Dubins")
    parser.add_argument(
        "--dino_ckpt_dir", type=str, required=False, default="/storage1/sibai/Active/ihab/research_new/checkpt_dino/outputs/dubins",
        help="Directory containing DINO-WM outputs/<env> (hydra.yaml + checkpoints/)"
    )
    parser.add_argument(
        "--config", type=str, default="/storage1/sibai/Active/ihab/research_new/dino_wm/train_HJ_configs.yaml",
        help="Path to the YAML config file with all hyperparameters"
    )
    # You can still define a few overrides if you like:
    parser.add_argument("--gamma-pyhj", type=float, default=None,
                        help="(Optional) override gamma_pyhj from the config")
    return parser.parse_args()

def merge_config_into_args(args):
    # 1) load the YAML into a dict
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    # 2) copy each config key/value onto args (hyphens→underscores)
    for key, val in cfg.items():
        name = key.replace('-', '_')
        # only set if the user didn’t already give it explicitly via CLI
        if not hasattr(args, name) or getattr(args, name) is None:
            setattr(args, name, val)
    return args

def save_best_fn(policy, epoch=epoch):
    torch.save(
        policy.state_dict(), 
        os.path.join(
            log_path+"/epoch_id_{}".format(epoch),
            "policy.pth"
        )
    )

class LatentDubinsEnv(gym.Env):
    """
    Wraps the classic Gym-based DubinsEnv into a Gymnasium-compatible Env.
    Encodes observations into DINO-WM latent space and uses info['h'] as reward.
    """
    def __init__(self, ckpt_dir: str, device: str):
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
        z = self._encode(obs)
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
        h_s = info.get('h', 0.0)
        z_next = self._encode(obs)
        return z_next, h_s, terminated, truncated, info

    # def _encode(self, obs):
    #     """
    #     Encode raw obs via DINO-WM into a flat latent vector.
    #     Supports obs as dict or tuple (visual, proprio).
    #     """
    #     # unpack obs
    #     if isinstance(obs, dict):
    #         visual = obs['visual']
    #         proprio = obs['proprio']
    #     elif isinstance(obs, (tuple, list)) and len(obs) == 2:
    #         visual, proprio = obs
    #     else:
    #         raise ValueError(f"Unexpected obs type: {type(obs)}")
    #     with torch.no_grad():
    #         # prepare tensors
    #         # visual_np = np.transpose(visual, (2, 0, 1))       # (C, H, W)
    #         # vis_t      = torch.from_numpy(visual_np)           # -> (C, H, W)
    #         # vis_t      = vis_t.unsqueeze(0).to(self.device)    # -> (1, C, H, W)
            
    #         # visual_np = np.transpose(visual, (2, 0, 1))  # (C, H, W)
    #         # vis_t = torch.from_numpy(visual_np)          # -> (C, H, W)
    #         # vis_t = vis_t.unsqueeze(0)                   # -> (1, C, H, W)  your B=1
    #         # vis_t = vis_t.unsqueeze(1)                   # -> (1, 1, C, H, W)  dummy T=1
    #         # vis_t = vis_t.to(self.device)
    #         # bring HWC→CHW and to float32, scale to [0,1]
            
    #         visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32)  # (C,H,W)
    #         visual_np /= 255.0                                          # normalize
    #         vis_t = torch.from_numpy(visual_np).unsqueeze(0)             # (1,C,H,W) float32
    #         vis_t = vis_t.to(self.device)       
                    
    #         # prop_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)
            
    #         prop_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)

    #         data = {'visual': vis_t, 'proprio': prop_t}
    #         lat = self.wm.encode_obs(data)
    #         # flatten visual patches and concat proprio
    #         z_vis = lat['visual'].reshape(1, -1)# check this: shape: (1, N_patches, E_dim)→ (1, N_patches*E_dim)
    #         z_prop = lat['proprio']  # (1, D_prop)
    #         z = torch.cat([z_vis, z_prop], dim=-1)
    #         return z.squeeze(0).cpu().numpy()
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
            z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches, E_dim) -> (1, N_patches*E_dim)
            z_prop = lat['proprio']  # (1, D_prop)
            
            # flatten visual patches and concatenate proprio
            z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim) torch.Size([1, 75264])
            z_prop = lat['proprio']  # (1, 1, D_prop) ([1, 1, 10])
            z_prop=z_prop.squeeze(0)
            
            # Concatenate both visual and proprio embeddings
            z = torch.cat([z_vis, z_prop], dim=-1)#torch.Size([1, 75274])
            return z.squeeze(0).cpu().numpy()#torch.size(75274,)


def main():
    args = parse_args()
    args = merge_config_into_args(args)
    actor_h = [int(x) for x in args.actor_hidden.split(',')]
    critic_h = [int(x) for x in args.critic_hidden.split(',')]

    # create vectorized envs
    train_envs = DummyVectorEnv([lambda: LatentDubinsEnv(args.dino_ckpt_dir, args.device)
                                 for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: LatentDubinsEnv(args.dino_ckpt_dir, args.device)
                                for _ in range(args.test_num)])

    # state/action shapes
    # state_shape = train_envs.envs[0].observation_space.shape
    # action_space = train_envs.envs[0].action_space
    # action_shape = action_space.shape or action_space.n
    # max_action = action_space.high if hasattr(action_space, 'high') else None
    
    # state_shape  = train_envs.observation_space.shape
    # action_space = train_envs.action_space
    # action_shape = action_space.shape or action_space.n
    # max_action   = action_space.high if hasattr(action_space, 'high') else None

    state_space = train_envs.observation_space[0]   # Box for env #0  || state_space Box(-inf, inf, (75274,), float32)
    action_space = train_envs.action_space[0]       # likewise || action_space Box(-1.0, 1.0, (1,), float32)
    state_shape  = state_space.shape
    action_shape = action_space.shape or action_space.n
    max_action   = action_space.high if hasattr(action_space, 'high') else None


    # Build critic
    critic_net = Net(
        state_shape, action_shape,
        hidden_sizes=args.critic_net,
        activation=getattr(torch.nn, args.critic_activation),
        concat=True,
        device=args.device
    )
    critic       = Critic(critic_net, device=args.device).to(args.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.weight_decay_pyhj
    )

    # Build actor
    actor_net   = Net(
        state_shape,
        hidden_sizes=args.control_net,
        activation=getattr(torch.nn, args.actor_activation),
        device=args.device
    )
    actor       = Actor(actor_net, action_shape, max_action=max_action,
                        device=args.device).to(args.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)

    # Assemble policy
    policy = avoid_DDPGPolicy_annealing(
        critic=critic,
        critic_optim=critic_optim,
        tau=args.tau,
        gamma=args.gamma_pyhj,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=action_space,
        actor=actor,
        actor_optim=actor_optim,
        actor_gradient_steps=args.actor_gradient_steps,
    )

    # Collectors & trainer call
    buffer          = VectorReplayBuffer(args.buffer_size, args.training_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector  = Collector(policy, test_envs)

    writer    = SummaryWriter(log_dir="runs/ddpg_hj_dino")
    tb_logger = TensorboardLogger(writer)

    offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.total_episodes,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size_pyhj,
        update_per_step=args.update_per_step,
        stop_fn=lambda r: False,
        save_best_fn=None,
        logger=tb_logger,
    )

if __name__ == "__main__":
    main()