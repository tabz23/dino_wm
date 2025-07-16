import argparse
import os
from pathlib import Path
import torch
import gymnasium as gym  # We use Gymnasium API for vector envs
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from typing import Any, Dict, List, Union, Optional, Tuple
import copy
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
        help="Which encoder to use: dino, r3m, vc1, resnet, dino_cls, etc."
    )
    
    parser.add_argument(
        "--with_finetune", action="store_true",
        help="Flag to enable fine-tuning of the vision encoder"
    )
    
    parser.add_argument(
        "--encoder_lr", type=float, default=1e-5,
        help="Learning rate for encoder fine-tuning"
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


def load_shared_world_model(ckpt_dir: str, device: str):
    """Load a single world model to be shared across all environments"""
    ckpt_dir = Path(ckpt_dir)
    hydra_cfg = ckpt_dir / 'hydra.yaml'
    snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
    
    train_cfg = OmegaConf.load(str(hydra_cfg))
    num_action_repeat = train_cfg.num_action_repeat
    
    wm = load_model(snapshot, train_cfg, num_action_repeat, device=device)
    print(f"Loaded shared world model from {ckpt_dir}")
    return wm

class LatentDubinsEnv(gym.Env):
    """
    Wraps the classic Gym-based DubinsEnv into a Gymnasium-compatible Env.
    Encodes observations into DINO-WM latent space and uses info['h'] as reward.
    """
    def __init__(self, shared_wm=None, ckpt_dir: str = None, device: str = None, 
                 with_proprio: bool = False, with_finetune: bool = False):
        super().__init__()
        # underlying Gym env
        self.env = DubinsEnv()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.with_proprio = with_proprio
        self.finetune_mode = with_finetune
        
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
        print("using finetune:", self.finetune_mode)
        z = self._encode(obs)
        
        # Always convert to numpy for observation space definition
        if torch.is_tensor(z):
            obs_shape = z.shape
            z_numpy = z.detach().cpu().numpy()
        else:
            z_numpy = z
            obs_shape = z.shape
        
        print(f"Example latent state z shape: {obs_shape}")
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = self.env.action_space

    def reset(self):
        """
        Reset underlying Gym env and encode obs to latent.
        Returns: (obs_latent, info_dict)
        
        NOTE: Always returns numpy arrays for compatibility with PyHJ framework,
        even in finetune mode. The TensorAwareReplayBuffer will handle conversion.
        """
        reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        z = self._encode(obs)
        
        # Always convert to numpy for PyHJ compatibility
        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()
        
        return z, {}

    def step(self, action):
        """
        Step in Gym env: returns (obs_latent, reward, terminated, truncated, info).
        Classic Gym returns (obs, reward, done, info).
        We map done->terminated and truncated=False.
        Reward is taken from info['h'].
        
        NOTE: Always returns numpy arrays for compatibility with PyHJ framework,
        even in finetune mode. The TensorAwareReplayBuffer will handle conversion.
        """
        obs_out, _, done, info = self.env.step(action)
        terminated = done
        truncated = False
        # extract obs if tuple
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        # override reward with safety metric
        h_s = info.get('h', 0.0) * 3 ##I multiplied by 3 to make HJ easier to learn
        z_next = self._encode(obs)
        
        # Always convert to numpy for PyHJ compatibility
        if torch.is_tensor(z_next):
            z_next = z_next.detach().cpu().numpy()
        
        return z_next, h_s, terminated, truncated, info

    def _encode(self, obs):
        """
        Encode raw obs via DINO-WM into a flat latent vector.
        Supports obs as dict or tuple (visual, proprio).
        
        In finetune mode, keeps gradients enabled and returns tensors.
        In normal mode, disables gradients and can return numpy arrays.
        """
        # unpack obs
        if isinstance(obs, dict):
            visual = obs['visual']
            proprio = obs['proprio']
        elif isinstance(obs, (tuple, list)) and len(obs) == 2:
            visual, proprio = obs
        else:
            raise ValueError(f"Unexpected obs type: {type(obs)}")
        
        # Use gradients if encoder is being finetuned or if we're in finetune mode
        use_gradients = self.finetune_mode or any(p.requires_grad for p in self.wm.encoder.parameters())
        context = torch.enable_grad() if use_gradients else torch.no_grad()
        
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
            if self.with_proprio:
                z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches, E_dim) -> (1, N_patches*E_dim)
                z_prop = lat['proprio']  # (1, D_prop)
                
                # flatten visual patches and concatenate proprio
                z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim) torch.Size([1, 75264])
                z_prop = lat['proprio']  # (1, 1, D_prop) ([1, 1, 10])
                z_prop = z_prop.squeeze(0)
                
                # Concatenate both visual and proprio embeddings
                z = torch.cat([z_vis, z_prop], dim=-1)  # torch.Size([1, 75274])
                z = z.squeeze(0)  # torch.size(75274,)
                
                return z
            
            else:
                z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim) torch.Size([1, 75264])
                z_vis = z_vis.squeeze(0)
                
                return z_vis


class TensorAwareReplayBuffer(VectorReplayBuffer):
    def __init__(
        self,
        size: int,
        buffer_num: int,
        stack_num: int = 1,
        ignore_obs_next: bool = True,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
        device: str = "cuda",
        finetune_mode: bool = False,
        shared_wm=None,
    ):
        super().__init__(
            total_size=size,
            buffer_num=buffer_num,
            stack_num=stack_num,
            ignore_obs_next=ignore_obs_next,
            save_only_last_obs=save_only_last_obs,
            sample_avail=sample_avail,
        )
        self.device = torch.device(device)
        self.finetune_mode = finetune_mode
        self.shared_wm = shared_wm

    def add(self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Override add to handle observations properly.
        Always stores as numpy arrays regardless of finetune mode.
        Returns: (ptr, ep_rew, ep_len, ep_idx) as expected by Collector.
        """
        # Convert any tensor observations to numpy for storage
        batch_copy = copy.deepcopy(batch)
        if hasattr(batch_copy, 'obs') and torch.is_tensor(batch_copy.obs):
            batch_copy.obs = batch_copy.obs.detach().cpu().numpy()
        if hasattr(batch_copy, 'obs_next') and torch.is_tensor(batch_copy.obs_next):
            batch_copy.obs_next = batch_copy.obs_next.detach().cpu().numpy()

        # Call parent add method and capture returned values
        ptr, ep_rew, ep_len, ep_idx = super().add(batch_copy, buffer_ids=buffer_ids)

        return ptr, ep_rew, ep_len, ep_idx

    def sample(self, batch_size: int) -> Batch:
        """
        Override sample to convert back to tensors when in finetune mode.
        This is where we handle the tensor conversion for training.
        """
        batch = super().sample(batch_size)

        if self.finetune_mode:
            # Convert numpy arrays back to tensors for gradient computation
            if hasattr(batch, 'obs') and isinstance(batch.obs, np.ndarray):
                batch.obs = torch.from_numpy(batch.obs).to(self.device).requires_grad_(True)
            if hasattr(batch, 'obs_next') and isinstance(batch.obs_next, np.ndarray):
                batch.obs_next = torch.from_numpy(batch.obs_next).to(self.device).requires_grad_(True)
            if hasattr(batch, 'act') and isinstance(batch.act, np.ndarray):
                batch.act = torch.from_numpy(batch.act).to(self.device)
            if hasattr(batch, 'rew') and isinstance(batch.rew, np.ndarray):
                batch.rew = torch.from_numpy(batch.rew).to(self.device)
            if hasattr(batch, 'done') and isinstance(batch.done, np.ndarray):
                batch.done = torch.from_numpy(batch.done).to(self.device)

        return batch
class TensorAwareCollector(Collector):
    """
    Modified collector that can handle tensor observations from the environment
    and convert them appropriately for the buffer.
    """
    
    def __init__(self, policy, env, buffer=None, preprocess_fn=None, exploration_noise=False, finetune_mode=False):
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)
        self.finetune_mode = finetune_mode
        
    def collect(self, n_step: int = 0, n_episode: int = 0, random: bool = False, 
                render: float = 0.0, no_grad: bool = True) -> Dict[str, Any]:
        """
        Override collect to handle tensor observations properly.
        The key insight is that we need gradients during policy execution in finetune mode,
        but the environment always returns numpy arrays for PyHJ compatibility.
        """
        
        # In finetune mode, we need gradients for policy execution
        if self.finetune_mode and not random:
            no_grad = False
            
        return super().collect(n_step, n_episode, random, render, no_grad)


class TensorAwareDDPGPolicy(avoid_DDPGPolicy_annealing):
    """
    Modified DDPG policy that can handle tensor observations for encoder finetuning.
    """
    
    def __init__(self, actor, critic, actor_optim, critic_optim, finetune_mode=False, **kwargs):
        print("actor:", actor)
        print("critic:", critic)
        print("actor_optim:", actor_optim)
        print("critic_optim:", critic_optim)
        print("kwargs:", kwargs)
        # Extract required arguments from kwargs with defaults
        tau = kwargs.pop('tau', 0.005)
        gamma = kwargs.pop('gamma', 0.99)
        exploration_noise = kwargs.pop('exploration_noise', GaussianNoise(sigma=0.1))
        estimation_step = kwargs.pop('estimation_step', 1)
        reward_normalization = kwargs.pop('reward_normalization', False)
        action_scaling = kwargs.pop('action_scaling', True)
        action_bound_method = kwargs.pop('action_bound_method', 'clip')
        actor_gradient_steps = kwargs.pop('actor_gradient_steps', 5)
        
        # Call parent __init__ with correct argument order
        super().__init__(
            critic=critic,
            critic_optim=critic_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=exploration_noise,
            reward_normalization=reward_normalization,
            estimation_step=estimation_step,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            actor=actor,
            actor_optim=actor_optim,
            actor_gradient_steps=actor_gradient_steps,
            **kwargs
        )
        self.finetune_mode = finetune_mode
        self.device='cuda'
        
    def process_fn(self, batch: Batch, buffer: TensorAwareReplayBuffer, indices: np.ndarray) -> Batch:
        """
        Override process_fn to handle tensor observations properly.
        In finetune mode, observations should already be tensors from the buffer.
        """
        if self.finetune_mode:
            # Ensure observations are tensors with gradients
            if hasattr(batch, 'obs') and not torch.is_tensor(batch.obs):
                batch.obs = torch.from_numpy(batch.obs).to(self.device).requires_grad_(True)
            elif hasattr(batch, 'obs') and torch.is_tensor(batch.obs) and not batch.obs.requires_grad:
                batch.obs = batch.obs.requires_grad_(True)
                
            if hasattr(batch, 'obs_next') and not torch.is_tensor(batch.obs_next):
                batch.obs_next = torch.from_numpy(batch.obs_next).to(self.device).requires_grad_(True)
            elif hasattr(batch, 'obs_next') and torch.is_tensor(batch.obs_next) and not batch.obs_next.requires_grad:
                batch.obs_next = batch.obs_next.requires_grad_(True)
        
        return super().process_fn(batch, buffer, indices)
    
    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None,
                model: str = "actor", **kwargs) -> Batch:
        """
        Override forward to handle tensor observations in finetune mode.
        """
        if self.finetune_mode and hasattr(batch, 'obs'):
            # Ensure observations are tensors for forward pass
            if not torch.is_tensor(batch.obs):
                batch.obs = torch.from_numpy(batch.obs).to(self.device)
            elif batch.obs.device != self.device:
                batch.obs = batch.obs.to(self.device)
        
        return super().forward(batch, state, model, **kwargs)
    
class FineTunablePolicy(avoid_DDPGPolicy_annealing):
    """
    Extended policy that re-encodes observations during training for fine-tuning.
    """
    def __init__(self, shared_wm=None, with_finetune=False, with_proprio=False, device='cuda', **kwargs):
        super().__init__(**kwargs)
        self.shared_wm = shared_wm
        self.with_finetune = with_finetune
        self.with_proprio = with_proprio
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Set encoder mode
        if self.shared_wm is not None:
            if self.with_finetune:
                self.shared_wm.train()
            else:
                self.shared_wm.eval()
    
    def _encode_raw_obs(self, raw_obs):
        """
        Re-encode raw observations with gradient flow for fine-tuning.
        """
        if self.shared_wm is None:
            raise ValueError("Shared world model not available for re-encoding")
        
        # unpack obs
        if isinstance(raw_obs, dict):
            visual = raw_obs['visual']
            proprio = raw_obs['proprio']
        elif isinstance(raw_obs, (tuple, list)) and len(raw_obs) == 2:
            visual, proprio = raw_obs
        else:
            raise ValueError(f"Unexpected obs type: {type(raw_obs)}")
        
        # prepare tensors
        visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32)  # (C, H, W)
        visual_np /= 255.0  # normalize to [0, 1]
        vis_t = torch.from_numpy(visual_np).unsqueeze(0)  # -> (1, C, H, W)
        vis_t = vis_t.unsqueeze(1)  # Add time dimension (1, 1, C, H, W)
        vis_t = vis_t.to(self.device)

        prop_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)
        prop_t = prop_t.unsqueeze(1)  # Add singleton dimension (1, 1, D_prop)
        
        data = {'visual': vis_t, 'proprio': prop_t}
        
        lat = self.shared_wm.encode_obs(data)
        
        # flatten visual patches and concat proprio
        if self.with_proprio:
            z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim)
            z_prop = lat['proprio'].squeeze(0)  # (1, D_prop)
            z = torch.cat([z_vis, z_prop], dim=-1)  # (1, total_dim)
            return z.squeeze(0)  # (total_dim,)
        else:
            z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim)
            return z_vis.squeeze(0)  # (N_patches * E_dim,)
    
    def learn(self, batch, **kwargs):
        """
        Override learn to re-encode observations if fine-tuning is enabled.
        """
        if self.with_finetune and self.shared_wm is not None:
            # Re-encode observations with gradient flow
            # Check if batch has raw_obs in info
            if hasattr(batch, 'info') and hasattr(batch.info, 'raw_obs'):
                batch_size = len(batch.obs)
                re_encoded_obs = []
                re_encoded_obs_next = []
                
                for i in range(batch_size):
                    # Re-encode current observation
                    raw_obs = batch.info.raw_obs[i]
                    if raw_obs is not None:
                        z = self._encode_raw_obs(raw_obs)
                        re_encoded_obs.append(z)
                    else:
                        # Fallback to original obs if raw_obs not available
                        re_encoded_obs.append(torch.from_numpy(batch.obs[i]).to(self.device))
                    
                    # Re-encode next observation (if available)
                    # Note: This is a simplified version. In practice, you might need to store
                    # raw_obs_next in the environment as well
                    re_encoded_obs_next.append(torch.from_numpy(batch.obs_next[i]).to(self.device))
                
                # Stack re-encoded observations
                if re_encoded_obs:
                    batch.obs = torch.stack(re_encoded_obs)
                    batch.obs_next = torch.stack(re_encoded_obs_next)
        
        # Call parent learn method
        return super().learn(batch, **kwargs)


# Set up matplotlib config
import os
os.environ['MPLCONFIGDIR'] = '/storage1/fs1/sibai/Active/ihab/tmp'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib.pyplot as plt
import wandb

def compute_hj_value(x, y, theta, policy, helper_env, args):
    """
    Compute the Hamilton-Jacobi filter value in latent space with strict device handling.
    """
    # Set precise state without advancing dynamics
    obs_dict, _ = helper_env.env.reset(state=[x, y, theta])
    z0 = helper_env._encode(obs_dict)
    
    # Ensure z0 is a tensor on the correct device
    if isinstance(z0, np.ndarray):
        z0 = torch.from_numpy(z0).float().to(args.device)
    elif isinstance(z0, torch.Tensor):
        z0 = z0.float().to(args.device)
    else:
        raise ValueError(f"Unexpected type for z0: {type(z0)}")
    
    # Create batch and ensure all components are on the right device
    batch = Batch(
        obs=z0.unsqueeze(0).to(args.device),  # Add batch dimension
        info=Batch()
    )
    
    # Ensure policy networks are on the correct device
    policy.actor_old.to(args.device)
    policy.critic.to(args.device)
    
    with torch.no_grad():
        # Get action from actor (old version)
        a_old = policy(batch, model="actor_old").act
        if isinstance(a_old, torch.Tensor):
            a_old = a_old.to(args.device)
        
        # Compute Q-value
        obs = batch.obs.to(args.device)
        q_val = policy.critic(obs, a_old)
        return q_val.cpu().item()
def plot_hj(policy, helper_env, thetas, args):
    """
    Plot the Hamilton-Jacobi safety filter in latent space with robust error handling.
    """
    xs = np.linspace(args.x_min, args.x_max, args.nx)
    ys = np.linspace(args.y_min, args.y_max, args.ny)
    
    # Create figures
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
                try:
                    # Print progress for debugging
                    print(f"Computing HJ value at x={x:.2f}, y={y:.2f}, theta={theta:.2f}")
                    
                    #Verify device consistency before computation
                    print(f"Policy device - actor_old: {next(policy.actor_old.parameters()).device}")
                    print(f"Policy device - critic: {next(policy.critic.parameters()).device}")
                    
                    val = compute_hj_value(x, y, theta, policy, helper_env, args)
                    vals[ix, iy] = val
                except Exception as e:
                    print(f"Error computing HJ value at x={x:.2f}, y={y:.2f}, theta={theta:.2f}: {str(e)}")
                    vals[ix, iy] = 0  # Default value
                    # Print stack trace for debugging
                    import traceback
                    traceback.print_exc()

        # Plot binary safe/unsafe
        axes1[i].imshow(
            (vals.T > 0),
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower",
            cmap='RdYlBu'
        )
        axes1[i].set_title(f"θ={theta:.2f} (safe mask)")
        axes1[i].set_xlabel("x")
        axes1[i].set_ylabel("y")

        # Plot continuous values
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
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 2) Load shared world model once
    shared_wm = load_shared_world_model(args.dino_ckpt_dir, args.device)
    
    # 3) init W&B + TB writer + logger
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_name = f"ddpg-{args.dino_encoder}-{timestamp}"
    if args.with_finetune:
        run_name += "-finetune"
    
    wandb.init(
        project=f"ddpg-hj-latent-dubins", 
        name=run_name,
        config=vars(args)
    )
    writer = SummaryWriter(log_dir=f"runs/ddpg_hj_latent/{run_name}/logs")
    wb_logger = WandbLogger()
    wb_logger.load(writer)
    logger = wb_logger

    # 4) Create environments with shared world model
    train_envs = DummyVectorEnv([
        lambda: LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio, 
                               with_finetune=args.with_finetune, device=args.device)
        for _ in range(args.training_num)
    ])
    
    test_envs = DummyVectorEnv([
        lambda: LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio, 
                               with_finetune=False, device=args.device)  # No finetune for test
        for _ in range(args.test_num)
    ])

    # 5) extract shapes & max_action
    state_space  = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    state_shape  = state_space.shape
    action_shape = action_space.shape or action_space.n
    max_action   = torch.tensor(action_space.high,
                                device=args.device,
                                dtype=torch.float32)

    # 6) build critic + actor
    critic_net = Net(state_shape, action_shape,
                     hidden_sizes=args.critic_net,
                     activation=getattr(torch.nn, args.critic_activation),
                     concat=True, device=args.device)
    critic = Critic(critic_net, device=args.device)
    

    actor_net   = Net(state_shape,
                      hidden_sizes=args.control_net,
                      activation=getattr(torch.nn, args.actor_activation),
                      device=args.device)
    actor       = Actor(actor_net, action_shape,
                        max_action=max_action,
                        device=args.device).to(args.device)

# 7) Create encoder optimizer if fine-tuning
    encoder_optimizer = torch.optim.Adam(shared_wm.encoder.parameters(), lr=args.encoder_lr) if args.with_finetune else None

    # 8) build DDPG policy
    policy = TensorAwareDDPGPolicy(
        actor=actor,
        critic=critic,
        actor_optim=torch.optim.Adam(actor.parameters(), lr=args.actor_lr),
        critic_optim=torch.optim.Adam(critic.parameters(), lr=args.critic_lr),
        tau=args.tau,
        gamma=args.gamma_pyhj,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        estimation_step=args.n_step,
        finetune_mode=args.with_finetune
    )

    # 9) Create tensor-aware buffer
    buffer     = TensorAwareReplayBuffer(
        size=args.buffer_size,
        buffer_num=args.training_num,
        device=args.device,
        finetune_mode=args.with_finetune,
        )

    # 10) Create collectors
    train_collector = Collector(
        policy, train_envs, buffer, 
        exploration_noise=True
    )
    
    test_collector = Collector(
        policy, test_envs
    )

    # 11) Create a helper env for HJ plotting
    helper_env = LatentDubinsEnv(
        shared_wm=shared_wm, 
        with_proprio=args.with_proprio,
        with_finetune=args.with_finetune
    )

    # 12) Policy update function with encoder finetuning
    def policy_update_fn(policy, batch, encoder_optimizer=None):
        """Custom policy update function that handles encoder finetuning"""
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        
        # Regular policy update
        losses = policy.learn(batch)
        
        # If encoder finetuning is enabled, step the encoder optimizer
        if encoder_optimizer is not None:
            encoder_optimizer.step()
        
        return losses

    # 13) Custom trainer function with encoder finetuning
    def custom_trainer(policy, train_collector, test_collector, max_epoch, 
                      step_per_epoch, step_per_collect, test_num, batch_size, 
                      update_per_step, encoder_optimizer=None, logger=None):
        """Custom trainer that handles encoder finetuning"""
        
        # Pre-collect random samples
        train_collector.collect(n_step=args.buffer_size // 10, random=True,no_grad=False)
        
        global_step = 0
        
        for epoch in range(max_epoch):
            # Collect training data
            collect_result = train_collector.collect(n_step=step_per_collect, no_grad=False)
            global_step += step_per_collect
            
            # Update policy
            if len(train_collector.buffer) >= batch_size:
                for _ in range(int(step_per_collect * update_per_step)):
                    batch = train_collector.buffer.sample(batch_size)
                    losses = policy_update_fn(policy, batch, encoder_optimizer)
                    
                    # Log losses
                    if logger is not None:
                        for key, value in losses.items():
                            logger.log_scalar(f"train/{key}", value, global_step)
            
            # Test policy
            # if epoch % args.test_interval == 0:
            #     test_result = test_collector.collect(n_episode=test_num)
                
            #     # Log test results
            #     if logger is not None:
            #         logger.log_scalar("test/reward", test_result['rews'].mean(), global_step)
            #         logger.log_scalar("test/length", test_result['lens'].mean(), global_step)
                
            #     print(f"Epoch {epoch}: test_reward={test_result['rews'].mean():.3f}, "
            #           f"test_length={test_result['lens'].mean():.1f}")
                
            # Plot HJ values periodically
            # if epoch % args.plot_interval == 0:
            thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
            fig1, fig2 = plot_hj(policy, helper_env, thetas, args)
            
            # Log plots to wandb
            wandb.log({
                f"hj_safe_mask_epoch_{epoch}": wandb.Image(fig1),
                f"hj_values_epoch_{epoch}": wandb.Image(fig2)
            })
            
            plt.close(fig1)
            plt.close(fig2)
    
        return policy

    # 14) Run training
    print("Starting DDPG training...")
    print(f"State shape: {state_shape}")
    print(f"Action shape: {action_shape}")
    print(f"Max action: {max_action}")
    print(f"Encoder finetuning: {args.with_finetune}")
    if args.with_finetune:
        print(f"Encoder optimizer: {encoder_optimizer is not None}")
    
    # Calculate max epochs from total episodes
    max_epoch = args.total_episodes // args.step_per_epoch
    
    # Run custom trainer
    policy = custom_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=max_epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        test_num=args.test_num,
        batch_size=args.batch_size_pyhj,
        update_per_step=args.update_per_step,
        encoder_optimizer=encoder_optimizer,
        logger=logger
    )

    # 15) Final evaluation and plotting
    print("Training completed. Running final evaluation...")
    
    # Final test
    final_test_result = test_collector.collect(n_episode=args.test_num * 2)
    print(f"Final test results:")
    print(f"  Average reward: {final_test_result['rews'].mean():.3f}")
    print(f"  Average length: {final_test_result['lens'].mean():.1f}")
    
    # Final HJ plot
    thetas = [0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]
    fig1, fig2 = plot_hj(policy, helper_env, thetas, args)
    
    # Save final plots
    wandb.log({
        "final_hj_safe_mask": wandb.Image(fig1),
        "final_hj_values": wandb.Image(fig2),
        "final_test_reward": final_test_result['rews'].mean(),
        "final_test_length": final_test_result['lens'].mean()
    })
    
    plt.close(fig1)
    plt.close(fig2)
    
    # 16) Save model
    ckpt_dir = f"runs/ddpg_hj_latent/{run_name}/final"
    model_save_path = ckpt_dir
    os.makedirs("models", exist_ok=True)
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'args': vars(args),
        'final_test_reward': final_test_result['rews'].mean(),
        'final_test_length': final_test_result['lens'].mean()
    }, model_save_path)
    
    print(f"Model saved to: {model_save_path}")
    
    # Clean up
    wandb.finish()
    writer.close()
    
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
    
    # python "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent(can_fine_tune_PVR)5.py" \
    # --dino_ckpt_dir "/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs/dubins/fully trained(prop repeated 3 times)" \
    # --config train_HJ_configs.yaml \
    # --dino_encoder r3m \
    # --with_finetune \
    # --encoder_lr 1e-5 \
    # --total-episodes 100000
    # --step-per-epoch 1000
    