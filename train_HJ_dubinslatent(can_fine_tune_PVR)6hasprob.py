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
IndexType = Union[slice, int, np.ndarray, List[int]]
# ENCODER LOGGING FUNCTIONS
def log_encoder_stats(shared_wm, encoder_optimizer, epoch, global_step):
    """
    Log encoder gradient norms and parameter changes with minimal overhead
    """
    if encoder_optimizer is None or shared_wm is None:
        return
    
    # 1. Log gradient norms
    total_grad_norm = 0
    param_count = 0
    param_with_grad_count = 0
    
    for param in shared_wm.encoder.parameters():
        param_count += 1
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
            param_with_grad_count += 1
    
    if param_with_grad_count > 0:
        total_grad_norm = (total_grad_norm ** 0.5)
        
        # Log to wandb
        import wandb
        wandb.log({
            "encoder/gradient_norm": total_grad_norm,
            "encoder/params_with_grad": param_with_grad_count,
            "encoder/total_params": param_count,
            "encoder/learning_rate": encoder_optimizer.param_groups[0]['lr'],
            "epoch": epoch,
            "global_step": global_step
        })
        
        print(f"Encoder: grad_norm={total_grad_norm:.6f}, params_with_grad={param_with_grad_count}/{param_count}")
    else:
        print(f"WARNING: No encoder gradients found! Total params: {param_count}")

def create_param_tracker(shared_wm):
    """Create a simple parameter tracker"""
    if shared_wm is None or not hasattr(shared_wm, 'encoder'):
        return None
        
    # Store initial parameters
    initial_params = {}
    for name, param in shared_wm.encoder.named_parameters():
        initial_params[name] = param.data.clone()
    print(f"Created parameter tracker for {len(initial_params)} encoder parameters")
    return initial_params

def log_param_changes(shared_wm, initial_params, epoch, global_step):
    """Log parameter changes since initialization"""
    if initial_params is None or shared_wm is None:
        return
        
    total_change = 0
    param_count = 0
    
    for name, param in shared_wm.encoder.named_parameters():
        if name in initial_params:
            change = (param.data - initial_params[name]).norm().item()
            total_change += change
            param_count += 1
    
    if param_count > 0:
        avg_change = total_change / param_count
        import wandb
        wandb.log({
            "encoder/avg_param_change": avg_change,
            "encoder/total_param_change": total_change,
            "epoch": epoch,
            "global_step": global_step
        })
        print(f"Encoder param changes: avg={avg_change:.8f}, total={total_change:.6f}")

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

    # DEBUG: Print final fine-tuning status
    print(f"=== ARGUMENT PARSING COMPLETE ===")
    print(f"args.with_finetune: {args.with_finetune} (type: {type(args.with_finetune)})")
    print(f"args.encoder_lr: {args.encoder_lr}")
    print(f"args.dino_encoder: {args.dino_encoder}")
    print("=================================")

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
    When with_finetune=True, returns raw observations for gradient flow.
    Otherwise, encodes observations into DINO-WM latent space.
    """
    def __init__(self, shared_wm=None, ckpt_dir: str = None, device: str = None, 
                 with_proprio: bool = False, with_finetune: bool = False):
        super().__init__()
        # underlying Gym env
        self.env = DubinsEnv()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.with_proprio = with_proprio
        self.finetune_mode = with_finetune
        self._debug_printed = False  # For debug prints
        
        # Use shared world model if provided, otherwise load new one
        if shared_wm is not None:
            self.wm = shared_wm
            print(f"Using shared world model (finetune_mode: {self.finetune_mode})")
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
            print(f"Loaded new world model from {ckpt_dir} (finetune_mode: {self.finetune_mode})")
        
        # Ensure world model is on correct device
        self.wm = self.wm.to(self.device)
        
        # probe a reset to set spaces
        reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        
        print("using proprio:", self.with_proprio)
        print("using finetune:", self.finetune_mode)
        
        if self.finetune_mode:
            # For finetuning, we return raw observations
            if isinstance(obs, dict):
                visual = obs['visual']
                proprio = obs['proprio']
            elif isinstance(obs, (tuple, list)) and len(obs) == 2:
                visual, proprio = obs
            else:
                raise ValueError(f"Unexpected obs type: {type(obs)}")
            
            # Define observation space as dict space for raw data
            from gymnasium.spaces import Dict
            self.observation_space = Dict({
                'visual': Box(low=0, high=255, shape=visual.shape, dtype=np.uint8),
                'proprio': Box(low=-np.inf, high=np.inf, shape=proprio.shape, dtype=np.float32)
            })
        else:
            # For non-finetuning, encode and define latent space
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
        """Reset underlying Gym env and return obs (raw or encoded based on mode)."""
        reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        
        if self.finetune_mode:
            # Return raw observations as dict
            if isinstance(obs, dict):
                return obs, {}
            elif isinstance(obs, (tuple, list)) and len(obs) == 2:
                visual, proprio = obs
                return {'visual': visual, 'proprio': proprio}, {}
            else:
                raise ValueError(f"Unexpected obs type: {type(obs)}")
        else:
            # Encode observations
            z = self._encode(obs)
            
            # Always convert to numpy for PyHJ compatibility
            if torch.is_tensor(z):
                z = z.detach().cpu().numpy()
            
            return z, {}

    def step(self, action):
        """Step in Gym env and return obs (raw or encoded based on mode)."""
        obs_out, _, done, info = self.env.step(action)
        terminated = done
        truncated = False
        # extract obs if tuple
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        # override reward with safety metric
        h_s = info.get('h', 0.0) * 3  # Multiplied by 3 to make HJ easier to learn
        
        if self.finetune_mode:
            # Return raw observations as dict
            if isinstance(obs, dict):
                obs_dict = obs
            elif isinstance(obs, (tuple, list)) and len(obs) == 2:
                visual, proprio = obs
                obs_dict = {'visual': visual, 'proprio': proprio}
            else:
                raise ValueError(f"Unexpected obs type: {type(obs)}")
            
            return obs_dict, h_s, terminated, truncated, info
        else:
            # Encode observations
            z_next = self._encode(obs)
            
            # Always convert to numpy for PyHJ compatibility
            if torch.is_tensor(z_next):
                z_next = z_next.detach().cpu().numpy()
            
            return z_next, h_s, terminated, truncated, info

    def _encode(self, obs):
        """
        Encode raw obs via DINO-WM into a flat latent vector.
        This should only be called when finetune_mode=False.
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
            
            # Ensure world model is in eval mode
            self.wm.eval()
            
            lat = self.wm.encode_obs(data)
            
            # flatten visual patches and concat proprio
            if self.with_proprio:
                z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches, E_dim) -> (1, N_patches*E_dim)
                z_prop = lat['proprio']  # (1, 1, D_prop)
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
        """Override add to handle observations properly."""
        # No conversion needed - store raw data as is
        return super().add(batch, buffer_ids=buffer_ids)

    def sample(self, batch_size: int) -> Batch:
        """Override sample to handle parent class return format."""
        # Call parent sample method - VectorReplayBuffer returns (batch, indices)
        sample_result = super().sample(batch_size)
        
        # Handle the tuple return from VectorReplayBuffer
        if isinstance(sample_result, tuple) and len(sample_result) >= 1:
            batch_data = sample_result[0]  # First element is the batch data
            
            # Ensure we have a proper Batch object
            if isinstance(batch_data, Batch):
                batch = batch_data
            elif isinstance(batch_data, dict):
                batch = Batch(batch_data)
            else:
                # Try to convert to Batch
                batch = Batch()
                try:
                    # Attempt to access attributes directly
                    for attr in ['obs', 'obs_next', 'act', 'rew', 'done', 'terminated', 'truncated', 'info', 'policy']:
                        if hasattr(batch_data, attr):
                            setattr(batch, attr, getattr(batch_data, attr))
                except Exception as e:
                    print(f"Error creating batch from data: {e}")
                    batch = Batch()
        else:
            batch = sample_result if isinstance(sample_result, Batch) else Batch(sample_result)

        # For non-finetuning mode, convert numpy arrays to tensors
        # For finetuning mode, keep observations as raw dicts
        if not self.finetune_mode:
            self._convert_batch_to_tensors(batch)

        return batch

    def __getitem__(self, index: Union[str, IndexType]) -> Batch:
        """Override getitem to ensure proper data format."""
        result = super().__getitem__(index)
        
        # Always return raw data - conversion happens in the policy
        return result

    def _convert_batch_to_tensors(self, batch: Batch) -> None:
        """Convert all numpy arrays in batch to tensors on the correct device."""
        for key in ['obs', 'obs_next', 'act', 'rew', 'done', 'terminated', 'truncated', 'returns', 'weight']:
            if hasattr(batch, key):
                value = getattr(batch, key)
                # Skip conversion for observations in finetune mode (keep as dict/Batch)
                if key in ['obs', 'obs_next'] and self.finetune_mode and isinstance(value, (dict, Batch)):
                    continue
                    
                if isinstance(value, np.ndarray):
                    if key in ['done', 'terminated', 'truncated']:
                        tensor_value = torch.from_numpy(value).bool().to(self.device)
                    else:
                        tensor_value = torch.from_numpy(value).float().to(self.device)
                    setattr(batch, key, tensor_value)
                elif torch.is_tensor(value):
                    if key in ['done', 'terminated', 'truncated']:
                        tensor_value = value.bool().to(self.device)
                    else:
                        tensor_value = value.float().to(self.device)
                    setattr(batch, key, tensor_value)
        
        # Handle nested Batch objects (like info, policy)
        for key in ['info', 'policy']:
            if hasattr(batch, key):
                value = getattr(batch, key)
                if isinstance(value, Batch):
                    self._convert_batch_to_tensors(value)
                elif isinstance(value, dict):
                    batch_value = Batch(value)
                    self._convert_batch_to_tensors(batch_value)
                    setattr(batch, key, batch_value)

class TensorAwareDDPGPolicy(avoid_DDPGPolicy_annealing):
    """Modified DDPG policy that can handle tensor observations for encoder finetuning."""
    
    def __init__(self, actor, critic, actor_optim, critic_optim, finetune_mode=False, 
                 device='cuda', shared_wm=None, with_proprio=False, **kwargs):
        # Extract required arguments from kwargs with defaults
        tau = kwargs.pop('tau', 0.005)
        gamma = kwargs.pop('gamma', 0.99)
        exploration_noise = kwargs.pop('exploration_noise', None)
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
        self.device = torch.device(device) if isinstance(device, str) else device
        self.shared_wm = shared_wm
        self.with_proprio = with_proprio
        self._debug_count = 0  # For debugging
        
        # Ensure all networks are on the correct device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.actor_old.to(self.device)
        self.critic_old.to(self.device)
        
    def _ensure_tensor_on_device(self, data, device=None):
        """Helper function to ensure data is a tensor on the correct device"""
        if device is None:
            device = self.device
            
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(device)
        elif torch.is_tensor(data):
            return data.float().to(device)
        else:
            return data
    
    def _ensure_batch_on_device(self, batch):
        """Ensure all tensors in batch are on the correct device"""
        for key in ['act', 'rew', 'done', 'terminated', 'truncated', 'returns']:
            if hasattr(batch, key):
                setattr(batch, key, self._ensure_tensor_on_device(getattr(batch, key)))
        return batch
    
    def _encode_observations(self, obs_dict, detach=False):
        """Encode raw observations using the world model.
        
        Args:
            obs_dict: Dictionary or Batch containing visual and proprio data
            detach: If True, detach the encoded observations from the computation graph
        """
        # Handle both dict and Batch types
        if isinstance(obs_dict, Batch):
            visual = obs_dict.visual
            proprio = obs_dict.proprio
        else:
            visual = obs_dict['visual']
            proprio = obs_dict['proprio']
            
        # Handle both single observations and batches
        if isinstance(visual, np.ndarray):
            if visual.ndim == 3:  # Single observation (H, W, C)
                visual = visual[np.newaxis, ...]  # Add batch dimension
                proprio = proprio[np.newaxis, ...]
            batch_size = visual.shape[0]
        else:
            raise ValueError(f"Unexpected visual type: {type(visual)}")
        
        # Prepare visual tensor
        visual_np = np.transpose(visual, (0, 3, 1, 2)).astype(np.float32)  # (B, H, W, C) -> (B, C, H, W)
        visual_np /= 255.0  # normalize to [0, 1]
        vis_t = torch.from_numpy(visual_np).unsqueeze(1).to(self.device)  # Add time dimension (B, 1, C, H, W)

        # Prepare proprio tensor
        if proprio.ndim == 1:  # Single observation
            proprio = proprio[np.newaxis, ...]
        prop_t = torch.from_numpy(proprio).unsqueeze(1).to(self.device)  # Add time dimension (B, 1, D_prop)
        
        data = {'visual': vis_t, 'proprio': prop_t}
        
        # Ensure world model is in train mode for gradients
        self.shared_wm.train()
        
        # Encode with gradients
        lat = self.shared_wm.encode_obs(data)
        
        # Flatten visual patches and concat proprio if needed
        if self.with_proprio:
            z_vis = lat['visual'].reshape(batch_size, -1)  # (B, N_patches * E_dim)
            z_prop = lat['proprio'].squeeze(1)  # (B, D_prop)
            z = torch.cat([z_vis, z_prop], dim=-1)  # (B, N_patches*E_dim + D_prop)
        else:
            z = lat['visual'].reshape(batch_size, -1)  # (B, N_patches * E_dim)
        
        # Detach if requested (for target networks)
        if detach:
            z = z.detach()
        
        # If single observation, remove batch dimension
        if batch_size == 1:
            z = z.squeeze(0)
            
        return z
    
    def _target_q(self, buffer, indices: np.ndarray) -> torch.Tensor:
        """Override _target_q to handle raw observations properly."""
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        
        # If finetuning, encode the observations with detach for target network
        if self.finetune_mode and hasattr(batch, 'obs_next'):
            obs_next = batch.obs_next
            if isinstance(obs_next, (dict, Batch)) and 'visual' in obs_next and 'proprio' in obs_next:
                # Encode observations without gradients for target network
                with torch.no_grad():
                    encoded_obs_next = self._encode_observations(obs_next, detach=True)
                batch.obs_next = encoded_obs_next
        
        # Get actions from actor_old with encoded observations
        with torch.no_grad():
            act = self(batch, model='actor_old', input='obs_next').act
        
        # Compute Q-value
        target_q = self.critic_old(batch.obs_next, act)
        
        return target_q
        
    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None,
                model: str = "actor", input: str = "obs", **kwargs) -> Batch:
        """Override forward to handle tensor observations and ensure device consistency."""
        batch = self._ensure_batch_on_device(batch)
        
        # Process observations if in finetune mode
        if self.finetune_mode and hasattr(batch, input):
            obs = getattr(batch, input)
            if isinstance(obs, (dict, Batch)) and 'visual' in obs and 'proprio' in obs:
                # For forward pass, encode without detaching
                encoded_obs = self._encode_observations(obs, detach=False)
                # Create new batch with encoded observations
                batch_new = Batch()
                for key in batch.keys():
                    if key == input:
                        setattr(batch_new, key, encoded_obs)
                    else:
                        setattr(batch_new, key, getattr(batch, key))
                batch = batch_new
        
        # Debug print occasionally
        if self._debug_count % 100 == 0 and self.finetune_mode:
            obs = getattr(batch, input)
            if torch.is_tensor(obs):
                print(f"Forward debug - obs type: {type(obs)}, obs shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
        self._debug_count += 1
        
        return super().forward(batch, state, model, input, **kwargs)
    
    def process_fn(self, batch: Batch, buffer, indices: np.ndarray) -> Batch:
        """Override process_fn to handle tensor observations properly."""
        batch = self._ensure_batch_on_device(batch)
        # Don't encode here - let the parent class handle it
        return super().process_fn(batch, buffer, indices)
    
import torch.nn.functional as F

def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
    if not self.finetune_mode:
        # If not fine-tuning, use the parent class's learn method
        return super().learn(batch, **kwargs)
    
    # For fine-tune mode, retain the graph for multiple backward passes
    # Compute target Q-values using the target networks
    with torch.no_grad():
        a_ = self(batch, model='actor_old', input='obs_next').act
        target_q = self.critic_old(batch.obs_next, a_).flatten()
        target_q = batch.rew + self._gamma * target_q * (1 - batch.terminated)
    
    # Compute current Q-values
    current_q = self.critic(batch.obs, batch.act).flatten()
    q_loss = F.mse_loss(current_q, target_q)
    
    # Update critic with retained graph
    self.critic_optim.zero_grad()
    q_loss.backward(retain_graph=True)  # Retain graph for actor loss
    self.critic_optim.step()
    
    # Compute actor loss
    a = self(batch).act
    actor_loss = -self.critic(batch.obs, a).mean()
    
    # Update actor
    self.actor_optim.zero_grad()
    actor_loss.backward()
    self.actor_optim.step()
    
    # Update target networks
    self.sync_weight()
    
    return {
        'loss/critic': q_loss.item(),
        'loss/actor': actor_loss.item(),
    }

    def exploration_noise(self, act: Union[np.ndarray, Batch], batch: Batch) -> Union[np.ndarray, Batch]:
        """Override exploration noise to handle device compatibility."""
        import warnings
        
        if self._noise is None:
            return act
            
        # Convert act to numpy if it's a tensor for noise addition
        if isinstance(act, torch.Tensor):
            act_np = act.detach().cpu().numpy()
            was_tensor = True
            original_device = act.device
        else:
            act_np = act
            was_tensor = False
            original_device = None
            
        if isinstance(act_np, np.ndarray):
            # Add noise to numpy array
            noise_sample = self._noise(act_np.shape)
            noisy_act = act_np + noise_sample
            
            # Convert back to tensor if original was tensor
            if was_tensor:
                noisy_act = torch.from_numpy(noisy_act).float().to(original_device)
            
            return noisy_act
        else:
            warnings.warn("Cannot add exploration noise to non-numpy_array action.")
            return act
        
def create_policy_and_trainer(args, state_shape, action_shape, max_action, shared_wm):
    """Create policy and trainer with proper device handling"""
    
    print(f"=== CREATING POLICY ===")
    print(f"Fine-tuning enabled: {args.with_finetune}")
    print(f"Encoder learning rate: {args.encoder_lr}")
    
    # Build critic + actor with explicit device placement
    critic_net = Net(state_shape, action_shape,
                     hidden_sizes=args.critic_net,
                     activation=getattr(torch.nn, args.critic_activation),
                     concat=True, device=args.device)
    critic = Critic(critic_net, device=args.device).to(args.device)
    
    actor_net = Net(state_shape,
                    hidden_sizes=args.control_net,
                    activation=getattr(torch.nn, args.actor_activation),
                    device=args.device)
    actor = Actor(actor_net, action_shape,
                  max_action=max_action,
                  device=args.device).to(args.device)

    # Create encoder optimizer if fine-tuning
    encoder_optimizer = None
    if args.with_finetune:
        print("🔥 ENABLING ENCODER FINE-TUNING")
        # Enable gradients for encoder parameters
        grad_enabled_count = 0
        total_params = 0
        for param in shared_wm.encoder.parameters():
            param.requires_grad = True
            grad_enabled_count += 1
            total_params += 1
        
        encoder_optimizer = torch.optim.Adam(shared_wm.encoder.parameters(), lr=args.encoder_lr)
        print(f"✅ Created encoder optimizer with lr={args.encoder_lr}")
        print(f"✅ Enabled gradients for {grad_enabled_count}/{total_params} encoder parameters")
    else:
        print("❌ ENCODER FINE-TUNING DISABLED")
        # Ensure encoder parameters don't require gradients
        for param in shared_wm.encoder.parameters():
            param.requires_grad = False

    # Create exploration noise
    exploration_noise = None
    if hasattr(args, 'exploration_noise') and args.exploration_noise > 0:
        exploration_noise = GaussianNoise(sigma=args.exploration_noise)
        print(f"✅ Created GaussianNoise with sigma={args.exploration_noise}")
    else:
        print("❌ Exploration noise disabled")

    # Build DDPG policy with shared_wm
    policy = TensorAwareDDPGPolicy(
        actor=actor,
        critic=critic,
        actor_optim=torch.optim.Adam(actor.parameters(), lr=args.actor_lr),
        critic_optim=torch.optim.Adam(critic.parameters(), lr=args.critic_lr),
        tau=args.tau,
        gamma=args.gamma_pyhj,
        exploration_noise=exploration_noise,
        estimation_step=args.n_step,
        finetune_mode=args.with_finetune,
        device=args.device,
        shared_wm=shared_wm,
        with_proprio=args.with_proprio
    )

    print("======================")
    return policy, encoder_optimizer

# Set up matplotlib config
import os
os.environ['MPLCONFIGDIR'] = '/storage1/fs1/sibai/Active/ihab/tmp'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib.pyplot as plt
import wandb

def compute_hj_value(x, y, theta, policy, helper_env, args):
    """Compute the Hamilton-Jacobi filter value in latent space."""
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
    """Plot the Hamilton-Jacobi safety filter in latent space."""
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
                    val = compute_hj_value(x, y, theta, policy, helper_env, args)
                    vals[ix, iy] = val
                except Exception as e:
                    print(f"Error computing HJ value at x={x:.2f}, y={y:.2f}, theta={theta:.2f}: {str(e)}")
                    vals[ix, iy] = 0  # Default value

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

def custom_trainer(policy, train_collector, test_collector, max_epoch, 
                  step_per_epoch, step_per_collect, test_num, batch_size, 
                  update_per_step, encoder_optimizer=None, logger=None, args=None, 
                  shared_wm=None, initial_encoder_params=None):
    """Custom trainer that handles encoder finetuning with proper error handling"""
    
    print("🚀 Starting training...")
    print(f"Fine-tuning mode: {args.with_finetune}")
    print(f"Encoder optimizer: {encoder_optimizer is not None}")
    
    # Pre-collect random samples
    try:
        train_collector.collect(600, random=True, no_grad=False)
        print(f"✅ Pre-collected 600 random samples. Buffer size: {len(train_collector.buffer)}")
    except Exception as e:
        print(f"❌ Error during pre-collection: {e}")
        import traceback
        traceback.print_exc()
        return policy
    
    global_step = 0
    
    for epoch in range(max_epoch):
        print(f"\n=== Epoch {epoch+1}/{max_epoch} ===")
        
        # Collect training data
        try:
            collect_result = train_collector.collect(n_step=step_per_collect, no_grad=False)
            global_step += step_per_collect
            print(f"📊 Collected {step_per_collect} steps. Buffer size: {len(train_collector.buffer)}")
        except Exception as e:
            print(f"❌ Error during data collection: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Update policy
        if len(train_collector.buffer) >= batch_size:
            print("🔄 Starting policy updates...")
            update_count = int(step_per_collect * update_per_step)
            
            total_actor_loss = 0
            total_critic_loss = 0
            successful_updates = 0
            
            for update_idx in range(update_count):
                try:
                    # Sample batch from buffer
                    batch = train_collector.buffer.sample(batch_size)
                    
                    # Get random indices for process_fn
                    indices = np.random.randint(0, len(train_collector.buffer), size=batch_size)
                    
                    # Process the batch (computes returns)
                    processed_batch = policy.process_fn(batch, train_collector.buffer, indices)
                    
                    # Check if returns field exists
                    if not hasattr(processed_batch, 'returns'):
                        print("⚠️  Warning: Batch missing 'returns' field after process_fn")
                        if hasattr(processed_batch, 'rew'):
                            processed_batch.returns = processed_batch.rew.clone()
                        else:
                            print("❌ Error: Cannot create returns field - no rewards available")
                            continue
                    
                    # Clear gradients for encoder if fine-tuning
                    if encoder_optimizer is not None:
                        encoder_optimizer.zero_grad()
                    
                    # Update policy
                    losses = policy.learn(processed_batch)
                    
                    # Step encoder optimizer and log gradients
                    if encoder_optimizer is not None:
                        # Log gradients every 100 updates
                        if update_idx % 100 == 0:
                            total_grad_norm = 0
                            param_count = 0
                            for param in shared_wm.encoder.parameters():
                                if param.grad is not None:
                                    param_norm = param.grad.data.norm(2)
                                    total_grad_norm += param_norm.item() ** 2
                                    param_count += 1
                            
                            if param_count > 0:
                                total_grad_norm = (total_grad_norm ** 0.5)
                                wandb.log({
                                    "encoder/gradient_norm_realtime": total_grad_norm,
                                    "global_step": global_step + update_idx
                                })
                                print(f"🔥 Encoder gradient norm: {total_grad_norm:.6f}")
                            else:
                                print("⚠️  No encoder gradients found!")
                        
                        encoder_optimizer.step()
                    
                    # Accumulate losses
                    total_actor_loss += losses.get('loss/actor', 0)
                    total_critic_loss += losses.get('loss/critic', 0)
                    successful_updates += 1
                    
                    # Log losses occasionally
                    if update_idx % 50 == 0:
                        print(f"  Update {update_idx}/{update_count}: {losses}")
                    
                    # Log to wandb
                    if logger is not None:
                        for key, value in losses.items():
                            try:
                                logger.log_scalar(f"train/{key}", value, global_step + update_idx)
                            except AttributeError:
                                wandb.log({f"train/{key}": value}, step=global_step + update_idx)
                            
                except Exception as e:
                    print(f"❌ Error during policy update {update_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Print epoch summary
            if successful_updates > 0:
                avg_actor_loss = total_actor_loss / successful_updates
                avg_critic_loss = total_critic_loss / successful_updates
                print(f"📈 Epoch {epoch+1} Summary:")
                print(f"   Successful updates: {successful_updates}/{update_count}")
                print(f"   Avg actor loss: {avg_actor_loss:.6f}")
                print(f"   Avg critic loss: {avg_critic_loss:.6f}")
                
                # Log epoch summary
                wandb.log({
                    "epoch_summary/avg_actor_loss": avg_actor_loss,
                    "epoch_summary/avg_critic_loss": avg_critic_loss,
                    "epoch_summary/successful_updates": successful_updates,
                    "epoch": epoch,
                    "global_step": global_step
                })
            
            print(f"✅ Completed {successful_updates} policy updates")
        else:
            print(f"⏳ Insufficient buffer size: {len(train_collector.buffer)} < {batch_size}")
        
        # Log encoder stats every 10 epochs
        if args.with_finetune and epoch % 10 == 0:
            log_encoder_stats(shared_wm, encoder_optimizer, epoch, global_step)
            log_param_changes(shared_wm, initial_encoder_params, epoch, global_step)
        
        # Plot HJ values every 5 epochs
        if epoch % 5 == 0:
            try:
                print("🎨 Generating HJ plots...")
                
                # Create helper env for plotting (always with finetune=False for plotting)
                helper_env = LatentDubinsEnv(
                    shared_wm=shared_wm,  # Use shared_wm directly
                    with_proprio=args.with_proprio,
                    with_finetune=False,  # Always False for plotting
                    device=args.device
                )
                
                thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
                fig1, fig2 = plot_hj(policy, helper_env, thetas, args)
                
                # Log plots to wandb with consistent keys
                wandb.log({
                    "HJ_Safe_Mask": wandb.Image(fig1),
                    "HJ_Values": wandb.Image(fig2),
                    "epoch": epoch,
                    "global_step": global_step
                }, step=global_step)
                
                plt.close(fig1)
                plt.close(fig2)
                print("✅ HJ plots generated and logged")
                
            except Exception as e:
                print(f"❌ Error during HJ plotting: {e}")
                import traceback
                traceback.print_exc()
    
    return policy

def main():
    # 1) Parse args + merge YAML
    args = get_args_and_merge_config()
    
    # Cast to the right types
    args.critic_lr = float(args.critic_lr)
    args.actor_lr = float(args.actor_lr)
    args.tau = float(args.tau)
    args.gamma_pyhj = float(args.gamma_pyhj)
    args.exploration_noise = float(args.exploration_noise)
    args.update_per_step = float(args.update_per_step)
    args.step_per_epoch = int(args.step_per_epoch)
    args.step_per_collect = int(args.step_per_collect)
    args.test_num = int(args.test_num)
    args.training_num = int(args.training_num)
    args.total_episodes = int(args.total_episodes)
    args.batch_size_pyhj = int(args.batch_size_pyhj)
    args.buffer_size = int(args.buffer_size)
    args.dino_ckpt_dir = os.path.join(args.dino_ckpt_dir, args.dino_encoder)
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {args.device}")
    
    # Final check of fine-tuning setting
    print(f"\n🔍 FINAL CONFIGURATION CHECK:")
    print(f"   Fine-tuning enabled: {args.with_finetune}")
    print(f"   Encoder learning rate: {args.encoder_lr}")
    print(f"   Exploration noise: {args.exploration_noise}")
    print(f"   Total episodes: {args.total_episodes}")
    print(f"   Step per collect: {args.step_per_collect}")
    print(f"   Step per epoch: {args.step_per_epoch}")
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 2) Load shared world model once
    shared_wm = load_shared_world_model(args.dino_ckpt_dir, args.device)
    print(f"🌍 Shared world model loaded on device: {next(shared_wm.parameters()).device}")
    
    # Create parameter tracker for encoder monitoring
    initial_encoder_params = create_param_tracker(shared_wm) if args.with_finetune else None
    
    # 3) Initialize W&B + TB writer + logger
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
    
    print(f"📊 Logging to wandb project: ddpg-hj-latent-dubins")
    print(f"📊 Run name: {run_name}")

    # 4) Create environments with shared world model
    print("\n🏗️  Creating training environments...")
    train_envs = DummyVectorEnv([
        lambda: LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio, 
                               with_finetune=args.with_finetune, device=args.device)  # CRUCIAL: Use args.with_finetune
        for _ in range(args.training_num)
    ])
    
    print("🏗️  Creating test environments...")
    test_envs = DummyVectorEnv([
        lambda: LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio, 
                               with_finetune=False, device=args.device)  # Test should be False
        for _ in range(args.test_num)
    ])

    # 5) Extract shapes & max_action
    # For finetuning mode, we need the encoded observation shape
    if args.with_finetune:
        # Get a sample observation to determine encoded shape
        sample_env = LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio, 
                                    with_finetune=False, device=args.device)
        sample_obs, _ = sample_env.reset()
        state_shape = sample_obs.shape
        print(f"📏 Encoded observation shape: {state_shape}")
    else:
        state_space = train_envs.observation_space[0]
        state_shape = state_space.shape
    
    action_space = train_envs.action_space[0]
    action_shape = action_space.shape or action_space.n
    max_action = torch.tensor(action_space.high, device=args.device, dtype=torch.float32)

    print(f"\n📏 Environment specs:")
    print(f"   State shape: {state_shape}")
    print(f"   Action shape: {action_shape}")
    print(f"   Max action: {max_action}")

    # 6) Create policy and trainer
    policy, encoder_optimizer = create_policy_and_trainer(
        args, state_shape, action_shape, max_action, shared_wm
    )
    
    print(f"\n🧠 Policy created. Networks on device:")
    print(f"   Actor: {next(policy.actor.parameters()).device}")
    print(f"   Critic: {next(policy.critic.parameters()).device}")
    print(f"   Actor_old: {next(policy.actor_old.parameters()).device}")
    print(f"   Critic_old: {next(policy.critic_old.parameters()).device}")

    # 7) Create tensor-aware buffer
    buffer = TensorAwareReplayBuffer(
        size=args.buffer_size,
        buffer_num=args.training_num,
        device=args.device,
        finetune_mode=args.with_finetune,
        shared_wm=shared_wm
    )

    # 8) Create collectors
    train_collector = Collector(
        policy, train_envs, buffer, 
        exploration_noise=True  # Enable exploration noise for training
    )
    
    test_collector = Collector(
        policy, test_envs,
        exploration_noise=False  # Disable for testing
    )

    # 9) Calculate max epochs from total episodes
    max_epoch = args.total_episodes // args.step_per_epoch
    print(f"\n🎯 Training configuration:")
    print(f"   Total epochs: {max_epoch}")
    print(f"   Steps per epoch: {args.step_per_epoch}")
    print(f"   Steps per collect: {args.step_per_collect}")
    print(f"   Updates per step: {args.update_per_step}")
    print(f"   Batch size: {args.batch_size_pyhj}")
    
    if args.with_finetune:
        print(f"🔥 Encoder fine-tuning: ENABLED")
        print(f"   Encoder optimizer: {encoder_optimizer is not None}")
        print(f"   Encoder requires_grad: {any(p.requires_grad for p in shared_wm.encoder.parameters())}")
    else:
        print(f"❌ Encoder fine-tuning: DISABLED")

    # 10) Run custom trainer
    print(f"\n🚀 Starting DDPG training...")
    try:
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
            logger=logger,
            args=args,
            shared_wm=shared_wm,
            initial_encoder_params=initial_encoder_params
        )
        print("🎉 Training completed successfully!")
    except Exception as e:
        print(f"💥 Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 11) Save model
    print("\n💾 Saving model...")
    try:
        ckpt_dir = f"runs/ddpg_hj_latent/{run_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        model_save_path = os.path.join(ckpt_dir, "model_final.pth")
        
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'args': vars(args),
            'encoder_requires_grad': any(p.requires_grad for p in shared_wm.encoder.parameters()) if args.with_finetune else False
        }, model_save_path)
        
        print(f"✅ Model saved to: {model_save_path}")
        
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    wandb.finish()
    writer.close()
    
    print("🏁 Training and evaluation completed!")

if __name__ == "__main__":
    main()
    
    
    # python "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent(can_fine_tune_PVR)6hasprob.py" \
    # --dino_ckpt_dir "/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs/dubins/fully trained(prop repeated 3 times)" \
    # --config train_HJ_configs.yaml \
    # --dino_encoder r3m \
    # --with_finetune \
    # --encoder_lr 1e-5 \
    # --total-episodes 100000
    # --step-per-epoch 1000