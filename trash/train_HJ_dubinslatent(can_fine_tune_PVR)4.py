import argparse
import os
from pathlib import Path
import torch
import gymnasium as gym  # We use Gymnasium API for vector envs
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from typing import Any, Dict, List, Union, Optional
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

    def add(self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None) -> None:
        batch_copy = copy.deepcopy(batch)
        if self.finetune_mode:
            if hasattr(batch_copy, 'info') and 'raw_obs' in batch_copy.info:
                raw_obs = batch_copy.info['raw_obs']  # Access raw_obs from info
                batch_copy.obs = self._encode_batch(raw_obs)  # Encode raw_obs
            if hasattr(batch_copy, 'info') and 'raw_obs_next' in batch_copy.info:
                raw_obs_next = batch_copy.info['raw_obs_next']  # Access raw_obs_next from info
                batch_copy.obs_next = self._encode_batch(raw_obs_next)
            print(f"[Debug] buffer.add: obs is_tensor={torch.is_tensor(batch_copy.obs) if hasattr(batch_copy, 'obs') else 'N/A'}, "
                f"requires_grad={batch_copy.obs.requires_grad if hasattr(batch_copy, 'obs') and torch.is_tensor(batch_copy.obs) else 'N/A'}")
        else:
            if hasattr(batch_copy, 'obs') and torch.is_tensor(batch_copy.obs):
                batch_copy.obs = batch_copy.obs.detach().cpu().numpy()
            if hasattr(batch_copy, 'obs_next') and torch.is_tensor(batch_copy.obs_next):
                batch_copy.obs_next = batch_copy.obs_next.detach().cpu().numpy()
        super().add(batch_copy, buffer_ids)

def _encode_batch(self, raw_obs):
    if raw_obs is None:
        return None
    batch_size = len(raw_obs) if isinstance(raw_obs, (list, tuple)) else raw_obs.shape[0]
    z_list = []
    for i in range(batch_size):
        obs = raw_obs[i]
        if isinstance(obs, dict):
            visual = obs['visual']
            proprio = obs['proprio'] if self.with_proprio else None
        elif isinstance(obs, (tuple, list)):
            if self.with_proprio:
                if len(obs) != 2:
                    raise ValueError(f"Expected obs to have 2 elements (visual, proprio), got {len(obs)}: {obs}")
                visual, proprio = obs
            else:
                if len(obs) != 1:
                    raise ValueError(f"Expected obs to have 1 element (visual) when with_proprio=False, got {len(obs)}: {obs}")
                visual = obs[0]
                proprio = None
        else:
            raise ValueError(f"Unexpected obs type: {type(obs)}, obs: {obs}")

        # Prepare visual tensor
        visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
        vis_t = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(1).to(self.device)

        # Prepare proprio tensor if applicable
        if self.with_proprio and proprio is not None:
            prop_t = torch.from_numpy(proprio).unsqueeze(0).unsqueeze(1).to(self.device)
        else:
            prop_t = None

        # Encode observation
        data = {'visual': vis_t}
        if prop_t is not None:
            data['proprio'] = prop_t

        with torch.enable_grad():
            lat = self.shared_wm.encode_obs(data)
            z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim)
            if self.with_proprio and 'proprio' in lat:
                z_prop = lat['proprio'].squeeze(0)  # (1, D_prop)
                z = torch.cat([z_vis, z_prop], dim=-1).squeeze(0)  # Concatenate visual and proprio
            else:
                z = z_vis.squeeze(0)  # Use only visual latent
            z_list.append(z)

    return torch.stack(z_list)

    def sample(self, batch_size: int) -> Batch:
        batch = super().sample(batch_size)
        if self.finetune_mode:
            if hasattr(batch, 'raw_obs') and batch.raw_obs is not None:
                batch.obs = self._encode_batch(batch.raw_obs)
            if hasattr(batch, 'raw_obs_next') and batch.raw_obs_next is not None:
                batch.obs_next = self._encode_batch(batch.raw_obs_next)
            if hasattr(batch, 'act') and isinstance(batch.act, np.ndarray):
                batch.act = torch.from_numpy(batch.act).to(self.device)
            if hasattr(batch, 'rew') and isinstance(batch.rew, np.ndarray):
                batch.rew = torch.from_numpy(batch.rew).to(self.device)
            if hasattr(batch, 'done') and isinstance(batch.done, np.ndarray):
                batch.done = torch.from_numpy(batch.done).to(self.device)
            print(f"[Debug] buffer.sample: obs is_tensor={torch.is_tensor(batch.obs) if hasattr(batch, 'obs') else 'N/A'}, requires_grad={batch.obs.requires_grad if hasattr(batch, 'obs') else 'N/A'}")
        return batch

class TensorAwareCollector(Collector):
    def __init__(self, policy, env, buffer=None, preprocess_fn=None, exploration_noise=False, finetune_mode=False, device="cpu", shared_wm=None):
        super().__init__(policy, env, buffer, preprocess_fn, exploration_noise)
        self.finetune_mode = finetune_mode
        self.device = torch.device(device)
        self.shared_wm = shared_wm  # Use provided shared_wm

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if self.finetune_mode:
            no_grad = False
            self.shared_wm.encoder.train()
        result = super().collect(
            n_step=n_step,
            n_episode=n_episode,
            random=random,
            render=render,
            no_grad=no_grad,
            gym_reset_kwargs=gym_reset_kwargs,
        )
        print(f"[Debug] collect: result keys={list(result.keys())}")
        if self.finetune_mode and hasattr(self.data, 'info') and 'raw_obs' in self.data.info:
            self.data.obs = self._reencode_observations(self.data.info['raw_obs'])
            if 'raw_obs_next' in self.data.info:
                self.data.obs_next = self._reencode_observations(self.data.info['raw_obs_next'])
            print(f"[Debug] collect: obs is_tensor={torch.is_tensor(self.data.obs)}, requires_grad={self.data.obs.requires_grad if torch.is_tensor(self.data.obs) else 'N/A'}")
        return result

    def _reencode_observations(self, raw_obs):
        batch_size = len(raw_obs) if isinstance(raw_obs, (list, tuple)) else raw_obs.shape[0]
        z_list = []
        for i in range(batch_size):
            obs = raw_obs[i]
            if isinstance(obs, dict):
                visual = obs['visual']
                proprio = obs['proprio']
            else:
                visual, proprio = obs  # Assuming obs is a tuple (visual, proprio)
            visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
            vis_t = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(1).to(self.device)
            prop_t = torch.from_numpy(proprio).unsqueeze(0).unsqueeze(1).to(self.device)
            data = {'visual': vis_t, 'proprio': prop_t}
            with torch.enable_grad():
                lat = self.shared_wm.encode_obs(data)
                z_vis = lat['visual'].reshape(1, -1)
                z_prop = lat['proprio'].squeeze(0)
                z = torch.cat([z_vis, z_prop], dim=-1).squeeze(0)
                z_list.append(z)
        return torch.stack(z_list)
        


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
    
    print(f"Loaded shared world model from {ckpt_dir}")
    return wm


class LatentDubinsEnv(gym.Env):
    """
    Wraps the classic Gym-based DubinsEnv into a Gymnasium-compatible Env.
    Encodes observations into DINO-WM latent space and uses info['h'] as reward.
    """
    def __init__(self, shared_wm=None, ckpt_dir: str = None, device: str = None, 
                 with_proprio: bool = False, finetune_mode: bool = False):
        super().__init__()
        # underlying Gym env
        self.env = DubinsEnv()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.with_proprio = with_proprio
        self.finetune_mode = finetune_mode
        
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
            if args.with_finetune:
                self.wm.encoder.train()
            print(f"Loaded new world model from {ckpt_dir}")
        
        # probe a reset to set spaces
        reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        
        print("using proprio:", self.with_proprio)
        z = self._encode(obs)
        
        if self.finetune_mode:
            # In finetune mode, we need to work with tensors
            if torch.is_tensor(z):
                obs_shape = z.shape
            else:
                obs_shape = z.shape
                z = torch.from_numpy(z).to(self.device)
                obs_shape = z.shape
        else:
            # In normal mode, convert to numpy
            if torch.is_tensor(z):
                z = z.detach().cpu().numpy()
            obs_shape = z.shape
        
        print(f"Example latent state z shape: {obs_shape}")
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = self.env.action_space

    def reset(self):
        reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        z = self._encode(obs)
        info = {'raw_obs': obs}
        if self.finetune_mode:
            print(f"[Debug] reset: z is_tensor={torch.is_tensor(z)}, requires_grad={z.requires_grad}")
            return z.detach().cpu().numpy(), info
        return z, info

    def step(self, action):
        obs_out, _, done, info = self.env.step(action)
        terminated = done
        truncated = False
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        h_s = info.get('h', 0.0) * 3
        z_next = self._encode(obs)
        info['raw_obs'] = obs
        if self.finetune_mode:
            print(f"[Debug] step: z_next is_tensor={torch.is_tensor(z_next)}, requires_grad={z_next.requires_grad}")
            return z_next.detach().cpu().numpy(), h_s, terminated, truncated, info
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
                
                # Only convert to numpy if not in finetune mode
                if not self.finetune_mode:
                    z = z.detach().cpu().numpy()
                return z
            
            else:
                z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim) torch.Size([1, 75264])
                z_vis = z_vis.squeeze(0)
                
                # Only convert to numpy if not in finetune mode
                if not self.finetune_mode:
                    z_vis = z_vis.detach().cpu().numpy()
                return z_vis


class TensorAwareDDPGPolicy(avoid_DDPGPolicy_annealing):
    def __init__(self, *args, finetune_mode=False, device="cuda", **kwargs):
        super().__init__(*args, **kwargs)
        self.finetune_mode = finetune_mode
        self.device = torch.device(device)

    def process_fn(self, batch: Batch, buffer: TensorAwareReplayBuffer, indices: np.ndarray) -> Batch:
        if self.finetune_mode:
            if hasattr(batch, 'obs') and not torch.is_tensor(batch.obs):
                batch.obs = torch.from_numpy(batch.obs).to(self.device).requires_grad_(True)
            if hasattr(batch, 'obs_next') and not torch.is_tensor(batch.obs_next):
                batch.obs_next = torch.from_numpy(batch.obs_next).to(self.device).requires_grad_(True)
            print(f"[Debug] process_fn: obs is_tensor={torch.is_tensor(batch.obs)}, requires_grad={batch.obs.requires_grad}")
        return super().process_fn(batch, buffer, indices)


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
    
    # Ensure z0 is numpy for this computation
    if torch.is_tensor(z0):
        z0 = z0.detach().cpu().numpy()
    
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
    
    encoder_optim = setup_encoder_finetuning(shared_wm, args)
    
    # 3) init W&B + TB writer + logger
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

    # 4) Create environments with shared world model
    train_envs = DummyVectorEnv([
        lambda: LatentDubinsEnv(
            shared_wm=shared_wm,
            with_proprio=args.with_proprio,
            finetune_mode=args.with_finetune  # Enable finetune mode
        )
        for _ in range(args.training_num)
    ])
    
    # No test env needed since no testing loop
    
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
    
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== Trainable Parameter Summary ===")
    print(f"  - Critic: {count_trainable_parameters(critic):,}")
    print(f"  - Actor:  {count_trainable_parameters(actor):,}")
    if args.with_finetune:
        print(f"  - Encoder: {sum(p.numel() for p in shared_wm.encoder.parameters() if p.requires_grad):,}")
        # Optional: list names of finetuned layers
        print("  - Encoder finetuned layers:")
        for name, param in shared_wm.encoder.named_parameters():
            if param.requires_grad:
                print(f"      {name} -> shape {tuple(param.shape)}")

    # 7) assemble your avoid‐DDPG policy
    policy = TensorAwareDDPGPolicy(
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
        finetune_mode=args.with_finetune,  # Enable finetune mode
       
    )

    # 8) hook into policy.learn to capture losses
    orig_learn = policy.learn
    policy.last_actor_loss  = 0.0
    policy.last_critic_loss = 0.0
    def learn_and_record(batch, **kw):
        if args.with_finetune and encoder_optim is not None:
            encoder_optim.zero_grad()  # Clear encoder grads

        metrics = orig_learn(batch, **kw)  # loss.backward() is done internally

        if args.with_finetune and encoder_optim is not None:
            grad_sum = sum(p.grad.abs().sum().item() for p in shared_wm.encoder.parameters() if p.grad is not None)
            print(f"[Finetune Check] Encoder total grad sum: {grad_sum:.4f}")
            encoder_optim.step()  # Apply encoder gradients

        policy.last_actor_loss  = metrics["loss/actor"]
        policy.last_critic_loss = metrics["loss/critic"]
        return metrics

    policy.learn = learn_and_record



    # 9) define train_fn to log those to W&B
    def train_fn(epoch: int, step_idx: int):
        wandb.log({
            "loss/actor":  policy.last_actor_loss,
            "loss/critic": policy.last_critic_loss,
        })

    # 10) collectors and replay buffer
    buffer          = buffer = TensorAwareReplayBuffer(
        size=args.buffer_size,
        buffer_num=args.training_num,
        device=args.device,
        finetune_mode=args.with_finetune,
        )
    train_collector = TensorAwareCollector(
        policy=policy,
        env=train_envs,
        buffer=buffer,
        exploration_noise=True,
        finetune_mode=args.with_finetune,  # Enable finetune mode
        shared_wm=shared_wm
        
    )
    print("collecting some initial data")
    train_collector.collect(10)
    print("initial data collected")

    # 11) choose headings & helper env (also uses shared model)
    thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    helper_env = LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio)

    # 12) main training loop, 1 epoch at a time
    log_path = Path(f"runs/ddpg_hj_latent/{args.dino_encoder}-{timestamp}")
    for epoch in range(1, args.total_episodes + 1):
        print(f"\n=== Epoch {epoch}/{args.total_episodes} ===")

        stats = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=None,  # no test loop
            max_epoch=1,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=0,
            batch_size=args.batch_size_pyhj,
            update_per_step=args.update_per_step,
            stop_fn=lambda r: False,
            train_fn=train_fn,        # log losses each epoch
            save_best_fn=None,
            logger=logger,
        )
        
        grad_sum = sum(p.grad.abs().sum().item() for p in shared_wm.encoder.parameters() if p.grad is not None)
        print(f"[Finetune Check] Encoder total grad sum: {grad_sum:.4f}")
        
        # Save policy checkpoint after each epoch
        ckpt_dir = log_path / f"epoch_id_{epoch}"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        torch.save(policy.state_dict(), ckpt_dir / "policy.pth")

        # Log other numeric stats to wandb
        numeric = {}
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                numeric[f"train/{k}"] = v
            elif isinstance(v, np.generic):
                numeric[f"train/{k}"] = float(v)
        wandb.log(numeric, step=epoch)

        # Plot latent-space HJ filter & log to wandb
        try:
            fig1, fig2 = plot_hj(policy, helper_env, thetas, args)
            wandb.log({
                "HJ_latent/binary":     wandb.Image(fig1),
                "HJ_latent/continuous": wandb.Image(fig2),
            })
            plt.close(fig1)
            plt.close(fig2)
            print("plotted")
        except Exception as e:
            print(f"Error plotting HJ values: {e}")

    print("Training complete.")


if __name__ == "__main__":
    main()



# python "train_HJ_dubinslatent(can_fine_tune_PVR)4.py" --with_finetune --dino_encoder r3m --finetune_lr 5e-6 --finetune_layers 2 --step-per-epoch 100 --nx 20 --ny 20
# python "train_HJ_dubinslatent(can_fine_tune_PVR)4.py" --step-per-epoch 100 --dino_encoder r3m --nx 20 --ny 20

# python "train_HJ_dubinslatent(can_fine_tune_PVR)4.py" --with_finetune --dino_encoder r3m --finetune_lr 5e-6 --finetune_layers 2 --step-per-epoch 1000  --nx 20 --ny 20 --total-episodes 50 --seed 0