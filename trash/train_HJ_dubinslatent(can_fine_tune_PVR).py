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
import os
import os
os.environ["WANDB_CONFIG_DIR"] = "/storage1/fs1/sibai/Active/ihab/tmp/.config"
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
        help="Which encoder to use: dino, r3m, vc1, etc."
    )
    
    parser.add_argument(
        "--with_finetune", action="store_true",
        help="Flag to enable fine-tuning of the vision backbone"
    )
    
    parser.add_argument(
        "--finetune_lr", type=float, default=1e-5,
        help="Learning rate for fine-tuning the vision backbone (typically much lower)"
    )
    
    parser.add_argument(
        "--finetune_layers", type=int, default=3,
        help="Number of last layers to unfreeze for fine-tuning (default: 3)"
    )
    
    parser.add_argument(
        "--progressive_unfreezing", action="store_true",
        help="Enable progressive unfreezing of layers during training"
    )
    
    parser.add_argument(
        "--unfreeze_schedule", type=str, default="25,50,75",
        help="Comma-separated epochs at which to unfreeze additional layers"
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


def get_encoder_layers(wm, encoder_type):
    """
    Get the encoder layers for different vision models.
    Returns a list of parameter groups that can be selectively unfrozen.
    """
    if encoder_type.lower() == 'dino':
        # For DINO, typically unfreeze the last few transformer blocks
        if hasattr(wm, 'encoder') and hasattr(wm.encoder, 'visual'):
            visual_encoder = wm.encoder.visual
            if hasattr(visual_encoder, 'blocks'):
                return visual_encoder.blocks  # Transformer blocks
            elif hasattr(visual_encoder, 'layers'):
                return visual_encoder.layers
    
    elif encoder_type.lower() == 'r3m':
        # For R3M (ResNet-based), unfreeze the last few residual blocks
        if hasattr(wm, 'encoder') and hasattr(wm.encoder, 'visual'):
            visual_encoder = wm.encoder.visual
            if hasattr(visual_encoder, 'layer4'):
                return [visual_encoder.layer4, visual_encoder.layer3, visual_encoder.layer2, visual_encoder.layer1]
    
    elif encoder_type.lower() == 'resnet':
        # For ResNet, unfreeze the last few layers
        if hasattr(wm, 'encoder') and hasattr(wm.encoder, 'visual'):
            visual_encoder = wm.encoder.visual
            if hasattr(visual_encoder, 'layer4'):
                return [visual_encoder.layer4, visual_encoder.layer3, visual_encoder.layer2, visual_encoder.layer1]
    
    elif encoder_type.lower() == 'vc1':
        # For VC1 (ViT-based), unfreeze the last few transformer blocks
        if hasattr(wm, 'encoder') and hasattr(wm.encoder, 'visual'):
            visual_encoder = wm.encoder.visual
            if hasattr(visual_encoder, 'blocks'):
                return visual_encoder.blocks
    
    # Fallback: try to find any encoder layers
    encoder_layers = []
    if hasattr(wm, 'encoder'):
        if hasattr(wm.encoder, 'visual'):
            encoder_layers.append(wm.encoder.visual)
        if hasattr(wm.encoder, 'layers'):
            encoder_layers.extend(wm.encoder.layers)
    
    return encoder_layers


def setup_finetune_optimizer(wm, encoder_type, finetune_lr, finetune_layers):
    """
    Set up optimizer for fine-tuning the vision backbone.
    Returns the optimizer and a list of unfrozen layers.
    """
    # First, freeze all parameters in the world model
    for param in wm.parameters():
        param.requires_grad = False
    
    # Get encoder layers
    encoder_layers = get_encoder_layers(wm, encoder_type)
    
    if not encoder_layers:
        print(f"Warning: Could not find encoder layers for {encoder_type}. Fine-tuning disabled.")
        return None, []
    
    # Unfreeze the last N layers
    unfrozen_layers = []
    finetune_params = []
    
    if isinstance(encoder_layers, list):
        # Take the last N layers
        layers_to_unfreeze = encoder_layers[-finetune_layers:] if finetune_layers <= len(encoder_layers) else encoder_layers
    else:
        # If it's a single module, just unfreeze it
        layers_to_unfreeze = [encoder_layers]
    
    for layer in layers_to_unfreeze:
        unfrozen_layers.append(layer)
        for param in layer.parameters():
            param.requires_grad = True
            finetune_params.append(param)
    
    if not finetune_params:
        print("Warning: No parameters found for fine-tuning.")
        return None, []
    
    # Create optimizer for fine-tuning parameters
    finetune_optimizer = torch.optim.AdamW(finetune_params, lr=finetune_lr, weight_decay=1e-4)
    
    print(f"Fine-tuning enabled: {len(finetune_params)} parameters unfrozen across {len(unfrozen_layers)} layers")
    
    return finetune_optimizer, unfrozen_layers


def progressive_unfreeze(wm, encoder_type, current_epoch, unfreeze_schedule, unfrozen_layers):
    """
    Progressively unfreeze more layers during training.
    """
    if current_epoch not in unfreeze_schedule:
        return unfrozen_layers, None
    
    encoder_layers = get_encoder_layers(wm, encoder_type)
    if not encoder_layers:
        return unfrozen_layers, None
    
    # Unfreeze one more layer
    current_unfrozen = len(unfrozen_layers)
    if current_unfrozen < len(encoder_layers):
        next_layer = encoder_layers[-(current_unfrozen + 1)]
        unfrozen_layers.append(next_layer)
        
        # Unfreeze parameters in the new layer
        new_params = []
        for param in next_layer.parameters():
            param.requires_grad = True
            new_params.append(param)
        
        # Create new optimizer with additional parameters
        all_params = []
        for layer in unfrozen_layers:
            all_params.extend(layer.parameters())
        
        new_optimizer = torch.optim.AdamW(all_params, lr=1e-5, weight_decay=1e-4)
        
        print(f"Progressive unfreezing: Added layer {current_unfrozen + 1}, total unfrozen: {len(unfrozen_layers)}")
        
        return unfrozen_layers, new_optimizer
    
    return unfrozen_layers, None


class LatentDubinsEnv(gym.Env):
    """
    Wraps the classic Gym-based DubinsEnv into a Gymnasium-compatible Env.
    Encodes observations into DINO-WM latent space and uses info['h'] as reward.
    """
    def __init__(self, ckpt_dir: str, device: str, with_proprio: bool, with_finetune: bool = False):
        super().__init__()
        # underlying Gym env
        self.env = DubinsEnv()
        self.device = torch.device(device)
        self.with_finetune = with_finetune
        
        # Paths for loading DINO-WM
        ckpt_dir = Path(ckpt_dir)
        hydra_cfg = ckpt_dir / 'hydra.yaml'
        snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
        # load train config and model weights
        train_cfg = OmegaConf.load(str(hydra_cfg))
        num_action_repeat = train_cfg.num_action_repeat
        self.wm = load_model(snapshot, train_cfg, num_action_repeat, device=self.device)
        
        # Set training mode for fine-tuning
        if with_finetune:
            self.wm.train()
        else:
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
        
        # For fine-tuning, we need gradients
        if self.with_finetune:
            # prepare tensors
            visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32)  # (C, H, W)
            visual_np /= 255.0  # normalize to [0, 1]
            vis_t = torch.from_numpy(visual_np).unsqueeze(0)  # -> (1, C, H, W)
            vis_t = vis_t.unsqueeze(1)  # Add time dimension (1, 1, C, H, W)
            vis_t = vis_t.to(self.device)

            prop_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)
            prop_t = prop_t.unsqueeze(1)  # Add singleton dimension (1, 1, D_prop)
            
            data = {'visual': vis_t, 'proprio': prop_t}
            lat = self.wm.encode_obs(data)
            
            if self.with_proprio:
                z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches, E_dim) -> (1, N_patches*E_dim)
                z_prop = lat['proprio']  # (1, 1, D_prop)
                z_prop = z_prop.squeeze(0)
                z = torch.cat([z_vis, z_prop], dim=-1)
                return z.squeeze(0)  # Keep gradients for fine-tuning
            else:
                z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim)
                return z_vis.squeeze(0)  # Keep gradients for fine-tuning
        else:
            # Original no-grad version
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
                lat = self.wm.encode_obs(data)
                
                if self.with_proprio:
                    z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches, E_dim) -> (1, N_patches*E_dim)
                    z_prop = lat['proprio']  # (1, 1, D_prop)
                    z_prop = z_prop.squeeze(0)
                    z = torch.cat([z_vis, z_prop], dim=-1)
                    return z.squeeze(0).cpu().numpy()
                else:
                    z_vis = lat['visual'].reshape(1, -1)  # (1, N_patches * E_dim)
                    return z_vis.squeeze(0).cpu().numpy()


import os
# point Matplotlib to /tmp (or any other writable dir)
os.environ['MPLCONFIGDIR'] = '/storage1/fs1/sibai/Active/ihab/tmp'
# make sure it exists
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

# now it's safe to import pyplot
import matplotlib.pyplot as plt
import wandb

# … all your existing imports …

from PyHJ.data import Batch

def compute_hj_value(x, y, theta, policy, helper_env, args):
    # set precise state without advancing dynamics
    obs_dict, _ = helper_env.env.reset(state=[x, y, theta])
    z0 = helper_env._encode(obs_dict)
    
    # Convert to numpy if it's a tensor (for fine-tuning case)
    if isinstance(z0, torch.Tensor):
        z0 = z0.detach().cpu().numpy()
    
    batch = Batch(obs=z0[None], info=Batch())
    a_old = policy(batch, model="actor_old").act
    q_val = policy.critic(batch.obs, a_old).cpu().item()
    return q_val

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
    
    # Parse unfreeze schedule
    if args.with_finetune and args.progressive_unfreezing:
        args.unfreeze_schedule = [int(x) for x in args.unfreeze_schedule.split(',')]
    else:
        args.unfreeze_schedule = []
    
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
    finetune_suffix = "_finetune" if args.with_finetune else ""
    wandb.init(
        project=f"ddpg-hj-latent-dubins", 
        name=f"ddpg-{args.dino_encoder}-{timestamp}{finetune_suffix}",
        config=vars(args)
    )
    writer    = SummaryWriter(log_dir=f"runs/ddpg_hj_latent/{args.dino_encoder}-{timestamp}{finetune_suffix}/logs")
    wb_logger = WandbLogger()
    wb_logger.load(writer)    # must load the TB writer
    logger    = wb_logger     # use W&B for offpolicy_trainer

    # 3) make your latent envs
    train_envs = DummyVectorEnv(
        [lambda: LatentDubinsEnv(args.dino_ckpt_dir, args.device, args.with_proprio, args.with_finetune)
         for _ in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: LatentDubinsEnv(args.dino_ckpt_dir, args.device, args.with_proprio, args.with_finetune)
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

    # 6) Setup fine-tuning optimizer if enabled
    finetune_optimizer = None
    unfrozen_layers = []
    if args.with_finetune:
        # Get the world model from the first training environment
        sample_env = train_envs.envs[0]
        finetune_optimizer, unfrozen_layers = setup_finetune_optimizer(
            sample_env.wm, args.dino_encoder, args.finetune_lr, args.finetune_layers
        )
        
        # Apply the same freezing/unfreezing to all environments
        for env in train_envs.envs + test_envs.envs:
            encoder_layers = get_encoder_layers(env.wm, args.dino_encoder)
            if encoder_layers:
                # Freeze all first
                for param in env.wm.parameters():
                    param.requires_grad = False
                # Then unfreeze the same layers
                layers_to_unfreeze = encoder_layers[-args.finetune_layers:] if args.finetune_layers <= len(encoder_layers) else encoder_layers
                for layer in layers_to_unfreeze:
                    for param in layer.parameters():
                        param.requires_grad = True

    # 7) assemble your avoid‐DDPG policy
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

    # 8) hook into policy.learn to capture losses and update fine-tuning
    orig_learn = policy.learn
    policy.last_actor_loss  = 0.0
    policy.last_critic_loss = 0.0
    policy.last_finetune_loss = 0.0
    
    def learn_and_record(batch, **kw):
        metrics = orig_learn(batch, **kw)
        policy.last_actor_loss  = metrics["loss/actor"]
        policy.last_critic_loss = metrics["loss/critic"]
        
        # Fine-tuning update
        if args.with_finetune and finetune_optimizer is not None:
            # The loss for fine-tuning should be the same as the critic loss
            # since we want the encoder to help the critic make better predictions
            finetune_loss = metrics["loss/critic"]
            policy.last_finetune_loss = finetune_loss
            
            # Update encoder parameters
            finetune_optimizer.zero_grad()
            # The gradients should already be computed from the critic loss
            finetune_optimizer.step()
        
        return metrics
    policy.learn = learn_and_record

    # 9) define train_fn to log those to W&B
    def train_fn(epoch: int, step_idx: int):
        log_dict = {
            "loss/actor":  policy.last_actor_loss,
            "loss/critic": policy.last_critic_loss,
        }
        if args.with_finetune:
            log_dict["loss/finetune"] = policy.last_finetune_loss
        wandb.log(log_dict)

    # 10) collectors
    buffer          = VectorReplayBuffer(args.buffer_size, args.training_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    print("collecting some data first")
    train_collector.collect(10000)
    print("done collecting some data first")
    test_collector  = Collector(policy, test_envs)

    # 11) choose headings & helper env
    thetas     = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    helper_env = LatentDubinsEnv(args.dino_ckpt_dir, args.device, args.with_proprio, args.with_finetune)

    # 12) training loop 
    log_path = Path(f"runs/ddpg_hj_latent/{args.dino_encoder}-{timestamp}{finetune_suffix}")
    
    for epoch in range(1, args.total_episodes + 1):
        print(f"\n=== Epoch {epoch}/{args.total_episodes} ===")

        # Progressive unfreezing
        if args.with_finetune and args.progressive_unfreezing and epoch in args.unfreeze_schedule:
            sample_env = train_envs.envs[0]
            unfrozen_layers, new_optimizer = progressive_unfreeze(
                sample_env.wm, args.dino_encoder, epoch, args.unfreeze_schedule, unfrozen_layers
            )
            if new_optimizer is not None:
                finetune_optimizer = new_optimizer
                # Apply the same unfreezing to all environments
                for env in train_envs.envs + test_envs.envs:
                    env_unfrozen_layers, _ = progressive_unfreeze(
                        env.wm, args.dino_encoder, epoch, args.unfreeze_schedule, 
                        unfrozen_layers[:-1]  # All but the last layer we just added
                    )

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
        
        # Add fine-tuning specific metrics
        if args.with_finetune:
            numeric["train/unfrozen_layers"] = len(unfrozen_layers)
            if finetune_optimizer is not None:
                numeric["train/finetune_lr"] = finetune_optimizer.param_groups[0]['lr']
        
        wandb.log(numeric, step=epoch)

        # c) save policy checkpoint
        ckpt_dir = log_path / f"epoch_id_{epoch}"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        torch.save(policy.state_dict(), ckpt_dir/"policy.pth")
        
        # Save fine-tuned encoder if enabled
        if args.with_finetune and finetune_optimizer is not None:
            sample_env = train_envs.envs[0]
            torch.save(sample_env.wm.state_dict(), ckpt_dir/"finetuned_encoder.pth")

        # d) plot latent‐space HJ filter & log
        fig1, fig2 = plot_hj(policy, helper_env, thetas, args)
        wandb.log({
            "HJ_latent/binary":     wandb.Image(fig1),
            "HJ_latent/continuous": wandb.Image(fig2),
        })
        plt.close(fig1); plt.close(fig2)

    print("Training complete.")
    
    # Save final models
    final_ckpt_dir = log_path / "final"
    final_ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(policy.state_dict(), final_ckpt_dir/"policy.pth")
    
    if args.with_finetune and finetune_optimizer is not None:
        sample_env = train_envs.envs[0]
        torch.save(sample_env.wm.state_dict(), final_ckpt_dir/"finetuned_encoder.pth")
        print(f"Fine-tuned encoder saved to {final_ckpt_dir/'finetuned_encoder.pth'}")

if __name__ == "__main__":
    main()


# Usage examples:
# For fine-tuning with proprioception:
# python "train_HJ_dubinslatent(can_fine_tune_PVR).py" --with_proprio --with_finetune --dino_encoder dino --finetune_lr 1e-5 --finetune_layers 3

# For fine-tuning without proprioception:
# python "train_HJ_dubinslatent(can_fine_tune_PVR).py" --with_finetune --dino_encoder r3m --finetune_lr 5e-6 --finetune_layers 2

# For progressive unfreezing:
# python "train_HJ_dubinslatent(can_fine_tune_PVR).py" --with_finetune --progressive_unfreezing --unfreeze_schedule "25,50,75" --dino_encoder vc1

# For comparison without fine-tuning:
# python "train_HJ_dubinslatent(can_fine_tune_PVR).py" --dino_encoder dino