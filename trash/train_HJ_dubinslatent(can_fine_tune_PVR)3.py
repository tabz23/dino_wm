
import argparse
import os
from pathlib import Path
import torch
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import torch.nn.functional as F

# PyHJ components for DDPG training
from PyHJ.data import Collector, VectorReplayBuffer, Batch
from PyHJ.trainer import offpolicy_trainer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import DDPGPolicy

# Load DINO-WM via plan.load_model
from plan import load_model

# Underlying Dubins Gym env
from env.dubins.dubins import DubinsEnv

import yaml
import matplotlib.pyplot as plt
import wandb

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

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg_parser = argparse.ArgumentParser()
    for key, val in sorted(cfg.items()):
        arg_t = args_type(val)
        cfg_parser.add_argument(f"--{key}", type=arg_t, default=arg_t(val))
    cfg_args = cfg_parser.parse_args(remaining)

    for key, val in vars(cfg_args).items():
        setattr(args, key.replace("-", "_"), val)

    return args

def load_shared_world_model(ckpt_dir: str, device: str):
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
    def __init__(self, shared_wm=None, ckpt_dir: str = None, device: str = 'cuda', with_proprio: bool = False):
        super().__init__()
        self.env = DubinsEnv()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.with_proprio = with_proprio

        if shared_wm is not None:
            self.wm = shared_wm
            print("Using shared world model")
        else:
            ckpt_dir = Path(ckpt_dir)
            hydra_cfg = ckpt_dir / 'hydra.yaml'
            snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
            train_cfg = OmegaConf.load(str(hydra_cfg))
            num_action_repeat = train_cfg.num_action_repeat
            self.wm = load_model(snapshot, train_cfg, num_action_repeat, device=self.device)
            self.wm.eval()
            print(f"Loaded new world model from {ckpt_dir}")

        # Define observation space as a dictionary matching raw observation
        visual_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        proprio_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({'visual': visual_space, 'proprio': proprio_space})
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        reset_out = self.env.reset(**kwargs)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        return obs, {}

    def step(self, action):
        obs_out, _, done, info = self.env.step(action)
        terminated = done
        truncated = False
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        h_s = info.get('h', 0.0) * 3  # Scale reward
        return obs, h_s, terminated, truncated, info

    def _encode(self, obs):
        # Used for plotting or external purposes
        visual = obs['visual']
        proprio = obs['proprio']
        visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
        vis_t = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(1).to(self.device)
        prop_t = torch.from_numpy(proprio).unsqueeze(0).unsqueeze(1).to(self.device)
        data = {'visual': vis_t, 'proprio': prop_t}
        with torch.no_grad():
            lat = self.wm.encode_obs(data)
            z_vis = lat['visual'].reshape(1, -1)
            if self.with_proprio:
                z_prop = lat['proprio'].squeeze(1)
                z = torch.cat([z_vis, z_prop], dim=-1)
            else:
                z = z_vis
            return z.squeeze(0).cpu().numpy()

class avoid_DDPGPolicy_annealing(DDPGPolicy):
    def __init__(self, critic, critic_optim, tau, gamma, exploration_noise, reward_normalization, estimation_step, action_space, actor, actor_optim, actor_gradient_steps, wm, with_proprio, finetune_mode=False):
        super().__init__(actor, actor_optim, critic, critic_optim, tau, gamma, exploration_noise, reward_normalization, estimation_step, action_space)
        self.wm = wm
        self.with_proprio = with_proprio
        self.finetune_mode = finetune_mode
        self.actor_gradient_steps = actor_gradient_steps

    def _encode_batch(self, raw_obs_batch, enable_grad=False):
        # raw_obs_batch is a dict with 'visual' and 'proprio' as NumPy arrays
        visuals = raw_obs_batch['visual']  # Shape: (B, H, W, C)
        proprios = raw_obs_batch['proprio']  # Shape: (B, D_prop)
        vis_t = torch.from_numpy(visuals).to(self.device).permute(0, 3, 1, 2) / 255.0  # (B, C, H, W)
        prop_t = torch.from_numpy(proprios).to(self.device)
        data = {'visual': vis_t.unsqueeze(1), 'proprio': prop_t.unsqueeze(1)}  # (B, 1, C, H, W) and (B, 1, D_prop)
        context = torch.enable_grad() if enable_grad else torch.no_grad()
        with context:
            lat = self.wm.encode_obs(data)
            z_vis = lat['visual'].reshape(len(visuals), -1)
            if self.with_proprio:
                z_prop = lat['proprio'].squeeze(1)
                z = torch.cat([z_vis, z_prop], dim=-1)
            else:
                z = z_vis
        return z

    def forward(self, batch, state=None, model="actor", input="obs", **kwargs):
        if model == "actor":
            raw_obs = batch[input]  # Dictionary with 'visual' and 'proprio'
            z = self._encode_batch(raw_obs, enable_grad=self.finetune_mode)
            return self.actor(z), state
        return super().forward(batch, state=state, model=model, input=input, **kwargs)

    def learn(self, batch, **kwargs):
        # Encode observations with gradients if finetuning
        enable_grad = self.finetune_mode
        z = self._encode_batch(batch.obs, enable_grad=enable_grad)
        z_next = self._encode_batch(batch.obs_next, enable_grad=enable_grad)

        # Compute target Q-value
        with torch.no_grad():
            a_next = self.actor(z_next)
            q_next = self.critic(z_next, a_next)
            target_q = batch.rew + (1 - batch.done) * self._gamma * q_next

        # Compute current Q-value
        act_tensor = torch.from_numpy(batch.act).to(self.device)
        q = self.critic(z, act_tensor)
        critic_loss = F.mse_loss(q, target_q)

        # Optimize critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor loss
        for _ in range(self.actor_gradient_steps):
            a = self.actor(z)
            q = self.critic(z, a)
            actor_loss = -q.mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        # Update target networks
        self.sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item(),
            "loss/encoder": critic_loss.item() if self.finetune_mode else 0.0
        }

def setup_encoder_finetuning(wm, args):
    if not args.with_finetune:
        return None
    # Freeze all encoder parameters initially
    for param in wm.encoder.parameters():
        param.requires_grad = False
    # Unfreeze the last few layers based on encoder type
    trainable_params = []
    encoder_layers = []

    if args.dino_encoder == "dino":
        if hasattr(wm.encoder, 'blocks'):
            encoder_layers = list(wm.encoder.blocks)[-args.finetune_layers:]
        elif hasattr(wm.encoder, 'transformer'):
            if hasattr(wm.encoder.transformer, 'layers'):
                encoder_layers = list(wm.encoder.transformer.layers)[-args.finetune_layers:]

    elif args.dino_encoder == "r3m":
        if hasattr(wm.encoder, 'convnet'):
            conv_layers = []
            for name, module in wm.encoder.convnet.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                    conv_layers.append(module)
            encoder_layers = conv_layers[-args.finetune_layers:]
        elif hasattr(wm.encoder, 'transformer'):
            encoder_layers = list(wm.encoder.transformer.layers)[-args.finetune_layers:]

    elif args.dino_encoder == "resnet":
        if hasattr(wm.encoder, 'layer4'):
            encoder_layers = [wm.encoder.layer4]
            if args.finetune_layers > 1 and hasattr(wm.encoder, 'layer3'):
                encoder_layers.insert(0, wm.encoder.layer3)
        elif hasattr(wm.encoder, 'layers'):
            encoder_layers = list(wm.encoder.layers)[-args.finetune_layers:]

    elif args.dino_encoder == "vc1":
        if hasattr(wm.encoder, 'blocks'):
            encoder_layers = list(wm.encoder.blocks)[-args.finetune_layers:]
        elif hasattr(wm.encoder, 'transformer'):
            if hasattr(wm.encoder.transformer, 'layers'):
                encoder_layers = list(wm.encoder.transformer.layers)[-args.finetune_layers:]

    elif args.dino_encoder == "scratch":
        if hasattr(wm.encoder, 'layers'):
            encoder_layers = list(wm.encoder.layers)[-args.finetune_layers:]
        elif hasattr(wm.encoder, 'blocks'):
            encoder_layers = list(wm.encoder.blocks)[-args.finetune_layers:]

    elif args.dino_encoder == "dino_cls":
        if hasattr(wm.encoder, 'blocks'):
            encoder_layers = list(wm.encoder.blocks)[-args.finetune_layers:]
        if hasattr(wm.encoder, 'head'):
            encoder_layers.append(wm.encoder.head)

    # Unfreeze selected layers
    for layer in encoder_layers:
        for param in layer.parameters():
            param.requires_grad = True
            trainable_params.append(param)

    if trainable_params:
        encoder_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.finetune_lr,
            weight_decay=args.weight_decay_pyhj if hasattr(args, 'weight_decay_pyhj') else 1e-4
        )
        print(f"Encoder finfining enabled:")
        print(f"  - Encoder: {args.dino_encoder}")
        print(f"  - Learning rate: {args.finetune_lr}")
        print(f"  - Layers to finetune: {args.finetune_layers}")
        num_trainable_params = sum(p.numel() for p in trainable_params)
        print(f"  - Trainable parameters: {num_trainable_params}")
        return encoder_optimizer
    else:
        print("Warning: No encoder layers found for finetuning")
        return None

def compute_hj_value(x, y, theta, policy, helper_env, args):
    obs_dict, _ = helper_env.env.reset(state=[x, y, theta])
    z = helper_env._encode(obs_dict)
    batch = Batch(obs=np.array([{'visual': obs_dict['visual'], 'proprio': obs_dict['proprio']}], dtype=object), info=Batch())
    a_old = policy(batch, model="actor_old").act
    q_val = policy.critic(torch.from_numpy(z).to(args.device, dtype=torch.float32).unsqueeze(0), torch.from_numpy(a_old).to(args.device, dtype=torch.float32)).cpu().item()
    return q_val

def plot_hj(policy, helper_env, thetas, args):
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
        axes1[i].imshow((vals.T > 0), extent=(args.x_min, args.x_max, args.y_min, args.y_max), origin="lower", cmap='RdYlBu')
        axes1[i].set_title(f"θ={theta:.2f} (safe mask)")
        axes1[i].set_xlabel("x")
        axes1[i].set_ylabel("y")
        im = axes2[i].imshow(vals.T, extent=(args.x_min, args.x_max, args.y_min, args.y_max), origin="lower", cmap='viridis')
        axes2[i].set_title(f"θ={theta:.2f} (HJ value)")
        axes2[i].set_xlabel("x")
        axes2[i].set_ylabel("y")
        fig2.colorbar(im, ax=axes2[i])
    fig1.tight_layout()
    fig2.tight_layout()
    return fig1, fig2

def main():
    args = get_args_and_merge_config()
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    shared_wm = load_shared_world_model(args.dino_ckpt_dir, args.device)

    encoder_optimizer = setup_encoder_finetuning(shared_wm, args)

    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    wandb.init(project=f"ddpg-hj-latent-dubins", name=f"ddpg-{args.dino_encoder}-{timestamp}", config=vars(args))
    writer = SummaryWriter(log_dir=f"runs/ddpg_hj_latent/{args.dino_encoder}-{timestamp}/logs")
    wb_logger = WandbLogger() 
    wb_logger.load(writer)
    logger = wb_logger

    train_envs = DummyVectorEnv([lambda: LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio) for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([lambda: LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio) for _ in range(args.test_num)])

    # Latent space shape (example, adjust based on wm.encoder output)
    latent_dim = 75264  # Based on previous output: [1, 75264] or [1, 75274] with proprio
    if args.with_proprio:
        latent_dim += 10  # Assuming proprio adds 10 dimensions
    state_shape = (latent_dim,)
    action_space = train_envs.action_space[0]
    action_shape = action_space.shape
    max_action = torch.tensor(action_space.high, device=args.device, dtype=torch.float32)

    # Define networks with latent space as input
    critic_net = Net(state_shape, action_shape, hidden_sizes=args.critic_net, activation=getattr(torch.nn, args.critic_activation), concat=True, device=args.device)
    critic = Critic(critic_net, device=args.device).to(args.device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay_pyhj)

    actor_net = Net(state_shape, hidden_sizes=args.control_net, activation=getattr(torch.nn, args.actor_activation), device=args.device)
    actor = Actor(actor_net, action_shape, max_action=max_action, device=args.device).to(args.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)

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
        wm=shared_wm,
        with_proprio=args.with_proprio,
        finetune_mode=args.with_finetune,
    )

    # Hook into policy.learn to capture losses
    orig_learn = policy.learn
    def learn_and_record(batch, **kwargs):
        metrics = orig_learn(batch, **kwargs)
        if encoder_optimizer is not None and args.with_finetune:
            encoder_optimizer.step()
            encoder_optimizer.zero_grad()
        return metrics
    policy.learn = learn_and_record

    # Define train_fn to log losses to W&B
    def train_fn(epoch: int, step_idx: int):
        log_dict = {
            "loss/actor": policy.last_actor_loss,
            "loss/critic": policy.last_critic_loss,
        }
        if encoder_optimizer is not None:
            log_dict["loss/encoder"] = policy.last_encoder_loss
            log_dict["finetune/learning_rate"] = encoder_optimizer.param_groups[0]['lr']
        wandb.log(log_dict)

    buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    print("Collecting some data first")
    train_collector.collect(n_step=10000)
    print("Done collecting some data first")
    test_collector = Collector(policy, test_envs)

    thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    helper_env = LatentDubinsEnv(shared_wm=shared_wm, with_proprio=args.with_proprio)

    log_path = Path(f"runs/ddpg_hj_latent/{args.dino_encoder}-{timestamp}")
    for epoch in range(1, args.total_episodes + 1):
        print(f"\n=== Epoch {epoch}/{args.total_episodes} ===")
        if args.with_finetune:
            shared_wm.encoder.train()
        else:
            shared_wm.encoder.eval()

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
            train_fn=train_fn,
            save_best_fn=None,
            logger=logger,
        )

        numeric = {f"train/{k}": float(v) for k, v in stats.items() if isinstance(v, (int, float, np.generic))}
        wandb.log(numeric, step=epoch)

        ckpt_dir = log_path / f"epoch_id_{epoch}"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        torch.save(policy.state_dict(), ckpt_dir / "policy.pth")
        if args.with_finetune and encoder_optimizer is not None:
            torch.save({
                'encoder_state_dict': shared_wm.encoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            }, ckpt_dir / "encoder_finetune.pth")

        try:
            shared_wm.encoder.eval()
            fig1, fig2 = plot_hj(policy, helper_env, thetas, args)
            wandb.log({
                "HJ_latent/binary": wandb.Image(fig1),
                "HJ_latent/continuous": wandb.Image(fig2),
            }, step=epoch)
            plt.close(fig1)
            plt.close(fig2)
        except Exception as e:
            print(f"Error plotting HJ values: {e}")

    print("Training complete.")

if __name__ == "__main__":
    main()




# python "train_HJ_dubinslatent(can_fine_tune_PVR)4.py" --with_finetune --dino_encoder r3m --finetune_lr 5e-6 --finetune_layers 2 --step-per-epoch 100 --nx 20 --ny 20
# python "train_HJ_dubinslatent(can_fine_tune_PVR)4.py" --step-per-epoch 100 --nx 20 --ny 20