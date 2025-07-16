import argparse
import os
from pathlib import Path
import torch
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import yaml
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from copy import deepcopy

# Load DINO-WM via plan.load_model
from plan import load_model

# Underlying Dubins Gym env
from env.dubins.dubins import DubinsEnv
from gymnasium.spaces import Box

# Set up matplotlib config
os.environ['MPLCONFIGDIR'] = '/storage1/fs1/sibai/Active/ihab/tmp'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

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
        help="Which encoder to use: dino, r3m, vc1, resnet, dino_cls."
    )
    parser.add_argument(
        "--with_finetune", action="store_true",
        help="Flag to fine-tune the encoder backbone"
    )
    parser.add_argument(
        "--encoder_lr", type=float, default=1e-5,
        help="Learning rate for the encoder fine-tuning"
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
    print(f"Loaded shared world model from {ckpt_dir}")
    return wm

class RawDubinsEnv(gym.Env):
    def __init__(self, device: str = None, with_proprio: bool = False):
        super().__init__()
        self.env = DubinsEnv()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.with_proprio = with_proprio
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        return obs, {}

    def step(self, action):
        obs_out, _, done, info = self.env.step(action)
        terminated = done
        truncated = False
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        h_s = info.get('h', 0.0) * 3  # Multiply by 3 to make HJ easier to learn
        return obs, h_s, terminated, truncated, info

class CustomReplayBuffer:
    def __init__(self, size: int, device: str):
        self.size = size
        self.device = torch.device(device)
        self.buffer = []
        self.position = 0

    def add(self, obs, act, rew, obs_next, done):
        data = (obs, act, rew, obs_next, done)
        if len(self.buffer) < self.size:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        obs_batch = [self.buffer[i][0] for i in indices]
        act_batch = [self.buffer[i][1] for i in indices]
        rew_batch = [self.buffer[i][2] for i in indices]
        obs_next_batch = [self.buffer[i][3] for i in indices]
        done_batch = [self.buffer[i][4] for i in indices]
        return obs_batch, act_batch, rew_batch, obs_next_batch, done_batch

    def __len__(self):
        return len(self.buffer)

def encode_batch(obs_batch, wm, device, with_proprio):
    visual_batch = []
    proprio_batch = []
    for obs in obs_batch:
        if isinstance(obs, dict):
            visual = obs['visual']
            proprio = obs['proprio']
        elif isinstance(obs, (tuple, list)) and len(obs) == 2:
            visual, proprio = obs
        else:
            raise ValueError(f"Unexpected obs type: {type(obs)}")
        visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
        vis_t = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(1).to(device)
        prop_t = torch.from_numpy(proprio).unsqueeze(0).unsqueeze(1).to(device)
        visual_batch.append(vis_t)
        proprio_batch.append(prop_t)
    visual_batch = torch.cat(visual_batch, dim=0)
    proprio_batch = torch.cat(proprio_batch, dim=0)
    data = {'visual': visual_batch, 'proprio': proprio_batch}
    lat = wm.encode_obs(data)
    if with_proprio:
        z_vis = lat['visual'].reshape(lat['visual'].shape[0], -1)
        z_prop = lat['proprio'].squeeze(1)
        z = torch.cat([z_vis, z_prop], dim=-1)
    else:
        z = lat['visual'].reshape(lat['visual'].shape[0], -1)
    return z

def compute_hj_value(x, y, theta, policy, helper_env, wm, device, args):
    obs_dict, _ = helper_env.env.reset(state=[x, y, theta])
    z = encode_batch([obs_dict], wm, device, args.with_proprio)
    z = z.to(device)
    with torch.no_grad():
        a_old = policy.actor_old(z)
        q_val = policy.critic(z, a_old).item()
    return q_val

def plot_hj(policy, helper_env, wm, thetas, args, device):
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
                vals[ix, iy] = compute_hj_value(x, y, theta, policy, helper_env, wm, device, args)
        axes1[i].imshow(
            (vals.T > 0),
            extent=(args.x_min, args.x_max, args.y_min, args.y_max),
            origin="lower",
            cmap='RdYlBu'
        )
        axes1[i].set_title(f"θ={theta:.2f} (safe mask)")
        axes1[i].set_xlabel("x")
        axes1[i].set_ylabel("y")
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

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, max_action):
        super().__init__()
        self.net = self.build_net(state_dim, action_dim, hidden_sizes, activation)
        self.max_action = max_action

    def build_net(self, input_dim, output_dim, hidden_sizes, activation):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(torch.nn, activation)())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        layers.append(torch.nn.Tanh())
        return torch.nn.Sequential(*layers)

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.net = self.build_net(state_dim + action_dim, 1, hidden_sizes, activation)

    def build_net(self, input_dim, output_dim, hidden_sizes, activation):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(torch.nn, activation)())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        return torch.nn.Sequential(*layers)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class AvoidDDPGPolicy:
    def __init__(self, actor, actor_optim, critic, critic_optim, tau, gamma, exploration_noise, device):
        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_old = deepcopy(actor)
        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_old = deepcopy(critic)
        self.tau = tau
        self.gamma = gamma
        self.exploration_noise = exploration_noise
        self.device = device
        self.actor_gradient_steps = 5  # Consistent with original policy

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def sync_weight(self):
        for target_param, param in zip(self.actor_old.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_old.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def learn(self, batch, wm, with_finetune, encoder_optim):
        obs_batch, act_batch, rew_batch, obs_next_batch, done_batch = batch
        z = encode_batch(obs_batch, wm, self.device, args.with_proprio)
        z_next = encode_batch(obs_next_batch, wm, self.device, args.with_proprio)
        act = torch.tensor(act_batch, dtype=torch.float32, device=self.device)
        rew = torch.tensor(rew_batch, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Critic loss (consistent with original HJ computation)
        with torch.no_grad():
            a_next = self.actor_old(z_next)
            target_q = self.critic_old(z_next, a_next)
            target_q = self.gamma * torch.minimum(rew, target_q) + (1 - self.gamma) * rew
        current_q = self.critic(z, act)
        critic_loss = torch.nn.functional.mse_loss(current_q, target_q)

        # Actor loss
        a = self.actor(z)
        actor_loss = -self.critic(z, a).mean()

        # Backpropagation
        self.critic_optim.zero_grad()
        if with_finetune:
            encoder_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        if with_finetune:
            encoder_optim.zero_grad()
        for _ in range(self.actor_gradient_steps):
            a = self.actor(z)
            actor_loss = -self.critic(z, a).mean()
            actor_loss.backward()
            self.actor_optim.step()
            if with_finetune:
                encoder_optim.step()

        self.sync_weight()

        return {"loss/actor": actor_loss.item(), "loss/critic": critic_loss.item()}

def main():
    args = get_args_and_merge_config()
    args.critic_lr = float(args.critic_lr)
    args.actor_lr = float(args.actor_lr)
    args.tau = float(args.tau)
    args.gamma_pyhj = float(args.gamma_pyhj)
    args.exploration_noise = float(args.exploration_noise)
    args.step_per_epoch = int(args.step_per_epoch)
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    shared_wm = load_shared_world_model(args.dino_ckpt_dir, device)
    if args.with_finetune:
        shared_wm.train()
    else:
        shared_wm.eval()

    train_envs = [RawDubinsEnv(device=device, with_proprio=args.with_proprio) for _ in range(args.training_num)]
    test_envs = [RawDubinsEnv(device=device, with_proprio=args.with_proprio) for _ in range(args.test_num)]

    state_dim = encode_batch([train_envs[0].reset()[0]], shared_wm, device, args.with_proprio).shape[1]
    action_dim = train_envs[0].action_space.shape[0]
    max_action = torch.tensor(train_envs[0].action_space.high, device=device, dtype=torch.float32)

    actor = Actor(state_dim, action_dim, args.control_net, args.actor_activation, max_action).to(device)
    critic = Critic(state_dim, action_dim, args.critic_net, args.critic_activation).to(device)

    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)

    if args.with_finetune:
        encoder_optim = torch.optim.AdamW(shared_wm.parameters(), lr=args.encoder_lr)
    else:
        encoder_optim = None

    policy = AvoidDDPGPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        tau=args.tau,
        gamma=args.gamma_pyhj,
        exploration_noise=args.exploration_noise,
        device=device
    )

    buffer = CustomReplayBuffer(args.buffer_size, device)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    wandb.init(
        project=f"ddpg-hj-latent-dubins",
        name=f"ddpg-{args.dino_encoder}-{timestamp}",
        config=vars(args)
    )
    writer = SummaryWriter(log_dir=f"runs/ddpg_hj_latent/{args.dino_encoder}-{timestamp}/logs")

    thetas = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
    helper_env = RawDubinsEnv(device=device, with_proprio=args.with_proprio)

    for epoch in range(1, args.total_episodes + 1):
        print(f"\n=== Epoch {epoch}/{args.total_episodes} ===")

        # Collect data
        for env in train_envs:
            obs, _ = env.reset()
            done = False
            while not done:
                z = encode_batch([obs], shared_wm, device, args.with_proprio)
                with torch.no_grad():
                    act = policy.actor(z).cpu().numpy().flatten()
                if np.random.rand() < args.exploration_noise:
                    act = np.random.uniform(-1, 1, act.shape)
                obs_next, rew, terminated, truncated, _ = env.step(act)
                done = terminated or truncated
                buffer.add(obs, act, rew, obs_next, done)
                obs = obs_next

        # Train
        for _ in range(args.step_per_epoch):
            if len(buffer) < args.batch_size_pyhj:
                continue
            batch = buffer.sample(args.batch_size_pyhj)
            metrics = policy.learn(batch, shared_wm, args.with_finetune, encoder_optim)
            wandb.log(metrics, step=epoch)

        # Plot HJ
        fig1, fig2 = plot_hj(policy, helper_env, shared_wm, thetas, args, device)
        wandb.log({
            "HJ_latent/binary": wandb.Image(fig1),
            "HJ_latent/continuous": wandb.Image(fig2),
        }, step=epoch)
        plt.close(fig1)
        plt.close(fig2)

    print("Training complete.")

if __name__ == "__main__":
    main()
    
    
# python "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/trainHJ_dubinslatent(canfinetunepvr)7.py" --dino_ckpt_dir "/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs/dubins/fully trained(prop repeated 3 times)" --config train_HJ_configs.yaml --dino_encoder r3m --with_finetune --encoder_lr 1e-5