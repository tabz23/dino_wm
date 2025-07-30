import argparse
import os
from pathlib import Path
import torch
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import functools

# Import your highway environment
from PyHJ.reach_rl_gym_envs.ra_highway_10d_with_render_onlycost_rew_is_cost_nodisturb import Highway_10D_game_Env2cost
# PyHJ.reach_rl_gym_envs.ra_highway_10d_with_render_onlycost_rew_is_cost_nodisturb
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
    parser = argparse.ArgumentParser("DDPG HJ on Highway Environment")
    parser.add_argument(
        "--config", type=str, default="/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_configs.yaml",
        help="Path to your flat YAML of hyperparameters"
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

def smart_controller(state):
    """
    Improved controller that:
    1. Drives toward +Y direction (theta ≈ π/2)
    2. Stays within road boundaries (X between 0 and 2)
    3. Avoids other cars by slight sideways movement
    4. Properly normalizes angles to avoid cosine symmetry issues
    """
    ego_pos = np.array([state[3], state[4]])  # ego x, y
    ego_theta = state[6]  # ego angle
    ego_vel = state[5]    # ego speed
    
    # Other car positions
    car0_pos = np.array([state[0], state[1]])  # disturbance car
    car2_pos = np.array([state[7], state[8]])  # obstacle car
    
    # Initialize actions
    accel = 0.0
    steer = 0.0
    
    # Primary goal: maintain forward direction (theta ≈ π/2)
    target_angle = np.pi / 2  # straight up (+Y direction)
    
    # Normalize ego_theta to [0, 2π) and compute angle difference
    ego_theta_norm = ego_theta % (2 * np.pi)
    angle_diff = target_angle - ego_theta_norm
    
    # Normalize angle difference to [-π, π]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    
    # Base steering to align toward π/2
    steer = np.clip(angle_diff * 0.5, -0.5, 0.5)  # Gentle steering
    
    # Ensure theta stays within environment bounds [π/4, 3π/4]
    theta_min, theta_max = np.pi / 4, 3 * np.pi / 4
    if ego_theta < theta_min:
        steer = max(steer, 0.5)  # Turn left to increase theta
    elif ego_theta > theta_max:
        steer = min(steer, -0.5)  # Turn right to decrease theta
    
    # BOUNDARY AVOIDANCE (highest priority)
    road_center = 1.0  # center of road (x = 1.0)
    boundary_margin = 0.2  # safety margin from boundaries
    
    if ego_pos[0] < boundary_margin:
        # Too close to left boundary (x=0) - steer right
        steer = min(steer, -0.8)  # Strong right turn (negative steering)
    elif ego_pos[0] > (2.0 - boundary_margin):
        # Too close to right boundary (x=2) - steer left
        steer = max(steer, 0.8)  # Strong left turn (positive steering)
    else:
        # Within safe boundaries - gentle correction toward center
        center_diff = road_center - ego_pos[0]  # positive if we need to go right
        center_correction = -center_diff * 0.2  # Negative for right, positive for left
        steer = np.clip(steer + center_correction, -0.5, 0.5)
    
    # COLLISION AVOIDANCE with other cars
    collision_distance = 1.2  # Distance to start avoiding
    avoidance_strength = 0.4
    
    # SPEED CONTROL
    target_speed = 2.0  # Moderate forward speed
    speed_diff = target_speed - ego_vel
    
    # Check if we're well-aligned (within ~17 degrees of forward)
    well_aligned = abs(angle_diff) < 0.3
    
    if ego_vel < 0.5 and well_aligned:
        # Too slow AND pointing right way - accelerate
        accel = 0.8
    elif ego_vel > 3.0:
        # Too fast - decelerate
        accel = -0.5
    elif not well_aligned:
        # Pointing wrong way - slow down to reorient
        accel = -0.3 if ego_vel > 1.0 else 0.1
    else:
        # Normal speed control when well aligned
        accel = np.clip(speed_diff * 0.5, -0.5, 0.5)
    
    # Reduce speed near boundaries for better control
    if ego_pos[0] < 0.3 or ego_pos[0] > 1.7:
        accel = min(accel, 0.2)
    
    # Final clipping
    accel = np.clip(accel, -1.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    
    # Return full action space (4D) with zeros for disturbance actions
    return np.array([accel, steer, 0.0, 0.0], dtype=np.float32)

class HighwayWrapper(gym.Env):
    """Wrapper for Highway environment to work with the HJ training"""
    def __init__(self, device: str = None):
        super().__init__()
        self.max_steps = 400  
        self.env = Highway_10D_game_Env2cost()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, state=None):
        self.current_step = 0  # reset step count
        if state is not None:
            obs, info = self.env.reset(initial_state=state)
        else:
            obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        # The environment expects 4D action but we only control first 2
        if len(action) == 2:
            full_action = np.array([action[0], action[1], 0.0, 0.0], dtype=np.float32)
        else:
            full_action = action
        obs, cost, done, truncated, info = self.env.step(full_action)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            truncated = True 
        # Use cost as reward for HJ (negative is unsafe)
        return obs, cost, done, truncated, info
    
    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)
    
    def close(self):
        self.env.close()

class OptimizedReplayBuffer:
    """Optimized replay buffer using pre-allocated tensors"""
    def __init__(self, size: int, device: str, obs_dim, action_dim):
        self.size = size
        self.device = torch.device(device)
        self.position = 0
        self.filled = 0
        
        # Pre-allocate tensors
        self.obs_buffer = torch.zeros((size, obs_dim), dtype=torch.float32, device=self.device)
        self.action_buffer = torch.zeros((size, action_dim), dtype=torch.float32, device=self.device)
        self.reward_buffer = torch.zeros((size, 1), dtype=torch.float32, device=self.device)
        self.next_obs_buffer = torch.zeros((size, obs_dim), dtype=torch.float32, device=self.device)
        self.done_buffer = torch.zeros((size, 1), dtype=torch.float32, device=self.device)

    def add(self, obs, act, rew, obs_next, done):
        # Convert numpy arrays to tensors and store
        self.obs_buffer[self.position] = torch.from_numpy(obs).float()
        self.action_buffer[self.position] = torch.from_numpy(act).float()
        self.reward_buffer[self.position] = rew
        self.next_obs_buffer[self.position] = torch.from_numpy(obs_next).float()
        self.done_buffer[self.position] = float(done)
        
        self.position = (self.position + 1) % self.size
        self.filled = min(self.filled + 1, self.size)

    def sample(self, batch_size: int):
        indices = torch.randint(0, self.filled, (batch_size,))
        
        return (
            self.obs_buffer[indices],
            self.action_buffer[indices],
            self.reward_buffer[indices],
            self.next_obs_buffer[indices],
            self.done_buffer[indices]
        )

    def __len__(self):
        return self.filled

def compute_hj_grid_vectorized(policy, env, fixed_state, args, device):
    """Vectorized computation of HJ values for a grid at fixed state (varying only ego x,y)"""
    xs = np.linspace(-0.3, 2.3, args.nx)  # Road boundaries
    ys = np.linspace(0.0, 20.0, args.ny)  # Y range
    
    # Create grid of states
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    states = []
    
    # For each grid point, create state with fixed values except ego x,y
    for i in range(args.nx):
        for j in range(args.ny):
            state = fixed_state.copy()
            state[3] = xx[i, j]  # ego x
            state[4] = yy[i, j]  # ego y
            states.append(state)
    
    states = np.array(states)
    
    # Convert states to tensors and evaluate in batches
    batch_size = min(256, len(states))
    all_values = []
    
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            batch_states = torch.from_numpy(states[i:i+batch_size]).float().to(device)
            a = policy.actor_old(batch_states)[:, :2]  # Only use first 2 actions
            q_vals = policy.critic(batch_states, a).squeeze(-1)
            all_values.append(q_vals.cpu().numpy())
    
    values = np.concatenate(all_values).reshape(args.nx, args.ny)
    return values

def plot_hj_highway(policy, env, fixed_states, args, device):
    """Plot HJ for highway environment with different fixed states"""
    n_samples = len(fixed_states)
    
    # Create figure with 2 rows: environment visualization and HJ heatmaps
    fig = plt.figure(figsize=(6*n_samples, 12))
    
    for idx, fixed_state in enumerate(fixed_states):
        # Top row: Environment visualization
        ax_env = plt.subplot(2, n_samples, idx + 1)
        
        # Reset env to the fixed state and render
        env.reset(state=fixed_state)
        rgb_array = env.render(mode="rgb_array")
        ax_env.imshow(rgb_array)
        ax_env.axis('off')
        ax_env.set_title(f'Environment State {idx+1}')
        
        # Bottom row: HJ heatmap
        ax_hj = plt.subplot(2, n_samples, idx + n_samples + 1)
        
        # Compute HJ values
        vals = compute_hj_grid_vectorized(policy, env, fixed_state, args, device)
        
        # Plot heatmap
        im = ax_hj.imshow(
            vals.T,
            extent=(-0.3, 2.3, 0.0, 20.0),
            origin="lower",
            cmap='RdYlBu',
            aspect='auto'
        )
        
        # Mark the current ego position
        ego_x, ego_y = fixed_state[3], fixed_state[4]
        ax_hj.plot(ego_x, ego_y, 'k*', markersize=15, label='Ego')
        
        # Mark other car positions
        car0_x, car0_y = fixed_state[0], fixed_state[1]
        car2_x, car2_y = fixed_state[7], fixed_state[8]
        ax_hj.plot(car0_x, car0_y, 'ro', markersize=10, label='Car0')
        ax_hj.plot(car2_x, car2_y, 'mo', markersize=10, label='Car2')
        
        ax_hj.set_xlabel("Ego X")
        ax_hj.set_ylabel("Ego Y")
        ax_hj.set_title(f"HJ Values (Safe > 0)")
        ax_hj.legend()
        ax_hj.grid(True, alpha=0.3)
        
        # Add road boundaries
        ax_hj.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax_hj.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
        
        plt.colorbar(im, ax=ax_hj)
    
    plt.tight_layout()
    return fig

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, max_action):
        super().__init__()
        self.net = self.build_net(state_dim, action_dim, hidden_sizes, activation)
        self.register_buffer('max_action', max_action)

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
    def __init__(self, actor, actor_optim, critic, critic_optim, tau, gamma, exploration_noise, device, actor_gradient_steps=1, action_space=None):
        self.actor = actor
        self.actor_optim = actor_optim
        self.actor_old = deepcopy(actor)
        self.critic = critic
        self.critic_optim = critic_optim
        self.critic_old = deepcopy(critic)
        self.tau = tau
        self.gamma = gamma
        self._noise = exploration_noise
        self.device = device
        
        self.actor_gradient_steps = actor_gradient_steps
        self.new_expl = True
        self.warmup = False
        self._n_step = 1
        self._rew_norm = False
        self.action_low = action_space.low[:2]  # Only first 2 actions
        self.action_high = action_space.high[:2]

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    @torch.no_grad()
    def sync_weight(self):
        """Soft-update the weight for the target network"""
        for target_param, param in zip(self.actor_old.parameters(), self.actor.parameters()):
            target_param.mul_(1 - self.tau).add_(param.data, alpha=self.tau)
        for target_param, param in zip(self.critic_old.parameters(), self.critic.parameters()):
            target_param.mul_(1 - self.tau).add_(param.data, alpha=self.tau)

    def _target_q(self, obs_next_batch):
        """Predict the value of a state"""
        with torch.no_grad():
            a_next = self.actor_old(obs_next_batch) 
            target_q = self.critic_old(obs_next_batch, a_next)
        return target_q

    def _nstep_return_approximated_avoid_Bellman_equation(self, rew, terminal_value, gamma):
        """
        Implements the avoid Bellman equation.
        Convention: negative is unsafe
        V = min(l(x), V(x'))
        """
        target_shape = terminal_value.shape
        bsz = target_shape[0]
        
        # Take the worst between safety now and safety in the future
        target_q = gamma * torch.minimum(
            rew.reshape(bsz, 1),  # safety now
            terminal_value  # safety in the future
        ) + (1 - gamma) * rew.reshape(bsz, 1)  # discount toward safety now
        
        return target_q.reshape(target_shape)

    def compute_nstep_return(self, obs_batch, act_batch, rew_batch, obs_next_batch, done_batch):
        """Compute the target q values using the avoid Bellman equation"""
        # Get target Q values
        target_q = self._target_q(obs_next_batch)
        
        # Apply the avoid Bellman equation
        returns = self._nstep_return_approximated_avoid_Bellman_equation(rew_batch, target_q, self.gamma)
        
        return returns

    def learn(self, batch):
        """Update critic network and actor network"""
        obs_batch, act_batch, rew_batch, obs_next_batch, done_batch = batch
        
        # Compute target returns using avoid Bellman equation
        target_returns = self.compute_nstep_return(obs_batch, act_batch, rew_batch, obs_next_batch, done_batch)
        
        # Critic update
        current_q = self.critic(obs_batch, act_batch).flatten()  # Only use first 2 actions
        target_q = target_returns.flatten()
        td = current_q - target_q
        critic_loss = td.pow(2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update
        actor_losses = []
        for _ in range(self.actor_gradient_steps):
            a = self.actor(obs_batch)  # Only output 2 actions
            actor_loss = -self.critic(obs_batch, a).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            actor_losses.append(actor_loss.item())
        
        final_actor_loss = actor_losses[-1]

        # Soft update the parameters
        self.sync_weight()
        
        return {
            "loss/actor": final_actor_loss,
            "loss/critic": critic_loss.item(),
        }

    def exploration_action(self, obs, use_smart_controller=False):
        """Get action with exploration"""
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        else:
            obs_tensor = obs
        
        with torch.no_grad():
            # Get action from actor
            act = self.actor(obs_tensor).cpu().numpy().squeeze()
            
            # Value-based exploration
            if self.new_expl and not use_smart_controller:
                rand_act = np.random.uniform(self.action_low, self.action_high, act.shape)
                rand_act_tensor = torch.from_numpy(rand_act).float().unsqueeze(0).to(self.device)
                
                value = self.critic(obs_tensor, rand_act_tensor).cpu().numpy().squeeze()
                
                # Where random actions would be safe (values >= 0): use the random action
                if value >= 0:
                    act = rand_act
            
            # Add Gaussian noise
            if self._noise is not None and self._noise > 0:
                noise = np.random.normal(0, self._noise, act.shape)
                act = act + noise
            
            # Warmup override
            if self.warmup:
                act = np.random.uniform(self.action_low, self.action_high, act.shape)
        
        return np.clip(act, self.action_low, self.action_high)

class ParallelEnvCollector:
    """Collects transitions from multiple envs in parallel with safety switching"""
    def __init__(self, envs, policy, device):
        self.envs = envs
        self.policy = policy
        self.device = device
        self.num_envs = len(envs)

    def simulate_dynamics(self, state, action):
        """Simulate one step of dynamics without modifying environment"""
        s = state.copy()
        dt, eps = 0.1, 0.1
        
        # Apply the same dynamics as in Highway_10D_game_Env2cost.step()
        # car0 disturbance (no disturbance in this version)
        s[1] = state[1] - dt * state[2]
        s[2] = state[2] + dt * eps * 0  # No disturbance
        
        # ego car
        s[3] = state[3] + dt * state[5] * np.cos(state[6])
        s[4] = state[4] + dt * state[5] * np.sin(state[6])
        s[5] = state[5] + dt * 2 * action[0]
        s[6] = state[6] + dt * 2 * action[1]
        
        # car2 obstacle (no disturbance)
        s[8] = state[8] + dt * state[9]
        s[9] = state[9] + dt * eps * 0  # No disturbance
        
        return s

    def collect_trajectories(self, buffer, max_steps: int=1000):
        """Collect transitions using smart controller with HJ safety switching"""
        obs_list = [env.reset()[0] for env in self.envs]
        done_flags = [False] * self.num_envs
        steps = 0
        
        # Track switching statistics
        total_smart_actions = 0
        total_hj_interventions = 0
        
        # Track collision statistics for switching policy
        switching_total_steps = 0
        switching_violations = 0
        switching_episodes = 0
        switching_episodes_with_violation = 0
        episode_had_violation = [False] * self.num_envs
        
        # Track collision statistics for pure HJ policy
        hj_total_steps = 0
        hj_violations = 0
        hj_episodes = 0
        hj_episodes_with_violation = 0
        
        # Decide which envs will use switching vs pure HJ
        # Half use switching (smart + HJ safety), half use pure HJ for diversity
        use_switching = [i % 2 == 0 for i in range(self.num_envs)]

        while steps < max_steps:
            active_idxs = [i for i, d in enumerate(done_flags) if not d]
            if not active_idxs:
                # Reset and track episode statistics
                for i in range(self.num_envs):
                    if use_switching[i]:
                        switching_episodes += 1
                        if episode_had_violation[i]:
                            switching_episodes_with_violation += 1
                    else:
                        hj_episodes += 1
                        if episode_had_violation[i]:
                            hj_episodes_with_violation += 1
                    
                    obs_list[i] = self.envs[i].reset()[0]
                    done_flags[i] = False
                    episode_had_violation[i] = False
                continue

            # Get actions for active envs
            for idx, env_idx in enumerate(active_idxs):
                env = self.envs[env_idx]
                obs = obs_list[env_idx]
                
                if use_switching[env_idx]:
                    # Smart controller with HJ safety switching
                    smart_action = smart_controller(obs)[:2]  # Only take first 2 actions
                    
                    # Simulate next state using exact dynamics
                    next_state = self.simulate_dynamics(obs, smart_action)
                    
                    # Get HJ value of the predicted next state
                    with torch.no_grad():
                        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
                        # Get the HJ value by computing V(x') = Q(x', π(x'))
                        next_action = self.policy.actor(next_state_tensor)
                        next_hj_value = self.policy.critic(next_state_tensor, next_action).squeeze().cpu().numpy()
                    
                    # Check hj safety
                    if next_hj_value < 0:
                        # Switch to pure HJ safe action (no exploration/noise)
                        with torch.no_grad():
                            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                            action = self.policy.actor(obs_tensor).cpu().numpy().squeeze()
                            action = np.clip(action, self.policy.action_low, self.policy.action_high)
                        total_hj_interventions += 1
                    else:
                        # Use smart controller action
                        action = smart_action
                        total_smart_actions += 1
                else:
                    # Pure HJ policy with exploration
                    action = self.policy.exploration_action(obs, use_smart_controller=False)
                
                # Step environment with chosen action
                obs_next, cost, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Track constraint violations (cost <= 0 means collision)
                if cost <= 0:
                    if use_switching[env_idx]:
                        switching_violations += 1
                        episode_had_violation[env_idx] = True
                    else:
                        hj_violations += 1
                        episode_had_violation[env_idx] = True
                
                # Track total steps for each policy type
                if use_switching[env_idx]:
                    switching_total_steps += 1
                else:
                    hj_total_steps += 1

                buffer.add(obs, action, cost, obs_next, done)
                steps += 1
                obs_list[env_idx] = obs_next
                done_flags[env_idx] = done

                if steps >= max_steps:
                    break
        
        # Calculate and print statistics
        print(f"\n  === Collection Statistics ===")
        
        # Switching policy statistics
        if use_switching.count(True) > 0 and switching_total_steps > 0:
            switching_steps = total_smart_actions + total_hj_interventions
            if switching_steps > 0:
                intervention_rate = 100 * total_hj_interventions / switching_steps
                print(f"  Switching Policy: {total_smart_actions} smart actions, "
                      f"{total_hj_interventions} HJ interventions ({intervention_rate:.1f}%)")
            
            violation_rate = 100 * switching_violations / switching_total_steps
            print(f"  Switching Policy Violations: {switching_violations}/{switching_total_steps} steps ({violation_rate:.2f}%)")
            
            if switching_episodes > 0:
                episode_violation_rate = 100 * switching_episodes_with_violation / switching_episodes
                print(f"  Switching Policy Episodes with Violations: {switching_episodes_with_violation}/{switching_episodes} ({episode_violation_rate:.1f}%)")
        
        # Pure HJ policy statistics
        if hj_total_steps > 0:
            hj_violation_rate = 100 * hj_violations / hj_total_steps
            print(f"  Pure HJ Policy Violations: {hj_violations}/{hj_total_steps} steps ({hj_violation_rate:.2f}%)")
            
            if hj_episodes > 0:
                hj_episode_violation_rate = 100 * hj_episodes_with_violation / hj_episodes
                print(f"  Pure HJ Policy Episodes with Violations: {hj_episodes_with_violation}/{hj_episodes} ({hj_episode_violation_rate:.1f}%)")
        
        # Return statistics for logging
        return {
            'switching_violation_rate': (100 * switching_violations / switching_total_steps) if switching_total_steps > 0 else 0,
            'switching_violations': switching_violations,
            'switching_total_steps': switching_total_steps,
            'hj_violation_rate': (100 * hj_violations / hj_total_steps) if hj_total_steps > 0 else 0,
            'hj_violations': hj_violations,
            'hj_total_steps': hj_total_steps,
            'intervention_rate': (100 * total_hj_interventions / (total_smart_actions + total_hj_interventions)) if (total_smart_actions + total_hj_interventions) > 0 else 0,
            'total_smart_actions': total_smart_actions,
            'total_hj_interventions': total_hj_interventions
        }
def generate_fixed_states_for_plotting():
    """Generate diverse fixed states for HJ plotting"""
    fixed_states = []
    
    # State 1: Cars spread out, ego in middle
    fixed_states.append(np.array([
        0.5, 15.0, 1.0,      # car0: left lane, far ahead
        1.0, 8.0, 2.0, np.pi/2,  # ego: center, middle
        1.5, 12.0, 1.5       # car2: right lane, ahead
    ]))

    # State 2: Cars close together, ego behind
    fixed_states.append(np.array([
        0.8, 10.0, 1.5,      # car0: left-center, middle
        1.0, 5.0, 2.5, np.pi/2,  # ego: center, behind
        1.2, 10.0, 1.5       # car2: right-center, middle
    ]))

    # State 3: Ego ahead, cars behind
    fixed_states.append(np.array([
        0.5, 5.0, 1.0,       # car0: left, behind
        1.5, 12.0, 2.0, np.pi/2, # ego: right, ahead
        1.0, 7.0, 1.5        # car2: center, middle
    ]))

    # State 4: Dense traffic scenario
    fixed_states.append(np.array([
        1.0, 8.0, 1.2,       # car0: center, middle
        0.5, 6.0, 1.8, np.pi/2,  # ego: left, slightly behind
        1.5, 9.0, 1.3        # car2: right, slightly ahead
    ]))

    # --- Additional Diverse Scenarios ---

    # State 5: Ego at far left, turning, with cars close on right
    fixed_states.append(np.array([
        1.2, 10.0, 1.0,      # car0: center-right, middle
        0.1, 6.0, 1.5, np.pi/3,  # ego: far left, slightly angled left
        1.5, 7.0, 1.2        # car2: far right, slightly behind
    ]))

    # State 6: Ego nearly stopped at center, cars approaching fast
    fixed_states.append(np.array([
        1.0, 13.0, 2.5,      # car0: center, very fast
        1.0, 10.0, 0.2, np.pi/2,  # ego: center, nearly stopped
        1.0, 12.0, 2.3       # car2: center, fast
    ]))

    # State 7: Ego near upper Y-bound, fast speed, curved orientation
    fixed_states.append(np.array([
        0.5, 18.0, 1.0,      # car0: behind, left
        1.8, 19.5, 3.0, np.pi/1.8,  # ego: far right, upper bound, curved
        1.5, 17.0, 1.5       # car2: slightly behind
    ]))

    # State 8: Ego at bottom Y-bound, heading straight, cars spread
    fixed_states.append(np.array([
        0.2, 5.0, 1.0,       # car0: far left, mid
        1.0, 0.5, 1.0, np.pi/2,  # ego: bottom, center
        1.8, 10.0, 1.0       # car2: far right, ahead
    ]))

    return fixed_states


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
    
    # Set default grid size for HJ plotting if not in config
    if not hasattr(args, 'nx'):
        args.nx = 50
    if not hasattr(args, 'ny'):
        args.ny = 50
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Enable cuDNN autotuner for better performance
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Create environments
    train_envs = [HighwayWrapper(device=device) for _ in range(args.training_num)]
    test_envs = [HighwayWrapper(device=device) for _ in range(args.test_num)]

    # Get dimensions
    dummy_obs, _ = train_envs[0].reset()
    state_dim = train_envs[0].observation_space.shape[0]
    action_dim = 2  # We only control 2 actions
    max_action = torch.tensor(train_envs[0].action_space.high[:2], device=device, dtype=torch.float32)

    # Initialize networks
    actor = Actor(state_dim, action_dim, args.control_net, args.actor_activation, max_action).to(device)
    critic = Critic(state_dim, action_dim, args.critic_net, args.critic_activation).to(device)

    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)

    policy = AvoidDDPGPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        tau=args.tau,
        gamma=args.gamma_pyhj,
        exploration_noise=args.exploration_noise,
        device=device,
        actor_gradient_steps=getattr(args, 'actor_gradient_steps', 1),
        action_space=train_envs[0].action_space
    )

    # Use optimized replay buffer
    buffer = OptimizedReplayBuffer(args.buffer_size, device, state_dim, action_dim)

    # Initialize parallel collector
    collector = ParallelEnvCollector(train_envs, policy, device)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_dir = Path(f"/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/conf/ddpg_hj_highway/{timestamp}/")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    wandb.init(
        project=f"ddpg-hj-highway",
        name=f"ddpg-highway-{timestamp}",
        config=vars(args)
    )
    writer = SummaryWriter(log_dir=log_dir / "logs")

    # Generate fixed states for plotting
    fixed_states = generate_fixed_states_for_plotting()
    
    # Create a single environment for plotting
    plot_env = HighwayWrapper(device=device)

    # Warm up buffer
    print(f"Warming up replay buffer (need ≥ {args.batch_size_pyhj} samples)...")
    while len(buffer) < 5000:
        collector.collect_trajectories(buffer, max_steps=1000)
    print(f"Replay buffer: {len(buffer)} samples.")

    for epoch in range(1, args.total_episodes + 1):
        print(f"\n=== Epoch {epoch}/{args.total_episodes} ===")
        
        # Collect data in parallel and get statistics
        collection_stats = collector.collect_trajectories(buffer, max_steps=6000)
        
        # Log collection statistics to wandb
        wandb.log({
            "collection/switching_violation_rate": collection_stats['switching_violation_rate'],
            "collection/hj_violation_rate": collection_stats['hj_violation_rate'],
            "collection/intervention_rate": collection_stats['intervention_rate'],
            "collection/switching_violations": collection_stats['switching_violations'],
            "collection/hj_violations": collection_stats['hj_violations'],
            "epoch": epoch
        })
        
        # Train with progress bar
        pbar = tqdm(range(args.step_per_epoch), desc=f"Epoch {epoch}", unit="step")
        for step in pbar:
            batch = buffer.sample(args.batch_size_pyhj)
            metrics = policy.learn(batch)
            
            pbar.set_postfix({
                "actor_loss": f"{metrics['loss/actor']:.8f}",
                "critic_loss": f"{metrics['loss/critic']:.8f}",
                "buffer": len(buffer)
            })
            
            # Log metrics to wandb
            wandb.log({**metrics, "epoch": epoch})
        pbar.close()

        # Plot HJ every few epochs
        if epoch % 3 == 0 or epoch == 1:
            print("Plotting HJ values...")
            fig = plot_hj_highway(policy, plot_env, fixed_states, args, device)
            
            # Log to wandb
            # wandb.log({
            #     "HJ_highway/combined": wandb.Image(fig),
            #     "epoch": epoch
            # })
            plt.close(fig)
            
            # Also create individual HJ plots for each fixed state
            for idx, fixed_state in enumerate(fixed_states):
                fig_individual = plt.figure(figsize=(12, 5))
                
                # Left: Environment visualization
                ax_env = plt.subplot(1, 2, 1)
                plot_env.reset(state=fixed_state)
                rgb_array = plot_env.render(mode="rgb_array")
                ax_env.imshow(rgb_array)
                ax_env.axis('off')
                ax_env.set_title(f'Environment State {idx+1}')
                
                # Right: HJ heatmap
                ax_hj = plt.subplot(1, 2, 2)
                vals = compute_hj_grid_vectorized(policy, plot_env, fixed_state, args, device)
                
                # Create a masked array where unsafe regions (vals < 0) are highlighted
                masked_vals = np.ma.masked_where(vals < 0, vals)
                
                
                '''
                ✅ What vmin and vmax do in imshow
They define the range of values that the colormap maps to colors.
vmin=-1: any value ≤ -1 will be mapped to the lowest color (e.g., deep red in 'RdYlBu')
vmax=1: any value ≥ 1 will be mapped to the highest color (e.g., deep blue)
Values between -1 and 1 are linearly interpolated within the colormap.
'''
                # Plot safe regions
                im = ax_hj.imshow(
                    vals.T,
                    extent=(-0.3, 2.3, 0.0, 20.0),
                    origin="lower",
                    cmap='RdYlBu',
                    aspect='auto',
                    vmin=-1,
                    vmax=1
                )
                
                # Add contour line at 0 (safety boundary)
                X, Y = np.meshgrid(np.linspace(-0.3, 2.3, args.nx), 
                                np.linspace(0, 20, args.ny))
                contour = ax_hj.contour(X, Y, vals.T, levels=[0], colors='black', linewidths=2)
                # ax_hj.clabel(contour, inline=True, fontsize=10, fmt='Safety Boundary')
                
                # Mark positions
                ego_x, ego_y = fixed_state[3], fixed_state[4]
                car0_x, car0_y = fixed_state[0], fixed_state[1]
                car2_x, car2_y = fixed_state[7], fixed_state[8]
                
                # ax_hj.plot(ego_x, ego_y, 'k*', markersize=20, label='Ego', markeredgewidth=2)
                ax_hj.plot(car0_x, car0_y, 'ro', markersize=12, label='Car0 (Disturbance)')
                ax_hj.plot(car2_x, car2_y, 'mo', markersize=12, label='Car2 (Obstacle)')
                
                # Add velocity vectors
                ego_v = fixed_state[5]
                ego_theta = fixed_state[6]
                ax_hj.arrow(ego_x, ego_y, 
                        0.3*ego_v*np.cos(ego_theta), 
                        0.3*ego_v*np.sin(ego_theta),
                        head_width=0.1, head_length=0.1, fc='black', ec='black')
                
                ax_hj.set_xlabel("Ego X Position")
                ax_hj.set_ylabel("Ego Y Position")
                ax_hj.set_title(f"HJ Value Function (Safe > 0, Unsafe < 0)")
                ax_hj.legend(loc='upper right')
                ax_hj.grid(True, alpha=0.3)
                
                # Add road boundaries
                ax_hj.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Road Edge')
                ax_hj.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax_hj)
                cbar.set_label('HJ Value (Safety Measure)', rotation=270, labelpad=20)
                
                # Add state information as text
                state_info = (f"Car0: x={car0_x:.2f}, y={car0_y:.2f}, v={fixed_state[2]:.2f}\n"
                            f"Ego: x={ego_x:.2f}, y={ego_y:.2f}, v={ego_v:.2f}, θ={ego_theta:.2f}\n"
                            f"Car2: x={car2_x:.2f}, y={car2_y:.2f}, v={fixed_state[9]:.2f}")
                ax_hj.text(0.02, 0.98, state_info, transform=ax_hj.transAxes, 
                        verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.tight_layout()
                
                wandb.log({
                    f"HJ_highway/state_{idx+1}": wandb.Image(fig_individual),
                    "epoch": epoch
                })
                plt.close(fig_individual)

        # Save checkpoints
        if epoch % 10 == 0 or epoch == args.total_episodes:
            epoch_dir = log_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'actor_state_dict': policy.actor.state_dict(),
                'critic_state_dict': policy.critic.state_dict(),
                'actor_optimizer_state_dict': policy.actor_optim.state_dict(),
                'critic_optimizer_state_dict': policy.critic_optim.state_dict(),
                'args': vars(args)
            }
            
            torch.save(checkpoint, epoch_dir / "checkpoint.pth")
            print(f"Saved checkpoint at epoch {epoch}")
            
            # Also save individual model files for easy loading
            torch.save(policy.actor.state_dict(), epoch_dir / "actor.pth")
            torch.save(policy.critic.state_dict(), epoch_dir / "critic.pth")

    # Final evaluation
    print("\nTraining complete. Running final evaluation...")
    
    # Create a comprehensive final visualization
    # fig_final = plt.figure(figsize=(20, 15))

    # wandb.log({
    #     "HJ_highway/final_evaluation": wandb.Image(fig_final),
    #     "epoch": args.total_episodes
    # })
    # plt.close(fig_final)
    
    # Close environments
    plot_env.close()
    for env in train_envs:
        env.close()
    for env in test_envs:
        env.close()
    
    print("Training and evaluation complete!")
    
    # Save final summary
    summary = {
        'total_episodes': args.total_episodes,
        'final_buffer_size': len(buffer),
        'model_path': str(log_dir / f"epoch_{args.total_episodes}"),
        'timestamp': timestamp
    }
    
    with open(log_dir / "training_summary.yaml", 'w') as f:
        yaml.dump(summary, f)
    
    wandb.finish()

if __name__ == "__main__":
    main()
    
    
    
# python "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/Train_HJ_highway.py"  --nx 50 --ny 50 --step-per-epoch 3000 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2
