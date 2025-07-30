"""
Test learned Hamilton-Jacobi safety filter in latent space for Dubins car
"""

import os
import argparse
from pathlib import Path
import torch
import numpy as np
import imageio
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import cv2

# Import required modules from your codebase
from plan import load_model
from env.dubins.dubins import DubinsEnv
from gymnasium.spaces import Box

# Set up matplotlib config
os.environ['MPLCONFIGDIR'] = '/storage1/fs1/sibai/Active/ihab/tmp'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)


class DubinsEnvForTesting:
    """Wrapper for DubinsEnv to match the interface expected by HJ code"""
    def __init__(self, device='cuda'):
        self.env = DubinsEnv()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = {"render_fps": 10}  # Add fps for video saving
        
    def reset(self, state=None):
        if state is not None:
            reset_out = self.env.reset(state=state)
        else:
            reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        return obs, {}
    
    def step(self, action):
        obs_out, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        h_s = info.get('h', 0.0) * 3  # Multiply by 3 to match training
        return obs, h_s, terminated, truncated, info
    
    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)
    
    def close(self):
        pass


class HJPolicyEvaluator:
    """Evaluates HJ value and provides safe actions using learned latent-space policy"""
    def __init__(self, actor_path, critic_path, wm, device='cuda', with_proprio=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.wm = wm
        self.with_proprio = with_proprio
        
        # Get num_hist from world model
        self.num_hist = self.wm.num_hist
        
        # Create dummy environment to get dimensions
        dummy_env = DubinsEnvForTesting(device)
        dummy_obs, _ = dummy_env.reset()
        
        # Get state dimension by encoding a dummy observation
        z = self.encode_observation(dummy_obs)
        state_dim = z.shape[1]
        action_dim = dummy_env.action_space.shape[0]
        max_action = torch.tensor(dummy_env.action_space.high, device=self.device, dtype=torch.float32)
        
        # Load actor and critic
        self.actor = self._load_actor(actor_path, state_dim, action_dim, max_action)
        self.critic = self._load_critic(critic_path, state_dim, action_dim)
        
        self.actor.eval()
        self.critic.eval()
        
        # History buffers for world model prediction
        self.obs_history = []
        self.action_history = []
    
    def _load_actor(self, path, state_dim, action_dim, max_action):
        """Load actor network"""
        # Recreate actor architecture from training code
        actor = Actor(state_dim, action_dim, [512, 512,512], 'ReLU', max_action).to(self.device)
        actor.load_state_dict(torch.load(path, map_location=self.device))
        return actor
    
    def _load_critic(self, path, state_dim, action_dim):
        """Load critic network"""
        # Recreate critic architecture from training code
        critic = Critic(state_dim, action_dim,[512, 512,512], 'ReLU').to(self.device)
        critic.load_state_dict(torch.load(path, map_location=self.device))
        return critic
    
    def encode_observation(self, obs):
        """Encode single observation to latent space (flattened for HJ network)"""
        if isinstance(obs, dict):
            visual = obs['visual']
            proprio = obs['proprio']
        else:
            visual, proprio = obs
        
        # Prepare visual data
        visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
        visual_tensor = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Prepare proprio data
        proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).unsqueeze(0).to(self.device)
        
        data = {'visual': visual_tensor, 'proprio': proprio_tensor}
        
        with torch.no_grad():
            lat = self.wm.encode_obs(data)
        
        # Store the full latent representation
        self.current_latent = lat
        
        # Flatten for HJ networks
        if self.with_proprio:
            z_vis = lat['visual'].reshape(lat['visual'].shape[0], -1)
            z_prop = lat['proprio'].squeeze(1)
            z = torch.cat([z_vis, z_prop], dim=-1)
        else:
            z = lat['visual'].reshape(lat['visual'].shape[0], -1)
        
        return z
    
    def get_hj_value(self, obs):
        """Get HJ value for current observation"""
        self.current_obs = obs
        z = self.encode_observation(obs)
        with torch.no_grad():
            action = self.actor(z)
            hj_value = self.critic(z, action).item()
        return hj_value
    
    def get_safe_action(self, obs):
        """Get safe action from HJ policy"""
        self.current_obs = obs
        z = self.encode_observation(obs)
        with torch.no_grad():
            action = self.actor(z).cpu().numpy().squeeze()
        return action
    
    def update_history(self, obs, action):
        """Update history buffers"""
        self.obs_history.append(obs)
        self.action_history.append(action)
        
        # Keep only num_hist frames
        if len(self.obs_history) > self.num_hist:
            self.obs_history.pop(0)
        if len(self.action_history) > self.num_hist:
            self.action_history.pop(0)
    
    def predict_next_state_value(self, obs, action):
        """Predict HJ value of next state using world model dynamics"""
        # Update history with current observation
        if len(self.obs_history) == 0:
            # Initialize history with current observation repeated
            for _ in range(self.num_hist):
                self.obs_history.append(obs)
                self.action_history.append(np.zeros_like(action))
        
        # Prepare observation history for world model
        visual_list = []
        proprio_list = []
        
        for hist_obs in self.obs_history:
            if isinstance(hist_obs, dict):
                visual = hist_obs['visual']
                proprio = hist_obs['proprio']
            else:
                visual, proprio = hist_obs
            
            # Check visual shape and ensure it has 3 channels
            if visual.shape[2] == 3:  # Channels last format
                visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
            else:  # Already in channels first format
                visual_np = visual.astype(np.float32) / 255.0
                ###here need to normalize image as well as in the img transform###
                ###here need to normalize image as well as in the img transform###
            visual_list.append(visual_np)
            proprio_list.append(proprio)
        
        # Stack into tensors
        visual_tensor = torch.from_numpy(np.stack(visual_list)).unsqueeze(0).to(self.device)
        proprio_tensor = torch.from_numpy(np.stack(proprio_list)).unsqueeze(0).to(self.device)
        
        # Prepare action history
        action_tensor = torch.from_numpy(np.stack(self.action_history)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode observation history with actions
            obs_dict = {'visual': visual_tensor, 'proprio': proprio_tensor}
            z_current = self.wm.encode(obs_dict, action_tensor)
            
            # Take only the history frames for prediction
            z_hist = z_current[:, :self.num_hist, :, :]
            
            # Predict next state
            z_pred = self.wm.predict(z_hist)
            
            # The predicted state already incorporates the action from history
            # Now we need to imagine one more step with the proposed action
            # Create a new sequence with the last predicted state
            visual_pred_list = visual_list[1:] + [visual_list[-1]]  # Shift and repeat last
            proprio_pred_list = proprio_list[1:] + [proprio_list[-1]]  # Shift and repeat last
            action_pred_list = self.action_history[1:] + [action]  # Shift and add new action
            
            # Prepare tensors for the shifted sequence
            visual_pred_tensor = torch.from_numpy(np.stack(visual_pred_list)).unsqueeze(0).to(self.device)
            proprio_pred_tensor = torch.from_numpy(np.stack(proprio_pred_list)).unsqueeze(0).to(self.device)
            action_pred_tensor = torch.from_numpy(np.stack(action_pred_list)).unsqueeze(0).to(self.device)
            
            # Encode and predict with the new action
            obs_pred_dict = {'visual': visual_pred_tensor, 'proprio': proprio_pred_tensor}
            z_with_action = self.wm.encode(obs_pred_dict, action_pred_tensor)
            z_next_pred = self.wm.predict(z_with_action[:, :self.num_hist, :, :])
            
            # Extract the last predicted state
            z_obs_next, _ = self.wm.separate_emb(z_next_pred[:, -1:, :, :])
            
            # Flatten for HJ evaluation
            if self.with_proprio:
                z_vis = z_obs_next['visual'].reshape(1, -1)
                z_prop = z_obs_next['proprio'].squeeze(1)
                z_next_flat = torch.cat([z_vis, z_prop], dim=-1)
            else:
                z_next_flat = z_obs_next['visual'].reshape(1, -1)
            
            # Get HJ value for predicted next state
            next_action = self.actor(z_next_flat)
            next_hj_value = self.critic(z_next_flat, next_action).item()
        
        return next_hj_value


class Actor(torch.nn.Module):
    """Actor network - must match training architecture"""
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
    """Critic network - must match training architecture"""
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


class PIDController:
    """PID controller for Dubins car to reach goal"""
    def __init__(self, kp_heading=2.0, kp_speed=0.5):
        self.kp_heading = kp_heading
        self.kp_speed = kp_speed
        self.goal = np.array([2.2, 2.2])  # From DubinsEnv
    
    def get_action(self, state):
        """Get PID control action based on current state"""
        # Extract position and heading from proprio state
        x, y, theta = state
        
        # Compute desired heading to goal
        dx = self.goal[0] - x
        dy = self.goal[1] - y
        desired_theta = np.arctan2(dy, dx)
        
        # Compute heading error (wrap to [-pi, pi])
        heading_error = desired_theta - theta
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # PID control for heading rate
        heading_rate = self.kp_heading * heading_error
        
        # Clip to action limits
        action = np.array([np.clip(heading_rate, -1.0, 1.0)], dtype=np.float32)
        
        return action


def load_world_model(ckpt_dir, device='cuda'):
    """Load the DINO world model"""
    ckpt_dir = Path(ckpt_dir)
    hydra_cfg = ckpt_dir / 'hydra.yaml'
    snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
    train_cfg = OmegaConf.load(str(hydra_cfg))
    num_action_repeat = train_cfg.num_action_repeat
    wm = load_model(snapshot, train_cfg, num_action_repeat, device=device)
    wm.eval()
    print(f"Loaded world model from {ckpt_dir}")
    print(f"World model action_dim: {wm.action_dim}, num_action_repeat: {wm.num_action_repeat}")
    print(f"Expected raw action dim: {wm.action_dim // wm.num_action_repeat}")
    return wm


def simulate_dubins_with_hj(hj_evaluator, env, mode="switching", max_steps=200, 
                           save_video=True, video_path=".", run_id=0,
                           safety_threshold=0.0):
    """
    Simulate Dubins environment with HJ safety filter
    
    Args:
        hj_evaluator: HJPolicyEvaluator instance
        env: DubinsEnvForTesting instance
        mode: "switching", "pid_only", or "safe_only"
        max_steps: Maximum simulation steps
        save_video: Whether to save video
        video_path: Path to save video
        run_id: Run identifier
        safety_threshold: HJ value threshold for safety (default 0.0)
    """
    
    # Create PID controller
    pid_controller = PIDController()
    
    # Reset environment
    obs, _ = env.reset()
    
    # Storage for video frames and metrics
    frames = []
    step_count = 0
    done = False
    hj_interventions = 0
    total_switches = 0
    min_hj_value = float('inf')
    constraint_violations = 0
    last_controller = None
    
    # Get initial state info
    if isinstance(obs, dict):
        initial_state = obs['proprio']
    else:
        initial_state = obs[1]
    
    print(f"\nStarting simulation in {mode} mode (Run {run_id})...")
    print(f"Initial state: {initial_state}")
    print(f"Initial HJ value: {hj_evaluator.get_hj_value(obs):.3f}")
    
    while not done and step_count < max_steps:
        # Get current HJ value
        current_hj_value = hj_evaluator.get_hj_value(obs)
        min_hj_value = min(min_hj_value, current_hj_value)
        
        # Get proprio state for PID
        if isinstance(obs, dict):
            proprio_state = obs['proprio']
        else:
            proprio_state = obs[1]
        
        # Determine action based on mode
        if mode == "safe_only":
            # Always use HJ safe policy
            action = hj_evaluator.get_safe_action(obs)
            using_hj = True
            
        elif mode == "pid_only":
            # Always use PID controller
            action = pid_controller.get_action(proprio_state)
            using_hj = False
            
        else:  # switching mode
            # Get PID action
            pid_action = pid_controller.get_action(proprio_state)
            
            # Predict HJ value of next state with PID action
            next_hj_value = hj_evaluator.predict_next_state_value(obs, pid_action)
            
            # Switch to safe controller if next state would be unsafe
            if next_hj_value < safety_threshold:
                action = hj_evaluator.get_safe_action(obs)
                using_hj = True
                hj_interventions += 1
                if last_controller == "PID":
                    total_switches += 1
                print(f"Step {step_count}: HJ intervention! Next HJ would be {next_hj_value:.3f}")
                last_controller = "HJ"
            else:
                action = pid_action
                using_hj = False
                if last_controller == "HJ":
                    total_switches += 1
                last_controller = "PID"
        
        # Step environment
        obs_next, cost, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update history for world model prediction
        hj_evaluator.update_history(obs_next, action)
        
        # Track constraint violations (negative cost means unsafe)
        if cost < 0:
            constraint_violations += 1
        
        # Render and save frame
        frame = env.render(mode="rgb_array")
        
        # Add HJ info overlay
        if frame is not None:
            # Add text overlay showing HJ value and controller
            frame_with_info = frame.copy()
            text_color = (0, 255, 0) if current_hj_value >= 0 else (0, 0, 255)
            controller_text = "HJ" if using_hj else "PID"
            
            cv2.putText(frame_with_info, f"HJ: {current_hj_value:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame_with_info, f"Controller: {controller_text}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_info, f"Step: {step_count}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if save_video:
                frames.append(frame_with_info)
        
        # Update obs for next iteration
        obs = obs_next
        step_count += 1
        
        # Print progress every 50 steps
        if step_count % 50 == 0:
            print(f"Step {step_count}: HJ={current_hj_value:.3f}, Cost={cost:.3f}, "
                  f"Controller={'HJ' if using_hj else 'PID'}")
    
    # Print summary
    print(f"\nSimulation ended after {step_count} steps")
    print(f"Minimum HJ value encountered: {min_hj_value:.3f}")
    print(f"Constraint violations: {constraint_violations}")
    if mode == "switching":
        print(f"HJ interventions: {hj_interventions} ({100*hj_interventions/max(1,step_count):.1f}% of steps)")
        print(f"Total controller switches: {total_switches}")
    
    # Save video if requested
    if save_video and frames:
        os.makedirs(video_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"dubins_hj_{mode}_run{run_id}_{timestamp}.mp4"
        full_video_path = os.path.join(video_path, video_name)
        
        with imageio.get_writer(full_video_path, fps=env.metadata["render_fps"]) as writer:
            for frame in frames:
                writer.append_data(frame)
        
        print(f"Video saved to: {full_video_path}")
    
    return {
        'steps': step_count,
        'violations': constraint_violations,
        'hj_interventions': hj_interventions if mode == "switching" else None,
        'min_hj': min_hj_value,
        'final_cost': cost
    }


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    wm_ckpt_dir = "/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/dubins/resnet"
    hj_ckpt_dir = "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/runs/ddpg_hj_latent/resnet-0729_2115/epoch_2"
    video_save_path = "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/dubins_test"
    
    actor_path = os.path.join(hj_ckpt_dir, "actor.pth")
    critic_path = os.path.join(hj_ckpt_dir, "critic.pth")
    
    # Load world model
    print("Loading world model...")
    wm = load_world_model(wm_ckpt_dir, device)
    
    # Create HJ evaluator
    print("Loading HJ policy...")
    hj_evaluator = HJPolicyEvaluator(actor_path, critic_path, wm, device, with_proprio=True)
    
    # Run simulations for each mode
    modes = ["switching", "pid_only", "safe_only"]
    num_runs_per_mode = 5
    
    all_results = {}
    
    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Running {num_runs_per_mode} simulations in {mode} mode")
        print(f"{'='*50}")
        
        mode_results = []
        
        for run_id in range(num_runs_per_mode):
            # Create fresh environment for each run
            env = DubinsEnvForTesting(device)
            
            # Run simulation
            results = simulate_dubins_with_hj(
                hj_evaluator=hj_evaluator,
                env=env,
                mode=mode,
                max_steps=200,
                save_video=True,
                video_path=video_save_path,
                run_id=run_id,
                safety_threshold=0.0
            )
            
            mode_results.append(results)
            env.close()
        
        all_results[mode] = mode_results
    
    # Print overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    for mode in modes:
        results = all_results[mode]
        avg_steps = np.mean([r['steps'] for r in results])
        avg_violations = np.mean([r['violations'] for r in results])
        avg_min_hj = np.mean([r['min_hj'] for r in results])
        
        print(f"\n{mode.upper()} MODE:")
        print(f"  Average steps: {avg_steps:.1f}")
        print(f"  Average violations: {avg_violations:.1f}")
        print(f"  Average minimum HJ: {avg_min_hj:.3f}")
        
        if mode == "switching":
            avg_interventions = np.mean([r['hj_interventions'] for r in results])
            avg_intervention_rate = 100 * avg_interventions / avg_steps
            print(f"  Average HJ interventions: {avg_interventions:.1f} ({avg_intervention_rate:.1f}%)")
    
    print(f"\nAll videos saved to: {video_save_path}")


if __name__ == "__main__":
    main()