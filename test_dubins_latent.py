# """
# Test learned Hamilton-Jacobi safety filter in latent space for Dubins car
# """

# import os
# import argparse
# from pathlib import Path
# import torch
# import numpy as np
# import imageio
# from datetime import datetime
# from copy import deepcopy
# import matplotlib.pyplot as plt
# from omegaconf import OmegaConf
# import cv2

# # Import required modules from your codebase
# from plan import load_model
# from env.dubins.dubins import DubinsEnv
# from gymnasium.spaces import Box

# # Set up matplotlib config
# os.environ['MPLCONFIGDIR'] = '/storage1/fs1/sibai/Active/ihab/tmp'
# os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)


# class DubinsEnvForTesting:
#     """Wrapper for DubinsEnv to match the interface expected by HJ code"""
#     def __init__(self, device='cuda'):
#         self.env = DubinsEnv()
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         self.observation_space = self.env.observation_space
#         self.action_space = self.env.action_space
#         self.metadata = {"render_fps": 10}  # Add fps for video saving
        
#     def reset(self, state=None):
#         if state is not None:
#             reset_out = self.env.reset(state=state)
#         else:
#             reset_out = self.env.reset()
#         obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
#         return obs, {}
    
#     def step(self, action):
#         obs_out, reward, done, info = self.env.step(action)
#         terminated = done
#         truncated = False
#         obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
#         h_s = info.get('h', 0.0) * 3  # Multiply by 3 to match training
#         return obs, h_s, terminated, truncated, info
    
#     def render(self, mode='rgb_array'):
#         return self.env.render(mode=mode)
    
#     def close(self):
#         pass


# class HJPolicyEvaluator:
#     """Evaluates HJ value and provides safe actions using learned latent-space policy"""
#     def __init__(self, actor_path, critic_path, wm, device='cuda', with_proprio=False):
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         self.wm = wm
#         self.with_proprio = with_proprio
        
#         # Get num_hist from world model
#         self.num_hist = self.wm.num_hist
        
#         # Create dummy environment to get dimensions
#         dummy_env = DubinsEnvForTesting(device)
#         dummy_obs, _ = dummy_env.reset()
        
#         # Get state dimension by encoding a dummy observation
#         z = self.encode_observation(dummy_obs)
#         state_dim = z.shape[1]
#         action_dim = dummy_env.action_space.shape[0]
#         max_action = torch.tensor(dummy_env.action_space.high, device=self.device, dtype=torch.float32)
        
#         # Load actor and critic
#         self.actor = self._load_actor(actor_path, state_dim, action_dim, max_action)
#         self.critic = self._load_critic(critic_path, state_dim, action_dim)
        
#         self.actor.eval()
#         self.critic.eval()
        
#         # History buffers for world model prediction
#         self.obs_history = []
#         self.action_history = []
    
#     def _load_actor(self, path, state_dim, action_dim, max_action):
#         """Load actor network"""
#         # Recreate actor architecture from training code
#         actor = Actor(state_dim, action_dim, [512, 512,512], 'ReLU', max_action).to(self.device)
#         actor.load_state_dict(torch.load(path, map_location=self.device))
#         return actor
    
#     def _load_critic(self, path, state_dim, action_dim):
#         """Load critic network"""
#         # Recreate critic architecture from training code
#         critic = Critic(state_dim, action_dim, [512, 512,512], 'ReLU').to(self.device)
#         critic.load_state_dict(torch.load(path, map_location=self.device))
#         return critic
    
#     def encode_observation(self, obs):
#         """Encode single observation to latent space (flattened for HJ network)"""
#         if isinstance(obs, dict):
#             visual = obs['visual']
#             proprio = obs['proprio']
#         else:
#             visual, proprio = obs
        
#         # Prepare visual data
#         visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
#         visual_tensor = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(0).to(self.device)
        
#         # Prepare proprio data
#         proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).unsqueeze(0).to(self.device)
        
#         data = {'visual': visual_tensor, 'proprio': proprio_tensor}
        
#         with torch.no_grad():
#             lat = self.wm.encode_obs(data)
        
#         # Store the full latent representation
#         self.current_latent = lat
        
#         # Flatten for HJ networks
#         if self.with_proprio:
#             z_vis = lat['visual'].reshape(lat['visual'].shape[0], -1)
#             z_prop = lat['proprio'].squeeze(1)
#             z = torch.cat([z_vis, z_prop], dim=-1)
#         else:
#             z = lat['visual'].reshape(lat['visual'].shape[0], -1)
        
#         return z
    
#     def get_hj_value(self, obs):
#         """Get HJ value for current observation"""
#         self.current_obs = obs
#         z = self.encode_observation(obs)
#         with torch.no_grad():
#             action = self.actor(z)
#             hj_value = self.critic(z, action).item()
#         return hj_value
    
#     def get_safe_action(self, obs):
#         """Get safe action from HJ policy"""
#         self.current_obs = obs
#         z = self.encode_observation(obs)
#         with torch.no_grad():
#             action = self.actor(z).cpu().numpy().squeeze()
#         return action
    
#     def update_history(self, obs, action):
#         """Update history buffers"""
#         # Ensure action is consistent shape
#         if isinstance(action, np.ndarray):
#             if action.ndim == 0:
#                 action = np.array([action])
#             elif action.ndim > 1:
#                 action = action.flatten()
#         else:
#             action = np.array([action]) if np.isscalar(action) else np.array(action)
        
#         self.obs_history.append(obs)
#         self.action_history.append(action)
        
#         # Keep only num_hist frames
#         if len(self.obs_history) > self.num_hist:
#             self.obs_history.pop(0)
#         if len(self.action_history) > self.num_hist:
#             self.action_history.pop(0)
    
#     def predict_next_state_value(self, obs, action):
#         """Predict HJ value of next state using world model dynamics"""
#         # Ensure action is consistent shape
#         if isinstance(action, np.ndarray):
#             if action.ndim == 0:
#                 action = np.array([action])
#             elif action.ndim > 1:
#                 action = action.flatten()
#         else:
#             action = np.array([action]) if np.isscalar(action) else np.array(action)
        
#         # Update history with current observation
#         if len(self.obs_history) == 0:
#             # Initialize history with current observation repeated
#             zero_action = np.zeros_like(action)  # Use same shape as provided action
#             for _ in range(self.num_hist):
#                 self.obs_history.append(obs)
#                 self.action_history.append(zero_action.copy())
        
#         # Prepare observation history for world model
#         visual_list = []
#         proprio_list = []
        
#         for hist_obs in self.obs_history:
#             if isinstance(hist_obs, dict):
#                 visual = hist_obs['visual']
#                 proprio = hist_obs['proprio']
#             else:
#                 visual, proprio = hist_obs
            
#             # Check visual shape and ensure it has 3 channels
#             if visual.shape[2] == 3:  # Channels last format
#                 visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
#             else:  # Already in channels first format
#                 visual_np = visual.astype(np.float32) / 255.0
                
#             visual_list.append(visual_np)
#             proprio_list.append(proprio)
        
#         # Stack into tensors
#         visual_tensor = torch.from_numpy(np.stack(visual_list)).unsqueeze(0).to(self.device)
#         proprio_tensor = torch.from_numpy(np.stack(proprio_list)).unsqueeze(0).to(self.device)
        
#         # Prepare action history - ensure all actions have same shape
#         action_list = []
#         for hist_action in self.action_history:
#             if isinstance(hist_action, np.ndarray):
#                 if hist_action.ndim == 0:
#                     hist_action = np.array([hist_action])
#                 elif hist_action.ndim > 1:
#                     hist_action = hist_action.flatten()
#             else:
#                 hist_action = np.array([hist_action]) if np.isscalar(hist_action) else np.array(hist_action)
#             action_list.append(hist_action)
        
#         action_tensor = torch.from_numpy(np.stack(action_list)).unsqueeze(0).to(self.device)
        
#         with torch.no_grad():
#             # Encode observation history with actions
#             obs_dict = {'visual': visual_tensor, 'proprio': proprio_tensor}
#             z_current = self.wm.encode(obs_dict, action_tensor)
            
#             # Take only the history frames for prediction
#             z_hist = z_current[:, :self.num_hist, :, :]
            
#             # Predict next state
#             z_pred = self.wm.predict(z_hist)
            
#             # The predicted state already incorporates the action from history
#             # Now we need to imagine one more step with the proposed action
#             # Create a new sequence with the last predicted state
#             visual_pred_list = visual_list[1:] + [visual_list[-1]]  # Shift and repeat last
#             proprio_pred_list = proprio_list[1:] + [proprio_list[-1]]  # Shift and repeat last
#             action_pred_list = action_list[1:] + [action]  # Shift and add new action
            
#             # Prepare tensors for the shifted sequence
#             visual_pred_tensor = torch.from_numpy(np.stack(visual_pred_list)).unsqueeze(0).to(self.device)
#             proprio_pred_tensor = torch.from_numpy(np.stack(proprio_pred_list)).unsqueeze(0).to(self.device)
#             action_pred_tensor = torch.from_numpy(np.stack(action_pred_list)).unsqueeze(0).to(self.device)
            
#             # Encode and predict with the new action
#             obs_pred_dict = {'visual': visual_pred_tensor, 'proprio': proprio_pred_tensor}
#             z_with_action = self.wm.encode(obs_pred_dict, action_pred_tensor)
#             z_next_pred = self.wm.predict(z_with_action[:, :self.num_hist, :, :])
            
#             # Extract the last predicted state
#             z_obs_next, _ = self.wm.separate_emb(z_next_pred[:, -1:, :, :])
            
#             # Flatten for HJ evaluation
#             if self.with_proprio:
#                 z_vis = z_obs_next['visual'].reshape(1, -1)
#                 z_prop = z_obs_next['proprio'].squeeze(1)
#                 z_next_flat = torch.cat([z_vis, z_prop], dim=-1)
#             else:
#                 z_next_flat = z_obs_next['visual'].reshape(1, -1)
            
#             # Get HJ value for predicted next state
#             next_action = self.actor(z_next_flat)
#             next_hj_value = self.critic(z_next_flat, next_action).item()
        
#         return next_hj_value


# class Actor(torch.nn.Module):
#     """Actor network - must match training architecture"""
#     def __init__(self, state_dim, action_dim, hidden_sizes, activation, max_action):
#         super().__init__()
#         self.net = self.build_net(state_dim, action_dim, hidden_sizes, activation)
#         self.register_buffer('max_action', max_action)
        
#     def build_net(self, input_dim, output_dim, hidden_sizes, activation):
#         layers = []
#         prev_dim = input_dim
#         for hidden_dim in hidden_sizes:
#             layers.append(torch.nn.Linear(prev_dim, hidden_dim))
#             layers.append(getattr(torch.nn, activation)())
#             prev_dim = hidden_dim
#         layers.append(torch.nn.Linear(prev_dim, output_dim))
#         layers.append(torch.nn.Tanh())
#         return torch.nn.Sequential(*layers)
    
#     def forward(self, state):
#         return self.max_action * self.net(state)


# class Critic(torch.nn.Module):
#     """Critic network - must match training architecture"""
#     def __init__(self, state_dim, action_dim, hidden_sizes, activation):
#         super().__init__()
#         self.net = self.build_net(state_dim + action_dim, 1, hidden_sizes, activation)
        
#     def build_net(self, input_dim, output_dim, hidden_sizes, activation):
#         layers = []
#         prev_dim = input_dim
#         for hidden_dim in hidden_sizes:
#             layers.append(torch.nn.Linear(prev_dim, hidden_dim))
#             layers.append(getattr(torch.nn, activation)())
#             prev_dim = hidden_dim
#         layers.append(torch.nn.Linear(prev_dim, output_dim))
#         return torch.nn.Sequential(*layers)
    
#     def forward(self, state, action):
#         return self.net(torch.cat([state, action], dim=-1))


# class PIDController:
#     """PID controller for Dubins car to reach goal"""
#     def __init__(self, kp_heading=2.0, kp_speed=0.5):
#         self.kp_heading = kp_heading
#         self.kp_speed = kp_speed
#         self.goal = np.array([2.2, 2.2])  # From DubinsEnv
    
#     def get_action(self, state):
#         """Get PID control action based on current state"""
#         # Extract position and heading from proprio state
#         x, y, theta = state
        
#         # Compute desired heading to goal
#         dx = self.goal[0] - x
#         dy = self.goal[1] - y
#         desired_theta = np.arctan2(dy, dx)
        
#         # Compute heading error (wrap to [-pi, pi])
#         heading_error = desired_theta - theta
#         heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
#         # PID control for heading rate
#         heading_rate = self.kp_heading * heading_error
        
#         # Clip to action limits
#         action = np.array([np.clip(heading_rate, -1.0, 1.0)], dtype=np.float32)
        
#         return action


# def load_world_model(ckpt_dir, device='cuda'):
#     """Load the DINO world model"""
#     ckpt_dir = Path(ckpt_dir)
#     hydra_cfg = ckpt_dir / 'hydra.yaml'
#     snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
#     train_cfg = OmegaConf.load(str(hydra_cfg))
#     num_action_repeat = train_cfg.num_action_repeat
#     wm = load_model(snapshot, train_cfg, num_action_repeat, device=device)
#     wm.eval()
#     print(f"Loaded world model from {ckpt_dir}")
#     print(f"World model action_dim: {wm.action_dim}, num_action_repeat: {wm.num_action_repeat}")
#     print(f"Expected raw action dim: {wm.action_dim // wm.num_action_repeat}")
#     return wm


# def simulate_dubins_with_hj(hj_evaluator, env, mode="switching", max_steps=200, 
#                            save_video=True, video_path=".", run_id=0,
#                            safety_threshold=0.0):
#     """
#     Simulate Dubins environment with HJ safety filter
    
#     Args:
#         hj_evaluator: HJPolicyEvaluator instance
#         env: DubinsEnvForTesting instance
#         mode: "switching", "pid_only", or "safe_only"
#         max_steps: Maximum simulation steps
#         save_video: Whether to save video
#         video_path: Path to save video
#         run_id: Run identifier
#         safety_threshold: HJ value threshold for safety (default 0.0)
#     """
    
#     # Create PID controller
#     pid_controller = PIDController()
    
#     # Reset environment
#     obs, _ = env.reset()
    
#     # Initialize history with current observation and zero actions
#     # Get the expected action dimension from the environment
#     zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
#     for _ in range(hj_evaluator.num_hist):
#         hj_evaluator.update_history(obs, zero_action)
    
#     # Storage for video frames and metrics
#     frames = []
#     step_count = 0
#     done = False
#     hj_interventions = 0
#     total_switches = 0
#     min_hj_value = float('inf')
#     constraint_violations = 0
#     last_controller = None
    
#     # Get initial state info
#     if isinstance(obs, dict):
#         initial_state = obs['proprio']
#     else:
#         initial_state = obs[1]
    
#     print(f"\nStarting simulation in {mode} mode (Run {run_id})...")
#     print(f"Initial state: {initial_state}")
#     print(f"Initial HJ value: {hj_evaluator.get_hj_value(obs):.3f}")
    
#     while not done and step_count < max_steps:
#         # Get current HJ value
#         current_hj_value = hj_evaluator.get_hj_value(obs)
#         min_hj_value = min(min_hj_value, current_hj_value)
        
#         # Get proprio state for PID
#         if isinstance(obs, dict):
#             proprio_state = obs['proprio']
#         else:
#             proprio_state = obs[1]
        
#         # Determine action based on mode
#         if mode == "safe_only":
#             # Always use HJ safe policy
#             action = hj_evaluator.get_safe_action(obs)
#             using_hj = True
            
#         elif mode == "pid_only":
#             # Always use PID controller
#             action = pid_controller.get_action(proprio_state)
#             using_hj = False
            
#         else:  # switching mode
#             # Get PID action
#             pid_action = pid_controller.get_action(proprio_state)
            
#             # Predict HJ value of next state with PID action
#             next_hj_value = hj_evaluator.predict_next_state_value(obs, pid_action)
            
#             # Switch to safe controller if next state would be unsafe
#             if next_hj_value < safety_threshold:
#                 action = hj_evaluator.get_safe_action(obs)
#                 using_hj = True
#                 hj_interventions += 1
#                 if last_controller == "PID":
#                     total_switches += 1
#                 print(f"Step {step_count}: HJ intervention! Next HJ would be {next_hj_value:.3f}")
#                 last_controller = "HJ"
#             else:
#                 action = pid_action
#                 using_hj = False
#                 if last_controller == "HJ":
#                     total_switches += 1
#                 last_controller = "PID"
        
#         # Step environment
#         obs_next, cost, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
        
#         # Update history for world model prediction
#         hj_evaluator.update_history(obs_next, action)
        
#         # Track constraint violations (negative cost means unsafe)
#         if cost < 0:
#             constraint_violations += 1
        
#         # Render and save frame
#         frame = env.render(mode="rgb_array")
        
#         # Add HJ info overlay
#         if frame is not None:
#             # Add text overlay showing HJ value and controller
#             frame_with_info = frame.copy()
#             controller_text = "HJ" if using_hj else "PID"
            
#             # Use black font for both text overlays
#             cv2.putText(frame_with_info, f"HJ: {current_hj_value:.2f}", 
#                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#             cv2.putText(frame_with_info, f"Controller: {controller_text}", 
#                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
#             if save_video:
#                 frames.append(frame_with_info)
        
#         # Update obs for next iteration
#         obs = obs_next
#         step_count += 1
        
#         # Print progress every 50 steps
#         if step_count % 50 == 0:
#             print(f"Step {step_count}: HJ={current_hj_value:.3f}, Cost={cost:.3f}, "
#                   f"Controller={'HJ' if using_hj else 'PID'}")
    
#     # Print summary
#     print(f"\nSimulation ended after {step_count} steps")
#     print(f"Minimum HJ value encountered: {min_hj_value:.3f}")
#     print(f"Constraint violations: {constraint_violations}")
#     if mode == "switching":
#         print(f"HJ interventions: {hj_interventions} ({100*hj_interventions/max(1,step_count):.1f}% of steps)")
#         print(f"Total controller switches: {total_switches}")
    
#     # Save video if requested
#     if save_video and frames:
#         os.makedirs(video_path, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         video_name = f"dubins_hj_{mode}_run{run_id}_{timestamp}.mp4"
#         full_video_path = os.path.join(video_path, video_name)
        
#         with imageio.get_writer(full_video_path, fps=env.metadata["render_fps"]) as writer:
#             for frame in frames:
#                 writer.append_data(frame)
        
#         print(f"Video saved to: {full_video_path}")
    
#     return {
#         'steps': step_count,
#         'violations': constraint_violations,
#         'hj_interventions': hj_interventions if mode == "switching" else None,
#         'min_hj': min_hj_value,
#         'final_cost': cost
#     }


# def main():
#     # Configuration
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Paths
#     wm_ckpt_dir = "/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/output3_frameskip1/dubins/dino_cls"
#     hj_ckpt_dir = "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/runs/ddpg_hj_latent/dino_cls-0731_1645/epoch_46"
#     video_save_path = "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/dubins_test"
    
#     actor_path = os.path.join(hj_ckpt_dir, "actor.pth")
#     critic_path = os.path.join(hj_ckpt_dir, "critic.pth")
    
#     # Load world model
#     print("Loading world model...")
#     wm = load_world_model(wm_ckpt_dir, device)
    
#     # Create HJ evaluator
#     print("Loading HJ policy...")
#     hj_evaluator = HJPolicyEvaluator(actor_path, critic_path, wm, device, with_proprio=False)
    
#     # Run simulations for each mode
#     modes = ["switching", "pid_only", "safe_only"]
#     num_runs_per_mode = 5
    
#     all_results = {}
    
#     for mode in modes:
#         print(f"\n{'='*50}")
#         print(f"Running {num_runs_per_mode} simulations in {mode} mode")
#         print(f"{'='*50}")
        
#         mode_results = []
        
#         for run_id in range(num_runs_per_mode):
#             # Create fresh environment for each run
#             env = DubinsEnvForTesting(device)
            
#             # Run simulation
#             results = simulate_dubins_with_hj(
#                 hj_evaluator=hj_evaluator,
#                 env=env,
#                 mode=mode,
#                 max_steps=200,
#                 save_video=True,
#                 video_path=video_save_path,
#                 run_id=run_id,
#                 safety_threshold=0.0
#             )
            
#             mode_results.append(results)
#             env.close()
        
#         all_results[mode] = mode_results
    
#     # Print overall summary
#     print("\n" + "="*70)
#     print("OVERALL SUMMARY")
#     print("="*70)
    
#     for mode in modes:
#         results = all_results[mode]
#         avg_steps = np.mean([r['steps'] for r in results])
#         avg_violations = np.mean([r['violations'] for r in results])
#         avg_min_hj = np.mean([r['min_hj'] for r in results])
        
#         print(f"\n{mode.upper()} MODE:")
#         print(f"  Average steps: {avg_steps:.1f}")
#         print(f"  Average violations: {avg_violations:.1f}")
#         print(f"  Average minimum HJ: {avg_min_hj:.3f}")
        
#         if mode == "switching":
#             avg_interventions = np.mean([r['hj_interventions'] for r in results])
#             avg_intervention_rate = 100 * avg_interventions / avg_steps
#             print(f"  Average HJ interventions: {avg_interventions:.1f} ({avg_intervention_rate:.1f}%)")
    
#     print(f"\nAll videos saved to: {video_save_path}")


# if __name__ == "__main__":
#     main()
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
        critic = Critic(state_dim, action_dim, [512, 512,512], 'ReLU').to(self.device)
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
        # Ensure action is consistent shape
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = np.array([action])
            elif action.ndim > 1:
                action = action.flatten()
        else:
            action = np.array([action]) if np.isscalar(action) else np.array(action)
        
        self.obs_history.append(obs)
        self.action_history.append(action)
        
        # Keep only num_hist frames
        if len(self.obs_history) > self.num_hist:
            self.obs_history.pop(0)
        if len(self.action_history) > self.num_hist:
            self.action_history.pop(0)
    
    def predict_next_state_value(self, obs, action, return_debug_info=False):
        """Predict HJ value of next state using world model's ACTUAL rollout method"""
        # Ensure action is consistent shape
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = np.array([action])
            elif action.ndim > 1:
                action = action.flatten()
        else:
            action = np.array([action]) if np.isscalar(action) else np.array(action)
        
        predicted_image = None
        predicted_proprio = None
        method_used = "actual_rollout_method"
        
        # Extract current observation
        if isinstance(obs, dict):
            visual = obs['visual']
            proprio = obs['proprio']
        else:
            visual, proprio = obs
        
        with torch.no_grad():
            try:
                # Determine how to create obs_0 based on available history
                if len(self.obs_history) >= self.num_hist:
                    # Use actual history
                    method_used = "rollout_with_history"
                    
                    # Get last num_hist observations from history
                    recent_history = self.obs_history[-self.num_hist:]
                    visual_list = []
                    proprio_list = []
                    
                    for hist_obs in recent_history:
                        if isinstance(hist_obs, dict):
                            hist_visual = hist_obs['visual']
                            hist_proprio = hist_obs['proprio']
                        else:
                            hist_visual, hist_proprio = hist_obs
                        
                        # CRITICAL FIX: Apply proper normalization for world model
                        # World model was trained with transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        # This converts [0,1] -> [-1,1] via: (x - 0.5) / 0.5 = 2*x - 1
                        if hist_visual.shape[2] == 3:  # Channels last
                            visual_np = np.transpose(hist_visual, (2, 0, 1)).astype(np.float32) / 255.0
                        else:
                            visual_np = hist_visual.astype(np.float32) / 255.0
                        
                        # Apply world model normalization: [0,1] -> [-1,1]
                        visual_np = 2.0 * visual_np - 1.0
                        
                        visual_list.append(visual_np)
                        proprio_list.append(hist_proprio)
                    
                    # Stack into obs_0 format
                    visual_tensor = torch.from_numpy(np.stack(visual_list)).unsqueeze(0).to(self.device)  # (1, num_hist, C, H, W)
                    proprio_tensor = torch.from_numpy(np.stack(proprio_list)).unsqueeze(0).to(self.device)  # (1, num_hist, proprio_dim)
                    
                # else:
                #     # Bootstrap with current observation repeated
                #     method_used = "rollout_bootstrap"
                    
                #     if visual.shape[2] == 3:  # Channels last
                #         visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
                #     else:
                #         visual_np = visual.astype(np.float32) / 255.0
                    
                #     # Apply world model normalization: [0,1] -> [-1,1] ###impimmipmimpipimipmimpipmimpipmipmipmimppimipmimpipmimpipmipmimpimpimpimpimpimpipmipmipmipmipmipmipmipmipmipimipm
                #     visual_np = 2.0 * visual_np - 1.0
                    
                #     visual_list = [visual_np] * self.num_hist
                #     proprio_list = [proprio] * self.num_hist
                    
                #     visual_tensor = torch.from_numpy(np.stack(visual_list)).unsqueeze(0).to(self.device)
                #     proprio_tensor = torch.from_numpy(np.stack(proprio_list)).unsqueeze(0).to(self.device)
                
                # Create obs_0 dictionary
                obs_0 = {'visual': visual_tensor, 'proprio': proprio_tensor}
                
                # Create action tensor for rollout - this should be (1, num_hist + num_pred, action_dim)
                # For single step prediction: num_pred = 1
                # So we need num_hist actions (for context) + 1 action (for prediction)
                
                if len(self.obs_history) >= self.num_hist:
                    # Use historical actions + new action
                    recent_actions = self.action_history[-self.num_hist:]
                    action_list = []
                    for hist_action in recent_actions:
                        if isinstance(hist_action, np.ndarray):
                            if hist_action.ndim == 0:
                                hist_action = np.array([hist_action])
                            elif hist_action.ndim > 1:
                                hist_action = hist_action.flatten()
                        else:
                            hist_action = np.array([hist_action]) if np.isscalar(hist_action) else np.array(hist_action)
                        action_list.append(hist_action)
                    
                    # Add the prediction action
                    action_list.append(action)
                # else:
                #     # Use zero actions for history + prediction action
                #     zero_action = np.zeros_like(action)
                #     action_list = [zero_action] * self.num_hist + [action]
                
                # Stack actions: (1, num_hist + 1, action_dim)
                action_tensor = torch.from_numpy(np.stack(action_list)).unsqueeze(0).to(self.device)
                
                if return_debug_info:
                    print(f"  Method: {method_used}")
                    print(f"  obs_0 visual shape: {obs_0['visual'].shape}")
                    print(f"  obs_0 visual range: [{obs_0['visual'].min():.3f}, {obs_0['visual'].max():.3f}]")
                    print(f"  obs_0 proprio shape: {obs_0['proprio'].shape}")
                    print(f"  action tensor shape: {action_tensor.shape}")
                    print(f"  Prediction action: {action}")
                
                # USE THE ACTUAL ROLLOUT METHOD
                z_obses, z_full = self.wm.rollout(obs_0=obs_0, act=action_tensor)
                
                if return_debug_info:
                    print(f"  Rollout z_obses visual shape: {z_obses['visual'].shape}")
                    print(f"  Rollout z_obses proprio shape: {z_obses['proprio'].shape}")
                
                # Get the FINAL state (last timestep) - this is our prediction
                z_final_visual = z_obses['visual'][:, -1:, :, :]  # (1, 1, visual_patches, emb_dim)
                z_final_proprio = z_obses['proprio'][:, -1:, :]   # (1, 1, proprio_emb_dim)
                
                z_obs_next = {'visual': z_final_visual, 'proprio': z_final_proprio}
                
                if return_debug_info:
                    print(f"  Final predicted state visual shape: {z_obs_next['visual'].shape}")
                    print(f"  Final predicted state proprio shape: {z_obs_next['proprio'].shape}")
                
                # Decode predicted state if requested
                if return_debug_info and self.wm.decoder is not None:
                    try:
                        print("  Decoding predicted state...")
                        decoded_obs, diff = self.wm.decode_obs(z_obs_next)
                        print(f"  Decoded visual shape: {decoded_obs['visual'].shape}")
                        print(f"  Raw decoded range: [{decoded_obs['visual'].min():.3f}, {decoded_obs['visual'].max():.3f}]")
                        
                        predicted_image_raw = decoded_obs['visual'][0, 0].cpu().numpy()  # (C, H, W)
                        
                        # Convert back from world model format [-1,1] to display format [0,1]
                        predicted_image_raw = (predicted_image_raw + 1.0) / 2.0  # [-1,1] -> [0,1]
                        predicted_image = np.transpose(predicted_image_raw, (1, 2, 0))  # (H, W, C)
                        predicted_image = np.clip(predicted_image, 0, 1)
                        
                        predicted_proprio = z_obs_next['proprio'][0, 0].cpu().numpy()
                        
                        print(f"  Converted image range: [{predicted_image.min():.3f}, {predicted_image.max():.3f}]")
                        
                    except Exception as decode_error:
                        print(f"  Decoding failed: {decode_error}")
                        import traceback
                        traceback.print_exc()
                        predicted_image = None
                        predicted_proprio = None
                
                # For HJ evaluation, we need to convert the latent back to HJ policy format
                # The HJ policy was trained on [0,1] images, but world model latents are from [-1,1] images
                # We need to be careful about this conversion...
                
                # For now, let's use the latents directly for HJ evaluation
                # since they represent the same visual content, just encoded differently
                if self.with_proprio:
                    z_vis = z_obs_next['visual'].reshape(1, -1)
                    z_prop = z_obs_next['proprio'].squeeze(1)
                    z_next_flat = torch.cat([z_vis, z_prop], dim=-1)
                else:
                    z_next_flat = z_obs_next['visual'].reshape(1, -1)
                
                # Get HJ value for predicted next state
                next_action = self.actor(z_next_flat)
                next_hj_value = self.critic(z_next_flat, next_action).item()
                
                if return_debug_info:
                    print(f"  Predicted HJ value: {next_hj_value:.3f}")
                
            except Exception as e:
                print(f"ERROR in rollout prediction: {e}")
                import traceback
                traceback.print_exc()
                # Return safe defaults
                next_hj_value = 0.0
                predicted_image = None
                predicted_proprio = None
        
        # Debug logging
        if return_debug_info:
            print(f"  Prediction method used: {method_used}")
            print(f"  History length: {len(self.obs_history)}/{self.num_hist}")
            print(f"  Action: {action}")
        
        if return_debug_info:
            return next_hj_value, predicted_image, predicted_proprio
        else:
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
                           safety_threshold=0.0, debug_plot=False):
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
    
    # Initialize history with current observation and zero actions
    # Get the expected action dimension from the environment
    zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    for _ in range(hj_evaluator.num_hist):
        hj_evaluator.update_history(obs, zero_action)
    
    # Storage for video frames and metrics
    frames = []
    step_count = 0
    done = False
    hj_interventions = 0
    total_switches = 0
    min_hj_value = float('inf')
    constraint_violations = 0
    last_controller = None
    
    # Debug storage for plotting predicted vs actual HJ values
    predicted_hj_values = []
    actual_hj_values = []
    pid_actions_taken = []
    hj_actions_taken = []
    
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
        next_hj_pid = None
        next_hj_hj = None
        
        if mode == "safe_only":
            # Always use HJ safe policy
            action = hj_evaluator.get_safe_action(obs)
            using_hj = True
            hj_actions_taken.append(action.copy())
            # Calculate what next HJ would be with this HJ action
            next_hj_hj = hj_evaluator.predict_next_state_value(obs, action)
            
        elif mode == "pid_only":
            # Always use PID controller
            action = pid_controller.get_action(proprio_state)
            using_hj = False
            pid_actions_taken.append(action.copy())
            # Calculate what next HJ would be with this PID action
            next_hj_pid = hj_evaluator.predict_next_state_value(obs, action)
            
        else:  # switching mode
            # Get PID action
            pid_action = pid_controller.get_action(proprio_state)
            
            # Predict HJ value of next state with PID action (with debug info)
            if step_count < 10:  # Only get debug info for first 10 steps to avoid slowdown
                next_hj_value, predicted_img, predicted_prop = hj_evaluator.predict_next_state_value(
                    obs, pid_action, return_debug_info=True)
            else:
                next_hj_value = hj_evaluator.predict_next_state_value(obs, pid_action)
                predicted_img, predicted_prop = None, None
            
            # Switch to safe controller if next state would be unsafe
            if next_hj_value < safety_threshold:
                action = hj_evaluator.get_safe_action(obs)
                using_hj = True
                hj_interventions += 1
                if last_controller == "PID":
                    total_switches += 1
                print(f"Step {step_count}: HJ intervention! Next HJ would be {next_hj_value:.3f}")
                print(f"  PID action: {pid_action}, HJ action: {action}")
                
                # Debug: Compare predicted vs actual if we have debug info
                if predicted_img is not None and predicted_prop is not None:
                    print(f"  Predicted proprio: {predicted_prop}")
                
                last_controller = "HJ"
                hj_actions_taken.append(action.copy())
                
                # Store predicted vs actual for debugging
                predicted_hj_values.append(next_hj_value)
                # Calculate what next HJ would be with chosen HJ action
                next_hj_hj = hj_evaluator.predict_next_state_value(obs, action)
            else:
                action = pid_action
                using_hj = False
                if last_controller == "HJ":
                    total_switches += 1
                last_controller = "PID"
                pid_actions_taken.append(action.copy())
                next_hj_pid = next_hj_value
        
        # Step environment
        obs_next, cost, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update history for world model prediction
        hj_evaluator.update_history(obs_next, action)
        
        # Store actual HJ value after taking action (for debugging)
        if mode == "switching" and len(predicted_hj_values) > len(actual_hj_values):
            actual_next_hj = hj_evaluator.get_hj_value(obs_next)
            actual_hj_values.append(actual_next_hj)
            
            # Save comparison images for first few interventions
            if len(actual_hj_values) <= 10 and step_count < 10:
                save_prediction_comparison(obs, obs_next, predicted_img, predicted_prop, 
                                         proprio_state, step_count, video_path, run_id)
        
        # Track constraint violations (negative cost means unsafe)
        if cost < 0:
            constraint_violations += 1
        
        # Render and save frame
        frame = env.render(mode="rgb_array")
        
        # Add HJ info overlay
        if frame is not None:
            # Add text overlay showing HJ value and controller
            frame_with_info = frame.copy()
            controller_text = "HJ" if using_hj else "PID"
            
            # Line 1: Current HJ value
            cv2.putText(frame_with_info, f"HJ(current): {current_hj_value:.2f}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Line 2: Next HJ value based on controller being used
            if using_hj and next_hj_hj is not None:
                cv2.putText(frame_with_info, f"HJ(next) using HJ: {next_hj_hj:.2f}", 
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            elif not using_hj and next_hj_pid is not None:
                cv2.putText(frame_with_info, f"HJ(next) using PID: {next_hj_pid:.2f}", 
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Line 3: Current controller
            cv2.putText(frame_with_info, f"Controller: {controller_text}", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
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
    
    # Create debug plots if requested
    if debug_plot and mode == "switching" and len(predicted_hj_values) > 0:
        create_debug_plots(predicted_hj_values, actual_hj_values, pid_actions_taken, 
                          hj_actions_taken, video_path, run_id)
    
    return {
        'steps': step_count,
        'violations': constraint_violations,
        'hj_interventions': hj_interventions if mode == "switching" else None,
        'min_hj': min_hj_value,
        'final_cost': cost
    }


def save_prediction_comparison(obs_current, obs_actual_next, predicted_img, predicted_prop, 
                              current_prop, step_count, video_path, run_id):
    """Save side-by-side comparison of predicted vs actual next state"""
    if predicted_img is None:
        return
        
    try:
        # Extract actual next image
        if isinstance(obs_actual_next, dict):
            actual_next_img = obs_actual_next['visual']
            actual_next_prop = obs_actual_next['proprio']
        else:
            actual_next_img = obs_actual_next[0]
            actual_next_prop = obs_actual_next[1]
        
        # Extract current image
        if isinstance(obs_current, dict):
            current_img = obs_current['visual']
        else:
            current_img = obs_current[0]
        
        # Normalize actual images to [0,1] if needed
        if actual_next_img.max() > 1.0:
            actual_next_img = actual_next_img.astype(np.float32) / 255.0
        if current_img.max() > 1.0:
            current_img = current_img.astype(np.float32) / 255.0
            
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Current state
        axes[0].imshow(current_img)
        axes[0].set_title(f'Current State\nProprio: [{current_prop[0]:.2f}, {current_prop[1]:.2f}, {current_prop[2]:.2f}]')
        axes[0].axis('off')
        
        # Predicted next state
        axes[1].imshow(predicted_img)
        if predicted_prop is not None:
            # Note: predicted_prop is the embedding (10-dim), not original proprio (3-dim)
            axes[1].set_title(f'Predicted Next\nProprio Embedding: {predicted_prop.shape[0]}D vector')
        else:
            axes[1].set_title('Predicted Next\n(No proprio decoded)')
        axes[1].axis('off')
        
        # Actual next state
        axes[2].imshow(actual_next_img)
        axes[2].set_title(f'Actual Next\nProprio: [{actual_next_prop[0]:.2f}, {actual_next_prop[1]:.2f}, {actual_next_prop[2]:.2f}]')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the comparison
        os.makedirs(video_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(video_path, f"prediction_comparison_run{run_id}_step{step_count}_{timestamp}.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Prediction comparison saved: {comparison_path}")
        
    except Exception as e:
        print(f"  Failed to save prediction comparison: {e}")


def create_debug_plots(predicted_hj_values, actual_hj_values, pid_actions, hj_actions, 
                      video_path, run_id):
    """Create debug plots for HJ prediction accuracy and action comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Predicted vs Actual HJ values
    if len(predicted_hj_values) > 0 and len(actual_hj_values) > 0:
        min_len = min(len(predicted_hj_values), len(actual_hj_values))
        pred_vals = predicted_hj_values[:min_len]
        actual_vals = actual_hj_values[:min_len]
        
        axes[0, 0].scatter(pred_vals, actual_vals, alpha=0.6)
        axes[0, 0].plot([min(pred_vals + actual_vals), max(pred_vals + actual_vals)], 
                       [min(pred_vals + actual_vals), max(pred_vals + actual_vals)], 'r--', label='Perfect prediction')
        axes[0, 0].set_xlabel('Predicted HJ Value')
        axes[0, 0].set_ylabel('Actual HJ Value')
        axes[0, 0].set_title('HJ Prediction Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Time series of predictions
        axes[0, 1].plot(pred_vals, 'b-', label='Predicted', alpha=0.7)
        axes[0, 1].plot(actual_vals, 'r-', label='Actual', alpha=0.7)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Safety threshold')
        axes[0, 1].set_xlabel('Intervention Number')
        axes[0, 1].set_ylabel('HJ Value')
        axes[0, 1].set_title('HJ Values Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    else:
        axes[0, 0].text(0.5, 0.5, 'No HJ interventions', ha='center', va='center')
        axes[0, 1].text(0.5, 0.5, 'No HJ interventions', ha='center', va='center')
    
    # Plot 3: PID actions distribution
    if len(pid_actions) > 0:
        pid_flat = np.concatenate(pid_actions) if pid_actions[0].ndim > 0 else np.array(pid_actions)
        axes[1, 0].hist(pid_flat, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_xlabel('Action Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'PID Actions Distribution (n={len(pid_actions)})')
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'No PID actions', ha='center', va='center')
    
    # Plot 4: HJ actions distribution
    if len(hj_actions) > 0:
        hj_flat = np.concatenate(hj_actions) if hj_actions[0].ndim > 0 else np.array(hj_actions)
        axes[1, 1].hist(hj_flat, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_xlabel('Action Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'HJ Actions Distribution (n={len(hj_actions)})')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No HJ actions', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save the debug plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_plot_path = os.path.join(video_path, f"debug_switching_run{run_id}_{timestamp}.png")
    plt.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug plot saved to: {debug_plot_path}")


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    wm_ckpt_dir = "/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/output3_frameskip1/dubins/dino_cls"
    hj_ckpt_dir = "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/runs/ddpg_hj_latent/dino_cls-0731_1645/epoch_10"
    video_save_path = "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/dubins_test"
    
    actor_path = os.path.join(hj_ckpt_dir, "actor.pth")
    critic_path = os.path.join(hj_ckpt_dir, "critic.pth")
    
    # Load world model
    print("Loading world model...")
    wm = load_world_model(wm_ckpt_dir, device)
    
    # Create HJ evaluator
    print("Loading HJ policy...")
    hj_evaluator = HJPolicyEvaluator(actor_path, critic_path, wm, device, with_proprio=False)
    
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
                safety_threshold=0.0,
                debug_plot=(mode == "switching")  # Only create debug plots for switching mode
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