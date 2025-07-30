import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from datetime import datetime
import random

# Add the repository root to Python path
repo_root = "/storage1/fs1/sibai/Active/ihab/research_new/Lipschitz_Continuous_Reachability_Learning"
os.chdir(repo_root)
os.environ["PYTHONPATH"] = repo_root + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, repo_root)

# Import the highway environment
from PyHJ.reach_rl_gym_envs.ra_highway_10d_with_render_onlycost_rew_is_cost_nodisturb import Highway_10D_game_Env2cost

# Set up matplotlib config
os.environ['MPLCONFIGDIR'] = '/storage1/fs1/sibai/Active/ihab/tmp'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)


def set_seed(seed):
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def smart_controller(state):
    """
    Smart controller from your training script that navigates while avoiding obstacles
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
        steer = max(steer, 0.8)  # Turn left to increase theta
    elif ego_theta > theta_max:
        steer = min(steer, -0.8)  # Turn right to decrease theta
    
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
    
    return np.array([accel, steer], dtype=np.float32)


class HJPolicyEvaluator:
    def __init__(self, checkpoint_path, device='cuda'):
        """Initialize the HJ policy evaluator with loaded models"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load configuration and models
        self.load_models()
        
    def load_models(self):
        """Load the actor and critic models from checkpoint"""
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path / "checkpoint.pth", map_location=self.device)
        args = checkpoint['args']
        
        # Initialize networks with same architecture as training
        state_dim = 10
        action_dim = 2
        max_action = torch.tensor([1.0, 1.0], device=self.device, dtype=torch.float32)
        
        # Create actor and critic with same architecture
        self.actor = Actor(
            state_dim, 
            action_dim, 
            args['control_net'], 
            args['actor_activation'], 
            max_action
        ).to(self.device)
        
        self.critic = Critic(
            state_dim, 
            action_dim, 
            args['critic_net'], 
            args['critic_activation']
        ).to(self.device)
        
        # Load state dicts
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Set to evaluation mode
        self.actor.eval()
        self.critic.eval()
        
        print(f"Loaded HJ policy from epoch {checkpoint['epoch']}")
        
    def get_hj_value(self, state):
        """Get the HJ value (critic value) for a given state"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action = self.actor(state_tensor)
            value = self.critic(state_tensor, action).squeeze().cpu().numpy()
        return float(value)
    
    def get_hj_value_with_action(self, state, action):
        """Get the HJ value for a given state-action pair"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
            value = self.critic(state_tensor, action_tensor).squeeze().cpu().numpy()
        return float(value)
    
    def get_safe_action(self, state):
        """Get the safe action from HJ policy"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).squeeze().cpu().numpy()
        return action


class HighwayEnvWithHJInfo(Highway_10D_game_Env2cost):
    """Extended highway environment that displays HJ values"""
    
    def __init__(self, hj_evaluator):
        super().__init__()
        self.hj_evaluator = hj_evaluator
        self.current_hj_value = 0.0
        self.using_hj_control = False
        self.switch_history = []  # Track when switches occur
        
    def render(self, mode="rgb_array"):
        """Extended render function that includes HJ value information"""
        fps = self.metadata["render_fps"]
        if self.fig is None:
            # Create figure with more space for legends outside the plot
            self.fig, self.ax = plt.subplots(figsize=(15, 8))
            self.ax.set_xlim(-0.5, 2.5)
            self.ax.set_ylim(-5, 20)
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('X Position')
            self.ax.set_ylabel('Y Position')

            # Draw road boundaries (no labels to avoid legend clutter)
            self.ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            self.ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5)

            # Initialize triangle car patches
            self.car_patches = []
            self.collision_circles = []
            self.target_patches = []  # Initialize target patches list
            colors = ['red', 'blue', 'orange']
            labels = ['Car 0 (Disturbance)', 'Car 1 (Ego)', 'Car 2 (Obstacle)']
            init_angles = [-np.pi/2, 0.0, np.pi/2]  # car0 down, ego dummy, car2 up
            
            for i, (angle, color, label) in enumerate(zip(init_angles, colors, labels)):
                verts = self._triangle_vertices(0, 0, angle)
                poly = plt.Polygon(verts, closed=True, color=color, alpha=0.8)
                self.ax.add_patch(poly)
                self.car_patches.append(poly)
                
                # Add collision detection circles (only for car 0 and car 2, not ego)
                if i != 1:  # Don't add circle for ego car
                    circle = plt.Circle((0, 0), 0.5, fill=False, color='red', 
                                  linestyle='--', linewidth=2, alpha=0.7)
                    self.ax.add_patch(circle)
                    self.collision_circles.append(circle)
                else:
                    self.collision_circles.append(None)  # Placeholder for ego car

            # Create custom legend outside the plot area
            # Car legend elements
            car_legend_elements = []
            car_legend_elements.append(plt.Polygon([[0,0]], color='red', alpha=0.8, label='Car 0 (Disturbance)'))
            car_legend_elements.append(plt.Polygon([[0,0]], color='blue', alpha=0.8, label='Car 1 (Ego)'))
            car_legend_elements.append(plt.Polygon([[0,0]], color='orange', alpha=0.8, label='Car 2 (Obstacle)'))
            car_legend_elements.append(plt.Rectangle((0,0), 1, 1, facecolor='green', alpha=0.3, label='Target Region (R≥0)'))
            car_legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.5, label='Road Boundaries'))
            
            # Collision zone legend elements
            zone_legend_elements = []
            zone_legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Collision (≤0.5)'))
            zone_legend_elements.append(plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, label='Warning (≤0.8)'))
            zone_legend_elements.append(plt.Line2D([0], [0], color='yellow', linestyle='--', linewidth=2, label='Safe (>0.8)'))
            
            # Place legends outside the plot area
            car_legend = self.ax.legend(handles=car_legend_elements, 
                                      title="Environment Elements",
                                      loc='center left', 
                                      bbox_to_anchor=(1.02, 0.7),
                                      fontsize=10)
            
            zone_legend = self.ax.legend(handles=zone_legend_elements, 
                                       title="Collision Zones",
                                       loc='center left', 
                                       bbox_to_anchor=(1.02, 0.3),
                                       fontsize=10)
            
            # Add the first legend back to the axes
            self.ax.add_artist(car_legend)

            # Info text placeholder - moved to avoid legend overlap
            self.info_text = self.ax.text(
                1.05, 1, "",
                transform=self.ax.transAxes,
                va='top', ha='left',
                fontsize=11,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black')
            )

            # Adjust layout to accommodate legends
            plt.tight_layout()
            plt.subplots_adjust(right=0.75)
            
            self.ax.grid(True, alpha=0.3)

        # Update triangle positions & orientations
        positions = [
            (self.state[0], self.state[1]),
            (self.state[3], self.state[4]),
            (self.state[7], self.state[8])
        ]
        thetas = [
            -np.pi/2,
             self.state[6],
            +np.pi/2
        ]
        
        # Update car positions
        for poly, (x, y), th in zip(self.car_patches, positions, thetas):
            verts = self._triangle_vertices(x, y, th)
            poly.set_xy(verts)

        # Update dynamic target region based on current state
        # Clear previous target region patches
        for patch in self.target_patches:
            patch.remove()
        self.target_patches = []
        
        # Target set: where reward >= 0
        # reward = 10 * min((s[3] - 1.0), (s[4] - s[8] - 1.0), (s[5] - s[9] - 0.2))
        # For reward >= 0, we need ALL three conditions:
        # 1. s[3] >= 1.0  (ego X >= 1.0)
        # 2. s[4] >= s[8] + 1.0  (ego Y >= car2_Y + 1.0)  
        # 3. s[5] >= s[9] + 0.2  (ego_speed >= car2_speed + 0.2)
        
        car2_y = self.state[8]
        car2_speed = self.state[9]
        
        # Target region bounds (only spatial part - speed constraint is separate)
        target_x_min = 1.0
        target_x_max = 2.0  # Road boundary
        target_y_min = car2_y + 1.0
        target_y_max = min(20, target_y_min + 5)  # Reasonable upper bound for visualization
        
        # Only draw if target region is within plot bounds and valid
        if target_y_min < 20 and target_x_min < 2.0 and target_y_max > target_y_min:
            target_width = target_x_max - target_x_min
            target_height = target_y_max - target_y_min
            
            # Check if ego satisfies speed condition
            ego_speed = self.state[5]
            speed_satisfied = ego_speed >= car2_speed + 0.2
            
            # Color target region based on speed condition satisfaction
            target_color = 'lightgreen' if speed_satisfied else 'lightcoral'
            target_alpha = 0.4 if speed_satisfied else 0.3
            
            target_rect = plt.Rectangle((target_x_min, target_y_min), target_width, target_height, 
                                  alpha=target_alpha, color=target_color, linestyle='--', linewidth=2)
            self.ax.add_patch(target_rect)
            self.target_patches.append(target_rect)
            
            # Add text annotation with speed requirement
            target_text = self.ax.text(target_x_min + 0.05, target_y_min + 0.1, 
                                     f'TARGET\nX≥1.0, Y≥{car2_y+1.0:.1f}\nv≥{car2_speed+0.2:.1f}', 
                                     fontsize=8, color='darkgreen' if speed_satisfied else 'darkred',
                                     weight='bold', va='bottom')
            self.target_patches.append(target_text)

        # Update collision circles for car 0 and car 2
        collision_positions = [
            (self.state[0], self.state[1]),  # Car 0
            None,                            # Ego (no circle)
            (self.state[7], self.state[8])   # Car 2
        ]
        
        for i, (circle, pos) in enumerate(zip(self.collision_circles, collision_positions)):
            if circle is not None and pos is not None:
                circle.center = pos
                # Change color based on distance to ego car
                ego_pos = np.array([self.state[3], self.state[4]])
                car_pos = np.array(pos)
                distance = np.linalg.norm(ego_pos - car_pos)
                
                if distance <= 0.5:
                    circle.set_color('red')
                    circle.set_alpha(0.9)
                    circle.set_linewidth(3)
                elif distance <= 0.8:
                    circle.set_color('orange')
                    circle.set_alpha(0.7)
                    circle.set_linewidth(2)
                else:
                    circle.set_color('black')
                    circle.set_alpha(0.5)
                    circle.set_linewidth(1)

        # Update info text with HJ value and controller status
        constraint_status = "VIOLATED!" if self.last_constraint <= 0 else "Safe"
        hj_status = "SAFE" if self.current_hj_value >= 0 else "UNSAFE"
        controller_status = "HJ (Safe)" if self.using_hj_control else "Smart"
        
        # Check if ego is in target set (reward >= 0)
        in_target = self.last_reward >= 0
        target_status = "IN TARGET!" if in_target else "Outside"
        
        distances = [
            np.linalg.norm(np.array([self.state[3], self.state[4]]) - np.array([self.state[0], self.state[1]])),
            np.linalg.norm(np.array([self.state[3], self.state[4]]) - np.array([self.state[7], self.state[8]]))
        ]
        
        # Target requirements
        car2_y = self.state[8]
        car2_speed = self.state[9]
        ego_x, ego_y, ego_speed = self.state[3], self.state[4], self.state[5]
        
        # Check individual target conditions
        x_ok = ego_x >= 1.0
        y_ok = ego_y >= car2_y + 1.0
        speed_ok = ego_speed >= car2_speed + 0.2
        
        txt = (f"HJ Value: {self.current_hj_value:.3f} ({hj_status})\n"
               f"Controller: {controller_status}\n"
               f"Reward: {self.last_reward:.2f} ({target_status})\n"
               f"Cost: {self.last_constraint:.2f} ({constraint_status})\n"
               f"Dist to Car0: {distances[0]:.2f}\n"
               f"Dist to Car2: {distances[1]:.2f}\n"
               f"Ego Speed: {self.state[5]:.2f}\n"
               f"Ego Angle: {self.state[6]:.2f}\n"
               f"Target: X≥1.0({'✓' if x_ok else '✗'}), Y≥{car2_y+1.0:.1f}({'✓' if y_ok else '✗'}), v≥{car2_speed+0.2:.1f}({'✓' if speed_ok else '✗'})")
        
        self.info_text.set_text(txt)
        
        # Color code the background based on safety status
        if self.last_constraint <= 0:
            self.info_text.set_bbox(dict(facecolor='red', alpha=0.3))
        elif self.current_hj_value < 0:
            self.info_text.set_bbox(dict(facecolor='yellow', alpha=0.3))
        elif in_target:
            self.info_text.set_bbox(dict(facecolor='green', alpha=0.3))
        else:
            self.info_text.set_bbox(dict(facecolor='white', alpha=0.8))

        if mode == "human":
            plt.pause(1.0 / fps)
            plt.draw()
        elif mode == "rgb_array":
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))


def simulate_with_hj_policy(hj_evaluator, mode="hj_only", max_steps=400, save_video=True, 
                           video_path="/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/conf/vids",
                           run_id=0):
    """
    Simulate the highway environment with HJ policy
    
    Args:
        hj_evaluator: HJPolicyEvaluator instance
        mode: "hj_only" or "switching" (between smart controller and HJ)
        max_steps: Maximum simulation steps
        save_video: Whether to save video
        video_path: Path to save video
        run_id: Run identifier for filename
    """
    
    # Create environment with HJ info display
    env = HighwayEnvWithHJInfo(hj_evaluator)
    
    # Reset environment
    state, _ = env.reset()
    
    # Storage for video frames
    frames = []
    step_count = 0
    done = False
    constraint_violations = 0
    hj_interventions = 0
    total_switches = 0
    
    print(f"\nStarting simulation in {mode} mode...")
    print(f"Initial state: {state}")
    print(f"Initial HJ value: {hj_evaluator.get_hj_value(state):.3f}")
    
    while not done and step_count < max_steps:
        # Get current HJ value
        current_hj_value = hj_evaluator.get_hj_value(state)
        env.current_hj_value = current_hj_value
        
        if mode == "hj_only":
            # Always use HJ policy
            action = hj_evaluator.get_safe_action(state)
            env.using_hj_control = True
        else:  # switching mode
            # First, get smart controller action
            smart_action = smart_controller(state)
            
            # Simulate one step ahead to check if smart action is safe
            # Create a temporary copy of state to simulate
            temp_state = state.copy()
            
            # Simple forward simulation (approximate)
            dt = 0.1
            # Update ALL cars, not just ego
            temp_state[1] = state[1] - dt * state[2]  # car0 y (moves backward)
            temp_state[3] = state[3] + dt * state[5] * np.cos(state[6])  # ego x
            temp_state[4] = state[4] + dt * state[5] * np.sin(state[6])  # ego y
            temp_state[5] = state[5] + dt * 2 * smart_action[0]  # ego velocity
            temp_state[6] = state[6] + dt * 2 * smart_action[1]  # ego theta
            temp_state[8] = state[8] + dt * state[9]  # car2 y (moves forward)
                        
            # Get HJ value of the predicted next state with smart action
            # next_hj_value = hj_evaluator.get_hj_value_with_action(state, smart_action)
            next_hj_value = hj_evaluator.get_hj_value(temp_state)
            # If next state would be unsafe (HJ < 0), use HJ policy instead
            if next_hj_value < 5:
                action = hj_evaluator.get_safe_action(state)
                env.using_hj_control = True
                hj_interventions += 1
                if step_count > 0:  # Don't count initial state
                    total_switches += 1
                print(f"Step {step_count}: HJ intervention! Next HJ would be {next_hj_value:.3f}, switching to safe policy")
            else:
                action = smart_action
                env.using_hj_control = False
        
        # Create full action (pad with zeros for adversary)
        full_action = np.array([action[0], action[1], 0.0, 0.0], dtype=np.float32)
        
        # Step environment
        state, cost, done, truncated, info = env.step(full_action)
        
        # Track constraint violations
        if cost <= 0:
            constraint_violations += 1
        
        # Render and save frame
        if save_video:
            frame = env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)
        else:
            env.render(mode="human")
        
        step_count += 1
        
        # Print progress every 50 steps
        if step_count % 50 == 0:
            print(f"Step {step_count}: HJ={current_hj_value:.3f}, Cost={cost:.3f}, "
                  f"Controller={'HJ' if env.using_hj_control else 'Smart'}")
    
    # Print summary
    print(f"\nSimulation ended after {step_count} steps")
    print(f"Total constraint violations: {constraint_violations}")
    if mode == "switching":
        print(f"HJ interventions: {hj_interventions} ({100*hj_interventions/step_count:.1f}% of steps)")
        print(f"Total controller switches: {total_switches}")
    print(f"Final HJ value: {hj_evaluator.get_hj_value(state):.3f}")
    print(f"Final cost: {cost:.3f}")
    
    # Save video if requested
    if save_video and frames:
        print(f"\nSaving video with {len(frames)} frames...")
        
        # Create directory if it doesn't exist
        os.makedirs(video_path, exist_ok=True)
        
        # Create video filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"highway_hj_{mode}_run{run_id}_{timestamp}.mp4"
        full_video_path = os.path.join(video_path, video_name)
        
        # Save frames as video using imageio
        with imageio.get_writer(full_video_path, fps=env.metadata["render_fps"]) as writer:
            for frame in frames:
                writer.append_data(frame)
        
        print(f"Video saved to: {full_video_path}")
    
    # Close environment
    env.close()
    
    return {
        'steps': step_count,
        'violations': constraint_violations,
        'hj_interventions': hj_interventions if mode == "switching" else None,
        'final_hj': hj_evaluator.get_hj_value(state),
        'final_cost': cost
    }



def main():
    parser = argparse.ArgumentParser(description="Evaluate HJ policy on Highway environment")
    parser.add_argument('--checkpoint_path', type=str, 
                        default="/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/conf/ddpg_hj_highway/0728_0037/epoch_300",
                        help='Path to checkpoint directory')
    parser.add_argument('--video_path', type=str,
                        default="/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/conf/vids",
                        help='Path to save videos')
    parser.add_argument('--mode', type=str, choices=['hj_only', 'switching', 'both'], default='switching',
                        help='Simulation mode: hj_only, switching, or both')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs per mode')
    parser.add_argument('--max_steps', type=int, default=400, help='Maximum steps per simulation')
    parser.add_argument('--seed', type=int, default=43, help='Random seed')
    parser.add_argument('--no_video', action='store_true', help='Disable video saving')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("=" * 80)
    print("HJ Policy Evaluation on Highway Environment")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Video path: {args.video_path}")
    print(f"Mode: {args.mode}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Max steps: {args.max_steps}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    
    # Load HJ policy
    print("\nLoading HJ policy...")
    hj_evaluator = HJPolicyEvaluator(args.checkpoint_path)
    
    # Determine which modes to run
    modes_to_run = []
    if args.mode == 'both':
        modes_to_run = ['hj_only', 'switching']
    else:
        modes_to_run = [args.mode]
    
    # Run simulations
    all_results = {}
    
    for mode in modes_to_run:
        print(f"\n\n{'='*60}")
        print(f"Running {mode} mode simulations")
        print(f"{'='*60}")
        
        mode_results = []
        
        for run_id in range(args.num_runs):
            print(f"\n--- Run {run_id + 1}/{args.num_runs} ---")
            
            try:
                results = simulate_with_hj_policy(
                    hj_evaluator=hj_evaluator,
                    mode=mode,
                    max_steps=args.max_steps,
                    save_video=not args.no_video,
                    video_path=args.video_path,
                    run_id=run_id
                )
                mode_results.append(results)
                
            except Exception as e:
                print(f"Error in simulation: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[mode] = mode_results
    
    # Print summary statistics
    print("\n\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    
    for mode, results in all_results.items():
        if results:
            print(f"\n{mode.upper()} MODE:")
            print("-" * 40)
            
            avg_steps = np.mean([r['steps'] for r in results])
            avg_violations = np.mean([r['violations'] for r in results])
            avg_final_hj = np.mean([r['final_hj'] for r in results])
            
            print(f"Average steps: {avg_steps:.1f}")
            print(f"Average constraint violations: {avg_violations:.1f}")
            print(f"Average final HJ value: {avg_final_hj:.3f}")
            
            if mode == 'switching':
                avg_interventions = np.mean([r['hj_interventions'] for r in results])
                intervention_rate = 100 * avg_interventions / avg_steps
                print(f"Average HJ interventions: {avg_interventions:.1f} ({intervention_rate:.1f}% of steps)")
            
            # Show individual run results
            print("\nIndividual runs:")
            for i, r in enumerate(results):
                print(f"  Run {i+1}: {r['steps']} steps, {r['violations']} violations, "
                      f"final HJ={r['final_hj']:.3f}")
                if mode == 'switching' and r['hj_interventions'] is not None:
                    print(f"         HJ interventions: {r['hj_interventions']}")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    
    if not args.no_video:
        print(f"\nVideos saved to: {args.video_path}")
    
    return all_results


if __name__ == "__main__":
    results = main()