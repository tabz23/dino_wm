from os import path
from typing import Optional

import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
import imageio
from os import path
from typing import Optional

import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
import imageio

import torch
from LCRL.policy import reach_avoid_game_DDPGPolicy as DDPGPolicy
from LCRL.utils.net.common import Net
from LCRL.utils.net.continuous import Actor, Critic
from LCRL.data import Batch
from LCRL.exploration import GaussianNoise
#!/usr/bin/env python3
import os
import sys

# 1) Change into your repo root
repo_root = "/storage1/fs1/sibai/Active/ihab/research_new/Lipschitz_Continuous_Reachability_Learning"
os.chdir(repo_root)

# 2) Make sure Python will look in that folder for imports
# Option A: adjust PYTHONPATH in the environment (for subprocesses)
os.environ["PYTHONPATH"] = repo_root + os.pathsep + os.environ.get("PYTHONPATH", "")

# Option B: (or in addition) prepend to sys.path for the current process
sys.path.insert(0, repo_root)

import random
import argparse
import numpy as np
import torch
import random

def set_seed(seed):
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Highway_10D_game_Env2(gym.Env):
    # |car 0 (disturbance)  target
    # |               car2
    # |        car1           
    # imagine that we have two three cars. The target region is the right area
    # car0: x,y,v
    # car1: x,y,v, theta, (ego agent)
    # car2: x,y,v
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,  # Reduced from 30 to 5 for slower video
    }

    def __init__(self):
        self.high = np.array([  1, 20, 2,
                                2, 20, 3, np.pi*3/4,
                                2, 20, 2], dtype=np.float32)
        self.low  = np.array([  0.0,  0.0,   0.5,
                                0.0,  0.0, 0.5, np.pi*1/4,
                                1.0,  0.0, 0.5], dtype=np.float32)

        self.action1_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action2_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space  = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)

        # Modified initial conditions to start ego car behind other cars
        self.initial_condition_high = np.array([   1.0, 15, 2,
                                                  2, 5, 3.0, np.pi*3/4,  # Ego starts lower Y (behind)
                                                  2, 10, 1.5], dtype=np.float32)
        self.initial_condition_low  = np.array([   0.0, 10, 0.5,
                                                  0.0, 0.0, 0.5, np.pi*1/4,  # Ego starts at Y=0-5
                                                  1.6, 5.0, 0.5], dtype=np.float32)

        # rendering state
        self.fig = None
        self.ax = None
        self.car_patches = []
        self.collision_circles = []  # For cost visualization
        self.target_patches = []  # For dynamic target region
        self.info_text = None

        # last step info
        self.last_reward = 0.0
        self.last_constraint = 0.0

    def step(self, act):
        s = self.state.copy()
        # reward: forward progress & speed advantage
        reward = 10 * min(
            (s[3] - 1.0),
            (s[4] - s[8] - 1.0),
            (s[5] - s[9] - 0.2)
        )
        # cost: collision avoidance
        const = 10 * min(
            np.sqrt((s[3]-s[0])**2 + (s[4]-s[1])**2) - 0.5,
            np.sqrt((s[3]-s[7])**2 + (s[4]-s[8])**2) - 0.5
        )

        # store for render
        self.last_reward     = float(reward)
        self.last_constraint = float(const)

        # dynamics
        dt, eps = 0.1, 0.1
        # car0 disturbance
        self.state[1] = s[1] - dt * s[2]
        self.state[2] = s[2] + dt * eps * act[2]
        # ego car
        self.state[3] = s[3] + dt * s[5] * np.cos(s[6])
        self.state[4] = s[4] + dt * s[5] * np.sin(s[6])
        self.state[5] = s[5] + dt * 2 * act[0]
        self.state[6] = s[6] + dt * 2 * act[1]
        # car2 obstacle
        self.state[8] = s[8] + dt * s[9]
        self.state[9] = s[9] + dt * eps * act[3]

        Done = False
        # ego out of bounds
        if (self.state[3] > self.high[3]+0.1 or self.state[3] < self.low[3]-0.1 or
            self.state[4] > self.high[4]+100 or self.state[4] < self.low[4]-10):
            Done = True
        # disturbance or obstacle out of bounds
        if (self.state[1] > self.high[1]+10 or self.state[1] < self.low[1]-10 or
            self.state[8] > self.high[8]+10 or self.state[8] < self.low[8]-10):
            Done = True
        # invalid speeds
        if (self.state[2] > self.high[2]+1 or self.state[2] < 0 or
            self.state[5] > self.high[5]+2 or self.state[5] < 0):
            Done = True

        return self.state.astype(np.float32), np.array(reward, np.float32), Done, False, {"constraint": const}

    def reset(self, initial_state=np.array([]), seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # return initial_state if it is assigned a value
        if len(initial_state) == 0:
            self.state = np.random.uniform(self.initial_condition_low, 
                                            self.initial_condition_high, 
                                            (10)).astype(np.float32)
        else:
            self.state = initial_state.astype(np.float32)

        self.last_reward = 0.0
        self.last_constraint = 0.0

        return self.state, {}

    def _triangle_vertices(self, x, y, theta, size=0.3):
        l = size
        w = size * 0.5
        return np.array([
            [x +  l*np.cos(theta),        y +  l*np.sin(theta)],
            [x + -l*0.5*np.cos(theta) -  w*np.sin(theta),
             y + -l*0.5*np.sin(theta) +  w*np.cos(theta)],
            [x + -l*0.5*np.cos(theta) +  w*np.sin(theta),
             y + -l*0.5*np.sin(theta) -  w*np.cos(theta)],
        ])

    def render(self, mode="human"):
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
                poly = Polygon(verts, closed=True, color=color, alpha=0.8)
                self.ax.add_patch(poly)
                self.car_patches.append(poly)
                
                # Add collision detection circles (only for car 0 and car 2, not ego)
                if i != 1:  # Don't add circle for ego car
                    circle = Circle((0, 0), 0.5, fill=False, color='red', 
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
            
            target_rect = Rectangle((target_x_min, target_y_min), target_width, target_height, 
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

        # Update info text with more details including target set status
        constraint_status = "VIOLATED!" if self.last_constraint <= 0 else "Safe"
        
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
        
        txt = (f"Reward: {self.last_reward:.2f} ({target_status})\n"
               f"Cost: {self.last_constraint:.2f} ({constraint_status})\n"
               f"Dist to Car0: {distances[0]:.2f}\n"
               f"Dist to Car2: {distances[1]:.2f}\n"
               f"Ego Speed: {self.state[5]:.2f}\n"
               f"Ego Angle: {self.state[6]:.2f}\n"
               f"Target: X≥1.0({'✓' if x_ok else '✗'}), Y≥{car2_y+1.0:.1f}({'✓' if y_ok else '✗'}), v≥{car2_speed+0.2:.1f}({'✓' if speed_ok else '✗'})")
        
        self.info_text.set_text(txt)
        if self.last_constraint <= 0:
            self.info_text.set_bbox(dict(facecolor='red', alpha=0.3))
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

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def load_ra_policy():
    """Load the pre-trained Reach-Avoid policy from the specified path."""
    class Args:
        task = 'ra_highway_Game-v2'
        state_shape = (10,)
        action_shape = (4,)
        action1_shape = (2,)
        action2_shape = (2,)
        max_action1 = 1.0
        max_action2 = 1.0
        control_net = [512, 512, 512]
        disturbance_net = [512, 512, 512]
        critic_net = [512, 512, 512]
        actor_activation = 'ReLU'
        critic_activation = 'ReLU'
        device = 'cpu'
        tau = 0.005
        gamma = 0.95
        exploration_noise = 0.1
        rew_norm = False
        n_step = 1
        actor_gradient_steps = 1

    args = Args()

    # Define activation functions
    actor_activation = torch.nn.ReLU
    critic_activation = torch.nn.ReLU

    # Define networks
    actor1_net = Net(args.state_shape, hidden_sizes=args.control_net, activation=actor_activation, device=args.device)
    actor1 = Actor(actor1_net, args.action1_shape, max_action=args.max_action1, device=args.device).to(args.device)
    actor1_optim = torch.optim.Adam(actor1.parameters(), lr=1e-4)  # Dummy optimizer

    actor2_net = Net(args.state_shape, hidden_sizes=args.disturbance_net, activation=actor_activation, device=args.device)
    actor2 = Actor(actor2_net, args.action2_shape, max_action=args.max_action2, device=args.device).to(args.device)
    actor2_optim = torch.optim.Adam(actor2.parameters(), lr=1e-4)  # Dummy optimizer

    critic_net = Net(args.state_shape, args.action_shape, hidden_sizes=args.critic_net, activation=critic_activation, concat=True, device=args.device)
    critic = Critic(critic_net, device=args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)  # Dummy optimizer

    # Create policy
    policy = DDPGPolicy(
        critic,
        critic_optim,
        tau=args.tau,
        gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=spaces.Box(low=-1, high=1, shape=(4,)),
        actor1=actor1,
        actor1_optim=actor1_optim,
        actor2=actor2,
        actor2_optim=actor2_optim,
        actor_gradient_steps=args.actor_gradient_steps,
    )

    # Load pre-trained model
    log_path = "/storage1/fs1/sibai/Active/ihab/research_new/Lipschitz_Continuous_Reachability_Learning/experiment_script/pretrained_neural_networks/ra_highway_Game-v2/ddpg_reach_avoid_actor_activation_ReLU_critic_activation_ReLU_game_gd_steps_1_tau_0.005_training_num_8_buffer_size_40000_c_net_512_3_a1_512_3_a2_512_3_gamma_0.95/noise_0.1_actor_lr_0.0001_critic_lr_0.001_batch_512_step_per_epoch_40000_kwargs_{}_seed_0"
    epoch_id = 200
    policy.load_state_dict(torch.load(log_path + '/epoch_id_{}/policy.pth'.format(epoch_id), map_location=args.device))
    policy.eval()  # Set to evaluation mode for inference
    return policy

def ra_controller(state, policy):
    """Generate control actions using the Reach-Avoid policy."""
    tmp_obs = np.array(state).reshape(1, -1)
    tmp_batch = Batch(obs=tmp_obs, info=Batch())
    with torch.no_grad():
        tmp = policy(tmp_batch, model="actor_old").act
    act = policy.map_action(tmp).cpu().numpy().flatten()
    return act[:2]  # Return only the control actions (first 2 dimensions)

def naive_controller(state):
    """Naive controller that just drives forward"""
    return np.array([0.5, 0.0], dtype=np.float32)

def smart_controller(state):
    """
    Improved controller that:
    1. Drives toward +Y direction (theta ≈ π/2)
    2. Stays within road boundaries (X between 0 and 2)
    3. Avoids other cars by slight sideways movement
    4. Properly normalizes angles to avoid cosine symmetry issues
    
    Coordinate system:
    - X: 0 (left) to 2 (right)
    - Y: forward direction (car moves in +Y)
    - theta: angle where π/2 points in +Y direction
    - positive steering (act[1] > 0) increases theta (turns left)
    - negative steering (act[1] < 0) decreases theta (turns right)
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
    # If angle_diff > 0: need to turn left (positive steering)
    # If angle_diff < 0: need to turn right (negative steering)
    steer = np.clip(angle_diff * 0.5, -0.5, 0.5)  # Gentle steering
    
    # Ensure theta stays within environment bounds [π/4, 3π/4]
    # If theta is outside bounds, steer to bring it back
    theta_min, theta_max = np.pi / 4, 3 * np.pi / 4
    if ego_theta < theta_min:
        steer = max(steer, 0.5)  # Turn left to increase theta
        print(f"WARNING: Theta {ego_theta:.3f} below min {theta_min:.3f}, steering left")
    elif ego_theta > theta_max:
        steer = min(steer, -0.5)  # Turn right to decrease theta
        print(f"WARNING: Theta {ego_theta:.3f} above max {theta_max:.3f}, steering right")
    
    # BOUNDARY AVOIDANCE (highest priority)
    road_center = 1.0  # center of road (x = 1.0)
    boundary_margin = 0.2  # safety margin from boundaries
    
    if ego_pos[0] < boundary_margin:
        # Too close to left boundary (x=0) - steer right
        steer = min(steer, -0.8)  # Strong right turn (negative steering)
        print(f"WARNING: Near left boundary at x={ego_pos[0]:.3f}, steering right")
    elif ego_pos[0] > (2.0 - boundary_margin):
        # Too close to right boundary (x=2) - steer left
        steer = max(steer, 0.8)  # Strong left turn (positive steering)
        print(f"WARNING: Near right boundary at x={ego_pos[0]:.3f}, steering left")
    else:
        # Within safe boundaries - gentle correction toward center
        center_diff = road_center - ego_pos[0]  # positive if we need to go right
        center_correction = -center_diff * 0.2  # Negative for right, positive for left
        steer = np.clip(steer + center_correction, -0.5, 0.5)
    
    # COLLISION AVOIDANCE with other cars
    collision_distance = 1.2  # Distance to start avoiding
    avoidance_strength = 0.4
    
    # Check distance to car 0 (disturbance car)
    # dist_to_car0 = np.linalg.norm(ego_pos - car0_pos)
    # if dist_to_car0 < collision_distance:
    #     relative_x = car0_pos[0] - ego_pos[0]
    #     if relative_x > 0:
    #         # Car 0 is to our right, steer left (positive steering)
    #         steer = max(steer, avoidance_strength)
    #     else:
    #         # Car 0 is to our left, steer right (negative steering)
    #         steer = min(steer, -avoidance_strength)
    #     print(f"Avoiding car 0: distance={dist_to_car0:.3f}, car at relative_x={relative_x:.3f}")
    
    # # Check distance to car 2 (obstacle car)
    # dist_to_car2 = np.linalg.norm(ego_pos - car2_pos)
    # if dist_to_car2 < collision_distance:
    #     relative_x = car2_pos[0] - ego_pos[0]
    #     if relative_x > 0:
    #         # Car 2 is to our right, steer left (positive steering)
    #         steer = max(steer, avoidance_strength)
    #     else:
    #         # Car 2 is to our left, steer right (negative steering)
    #         steer = min(steer, -avoidance_strength)
    #     print(f"Avoiding car 2: distance={dist_to_car2:.3f}, car at relative_x={relative_x:.3f}")
    
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

def zero_adversary_action():
    """No adversarial action"""
    return np.array([0.0, 0.0], dtype=np.float32)

def random_adversary_action():
    """Random adversarial action"""
    return np.random.uniform(-0.5, 0.5, 2).astype(np.float32)

def simulate_environment(controller_type="naive", adversary_type="zero", run_id=0, save_video=True, 
                         save_path="/storage1/fs1/sibai/Active/ihab/research_new/Lipschitz_Continuous_Reachability_Learning/LCRL/reach_rl_gym_envs/vids"):
    """
    Simulate the highway environment with different controller and adversary options
    
    Args:
        controller_type: "naive", "smart", or "ra"
        adversary_type: "zero" or "random"
        save_video: Whether to save frames for video
        save_path: Directory to save video
    """
    
    # Create environment
    env = Highway_10D_game_Env2()
    
    # Reset environment
    state, _ = env.reset()
    
    # Choose controller
    if controller_type == "naive":
        controller = naive_controller
        print("Using naive controller (drives straight)")
    elif controller_type == "smart":
        controller = smart_controller
        print("Using smart controller (improved evasive)")
    elif controller_type == "ra":
        policy = load_ra_policy()
        controller = lambda state: ra_controller(state, policy)
        print("Using Reach-Avoid (RA) policy controller")
    else:
        raise ValueError(f"Unknown controller_type: {controller_type}")
    
    # Choose adversary
    if adversary_type == "zero":
        adversary = zero_adversary_action
        print("Using zero adversarial action")
    else:
        adversary = random_adversary_action
        print("Using random adversarial action")
    
    # Storage for frames if saving video
    frames = []
    step_count = 0
    done = False
    constraint_violations = 0
    target_reached = False
    first_target_step = None
    
    print(f"\nStarting simulation...")
    print(f"Initial state: {state}")
    print(f"Ego car starts at: ({state[3]:.2f}, {state[4]:.2f})")
    print(f"Car 0 at: ({state[0]:.2f}, {state[1]:.2f})")
    print(f"Car 2 at: ({state[7]:.2f}, {state[8]:.2f})")
    
    while not done and step_count < 500:  # Max steps to prevent infinite loops
        # Get actions
        control_action = controller(state)
        adversary_action = adversary()
        
        # Combine actions
        full_action = np.concatenate([control_action, adversary_action])
        
        # Step environment
        state, reward, done, truncated, info = env.step(full_action)
        constraint_value = info["constraint"]
        
        # Check if target is reached (reward >= 0)
        if reward >= 0 and not target_reached:
            target_reached = True
            first_target_step = step_count
            # print(f"\n*** TARGET REACHED AT STEP {step_count} ***")
            # print(f"Reward: {reward:.4f} (≥ 0)")
            # print(f"Ego position: ({state[3]:.2f}, {state[4]:.2f})")
            # print(f"Mission accomplished! (RA controller can now terminate)")
        
        # Render environment
        if save_video:
            frame = env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)
        else:
            env.render(mode="human")
        
        # Track constraint violations (constraint violation when <= 0)
        if constraint_value <= 0:
            constraint_violations += 1
            # if constraint_violations == 1:  # First violation
                # print(f"\n*** FIRST CONSTRAINT VIOLATION AT STEP {step_count} ***")
                # print(f"Constraint value: {constraint_value:.4f}")
                # print(f"Ego position: ({state[3]:.2f}, {state[4]:.2f})")
        
        # Print progress
        # if step_count % 50 == 0:
            # print(f"Step {step_count}: Reward={reward:.3f}, Constraint={constraint_value:.3f}")
            # print(f"  Ego: ({state[3]:.2f}, {state[4]:.2f}), θ={state[6]:.2f}, v={state[5]:.2f}")
            # print(f"  Violations so far: {constraint_violations}")
            # if target_reached:
            #     print(f"  Target reached at step {first_target_step}")
        
        step_count += 1
    
    # print(f"\nSimulation ended after {step_count} steps")
    # print(f"Done: {done}")
    # print(f"Target reached: {target_reached}" + (f" (step {first_target_step})" if target_reached else ""))
    # print(f"Total constraint violations: {constraint_violations}")
    # print(f"Final state: {state}")
    # print(f"Final reward: {reward}")
    # print(f"Final constraint: {constraint_value}")
    
    # if controller_type == "ra" and target_reached:
    #     print(f"\n*** RA CONTROLLER ANALYSIS ***")
    #     print(f"✓ Successfully navigated around obstacles")
    #     print(f"✓ Reached target set (reward ≥ 0) at step {first_target_step}")
    #     print(f"✓ Mission accomplished - RA objective completed")
    #     if done and step_count < 500:
    #         print(f"✓ Episode terminated after reaching target (normal RA behavior)")
    #     if constraint_violations == 0:
    #         print(f"✓ Zero constraint violations - perfect safety!")
    #     else:
    #         print(f"⚠ {constraint_violations} constraint violations occurred")
    
    # Save video if requested
    if save_video and frames:
        print(f"\nSaving video with {len(frames)} frames...")
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Create video filename
        video_name = f"highway_sim_{controller_type}_{adversary_type}_run{run_id}.mp4"
        video_path = os.path.join(save_path, video_name)
        
        # Save frames as video using imageio with lower fps
        with imageio.get_writer(video_path, fps=env.metadata["render_fps"]) as writer:
            for frame in frames:
                writer.append_data(frame)
        
        print(f"Video saved to: {video_path}")
    
    # Close environment
    env.close()
    
    return step_count, reward, constraint_value, constraint_violations, target_reached

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with a specified seed.")
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set the seed
    set_seed(args.seed)
    print("seed",args.seed)
    
    print("Highway Environment Simulation")
    print("=" * 50)
    
    # Run different combinations including RA policy
    scenarios = [
         ("ra", "random", "Reach-Avoid controller with random adversary"),
        ("naive", "zero", "Naive controller with no adversary"),
        ("naive", "random", "Naive controller with random adversary"),
        ("smart", "zero", "Smart controller with no adversary"),
        ("smart", "random", "Smart controller with random adversary"),
        ("ra", "zero", "Reach-Avoid controller with no adversary"),
        
    ]
    num_runs = 3  # Number of videos to save per scenario
    
    for controller, adversary, description in scenarios:
        print(f"\n\n{description}")
        print("-" * len(description))
        
        # Run the scenario multiple times
        for run_id in range(num_runs):
            print(f"\nRun {run_id + 1} of {num_runs}")
            try:
                steps, final_reward, final_constraint, violations, reached_target = simulate_environment(
                    controller_type=controller,
                    adversary_type=adversary,
                    run_id=run_id,  # Pass the run identifier
                    save_video=True
                )
                
                print(f"Results: {steps} steps, reward: {final_reward:.3f}, "
                    f"constraint: {final_constraint:.3f}, violations: {violations}, target: {reached_target}")
            
            except Exception as e:
                print(f"Error in simulation: {e}")
    
    print("\nAll simulations completed!")