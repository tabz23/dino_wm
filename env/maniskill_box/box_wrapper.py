import gymnasium as gym
import mani_skill.envs
import torch
from pathlib import Path
import numpy as np
import multiprocessing as mp
#obs_mode = "state"
#.get_state()
#[0:13] - Box
#[13:26] - Unknown
#[26:76] - proprio
#obs
#[0:50] - proprio
#[50:64] - right_tcp_pos, left_tcp_pos
#[64:71] - box
#[71:77] - tcp_to_box_pos 

def aggregate_dct(dcts):
        full_dct = {}
        for dct in dcts:
            for key, value in dct.items():
                if key not in full_dct:
                    full_dct[key] = []
                full_dct[key].append(value)
        for key, value in full_dct.items():
            if isinstance(value[0], torch.Tensor):
                full_dct[key] = torch.stack(value)
            else:
                full_dct[key] = np.stack(value)
        return full_dct

class ManiskillWrapper(gym.Env):
    def __init__(
            self,
            env_name = "UnitreeG1TransportBox-v1",
            obs_mode = "state",
            size = (224,224),
            
    ):
        self._env = gym.make(env_name, obs_mode = obs_mode, sensor_configs = dict(width = size[0], height = size[1]))
        self.action_dim = self._env.action_space.shape[0]
        
    @property
    def action_space(self):
        return self._env.action_space
    
    def calculate_cost(env, z_threshold = 0.6):
        box_z = env.unwrapped.box.pose.p[0,2].item()
        if box_z < z_threshold:
            return 1.0
        return 0.0
       
    def sample_random_init_goal_states(self,seed):
        rs = np.random.RandomState(seed)
        obs, _ = self._env.reset()
        full_state = self._env.unwrapped.get_state().cpu().numpy()

        init_state = full_state.copy()
        goal_state = full_state.copy()

        if len(init_state.shape) > 1:
            init_flat = init_state[0]
            goal_flat = goal_state[0] 
        else:
            init_flat = init_state
            goal_flat = goal_state
        box_xy = rs.uniform([-0.05, -0.05], [0.05, 0.2], 2)
       
        init_flat[0:3] = [box_xy[0], box_xy[1], 0.75061]
        init_flat[3:7] = self._random_quaternion_z_only(rs)

        ##add in the final proprio position as well as the final position of the box
        goal_flat[0:13] = [1.0878e-01,  3.2413e-01,  7.5867e-01,  6.8007e-01,  3.4473e-02,
        -3.6622e-02,  7.3142e-01, -5.0470e-01,  8.1800e-01, -2.9583e-01,
        -4.2026e+00, -5.7218e-02,  1.6107e-01]

        goal_flat[26:76] = [4.6995e-01, -6.5044e-02, -2.0190e-01,  1.2882e+00,
        -2.6454e-01,  4.5111e-02,  5.4679e-01,  2.0209e-01,  4.0685e-01,
        -2.0531e-01, -1.6171e+00, -5.2359e-01,  3.0008e-01,  9.5611e-02,
         2.1791e-01,  6.2726e-01, -3.0000e-01, -1.0000e+00,  1.7502e-05,
         3.5060e-07,  5.6523e-02,  1.8400e+00,  1.8400e+00, -5.2480e-05,
        -8.7245e-01,  1.2882e+00, -7.9382e-01,  1.2051e+00,  2.3846e-01,
        -4.8248e-02,  5.8930e-01,  1.2138e+00,  1.2230e+00,  1.2345e+00,
         3.5968e-01, -1.2002e+00, -7.7368e-05, -5.9855e-04,  1.4155e+00,
        -2.5819e+00,  8.1271e-02,  6.2267e-05,  1.3574e-04, -1.4242e-04,
         6.1616e-06, -1.3426e+00, -5.4372e-05,  3.5461e-06,  4.3217e-04,
         2.9983e+00]
        
        return init_state, goal_state
    
    def _random_quaternion_z_only(self, rs):
        angle = rs.uniform(0, np.pi/6)
        w = np.cos(angle/2)
        x = 0.0
        y = 0.0
        z = np.sin(angle/2)

        return np.array([w, x, y, z])
    
    def eval_state(self, goal_state, cur_state):
        if isinstance(cur_state, torch.Tensor):
            cur_state = cur_state.cpu().numpy()
        if isinstance(goal_state, torch.Tensor):
            goal_state = goal_state.cpu().numpy()

        box_cur = cur_state[0:3]
        robot_cur = cur_state[26:76]
        cur = np.concatenate([box_cur, robot_cur])

        box_goal = goal_state[0:3]
        robot_goal = goal_state[26:76]
        goal = np.concatenate([box_goal, robot_goal])
        poss_diff = np.linalg.norm(cur - goal)
        success = poss_diff < 0.1

        state_dist = np.linalg.norm(goal_state - cur_state)
        return{
            'success': success,
            'state_dist' : state_dist
        }
    
    def prepare(self, seed, init_state):
        self._env.reset(seed = seed)
        if len(init_state.shape) ==1:
            state_with_batch = init_state.reshape(1, -1)
        else:
            state_with_batch = init_state
        self._env.unwrapped.set_state(state_with_batch)
        image = self._env.unwrapped.render_sensors().squeeze(0)[:, -224:, :]
        state = self._env.unwrapped.get_state()
        obs = {
            'proprio': state[0][26:76],
            'visual': image
        }
        return obs, state
    
    def step_multiple(self, actions):
        obses = []
        rewards = []
        dones = []
        infos = []

        for action in actions:
            obs, reward, truncated, terminated, info = self._env.step(action)
            visual = self._env.unwrapped.render_sensors().squeeze(0)[:,-224:,:]
            state = self._env.unwrapped.get_state()
            proprio = obs[0][0:50]
            obs = {'visual' : visual, 'proprio': proprio}
            obses.append(obs)
            rewards.append(reward)
            done = terminated or truncated
            dones.append(done)
            info['state'] = state
            infos.append(info)
            if terminated or truncated:
                break

        obses = aggregate_dct(obses)
        rewards = np.array(rewards)
        dones = np.array(dones)
        infos = aggregate_dct(infos)

        return obses, rewards, dones, infos 
    
    def rollout(self, seed, init_state, actions):
        obs, state = self.prepare(seed, init_state)  

        obses, rewards, dones, infos = self.step_multiple(actions)
        
        initial_visual = obs['visual'].unsqueeze(0) 
        initial_proprio = obs['proprio'].unsqueeze(0)
    
        result_obses = {
            'visual': torch.cat([initial_visual, obses['visual']], dim=0),
            'proprio': torch.cat([initial_proprio, obses['proprio']], dim=0)
        }
    
        initial_state = state.unsqueeze(0)
        states = torch.cat([initial_state, infos['state']], dim=0)
    

        return result_obses, states
    
    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, action):
        return self._env.step(action)

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def close(self):
        return self._env.close()

    @property
    def observation_space(self):
        return self._env.observation_space