# import gymnasium as gym
# import mani_skill.envs
# from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
# import torch
# import argparse
# from pathlib import Path

# def collect_episodes(name, num_episodes, max_steps, obs_mode, output_dir):
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)
#     obses_dir = output_path / "obses"
#     obses_dir.mkdir(exist_ok=True)

#     all_actions = []
#     all_states = []
#     all_costs = []
#     seq_lengths = []

#     for episode_idx in range(num_episodes):
#         print(f"Episode {episode_idx + 1}/{num_episodes}")
#         #Initalize environment 
#         env = gym.make(name, obs_mode = obs_mode, sensor_configs = dict(width = 224, height = 224))
#         env = FlattenRGBDObservationWrapper(env)
        
#         episode_actions = []
#         episode_states = []
#         episode_costs = []
#         episode_obs = []
#         obs, _ = env.reset()

#         for step in range(max_steps):
#             #Collect data
#             episode_states.append(obs['state'][0])
#             episode_obs.append(obs['rgb'][0][:,:,  3:6])

#             action = env.action_space.sample()
#             episode_actions.append(torch.tensor(action, dtype = torch.float32))

#             obs, _, terminated, truncated, _ = env.step(action)
#             cost =calculate_cost(env)
#             episode_costs.append(torch.tensor(cost,dtype = torch.float32))

#             if terminated or truncated:
#                 break

#         env.close()

#         all_actions.append(torch.stack(episode_actions))
#         all_states.append(torch.stack(episode_states))
#         all_costs.append(torch.stack(episode_costs))
#         seq_lengths.append(len(episode_actions))

#         torch.save(torch.stack(episode_obs).cpu(), obses_dir / f"episode_{episode_idx}.pth")

#     #Pad data so that it's all the same size
#     max_len = max(seq_lengths)

    
#     padded_actions = torch.zeros(num_episodes, max_len, all_actions[0].shape[-1])
#     padded_states = torch.zeros(num_episodes, max_len, all_states[0].shape[-1])
#     padded_costs = torch.zeros(num_episodes, max_len)

#     for i, (actions, states, costs) in enumerate(zip(all_actions, all_states, all_costs)):
#         length = len(actions)
#         padded_actions[i, :length] = actions
#         padded_states[i, :length] = states
#         padded_costs[i, :length] = costs
#     #Save data
#     torch.save(padded_actions, output_path / "actions.pth")
#     torch.save(padded_states, output_path / "states.pth")
#     torch.save(torch.tensor(seq_lengths), output_path/ "seq_lengths.pth")
#     torch.save(padded_costs, output_path / "costs.pth")

#     print(f"Saved {num_episodes} episodes to {output_path}") 

# def calculate_cost(env, collision_threshold = 1e-6):
#     #Get objects from environment
#     bowl = env.unwrapped.bowl
#     scene = env.unwrapped.scene
#     robot = env.unwrapped.agent.robot
#     #Find the correct hand link
#     all_links = robot.get_links()
#     right_hand_link = next((link for link in all_links if link.name == 'right_palm_link'), None)
#     #Calculate contact forces
#     contact_forces = scene.get_pairwise_contact_forces(right_hand_link, bowl)
#     if contact_forces is not None and len(contact_forces) > 0:
#         forces_magnitudes = torch.norm(contact_forces, dim = -1)
#         total_force = torch.sum(forces_magnitudes).item()
#         # Returns 0.8 if there is a collision
#         if total_force > collision_threshold:
#             return 0.8
#     return 0.0

# def main():
#     #Added argparser so that all variables can be changed easily 
#     parser = argparse.ArgumentParser(description= ' Record Maniskill')
#     parser.add_argument('--name', type= str, default= "UnitreeG1PlaceAppleInBowl-v1", help = 'Name of environment')
#     parser.add_argument('--num-episodes', type = int, default = 2000)
#     parser.add_argument('--max-steps', type = int, default = 300, help = 'Number of steps')
#     parser.add_argument('--obs-mode',type = str, default = "rgb+depth", help = 'Observation mode')
#     parser.add_argument('--output-dir', type = str, default = '/storage1/sibai/Active/ihab/research_new/datasets_dino/maniskill')
#     args = parser.parse_args()
    
#     collect_episodes(args.name, args.num_episodes, args.max_steps, args.obs_mode, args.output_dir)

# if __name__ == "__main__":
#     main()

import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
import torch
import argparse
from pathlib import Path

def collect_episodes(name, num_episodes, max_steps, obs_mode, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    obses_dir = output_path / "obses"
    obses_dir.mkdir(exist_ok=True)

    all_actions = []
    all_states = []
    all_costs = []
    seq_lengths = []

    for episode_idx in range(num_episodes):
        print(f"Episode {episode_idx + 1}/{num_episodes}")
        #Initalize environment 
        env = gym.make(name, obs_mode = obs_mode, sensor_configs = dict(width = 224, height = 224))
        env = FlattenRGBDObservationWrapper(env)
        
        episode_actions = []
        episode_states = []
        episode_costs = []
        episode_obs = []


        obs, _ = env.reset()

        for step in range(max_steps):
            #Collect data
            state = env.get_state()
            if len(state.shape) > 1:
                state = state.squeeze(0)
            episode_states.append(state)
            episode_obs.append(obs['rgb'][0][:,:,  3:6])

            action = env.action_space.sample()
            episode_actions.append(torch.tensor(action, dtype = torch.float32))

            obs, _, terminated, truncated, _ = env.step(action)
            cost =calculate_cost(env)
            episode_costs.append(torch.tensor(cost,dtype = torch.float32))

            if terminated or truncated:
                break

        env.close()

        all_actions.append(torch.stack(episode_actions))
        all_states.append(torch.stack(episode_states))
        all_costs.append(torch.stack(episode_costs))
        seq_lengths.append(len(episode_actions))

        torch.save(torch.stack(episode_obs).cpu(), obses_dir / f"episode_{episode_idx}.pth")

    #Pad data so that it's all the same size
    max_len = max(seq_lengths)

    
    padded_actions = torch.zeros(num_episodes, max_len, all_actions[0].shape[-1])
    padded_states = torch.zeros(num_episodes, max_len, all_states[0].shape[-1])
    padded_costs = torch.zeros(num_episodes, max_len)

    for i, (actions, states, costs) in enumerate(zip(all_actions, all_states, all_costs)):
        length = len(actions)
        padded_actions[i, :length] = actions
        padded_states[i, :length] = states
        padded_costs[i, :length] = costs
    #Save data
    torch.save(padded_actions, output_path / "actions.pth")
    torch.save(padded_states, output_path / "states.pth")
    torch.save(torch.tensor(seq_lengths), output_path/ "seq_lengths.pth")
    torch.save(padded_costs, output_path / "costs.pth")

    print(f"Saved {num_episodes} episodes to {output_path}") 

def calculate_cost(env, collision_threshold = 1e-6):
    #Get objects from environment
    bowl = env.unwrapped.bowl
    scene = env.unwrapped.scene
    robot = env.unwrapped.agent.robot
    #Find the correct hand link
    all_links = robot.get_links()
    right_hand_link = next((link for link in all_links if link.name == 'right_palm_link'), None)
    #Calculate contact forces
    contact_forces = scene.get_pairwise_contact_forces(right_hand_link, bowl)
    if contact_forces is not None and len(contact_forces) > 0:
        forces_magnitudes = torch.norm(contact_forces, dim = -1)
        total_force = torch.sum(forces_magnitudes).item()
        # Returns 0.8 if there is a collision
        if total_force > collision_threshold:
            return 0.8
    return 0.0

def main():
    #Added argparser so that all variables can be changed easily 
    parser = argparse.ArgumentParser(description= ' Record Maniskill')
    parser.add_argument('--name', type= str, default= "UnitreeG1PlaceAppleInBowl-v1", help = 'Name of environment')
    parser.add_argument('--num-episodes', type = int, default = 10)
    parser.add_argument('--max-steps', type = int, default = 300, help = 'Number of steps')
    parser.add_argument('--obs-mode',type = str, default = "rgb+depth", help = 'Observation mode')
    parser.add_argument('--output-dir', type = str, default = '/storage1/sibai/Active/ihab/research_new/datasets_dino/maniskill')
    args = parser.parse_args()
    
    collect_episodes(args.name, args.num_episodes, args.max_steps, args.obs_mode, args.output_dir)

if __name__ == "__main__":
    main()