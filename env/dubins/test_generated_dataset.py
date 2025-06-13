import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import time

def visualize_dubins_trajectories(
    data_dir="data/dubins_dataset",
    num_traj=1,             # Number of trajectories to visualize
    # show_steps=40,          # Number of steps per trajectory to show
    fps=15,                 # Animation speed
):
    obs_dir = Path(data_dir) / "obses"
    actions = torch.load(Path(data_dir) / "actions.pth")
    states = torch.load(Path(data_dir) / "states.pth")
    seq_lengths = torch.load(Path(data_dir) / "seq_lengths.pth")


    print(f"Loaded {len(seq_lengths)} trajectories.")
    n_vis = min(num_traj, len(seq_lengths))
    idxs = np.random.choice(len(seq_lengths), size=n_vis, replace=False)

    for i, idx in enumerate(idxs):
        print(f"\nTrajectory {idx}: Length = {seq_lengths[idx].item()}")
        traj_states = states[idx][:seq_lengths[idx]]
        print("  State [first]:", traj_states[0].numpy())
        print("  State [last]:", traj_states[-1].numpy())

        imgs = torch.load(obs_dir / f"episode_{idx:03d}.pth")  # [T, H, W, C] uint8
        T = imgs.shape[0]
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.title(f"Dubins Trajectory {idx}")

        for t in range(T):
            ax.clear()
            ax.imshow(imgs[t].numpy())
            ax.set_axis_off()
            ax.set_title(f"Traj {idx} - Step {t+1}/{T}")
            plt.pause(1.0 / fps)
        plt.show(block=False)
        time.sleep(1)
        plt.close(fig)
    print(f"seq_lengths.shape: {seq_lengths.shape}")
    print(f"actions type: {type(actions)}, length: {len(actions)}")
    print(f"states type: {type(states)}, length: {len(states)}")
    print(f"actions[0] shape: {actions[0].shape}")  # first trajectory
    print(f"states[0] shape: {states[0].shape}")
    print(imgs.shape)

if __name__ == "__main__":
    visualize_dubins_trajectories()
