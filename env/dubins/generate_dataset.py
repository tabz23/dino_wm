import torch
import numpy as np
import os
from tqdm import trange
from dubins import DubinsEnv  # Make sure this import is correct!
from pathlib import Path

def collect_dubins_dataset(
    outdir="/storage1/sibai/Active/ihab/research_new/datasets_dino/dubins",
    n_traj=5000,
    min_traj_len=349,
    max_traj_len=350,
    render_size=255,
    seed=42
):
    np.random.seed(seed)
    os.makedirs(outdir, exist_ok=True)
    obs_dir = Path(outdir) / "obses"
    obs_dir.mkdir(parents=True, exist_ok=True)
    actions_list = []
    states_list = []
    seq_lengths = []

    for idx in trange(n_traj, desc="Collecting Dubins Trajectories"):
        env = DubinsEnv(seed=seed + idx)
        obs, full_state = env.reset()
        visuals = []
        actions = []
        states = []
        done = False
        t = 0
        max_len = np.random.randint(min_traj_len, max_traj_len + 1)
        while not done and t < max_len:
            # random action
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            img = obs["visual"]
            visuals.append(img)
            actions.append(action)
            states.append(info["state"])
            t += 1
        # Save trajectory visuals (images)
        visuals = np.stack(visuals).astype(np.uint8)  # [T, H, W, C]
        torch.save(torch.from_numpy(visuals), obs_dir / f"episode_{idx:03d}.pth")
        # Save actions and states as tensors (not stacking across trajectories)
        actions_list.append(torch.tensor(np.stack(actions), dtype=torch.float32))
        states_list.append(torch.tensor(np.stack(states), dtype=torch.float32))
        seq_lengths.append(t)

    # Save lists of tensors (not stacked!)
    torch.save(actions_list, Path(outdir) / "actions.pth")
    torch.save(states_list, Path(outdir) / "states.pth")
    torch.save(torch.tensor(seq_lengths, dtype=torch.long), Path(outdir) / "seq_lengths.pth")
    print(f"Saved {n_traj} trajectories to {outdir}")

if __name__ == "__main__":
    collect_dubins_dataset()
