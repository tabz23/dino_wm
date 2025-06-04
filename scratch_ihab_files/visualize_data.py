import os
# 1. Redirect Matplotlib’s cache to a scratch directory (create it if necessary)
os.environ["MPLCONFIGDIR"] = "/storage1/sibai/Active/ihab/tmp/matplotlib_cache"
# Ensure the directory exists:
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import torch
from pathlib import Path
import re
from torch.utils.data import Dataset

import matplotlib
# Use the “Agg” backend so we can save images without a display
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 2. Directory where we’ll save outputs
save_dir = "/storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/trash"
os.makedirs(save_dir, exist_ok=True)

# 3. Load the global episode-wise tensors
actions     = torch.load("/storage1/sibai/Active/ihab/research_new/datasets_dino/point_maze/actions.pth")
states      = torch.load("/storage1/sibai/Active/ihab/research_new/datasets_dino/point_maze/states.pth")
seq_lengths = torch.load("/storage1/sibai/Active/ihab/research_new/datasets_dino/point_maze/seq_lengths.pth")

# 4. Load and sort observation files
obs_dir = Path("/storage1/sibai/Active/ihab/research_new/datasets_dino/point_maze/obses")
def episode_index(fn):
    return int(re.search(r"\d+", fn.name).group())
obs_paths = sorted(obs_dir.glob("episode_*.pth*"), key=episode_index)

assert len(obs_paths) == len(seq_lengths) == len(states), "Mismatch in number of episodes"

# 5. Define the EpisodeDataset class
class EpisodeDataset(Dataset):
    def __init__(self, states, actions, obs_paths):
        self.states = states        # shape: [num_episodes, T, state_dim]
        self.actions = actions      # shape: [num_episodes, T, action_dim]
        self.obs_paths = obs_paths  # list of Path objects

    def __len__(self):
        return len(self.obs_paths)

    def __getitem__(self, idx):
        return {
            "states":  self.states[idx],              # [T, state_dim]
            "actions": self.actions[idx],             # [T, action_dim]
            "obs":     torch.load(self.obs_paths[idx])  # [T, H, W, 3]
        }

# 6. Instantiate dataset and grab the first episode
dataset = EpisodeDataset(states, actions, obs_paths)
sample = dataset[0]

# 7. Save first 3 timesteps of state/action to a text file
text_path = os.path.join(save_dir, "episode0_data.txt")
with open(text_path, "w") as f:
    f.write(f"Total episodes: {len(dataset)}\n")
    f.write(f"States shape: {sample['states'].shape}\n")
    f.write(f"Actions shape: {sample['actions'].shape}\n")
    f.write(f"Obs shape: {sample['obs'].shape}\n\n")
    f.write("First 3 timesteps:\n")
    for t in range(min(3, sample["states"].shape[0])):
        f.write(f"Timestep {t}:\n")
        f.write(f"  State:  {sample['states'][t].tolist()}\n")
        f.write(f"  Action: {sample['actions'][t].tolist()}\n")
print(f"Data saved to {text_path}")

# 8. Save the first observation frame as a PNG
obs_img = sample["obs"][0].numpy().astype("uint8")  # [224, 224, 3]
image_path = os.path.join(save_dir, "episode0_obs0.png")
plt.imshow(obs_img)
plt.axis("off")
plt.savefig(image_path, bbox_inches="tight")
plt.close()
print(f"Image saved to {image_path}")
