import torch
import numpy as np
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional
from traj_dset import TrajDataset, TrajSlicerDataset, get_train_val_sliced, split_traj_datasets


class CarlaDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        normalize_states: bool = True,
        n_rollout: Optional[int] = None,
        with_costs: bool = True
    ):
        self.data_path = Path(data_path)
        self.transform = transform 

        self.states = [traj.float() for traj in torch.load(self.data_path / "states.pth")]
        self.actions = [traj.float() for traj in torch.load(self.data_path / "actions.pth")]
        self.seq_lengths = torch.load(self.data_path / "seq_lengths.pth").long()
        self.costs = [traj.float() for traj in torch.load(self.data_path / "costs.pth")]

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = min(n_rollout, len(self.seq_lengths))
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        
        self.proprios = [traj[:, :6] for traj in self.states] 

        self.action_dim = self.actions[0].shape[-1]
        self.state_dim = self.states[0].shape[-1]
        self.proprio_dim = self.proprios[0].shape[-1]
        print(f"Action dim: {self.action_dim}, State dim: {self.state_dim}, Proprio dim: {self.proprio_dim}")
        
        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self.get_data_mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios, self.seq_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = [
            (traj - self.action_mean) / self.action_std for traj in self.actions
        ]
        self.proprios = [
            (traj - self.proprio_mean) / self.proprio_std for traj in self.proprios
        ]



    def get_seq_length(self, idx):
        return self.seq_lengths[idx].item()

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.get_seq_length(i)
            result.append(self.actions[i][:T, :])
        return torch.cat(result, dim = 0)

    def get_frames(self, idx, frames):
        obs_file = self.data_path / "obses" / f"episode_{int(idx):03d}.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = torch.load(obs_file, map_location=device)
        images = images.float() / 255.0
        images = rearrange(images, "T H W C -> T C H W")
        if self.transform:
            images = self.transform(images)

        image = images[frames]
        actions = self.actions[idx][frames]
        full_states = self.states[idx][frames]
        proprio_states = self.proprios[idx][frames]
            
        obs = {
            "visual" : image,
            "proprio" : proprio_states
        }

        return obs, actions, full_states, {}
        

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))
    
    def __len__(self):
        return len(self.seq_lengths)
    
    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0
        
    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj][:traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0)
        return data_mean, data_std
        
def load_carla_slice_train_val(
        data_path,
        n_rollout = 50,
        normalize_action = True,
        split_ratio = 0.8,
        num_hist = 0,
        num_pred = 0,
        frameskip = 0,
):
    dset = CarlaDataset(
        data_path=data_path,
        normalize_action=normalize_action,
        n_rollout=n_rollout,
    )
    
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset, 
        train_fraction=split_ratio, 
        num_frames=num_hist + num_pred, 
        frameskip=frameskip
    )

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset