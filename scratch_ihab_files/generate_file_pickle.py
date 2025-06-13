#!/usr/bin/env python3
"""
make_pusht_goal_file.py

This script:
  1. Generates plan_targets.pkl for Pusht with:
       - n_evals   = 1
       - goal_H    = 5
       - frameskip = 5
       - output    = /storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/plan_targets.pkl
  2. After saving, it reloads that pickle and prints out each key
     (and sub‐key) along with their types and shapes, in the
     same nested format you saw earlier.

Usage:
    python make_pusht_goal_file.py

No arguments needed (all settings are hard‐coded for n_evals=1, goal_H=5, frameskip=5).
"""

import pickle
import numpy as np
import torch
from pathlib import Path
import sys
import os
sys.path.append("/storage1/sibai/Active/ihab/research_new/dino_wm")
from env.pusht.pusht_wrapper import PushTWrapper


def describe(item):
    """Return a short description of type and shape (if Tensor/ndarray)."""
    if isinstance(item, torch.Tensor):
        return f"Tensor(shape={tuple(item.shape)}, dtype={item.dtype})"
    elif isinstance(item, np.ndarray):
        return f"ndarray(shape={item.shape}, dtype={item.dtype})"
    elif isinstance(item, dict):
        return f"dict (len={len(item)})"
    else:
        return f"{type(item).__name__}"


def print_pickle_structure(pickle_path):
    """
    Load the pickle at `pickle_path` and print its contents in a nested format:

    dict (len=6)
      └─ obs_0: dict (len=2)
          └─ visual: ndarray(shape=(1, 1, 224, 224, 3), dtype=uint8)
          └─ proprio: ndarray(shape=(1, 1, 4), dtype=float32)
      └─ obs_g: dict (len=2)
          └─ visual: ndarray(shape=(1, 1, 224, 224, 3), dtype=uint8)
          └─ proprio: ndarray(shape=(1, 1, 4), dtype=float32)
      └─ state_0: ndarray(shape=(1, 7), dtype=float32)
      └─ state_g: ndarray(shape=(1, 7), dtype=float32)
      └─ gt_actions: Tensor(shape=(1, 25, 2), dtype=torch.float32)
      └─ goal_H: int
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # Top‐level
    print(f"dict (len={len(data)})")
    for key, value in data.items():
        desc = describe(value)
        print(f"  └─ {key}: {desc}")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                sub_desc = describe(sub_value)
                print(f"          └─ {sub_key}: {sub_desc}")
    print()  # final newline


def main():
    # Hard‐coded parameters:
    N_EVALS = 1
    GOAL_H = 5
    FRAMESKIP = 5
    OUTPUT_PATH = Path(
        "/storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/plan_targets.pkl"
    )

    # 1. Instantiate the Pusht environment wrapper
    env = PushTWrapper(with_velocity=True, with_target=True)

    # 2. Determine dims
    action_dim = env.action_dim                       # e.g. 2 for Pusht
    sample_state = env.sample_random_init_goal_states(seed=0)[0]
    state_dim = sample_state.shape[0]                 # e.g. 7

    # Get proprio_dim by doing one reset
    init_s, _ = env.sample_random_init_goal_states(seed=0)
    env.seed(0)
    env.reset_to_state = init_s
    obs0_once, _ = env.reset()
    proprio_dim = obs0_once["proprio"].shape[0]       # e.g. 4

    # 3. Sample states & observations for the single seed
    state_0_list = []
    state_g_list = []
    obs_0_visual_list  = []
    obs_0_proprio_list = []
    obs_g_visual_list  = []
    obs_g_proprio_list = []

    rng = np.random.RandomState(42)

    # Only one iteration since N_EVALS = 1
    init_state, goal_state = env.sample_random_init_goal_states(seed=0)
    state_0_list.append(init_state.copy())
    state_g_list.append(goal_state.copy())

    env.seed(rng.randint(0, 10_000_000))
    env.reset_to_state = init_state
    obs0, _ = env.reset()

    env.seed(rng.randint(0, 10_000_000))
    env.reset_to_state = goal_state
    obsg, _ = env.reset()

    obs_0_visual_list.append(obs0["visual"].copy())
    obs_0_proprio_list.append(obs0["proprio"].copy())
    obs_g_visual_list.append(obsg["visual"].copy())
    obs_g_proprio_list.append(obsg["proprio"].copy())

    # 4. Stack into batch form (with batch=1)
    # 4a. Visual: (1, 1, H, W, C)
    vis0_np = np.stack(obs_0_visual_list, axis=0)   # (1, 224, 224, 3)
    vis0_np = vis0_np.reshape(N_EVALS, 1, *vis0_np.shape[1:])  
    # Now vis0_np: (1, 1, 224, 224, 3)

    visg_np = np.stack(obs_g_visual_list, axis=0)   # (1, 224, 224, 3)
    visg_np = visg_np.reshape(N_EVALS, 1, *visg_np.shape[1:])
    # Now visg_np: (1, 1, 224, 224, 3)

    # 4b. Proprio: (1, 1, proprio_dim)
    prop0_np = np.stack(obs_0_proprio_list, axis=0)   # (1, proprio_dim)
    prop0_np = prop0_np.reshape(N_EVALS, 1, proprio_dim)

    propg_np = np.stack(obs_g_proprio_list, axis=0)   # (1, proprio_dim)
    propg_np = propg_np.reshape(N_EVALS, 1, proprio_dim)

    obs_0 = {
        "visual":  vis0_np.astype(np.uint8),
        "proprio": prop0_np.astype(np.float32),
    }
    obs_g = {
        "visual":  visg_np.astype(np.uint8),
        "proprio": propg_np.astype(np.float32),
    }

    # 4c. States: (1, state_dim)
    state_0 = np.stack(state_0_list, axis=0).astype(np.float32)  # (1, 7)
    state_g = np.stack(state_g_list, axis=0).astype(np.float32)  # (1, 7)

    # 4d. gt_actions: zeros of shape (1, 25, 2)
    gt_actions = torch.zeros((N_EVALS, GOAL_H * FRAMESKIP, action_dim), dtype=torch.float32)

    # 5. Package into dict & dump to pickle
    data_to_save = {
        "obs_0":      obs_0,
        "obs_g":      obs_g,
        "state_0":    state_0,
        "state_g":    state_g,
        "gt_actions": gt_actions,
        "goal_H":     GOAL_H
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Saved plan_targets.pkl with batch=1 at {OUTPUT_PATH}\n")

    # 6. Reload and print its structure + shapes
    print_pickle_structure(OUTPUT_PATH)


if __name__ == "__main__":
    main()
