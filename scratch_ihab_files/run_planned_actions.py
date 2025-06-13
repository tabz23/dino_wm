import pickle
import numpy as np

import cv2

import imageio
from einops import rearrange
import sys
import os
sys.path.append("/storage1/sibai/Active/ihab/research_new/dino_wm")
from env.pusht.pusht_wrapper import PushTWrapper

# --- Preprocessor utility (must use same as in planner) ---
class DummyPreprocessor:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def denormalize_actions(self, x):
        # x: (..., action_dim)
        return x * self.std + self.mean

def load_pickles(actions_path, targets_path):
    with open(actions_path, "rb") as f:
        actions = pickle.load(f)
    with open(targets_path, "rb") as f:
        targets = pickle.load(f)
    return actions, targets

def save_video(frames, filename, fps=10):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved video to {filename}")

def replay_with_step(env, init_state, primitive_actions, video_path):
    obs, state = env.prepare(seed=99, init_state=init_state)
    frames = [env.render("rgb_array")]
    for a in primitive_actions:
        obs, reward, done, info = env.step(a)
        frames.append(env.render("rgb_array"))
        if done:
            break
    save_video(frames, video_path)

def replay_with_step_multiple(env, init_state, primitive_actions, frameskip, video_path):
    obs, state = env.prepare(seed=99, init_state=init_state)
    frames = [env.render("rgb_array")]
    num_steps = len(primitive_actions) // frameskip
    for i in range(num_steps):
        prim_chunk = primitive_actions[i * frameskip : (i + 1) * frameskip]
        obses, rewards, dones, infos = env.step_multiple(prim_chunk)
        for step in range(frameskip):
            frames.append(env.render("rgb_array"))
            if dones[step]:
                break
    save_video(frames, video_path)

def replay_with_rollout(env, init_state, primitive_actions, video_path):
    # Rollout returns all obs, but we'll re-execute to get frames
    obs, state = env.prepare(seed=99, init_state=init_state)
    frames = [env.render("rgb_array")]
    for a in primitive_actions:
        obs, reward, done, info = env.step(a)
        frames.append(env.render("rgb_array"))
        if done:
            break
    save_video(frames, video_path)

if __name__ == "__main__":
    # ==== SETUP ====
    actions_path = "/storage1/sibai/Active/ihab/research_new/dino_wm/plan_outputs/20250604183501_pusht_gH5/planned_actions.pkl"
    targets_path = "/storage1/sibai/Active/ihab/research_new/dino_wm/plan_outputs/20250604183501_pusht_gH5/plan_targets.pkl"##check plan.py i changed it to return this
    actions, targets = load_pickles(actions_path, targets_path)
    actions = actions[0]  # remove batch dim: (planner_horizon, meta_action_dim)
    init_state = targets['state_0'][0]

    # ---- Get correct env/action sizes ----
    env = PushTWrapper()
    action_dim = env.action_dim  # e.g., 2 for PushT
    planner_horizon, meta_action_dim = actions.shape
    frameskip = meta_action_dim // action_dim

    action_mean = [-0.0087, 0.0068]##from the dataset python file
    action_std = [0.2019, 0.2002]##from the dataset python file


    preprocessor = DummyPreprocessor(action_mean, action_std)

    # ==== Rearrange and denormalize actions ====
    # actions: (planner_horizon, meta_action_dim)
    primitive_actions = rearrange(
        actions, "t (f d) -> (t f) d", f=frameskip, d=action_dim
    )
    primitive_actions = preprocessor.denormalize_actions(primitive_actions)

    # ========== Replays ==========
    print("Replaying with step()...")
    replay_with_step(
        env=PushTWrapper(),  # new instance to avoid state carryover
        init_state=init_state,
        primitive_actions=primitive_actions,
        video_path="scratch_ihab_files/videos/pusht_replay_step.mp4"
    )

    print("Replaying with step_multiple()...")
    replay_with_step_multiple(
        env=PushTWrapper(),
        init_state=init_state,
        primitive_actions=primitive_actions,
        frameskip=frameskip,
        video_path="scratch_ihab_files/videos/pusht_replay_step_multiple().mp4"
    )

    print("Replaying with rollout() (actually just stepping)...")
    replay_with_rollout(
        env=PushTWrapper(),
        init_state=init_state,
        primitive_actions=primitive_actions,
        video_path="scratch_ihab_files/videos/pusht_replay_rollout().mp4"
    )
