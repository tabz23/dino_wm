
# -------------------------------------------------------------------------
import os, cv2, torch, argparse, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from matplotlib.colors import LinearSegmentedColormap

# -------------------------------------------------------------------------
# 0.  Project‑specific imports (adjust if you moved code around)
# -------------------------------------------------------------------------
from plan import load_model
from datasets.img_transforms import default_transform     # resize+crop+normalise
from datasets.traj_dset import (
    TrajDataset, TrajSlicerDataset,               # base helpers
    get_train_val_sliced_with_cost, get_train_val_sliced,
    split_traj_datasets
)
from einops import rearrange
from typing import Optional, Callable
# -------------------------------------------------------------------------
# 1.  Utility helpers
# -------------------------------------------------------------------------

def unnormalize(img: torch.Tensor) -> torch.Tensor:
    """Undo `Normalize([0.5],[0.5])`:  [-1,1] ➝ [0,1] (CHW)."""
    return img * 0.5 + 0.5


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Assumes img∈[0,1], returns HWC uint8."""
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

# -------------------------------------------------------------------------
# 2.  Occlusion helpers
# -------------------------------------------------------------------------

def generate_occlusion_mask(image_shape, patch_size, stride, i, j):
    y0, x0 = i * stride, j * stride
    y1, x1 = min(y0 + patch_size, image_shape[0]), min(x0 + patch_size, image_shape[1])
    mask = np.ones(image_shape, dtype=np.float32)
    mask[y0:y1, x0:x1, :] = 0
    return mask, (y0, y1, x0, x1)


def apply_occlusion(image, patch_size, stride, i, j,
                    method="black", blur_sigma=5):
    mask, (y0, y1, x0, x1) = generate_occlusion_mask(
        image.shape, patch_size, stride, i, j
    )

    if method == "black":
        return image * mask

    if method == "blur":
        out = image.copy()
        patch = to_uint8(out[y0:y1, x0:x1, :])
        blurred = cv2.GaussianBlur(patch, (0, 0), blur_sigma)
        out[y0:y1, x0:x1, :] = blurred.astype(np.float32) / 255.0
        return out

    raise ValueError(f"Unknown occlusion method {method!r}")

# -------------------------------------------------------------------------
# 3.  Visualisation helpers
# -------------------------------------------------------------------------

def visualize_saliency(image, saliency_map,
                       predicted_class, original_prob,
                       save_path):
    """Save original, heat‑map, and overlay panels to `save_path`."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # -------- Original --------
    img_vis = to_uint8(unnormalize(image).permute(1, 2, 0).cpu().numpy())
    axes[0].imshow(img_vis, interpolation="bilinear", resample=True)
    axes[0].set_title(
        f"Original\nPredicted: {'Unsafe' if predicted_class else 'Safe'}"
        f"\nConfidence: {original_prob:.3f}"
    )
    axes[0].axis("off")

    # -------- Heat‑map --------
    im1 = axes[1].imshow(
        saliency_map, cmap="hot", interpolation="bilinear", resample=True
    )
    axes[1].set_title("Saliency (red = important)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # -------- Overlay --------
    sal_norm = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min() + 1e-8
    )
    cmap = LinearSegmentedColormap.from_list(
        "custom", [(0, 0, 1, 0.0), (1, 0, 0, 0.7)]
    )
    axes[2].imshow(img_vis, interpolation="bilinear", resample=True)
    axes[2].imshow(
        sal_norm, cmap=cmap, alpha=0.5, interpolation="bilinear", resample=True
    )
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_occlusion_grid(image, patch_size=16, stride=8,
                          output_path=None, method="black", blur_sigma=5):
    """3×3 montage showing exemplar occlusions."""
    img_np = unnormalize(image).permute(1, 2, 0).cpu().numpy()
    H, W = img_np.shape[:2]
    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1

    positions = [
        (0, 0, "Top‑left"),      (0, n_cols // 2, "Top‑centre"),    (0, n_cols - 1, "Top‑right"),
        (n_rows // 2, 0, "Mid‑left"), (n_rows // 2, n_cols // 2, "Centre"), (n_rows // 2, n_cols - 1, "Mid‑right"),
        (n_rows - 1, 0, "Bottom‑left"), (n_rows - 1, n_cols // 2, "Bottom‑centre"), (n_rows - 1, n_cols - 1, "Bottom‑right")
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    for k, (i, j, label) in enumerate(positions):
        occ = apply_occlusion(
            img_np, patch_size, stride, i, j,
            method=method, blur_sigma=blur_sigma
        )
        axes[k].imshow(to_uint8(occ), interpolation="bilinear", resample=True)
        axes[k].add_patch(
            plt.Rectangle((j * stride, i * stride), patch_size, patch_size,
                          linewidth=2, edgecolor="red", facecolor="none")
        )
        axes[k].set_title(label, fontsize=10)
        axes[k].axis("off")

    plt.suptitle(f"Occlusion examples ({method}, {patch_size}px)", fontsize=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# -------------------------------------------------------------------------
# 4.  Occlusion‑based saliency computation
# -------------------------------------------------------------------------

def compute_saliency_map(model, image, proprio,
                         patch_size=16, stride=8,
                         device="cuda",
                         save_examples=False, sample_idx=0,
                         out_dir=None,
                         method="black", blur_sigma=5):
    """Returns (saliency_map [H×W], predicted_class, original_prob)."""
    model.eval()

    with torch.no_grad():
        obs = {
            "visual":  image.unsqueeze(0).unsqueeze(0).to(device),
            "proprio": proprio.unsqueeze(0).unsqueeze(0).to(device)
        }
        logits = model(obs)
        probs  = F.softmax(logits, dim=-1)
        pred_cls  = logits.argmax(dim=-1).item()
        base_conf = probs[0, pred_cls].item()

    H, W = image.shape[1:]
    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1
    scores = np.zeros((n_rows, n_cols), dtype=np.float32)

    img_np = unnormalize(image).permute(1, 2, 0).cpu().numpy()

    sample_pos = []
    if save_examples:
        sample_pos = [
            (0, 0, "tl"), (0, n_cols - 1, "tr"),
            (n_rows // 2, n_cols // 2, "ctr"),
            (n_rows - 1, 0, "bl"), (n_rows - 1, n_cols - 1, "br")
        ]

    for i in tqdm(range(n_rows), desc="Computing saliency"):
        for j in range(n_cols):
            occ_img = apply_occlusion(
                img_np, patch_size, stride, i, j,
                method=method, blur_sigma=blur_sigma
            )
            occ_tensor = torch.from_numpy(occ_img).permute(2, 0, 1).float()
            with torch.no_grad():
                obs_occ = {
                    "visual":  occ_tensor.unsqueeze(0).unsqueeze(0).to(device),
                    "proprio": proprio.unsqueeze(0).unsqueeze(0).to(device)
                }
                occ_logits = model(obs_occ)
                occ_prob = F.softmax(occ_logits, dim=-1)[0, pred_cls].item()
            scores[i, j] = base_conf - occ_prob

            # optional snapshots
            if save_examples and out_dir and any(p[:2] == (i, j) for p in sample_pos):
                tag = [p[2] for p in sample_pos if p[0] == i and p[1] == j][0]
                plt.figure(figsize=(4, 4))
                plt.imshow(to_uint8(occ_img), interpolation="bilinear", resample=True)
                plt.axis("off")
                plt.title(f"{tag} drop={scores[i, j]:.3f}")
                plt.savefig(Path(out_dir) / f"occ_{sample_idx:03d}_{tag}_{method}.png",
                            dpi=150, bbox_inches="tight")
                plt.close()

    saliency = cv2.resize(scores, (W, H), interpolation=cv2.INTER_LINEAR)
    return saliency, pred_cls, base_conf

# -------------------------------------------------------------------------
# 5.  Failure‑classifier definition
# -------------------------------------------------------------------------

class FailureClassifier(torch.nn.Module):
    def __init__(self, sample, wm, args):
        super().__init__()
        self.wm   = wm
        self.args = args
        if self.args.freeze_wm:
            for p in self.wm.parameters():
                p.requires_grad_(False)

        hid_dim = self.encode(sample).shape[-1]
        if self.args.single_layer_classifier:
            self.head = torch.nn.Linear(hid_dim, 2)
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, 512), torch.nn.ReLU(),
                torch.nn.Linear(512, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, 2)
            )

    # ----- Encode one timestep (B×C×H×W  +  B×D) -----
    def encode(self, obs):
        vis, prop = obs["visual"], obs["proprio"]
        if vis.dim()  == 4: vis  = vis.unsqueeze(1)   # add T
        if prop.dim() == 2: prop = prop.unsqueeze(1)
        lat = self.wm.encode_obs({"visual": vis, "proprio": prop})
        if self.args.with_proprio:
            return torch.cat(
                [lat["visual"].reshape(lat["visual"].size(0), -1),
                 lat["proprio"].reshape(lat["proprio"].size(0), -1)],
                dim=-1
            )
        return lat["visual"].reshape(lat["visual"].size(0), -1)

    def forward(self, obs):
        return self.head(self.encode(obs))

def load_data(task: str, root: str, seed=42, n_rollout=None, split_ratio=0.9):
    """
    Returns (train_slices, val_slices, train_slices_cost, val_slices_cost)
    where slices are `TrajSlicerDataset` objects created by
    `get_train_val_sliced_with_cost`.
    """
    if "dubins" in task:
        from datasets.dubins_dset import PointMazeDataset as Dataset
    elif "maniskill" in task:
        from datasets.maniskill_dset import ManiSkillDataset as Dataset
    elif "carla" in task:
        from datasets.carla_dset import CarlaDataset as Dataset
    elif "cargoal" in task:
        from datasets.cargoal_dset import PointMazeDataset as Dataset
    else:
        raise ValueError("dataset not supported")

    dset_path = Path(root) / task
    base_dset = Dataset(
        str(dset_path),
        transform=default_transform(),
        normalize_action=True,
        normalize_states=True,
        n_rollout=n_rollout,
        with_costs=True
    )
    cost_dset = Dataset(
        str(dset_path),
        transform=default_transform(),
        normalize_action=True,
        normalize_states=True,
        n_rollout=n_rollout,
        with_costs=True,
        only_cost=True
    )

    train_s, val_s, train_c, val_c = get_train_val_sliced_with_cost(
        traj_dataset=base_dset,
        traj_dataset_cost=cost_dset,
        train_fraction=split_ratio,
        num_frames=1,
        random_seed=seed,
        frameskip=1
    )
    return train_s, val_s, train_c, val_c

# -------------------------------------------------------------------------
# 7.  Main
# -------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Occlusion‑based saliency")
    p.add_argument("--task",         default="maniskill3000classif")
    p.add_argument("--data_path",    required=True)
    p.add_argument("--backbone",     default="r3m")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--with_proprio", action="store_true")
    p.add_argument("--num_samples",  type=int, default=10)
    p.add_argument("--patch_size",   type=int, default=16)
    p.add_argument("--stride",       type=int, default=8)
    p.add_argument("--seed",         type=int, default=1)
    p.add_argument("--occlusion_method", choices=["black", "blur"], default="black")
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--blur_sigma",   type=float, default=10.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---------- world‑model ----------
    ckpt_dir = Path(f"/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs/dubins/fully_trained_prop_repeated_3_times/{args.backbone}")
    hydra_cfg = ckpt_dir / 'hydra.yaml'
    wm_snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
    

    train_cfg = OmegaConf.load(str(hydra_cfg))
    num_action_repeat = train_cfg.num_action_repeat
    
    wm = load_model(wm_snapshot, train_cfg, num_action_repeat, device=args.device)
    wm.eval()
    # ---------- classifier ----------
    cls_sub = "classifier-with-proprio" if args.with_proprio else "classifier-without-proprio"
    cls_path = ckpt_dir / cls_sub / f"dubins1800_withcost_{args.backbone}" / "failure_classifier.pth"
    # build dummy obs to size FC layer
    _, val_slices, _, _ = load_data(args.task, args.data_path, args.seed)
    val_loader = torch.utils.data.DataLoader(val_slices, batch_size=1, shuffle=True)
    dummy_obs, _, _, _ = next(iter(val_loader))
    for k in dummy_obs: dummy_obs[k] = dummy_obs[k].to(args.device)

    class Namespace:  # tiny holder
        freeze_wm = True
        with_proprio = args.with_proprio
        single_layer_classifier = True
    fc = FailureClassifier(dummy_obs, wm, Namespace()).to(args.device)
    fc.load_state_dict(torch.load(cls_path, map_location=args.device)["model_state_dict"])
    fc.eval()

    # ---------- gather samples ----------
    print(f"[INFO] Generating saliency maps for {args.num_samples} frames …")
    safe, unsafe = [], []
    for obs, _, _, cost in val_loader:
        obs_gpu = {k: v.to(args.device) for k, v in obs.items()}
        pred = fc(obs_gpu).argmax(dim=-1).item()
        true_lbl = cost.item()
        if pred == 0 and len(safe) < args.num_samples // 2:
            safe.append((obs, true_lbl))
        elif pred == 1 and len(unsafe) < args.num_samples // 2:
            unsafe.append((obs, true_lbl))
        if len(safe) + len(unsafe) >= args.num_samples: break

    samples = safe + unsafe
    for idx, (obs, true_lbl) in enumerate(samples):
        img  = obs["visual"].squeeze(0).squeeze(0)
        prop = obs["proprio"].squeeze(0).squeeze(0)

        sal, pred_cls, conf = compute_saliency_map(
            fc, img, prop,
            patch_size=args.patch_size, stride=args.stride,
            device=args.device,
            save_examples=(idx < 5), sample_idx=idx,
            out_dir=args.output_dir,
            method=args.occlusion_method, blur_sigma=args.blur_sigma
        )

        out_png = Path(args.output_dir) / (
            f"sal_{idx:03d}_pred{'unsafe' if pred_cls else 'safe'}_"
            f"true{'unsafe' if true_lbl else 'safe'}.png"
        )
        visualize_saliency(img, sal, pred_cls, conf, out_png)

        if idx < 5:
            grid_png = Path(args.output_dir) / (
                f"grid_{idx:03d}_{args.occlusion_method}.png"
            )
            create_occlusion_grid(
                img, patch_size=args.patch_size, stride=args.stride,
                output_path=grid_png,
                method=args.occlusion_method, blur_sigma=args.blur_sigma
            )
        print(f"[{idx+1}/{len(samples)}] saved {out_png.name}")

    print(f"\n✅  All outputs saved under: {args.output_dir}")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()


# python saliency.py  --task maniskill3000classif --data_path "/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino" --backbone r3m  --occlusion_method blur  --blur_sigma 11   --num_samples 20 --output_dir /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/saliency --seed 3

# python saliency.py  --task dubins1800_withcost --data_path "/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino" --backbone r3m  --occlusion_method blur  --blur_sigma 11   --num_samples 20 --output_dir /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/saliency --seed 3
# python saliency.py  --task dubins1800_withcost --data_path "/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino" --backbone dino  --occlusion_method black  --blur_sigma 11   --num_samples 20 --output_dir /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/saliency --seed 3 --patch_size 20 --stride 10

#fix     ckpt_dir = Path(f"/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs1/dubins/{args.backbone}")
#fix     cls_path = ckpt_dir / cls_sub / f"maniskillnew_{args.backbone}" / "failure_classifier.pth"