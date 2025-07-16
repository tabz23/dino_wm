from plan import load_model
from omegaconf import OmegaConf
import argparse
import yaml
from pathlib import Path
import torch

def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

def get_args_and_merge_config():
    # 1) Top‐level parser for the flags you always need
    parser = argparse.ArgumentParser("Failure Classifier Training")
    parser.add_argument(
        "--dino_ckpt_dir", type=str,
        default="/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs",
        help="Where to find the DINO-WM checkpoints"
    )
    parser.add_argument(
        "--config", type=str, default="train_failure_classifier.yaml",
        help="Path to your flat YAML of hyperparameters"
    )
    
    parser.add_argument(
    "--with_proprio", action="store_true",
    help="Flag to include proprioceptive information in latent encoding"
    )
    
    parser.add_argument(
    "--dino_encoder", type=str, default="dino",
    help="Which encoder to use: dino, r3m, vc1, etc."
    )
    
    parser.add_argument(
    "--task", type=str, default="dubins", choices=["dubins", "cargoal", "maniskill", "carla", "pusht"],
    help="Which task to perform: dubins, other_task, etc."
    )
    
    args, remaining = parser.parse_known_args()

    # 2) Load all keys & values from the YAML (no `defaults:` wrapper needed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)  # e.g. {'batch_size':16, 'critic-lr':1e-3, ...}

    # 3) Dynamically build a second parser so each key adopts the right type
    cfg_parser = argparse.ArgumentParser()
    for key, val in sorted(cfg.items()):
        arg_t = args_type(val)
        cfg_parser.add_argument(f"--{key}", type=arg_t, default=arg_t(val))
    cfg_args = cfg_parser.parse_args(remaining)

    # 4) Merge everything back into the top‐level args namespace
    for key, val in vars(cfg_args).items():
        setattr(args, key.replace("-", "_"), val)

    return args

class FailureClassifier(torch.nn.Module):
    def __init__(self, wm, args):
        super(FailureClassifier, self).__init__()
        self.wm = wm
        self.args = args
        if self.args.freeze_wm:
            for p in self.wm.parameters():
                p.requires_grad = False
        ''' As hussein said, we need a single layer classifier to evaluate the performance of the visual representation. Here I didn't use sigmoid type output because I notice you concated the representations of all tokens together to construct the final representation. If we use weighted sum to calculate the final value. The trained weight will be very small, which is not good intuitively.
        
        For multi-layer one, I use the same setting used in the latent safety filter paper
        '''
        if self.args.single_layer_classifier:
            self.head = torch.nn.Linear(self.args.latent_dim, 2)
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(self.args.latent_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 2)
            )

    def forward(self, obs):
        rep = self.encode(obs)
        return self.head(rep)

    def encode(self, obs):
        vis = obs['visual'].unsqueeze(1)
        proprio = obs['proprio'].unsqueeze(1)
        B = vis.shape[0]
        data = {'visual': vis, 'proprio': proprio}
        latent = self.wm.encode(data)
        if self.args.with_proprio:
            lat_vis = latent['visual'].reshape(B, -1)
            lat_prop = latent['proprio'].reshape(B, -1)
            rep = torch.cat([lat_vis, lat_prop], dim=-1)
        else:
            rep = latent['visual'].reshape(B, -1)
        return rep

class FailureClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, mode = "train"):
        path = data_path / f"{mode}"
        self.data_path = Path(path)
        # need to store data in three dirs: train, val, test
        # then three pth files store visual, proprio, labels
        # How do we collect the safe trajectories in each task?
        self.visual = torch.load(self.data_path / "visual.pth") 
        self.proprio = torch.load(self.data_path / "proprio.pth")
        self.labels = torch.load(self.data_path / "labels.pth")

    def __len__(self):
        return len(self.visual)

    def __getitem__(self, idx):
        visual = self.visual[idx]
        proprio = self.proprio[idx]
        visual = torch.tensor(visual, dtype=torch.float32)
        proprio = torch.tensor(proprio, dtype=torch.float32)
        obs = {'visual': visual/255, 'proprio': proprio}
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'obs': obs, 'labels': label}

def train(fc, train_dataloader, val_dataloader, args):
    optimizer = torch.optim.Adam(fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_stats = {'loss':[]}
    val_stats = {'loss':[], 'accuracy':[]}
    for epoch in range(args.epochs):
        fc.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            obs = batch['obs'].to(args.device)
            labels = batch['labels'].to(args.device)
            logits = fc(obs)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            train_stats['loss'].append(loss.item())
            loss.backward()
            optimizer.step()
        if (epoch + 1) % args.eval_freq == 0:
            stats = test(fc, val_dataloader, args)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {stats['loss']:.4f}, Accuracy: {stats['accuracy']:.4f}")
            val_stats['loss'].append(stats['loss'])
            val_stats['accuracy'].append(stats['accuracy'])
    return train_stats, val_stats

def test(fc, test_dataloader, args):
    eval_stats = {}
    fc.eval()
    total_preds = torch.tensor([], dtype=torch.long, device=args.device)
    total_labels = torch.tensor([], dtype=torch.long, device=args.device)
    with torch.no_grad():
        for batch in test_dataloader:
            obs = batch['obs'].to(args.device)
            labels = batch['labels'].to(args.device)
            logits = fc(obs)
            preds = logits.argmax(dim=-1)
            total_preds = torch.cat((total_preds, preds), dim=0)
            total_labels = torch.cat((total_labels, labels), dim=0)
        loss = torch.nn.functional.cross_entropy(total_preds, total_labels)/len(test_dataloader)
        eval_stats['loss'] = loss.item()
        acc = (total_preds == total_labels).float().mean().item()
        eval_stats['accuracy'] = acc
    return eval_stats

def main():
    args = get_args_and_merge_config()
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir = ckpt_dir / f"{args.task}"
    hydra_cfg = ckpt_dir / 'hydra.yaml'
    snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
    # load train config and model weights
    train_cfg = OmegaConf.load(str(hydra_cfg))
    num_action_repeat = train_cfg.num_action_repeat
    wm = load_model(snapshot, train_cfg, num_action_repeat, device=args.device)
    wm.eval()
    fc = FailureClassifier(wm, args).to(args.device)
    train_data = FailureClassifierDataset(args.data_path, mode="train") 
    val_data = FailureClassifierDataset(args.data_path, mode="val")
    test_data = FailureClassifierDataset(args.data_path, mode="test")
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    train_stats, val_stats = train(fc, train_dataloader, val_dataloader, args)
    eval_stats = test(fc, test_dataloader, args)


    # ckpts and stats will be saved to the original ckpt_dir
    torch.save(
        {
            'model_state_dict': fc.state_dict(),
        },
        ckpt_dir / 'classifier' / 'failure_classifier.pth'
    )

    torch.save({
        'train_stats': train_stats,
        'val_stats': val_stats,
        'eval_stats': eval_stats
    }, ckpt_dir / 'classifier' / 'stats.pth')
