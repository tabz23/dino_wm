from plan import load_model
from omegaconf import OmegaConf
import argparse
import yaml
from pathlib import Path
import torch
from datasets.traj_dset import split_traj_datasets, get_train_val_sliced_with_cost
from datasets.img_transforms import default_transform
from tqdm import tqdm, trange
import wandb
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from copy import deepcopy

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
        default="/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs2",
        help="Where to find the DINO-WM checkpoints"
    )
    parser.add_argument(
        "--config", type=str, default="train_failure_classifier.yaml",
        help="Path to your flat YAML of hyperparameters"
    )
    
    parser.add_argument(
    "--without_proprio", default=False, action="store_true",
    help="Flag to include proprioceptive information in latent encoding"
    )
    
    parser.add_argument(
    "--dino_encoder", type=str, default="dino",
    help="Which encoder to use: dino, r3m, vc1, etc."
    )
    
    parser.add_argument(
    "--task", type=str, default="maniskillnew",
    help="Which task to perform: dubins, other_task, etc."
    )

    parser.add_argument(
    "--finetune", default=False, action='store_true',
    )

    parser.add_argument(
    "--single_layer_classifier", default=False, action='store_true',
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
    def __init__(self, sample, wm, args):
        super(FailureClassifier, self).__init__()
        self.wm = wm
        self.args = args
        if self.args.freeze_wm:
            for p in self.wm.parameters():
                p.requires_grad = False
        ''' As hussein said, we need a single layer classifier to evaluate the performance of the visual representation. Here I didn't use sigmoid type output because I notice you concated the representations of all tokens together to construct the final representation. If we use weighted sum to calculate the final value. The trained weight will be very small, which is not good intuitively.
        
        For multi-layer one, I use the same setting used in the latent safety filter paper
        '''
        hidden_dim = self.encode(sample).shape[-1]
        print(hidden_dim)
        if self.args.single_layer_classifier:
            self.head = torch.nn.Linear(hidden_dim, 2)
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1)
            )

    def forward(self, obs):
        rep = self.encode(obs)
        return self.head(rep)

    def encode(self, obs):
        vis = obs['visual'].unsqueeze(1)
        proprio = obs['proprio'].unsqueeze(1)
        B = vis.shape[0]
        data = {'visual': vis, 'proprio': proprio}
        latent = self.wm.encode_obs(data)
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
        # DEBUG: print your min/max once
        # print(f"[DEBUG] visual raw min={visual.min():.0f}, max={visual.max():.0f}")
        proprio = torch.tensor(proprio, dtype=torch.float32)
        obs = {'visual': visual/255, 'proprio': proprio}
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'obs': obs, 'labels': label}

def load_data(
        task,
        data_path,
        seed = 42,
        transform = default_transform,
        n_rollout = None,
        normalize_action = False,
        normalize_states = True,
        split_ratio = 0.9,
        num_hist = 1,
        num_pred = 0,
        frameskip = 1,
        with_costs = True,
        ):
    if "dubins" in task:
        from datasets.dubins_dset import PointMazeDataset as Dataset
    elif "maniskill" in task:
        from datasets.maniskill_dset import ManiSkillDataset as Dataset
    elif "carla" in task:
        from datasets.carla_dset import CarlaDataset as Dataset
    elif "cargoal" in task:
        from datasets.cargoal_dset import PointMazeDataset as Dataset
    else:
        raise("dataset not supported")
    path = Path(data_path) / task
    dset = Dataset(
        data_path=path,
        transform=default_transform(),
        normalize_action=normalize_action,
        normalize_states=normalize_states,
        n_rollout=n_rollout,
        with_costs=with_costs
    )
    dset_cost = Dataset(
        data_path=path,
        transform=default_transform(),
        normalize_action=normalize_action,
        normalize_states=normalize_states,
        n_rollout=n_rollout,
        with_costs=with_costs,
        only_cost=True,
    )
    train_slices, val_slices, train_slices_cost, val_slices_cost = get_train_val_sliced_with_cost(
        traj_dataset=dset, 
        traj_dataset_cost=dset_cost, 
        train_fraction=split_ratio, 
        num_frames=num_hist + num_pred, 
        random_seed=seed,
        frameskip=frameskip
    )
    return train_slices, val_slices, train_slices_cost, val_slices_cost

def train(fc, train_dataloader, val_dataloader, args, logger):
    optimizer = torch.optim.Adam(fc.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    train_stats = {'loss':[]}
    val_stats = {'val_loss':[], 'val_accuracy':[],'TN':[],'FP':[],'FN':[],'TP':[],'Safe ACC':[],'Unsafe ACC':[]}
    fc.train()
    for _ in trange(args.epochs):
        for idx, batch in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            obs, act, state, cost = batch
            if cost.sum() == 0:
                continue
            for k, v in obs.items():
                obs[k] = v.to(args.device).squeeze(1)
            labels = cost.to(args.device).squeeze(1)
            logits = fc(obs)
            # print(labels.sum(),labels.shape[0]-labels.sum(),labels.sum()/labels.shape[0])
            if args.single_layer_classifier:
                loss = torch.nn.functional.cross_entropy(logits, labels)
            else:
                labels[labels == 1] = -1
                labels[labels == 0] = 1
                labels = labels.float()
                loss = torch.nn.functional.mse_loss(logits,labels)
            train_stats['loss'].append(loss.item())
            loss.backward()
            optimizer.step()
            logger.log({"train_loss":loss.item()})
            # if (idx == 100) or ((idx + 1) % args.eval_freq == 0):
            if ((idx + 1) % args.eval_freq == 0):
                stats = test(fc, val_dataloader, args, logger)
                print(f"Loss: {stats['loss']:.4f}, Accuracy: {stats['accuracy']:.4f}")
                val_stats['val_loss'].append(stats['loss'])
                val_stats['val_accuracy'].append(stats['accuracy'])
                val_stats['TN'].append(stats['TN'])
                val_stats['TP'].append(stats['TP'])
                val_stats['FP'].append(stats['FP'])
                val_stats['FN'].append(stats['FN'])
                val_stats['Unsafe ACC'].append(stats['Unsafe ACC'])
                val_stats['Safe ACC'].append(stats['Safe ACC'])
                logger.log(stats)
    return train_stats, val_stats

def test(fc, test_dataloader, args, logger):
    eval_stats = {}
    fc.eval()
    total_preds = torch.tensor([], dtype=torch.long, device=args.device)
    total_logits = torch.tensor([], dtype=torch.long, device=args.device)
    total_labels = torch.tensor([], dtype=torch.long, device=args.device)
    with torch.no_grad():
        for batch in test_dataloader:
            obs, act, state, cost = batch
            for k, v in obs.items():
                obs[k] = v.to(args.device).squeeze(1)
            labels = cost.to(args.device).squeeze(1)
            logits = fc(obs)
            if args.single_layer_classifier:
                preds = logits.argmax(dim=-1)
            else:
                preds = torch.zeros_like(logits)
                preds[logits <= 0] = 1
                preds[logits > 0] = 0
            total_preds = torch.cat((total_preds, preds), dim=0)
            total_logits = torch.cat((total_logits, logits), dim=0)
            total_labels = torch.cat((total_labels, labels), dim=0)
        
        if args.single_layer_classifier:
            loss = torch.nn.functional.cross_entropy(total_logits, total_labels)/len(test_dataloader)
        else:
            targets = deepcopy(total_labels)
            targets[targets == 0] = 1
            targets[targets == 1] = -1
            loss = torch.nn.functional.mse_loss(total_logits,targets.float())
        eval_stats['loss'] = loss.item()
        acc = (total_preds == total_labels).float().mean().item()
        total_preds = total_preds.detach().cpu().numpy()
        total_labels = total_labels.detach().cpu().numpy()
        cm = confusion_matrix(total_labels, total_preds)
        tn, fp, fn, tp = cm.ravel()
        eval_stats['TN'] = tn
        eval_stats['FP'] = fp
        eval_stats['FN'] = fn
        eval_stats['TP'] = tp
        eval_stats['Unsafe ACC'] = tp / (tp + fn)
        eval_stats['Safe ACC'] = tn / (tn + fp)
        eval_stats['accuracy'] = acc
    return eval_stats

def main():
    print("started")
    wandb.login()
    args = get_args_and_merge_config()
    if args.without_proprio:
        args.with_proprio = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_data, val_data, train_data_cost, _ = load_data(args.task,args.data_path,args.seed)
    label = torch.zeros(len(train_data))

    for idx, (_, _, _, cost) in enumerate(tqdm(train_data_cost)):
    # for idx, (_, _, _, cost) in enumerate(tqdm(train_data)):
        label[idx] = cost[0]
    sum_unsafe = label.sum()
    weight_safe = 0.5/(len(train_data)-sum_unsafe)
    weight_unsafe = 0.5/sum_unsafe

    weights = torch.zeros(len(train_data))
    weights[label == 0] = weight_safe
    weights[label == 1] = weight_unsafe
    sampler_train = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=len(train_data), replacement=True)
    train_dataloader = torch.utils.data.DataLoader(
        # train_data, batch_size=args.batch_size, shuffle=True
        train_data, batch_size=args.batch_size, sampler = sampler_train
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False
    )
    _, _, _, cost = next(iter(train_dataloader))
    print(cost.sum()/cost.shape[0])


    backbones = ["r3m","vc1","resnet","dino","dino_cls","scratch","full_scratch"]
    # backbones = ["full_scratch"]
    if args.finetune:
        backbones = backbones[:-1]
    save_path = Path(args.save_path)
    for backbone in backbones:
        ckpt_dir = Path(args.dino_ckpt_dir)
        if "maniskill" in args.task:
            ckpt_dir = Path(f"/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/maniskill/{backbone}")
            if backbone == "full_scratch":
                ckpt_dir = Path(f"/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/maniskill/vc1")
        elif "carla" in args.task:
            ckpt_dir = Path(f"/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/carla/{backbone}")
            if backbone == "full_scratch":
                ckpt_dir = Path(f"/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/carla/vc1")
        elif "dubins" in args.task:
            ckpt_dir = Path(f"/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/dubins/{backbone}")
            if backbone == "full_scratch":
                ckpt_dir = Path(f"/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/dubins/vc1")
        elif "cargoal" in args.task:
            ckpt_dir = Path(f"/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/cargoal/{backbone}")
            if backbone == "full_scratch":
                ckpt_dir = Path(f"/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/cargoal/vc1")
        else:
            ckpt_dir = ckpt_dir / f"{args.task}"
        hydra_cfg = ckpt_dir / 'hydra.yaml'
        snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'

        if args.finetune:
            backbone = f"{backbone}_ft"
        training_type = "linear-probing"
        if not args.single_layer_classifier:
            training_type = "MLP"
        config = vars(args)
        config["backbone"] = backbone
        config["ckpt_dir"] = ckpt_dir
        if args.with_proprio:
            flag = "with-proprio"
        else:
            flag = "without-proprio"
        print(config)
        if not os.path.exists(save_path / training_type / args.task  / backbone / flag):
            os.makedirs(save_path / training_type / args.task  / backbone / flag)
        logger = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="i-k-tabbara-washington-university-in-st-louis",
            # Set the wandb project where this run will be logged.
            project=f"{training_type}-{flag}-{args.task}",
            dir= save_path / training_type / args.task  / backbone / flag ,
            # Track hyperparameters and run metadata.
            config=config,
            name=f"{backbone}"
        )
        # load train config and model weights
        train_cfg = OmegaConf.load(str(hydra_cfg))
        num_action_repeat = train_cfg.num_action_repeat
        wm = load_model(snapshot, train_cfg, num_action_repeat, device=args.device)
        wm.eval()
        if backbone == "full_scratch":
            for m in wm.modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    torch.nn.init.xavier_uniform_(m.weight)
            for p in wm.parameters():
                p.requires_grad = True
        if args.finetune or (backbone == "full_scratch"):
            args.freeze_wm = False
        else:
            args.freeze_wm = True

        obs, _, _, _ = next(iter(val_dataloader))
        for k, v in obs.items():
            obs[k] = v.to(args.device).squeeze(1)
        fc = FailureClassifier(obs, wm, args).to(args.device)
        train_stats, val_stats = train(fc, train_dataloader, val_dataloader, args, logger)
        # eval_stats = test(fc, val_dataloader, args, logger)


        # ckpts and stats will be saved to the original ckpt_dir
        torch.save(
            {
                'model_state_dict': fc.state_dict(),
            },
            save_path / training_type / args.task  / backbone / flag / 'failure_classifier.pth'
        )

        torch.save({
            'train_stats': train_stats,
            'val_stats': val_stats,
            # 'eval_stats': eval_stats
        }, save_path / training_type / args.task  / backbone / flag / 'stats.pth')

        wandb.finish()

if __name__ == "__main__":
    main()

    # python train_failure_classifier.py --task maniskillnew
    # bsub -q gpu-compute < bsub.sh
    # bsub -gpu "num=1" -R "rusage[mem=40]" -q gpu-compute-debug -Is /bin/bash 
    # bsub -Is /bin/bash 
    # bsub -G compute-sibai < script_yuxuan.sh
    # bsub -n 12 -q general-interactive -Is -G compute-sibai -R 'rusage[mem=102GB]' -M 100GB -R 'gpuhost' -gpu "num=1:gmem=30G" -a 'docker(continuumio/anaconda3:2021.11)' -env "LSF_DOCKER_VOLUMES=/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active,LSF_DOCKER_SHM_SIZE=32g" /bin/bash


    # train_dataloader = torch.utils.data.DataLoader(
    #     # train_data, batch_size=args.batch_size, shuffle=True
    #     train_data, batch_size=64, sampler = sampler_train
    # )
    # _,_,_,cost=next(iter(train_dataloader))

    # cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm