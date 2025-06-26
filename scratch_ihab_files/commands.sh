#!/bin/bash


### bsub -n 16 -gpu "num=1:gmodel=NVIDIAA100_SXM4_80GB" -R "rusage[mem=100]" -q gpu-compute-debug -Is /bin/bash 
###bsub -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAGeForceGTX1080Ti" -q interactive -Is /bin/bash

##run below using source /path.sh
source activate
conda activate dino_wm
cd /storage1/sibai/Active/ihab/research_new/dino_wm
export DATASET_DIR=/storage1/sibai/Active/ihab/research_new/datasets_dino
export TORCH_HOME=/storage1/sibai/Active/ihab/tmp/torch
# export HOME=/storage1/sibai/Active/ihab/tmp/

python ppo.py --env_id="UnitreeG1PlaceAppleInBowl-v1"   --num_envs=512 --update_epochs=8 --num_minibatches=32   --total_timesteps=100_000_000 --num-steps=100 --num-eval-steps=100 --checkpoint="/storage1/sibai/Active/ihab/research_new/ManiSkill/examples/baselines/ppo/runs/UnitreeG1PlaceAppleInBowl-v1__ppo__8__1750699220/ckpt_6326.pt"

# python train.py --config-name "train.yaml" env=dubins frameskip=5 num_hist=3
# accelerate launch train.py --config-name "train copy.yaml" env=pusht frameskip=5 num_hist=3

# accelerate launch train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3
# python train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3
# python train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3

# accelerate launch plan.py --config-name "plan_pusht copy.yaml" model_name=pusht
# accelerate launch plan.py --config-name "plan_dubins.yaml" model_name=dubins

