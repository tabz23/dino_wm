#!/bin/bash


### bsub -gpu "num=1:gmodel=NVIDIAA100_SXM4_80GB" -R "rusage[mem=40]" -q gpu-compute-debug -Is /bin/bash 
###bsub -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAGeForceGTX1080Ti" -q interactive -Is /bin/bash

##run below using source /path.sh
source activate
conda activate dino_wm
cd /storage1/sibai/Active/ihab/research_new/dino_wm
export DATASET_DIR=/storage1/sibai/Active/ihab/research_new/datasets_dino
export TORCH_HOME=/storage1/sibai/Active/ihab/tmp/torch
# export HOME=/storage1/sibai/Active/ihab/tmp/


# python train.py --config-name "train copy.yaml" env=point_maze frameskip=5 num_hist=3
# accelerate launch train.py --config-name "train copy.yaml" env=pusht frameskip=5 num_hist=3

# accelerate launch train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3
# python train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3
# python train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3

# accelerate launch plan.py --config-name "plan_pusht copy.yaml" model_name=pusht
# accelerate launch plan.py --config-name "plan_dubins.yaml" model_name=dubins

