#!/bin/bash


### bsub -gpu "num=1:gmodel=NVIDIAA100_SXM4_80GB" -R "rusage[mem=40]" -q gpu-compute-debug -Is /bin/bash 
###bsub -gpu "num=1:mode=exclusive_process:gmodel=NVIDIAGeForceGTX1080Ti" -q interactive -Is /bin/bash

##run below using source /path.sh
source activate
conda activate dino_wm
cd /storage1/sibai/Active/ihab/research_new/dino_wm
export DATASET_DIR=/storage1/sibai/Active/ihab/research_new/datasets_dino
export TORCH_HOME=/storage1/sibai/Active/ihab/tmp/torch


# python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3

# python plan.py --config-name "plan_pusht copy.yaml" model_name=pusht