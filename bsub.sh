#!/bin/bash
#BSUB -o /storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/hj/output_%J.log
#BSUB -e /storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/hj/error_%J.log
#BSUB -R "rusage[mem=40]"
#BSUB -gpu "num=1"
#BSUB -J PythonGPUJob
export PATH="/storage1/sibai/Active/yuxuan/anaconda3/bin:$PATH"
export WANDB_API_KEY=7893bf6676aaa0213e6da2edbc8f4b42fa816084
export MUJOCO_GL=egl
source activate 
conda activate dino
wandb login
cd /storage1/sibai/Active/ihab/research_new/dino_wm
python train_hj_cargoal.py