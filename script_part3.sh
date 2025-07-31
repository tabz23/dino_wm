#!/bin/bash
#BSUB -n 12
#BSUB -q general
#BSUB -R 'rusage[mem=102GB]'
#BSUB -M 100GB
#BSUB -R 'gpuhost'
#BSUB -gpu "num=1:gmem=30G"
#BSUB -a 'docker(continuumio/anaconda3:2021.11)'
#BSUB -W 600:00
#BSUB -J dino_wm_job
#BSUB -oo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/output%J.log
#BSUB -eo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/error%J.log
#BSUB -N
#BSUB -u y.yuxuan@wustl.edu
#BSUB -env "LSF_DOCKER_VOLUMES=/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active,LSF_DOCKER_SHM_SIZE=32g"

# bsub -n 12 -q general-interactive -Is -G compute-sibai -R 'rusage[mem=102GB]' -M 100GB -R 'gpuhost' -gpu "num=1:gmem=30G" -a 'docker(continuumio/anaconda3:2021.11)' -env "LSF_DOCKER_VOLUMES=/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active,LSF_DOCKER_SHM_SIZE=32g" /bin/bash
# xxxxx   xxx BSUB -q general-interactive
# xxxxx xx  BSUB -W 600:00

# # Environment variables for your run
export CONDA_ENVS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/envs"
export CONDA_PKGS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/pkgs"
export PATH="/opt/conda/bin:$PATH"
export TORCH_HOME=/storage1/fs1/sibai/Active/ihab/tmp/torch
export WANDB_API_KEY=7893bf6676aaa0213e6da2edbc8f4b42fa816084
source /opt/conda/etc/profile.d/conda.sh
conda activate dino_wm_ris
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm
wandb login

# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task maniskillnew --single_layer_classifier
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task dubins1800_withcost --single_layer_classifier
#  python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task cargoalnewshort --single_layer_classifier
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task carla_distance_cost --single_layer_classifier

# without_proprio
python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task dubins1800_withcost --without_proprio --single_layer_classifier
python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task maniskillnew --without_proprio --single_layer_classifier
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task cargoalnewshort --without_proprio --single_layer_classifier
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task carla_distance_cost --without_proprio --single_layer_classifier


# mani and dubins failed
# training MLP
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task maniskillnew 
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task dubins1800_withcost 
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task cargoalnewshort 
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task carla_distance_cost
