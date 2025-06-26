#!/bin/bash
#BSUB -n 16
#BSUB -q general-interactive
#BSUB -R 'rusage[mem=102GB]'
#BSUB -M 100GB
#BSUB -R 'gpuhost'
#BSUB -gpu "num=1:gmem=30"
#BSUB -a 'docker(continuumio/anaconda3:2021.11)'
#BSUB -W 400
#BSUB -J dino_wm_job
#BSUB -oo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_new/output%J.log
#BSUB -eo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_new/error%J.log
#BSUB -N
#BSUB -u i.k.tabbara@wustl.edu
#BSUB -env "LSF_DOCKER_VOLUMES=/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active,LSF_DOCKER_SHM_SIZE=64g"


# Environment variables for your run
export CONDA_ENVS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/envs"
export CONDA_PKGS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/pkgs"
export PATH="/opt/conda/bin:$PATH"
export NETRC=/storage1/fs1/sibai/Active/ihab/tmp/.netrc
export DATASET_DIR=/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino
export TMPDIR=/storage1/fs1/sibai/Active/ihab/tmp
export TORCH_HOME=/storage1/fs1/sibai/Active/ihab/tmp/torch
export WANDB_CONFIG_DIR=/storage1/fs1/sibai/Active/ihab/tmp/.config
# export ACCELERATE_DISABLE_HOST_CHECK=1
# export WANDB_MODE=offline
source /opt/conda/etc/profile.d/conda.sh
conda activate dino_wm_ris
export SSL_CERT_FILE=$(python -m certifi)
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm
df -h /dev/shm
# Launch training
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3 encoder=vc1 decoder=transposed_conv
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3 encoder=dino_cls decoder=transposed_conv
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3 encoder=resnet decoder=transposed_conv


python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3 encoder=scratch model.train_encoder=true decoder=transposed_conv 
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3 encoder=r3m decoder=transposed_conv
# python train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3 encoder=dino decoder=vqvae training.batch_size=16 ##USE PYTHON MESH ACCELERATE




