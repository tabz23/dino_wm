#!/bin/bash
#BSUB -n 12
#BSUB -q general-interactive
#BSUB -R 'rusage[mem=102GB]'
#BSUB -M 100GB
#BSUB -R 'gpuhost'
#BSUB -gpu "num=1:gmem=30G"
#BSUB -a 'docker(continuumio/anaconda3:2021.11)'
#BSUB -W 24:00
#BSUB -J dino_wm_job
#BSUB -oo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_new/output%J.log
#BSUB -eo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_new/error%J.log
#BSUB -N
#BSUB -u i.k.tabbara@wustl.edu
#BSUB -env "LSF_DOCKER_VOLUMES=/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active,LSF_DOCKER_SHM_SIZE=32g"


# Environment variables for your run
export CONDA_ENVS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/envs"
export CONDA_PKGS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/pkgs"
export PATH="/opt/conda/bin:$PATH"
export NETRC=/storage1/fs1/sibai/Active/ihab/tmp/.netrc
export DATASET_DIR=/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino
export TMPDIR=/storage1/fs1/sibai/Active/ihab/tmp
export TORCH_HOME=/storage1/fs1/sibai/Active/ihab/tmp/torch
export WANDB_CONFIG_DIR=/storage1/fs1/sibai/Active/ihab/tmp/.config
df -h /dev/shm
# export ACCELERATE_DISABLE_HOST_CHECK=1
# export WANDB_MODE=offline
source /opt/conda/etc/profile.d/conda.sh
conda activate dino_wm_ris
export SSL_CERT_FILE=$(python -m certifi)
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm

#-m compute1-exec-358
# Launch training

python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinstruth.py --seed 3 --gamma-pyhj 0.99
python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinstruth.py --seed 4 --gamma-pyhj 0.99
python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinstruth.py --seed 5 --gamma-pyhj 0.99
python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinstruth.py --seed 6 --gamma-pyhj 0.99
python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinstruth.py --seed 7 --gamma-pyhj 0.99


# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=cargoal frameskip=5 num_hist=3 encoder=dino_cls decoder=transposed_conv proprio_emb_dim=50
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=cargoal frameskip=5 num_hist=3 encoder=vc1 decoder=transposed_conv proprio_emb_dim=50
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=cargoal frameskip=5 num_hist=3 encoder=resnet decoder=transposed_conv proprio_emb_dim=50
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=cargoal frameskip=5 num_hist=3 encoder=scratch model.train_encoder=true decoder=transposed_conv proprio_emb_dim=50
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=cargoal frameskip=5 num_hist=3 encoder=r3m decoder=transposed_conv proprio_emb_dim=50
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=cargoal frameskip=5 num_hist=3 encoder=dino decoder=vqvae training.batch_size=32 proprio_emb_dim=50##USE PYTHON MESH ACCELERATE


# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3 encoder=dino_cls decoder=transposed_conv proprio_emb_dim=100
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3 encoder=vc1 decoder=transposed_conv proprio_emb_dim=100
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3 encoder=resnet decoder=transposed_conv proprio_emb_dim=100
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3 encoder=scratch model.train_encoder=true decoder=transposed_conv proprio_emb_dim=100
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3 encoder=r3m decoder=transposed_conv proprio_emb_dim=100
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3 encoder=dino decoder=vqvae training.batch_size=32 proprio_emb_dim=100 ##USE PYTHON MESH ACCELERATE








# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder vc1  --seed 1 --gamma-pyhj 0.98
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder dino_cls --seed 1 --gamma_pyhj 0.98
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder resnet  --seed 1 --gamma-pyhj 0.98
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder scratch  --seed 1 --gamma-pyhj 0.98


# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder dino_cls --seed 1 --gamma-pyhj 0.999

# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder vc1  --seed 1 --gamma-pyhj 0.98


# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder dino  --seed 1 --gamma-pyhj 0.98


# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder r3m  --seed 1 --gamma-pyhj 0.98
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder dino  --seed 1 --gamma-pyhj 0.98


# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder resnet  --seed 1 --gamma-pyhj 0.999
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder scratch  --seed 1 --gamma-pyhj 0.999
# python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder r3m  --seed 1 --gamma-pyhj 0.999


#bjobs -q 787317