#BSUB -o /storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs/output%J.log
#BSUB -e /storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs/error%J.log
#BSUB -gpu "num=1:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -R "rusage[mem=20]"
#BSUB -J PythonGPUJob

# source /storage1/sibai/Active/ihab/miniconda3/bin/activate
# source /storage1/fs1/sibai/Active/ihab/miniconda3/bin/activate
source activate
conda activate dino_wm
cd /storage1/sibai/Active/ihab/research_new/dino_wm
export DATASET_DIR=/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino
export TORCH_HOME=/storage1/fs1/sibai/Active/ihab/tmp/torch
python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=pusht frameskip=5 num_hist=3
#WANDB_MODE=disabled python train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3
#WANDB_MODE=disabled accelerate launch train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3

# bsub -q gpu-compute < /storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/script.sh

# LSF_DOCKER_VOLUMES="/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active" bsub -n 1 -q general-interactive -Is -a 'docker(ubuntu)' /bin/bash
# export CONDA_DIR=/storage1/fs1/sibai/Active/ihab/miniconda3
# export PATH=$CONDA_DIR/bin:$PATH



# export LSF_DOCKER_VOLUMES="/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active"
# bsub -Is -q general-interactive -a 'docker(continuumio/anaconda3:2021.11)' /bin/bash
# export CONDA_ENVS_DIRS="/storage1/fs1/sibai/Active/ihab/miniconda3/envs"
# conda info --envs
# conda activate dino_wm  #but still cant use it

#conda env export --name dino_wm > dino_wm.yml





# export LSF_DOCKER_VOLUMES="/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active"
# bsub -Is -q general-interactive -a 'docker(continuumio/anaconda3:2021.11)' /bin/bash
# Run on the login node *once*

# mkdir -p /storage1/fs1/sibai/Active/ihab/conda/envs
# mkdir -p /storage1/fs1/sibai/Active/ihab/conda/pkgs

###### HERE 
export LSF_DOCKER_VOLUMES="/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active" 
export LSF_DOCKER_SHM_SIZE='64g'  
bsub -n 10 -Is -q general-interactive -R 'rusage[mem=102GB]' -M 100GB -R 'gpuhost' -gpu "num=1:gmem=30G"  -a 'docker(continuumio/anaconda3:2021.11)'  /bin/bash 
# bsub -n 12 -Is -q general-interactive -R 'rusage[mem=32GB]' -M 30 -R 'gpuhost' -gpu "num=1:gmem=10G"  -a 'docker(continuumio/anaconda3:2021.11)'  /bin/bash 
# bsub -n 12 -Is -q general-interactive -R 'rusage[mem=32GB]' -M 30 -R 'gpuhost' -gpu "num=1:gmem=10G" -a 'docker(nvidia/cuda:11.8.0-base-ubuntu22.04)' /bin/bash
# bsub -n 12 -Is -q general-interactive -R 'rusage[mem=32GB]' -M 30 -R 'gpuhost' -gpu "num=1:gmem=1G" -a 'docker(nvcr.io/nvidia/pytorch:23.10-py3)' /bin/bash
bsub -n 12 -Is -q general-interactive -R 'rusage[mem=220GB]' -M 200 -R 'gpuhost' -gpu "num=2:gmem=5G" -a 'docker(nvcr.io/nvidia/pytorch:22.10-py3)' /bin/bash 
export CONDA_ENVS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/envs" \
export CONDA_PKGS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/pkgs" \
export PATH="/opt/conda/bin:$PATH" \
export NETRC=/storage1/fs1/sibai/Active/ihab/tmp/.netrc \
export DATASET_DIR=/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino \
export TORCH_HOME=/storage1/fs1/sibai/Active/ihab/tmp/torch \
export WANDB_CONFIG_DIR=/storage1/fs1/sibai/Active/ihab/tmp/.config  

# export TMPDIR=/storage1/fs1/sibai/Active/ihab/tmp \
bhosts -w general-interactive -gpu
df -h /dev/shm
bjobs -l 663370 #check ram w hek 5bar

source /opt/conda/etc/profile.d/conda.sh 
conda activate dino_wm_ris    
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm


python train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3
python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=point_maze frameskip=5 num_hist=3
# python -m accelerate.commands.launch train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3 encoder=dino decoder=vqvae



LSF_DOCKER_VOLUMES='/storage2/fs1/dt-summer-corp/Active/:/storage2/fs1/dt-summer-corp/Active/' \
LSF_DOCKER_PORTS='8501:8501' \
LSF_DOCKER_SHM_SIZE='4g' \
bsub -n 1 -Is -q artsci-interactive \
     -G compute-dt-summer-corp \
     -R 'rusage[mem=32GB]' -M 30GB \
     -R 'gpuhost' -gpu "num=1:gmodel=NVIDIAA10080GBPCIe:gmem=78G" \
     -m compute1-exec-370.ris.wustl.edu \
     -R 'select[port8501=1]' \
     -a 'docker(ahmad8742/urban:latest)' /bin/bash


# bsub -n 1 -Is -q general-interactive -R 'rusage[mem=32GB]' -M 30GB -R 'gpuhost' -gpu "num=3:gmem=78G" -a 'docker(continuumio/anaconda3:2021.11)' /bin/bash

# which conda
# source /opt/conda/etc/profile.d/conda.sh
# conda activate


# NVIDIAA40
# NVIDIAA10080GBPCIe
# NVIDIAA100_SXM4_80GB



bsub -n 5  -q general -R 'rusage[mem=52GB]' -M 50GB -R 'gpuhost' -gpu "num=1:gmem=15G"  -a 'docker(continuumio/anaconda3:2021.11)'  /bin/bash -c python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent.py --dino_encoder r3m  --seed 1 --gamma-pyhj 0.98

