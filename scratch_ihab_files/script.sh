#BSUB -o /storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs/output%J.log
#BSUB -e /storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs/error%J.log
#BSUB -gpu "num=1:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -R "rusage[mem=20]"
#BSUB -J PythonGPUJob

# source /storage1/sibai/Active/ihab/miniconda3/bin/activate
source activate
conda activate dino_wm
cd /storage1/sibai/Active/ihab/research_new/dino_wm
export DATASET_DIR=/storage1/sibai/Active/ihab/research_new/datasets_dino
export TORCH_HOME=/storage1/sibai/Active/ihab/tmp/torch
#accelerate launch train.py --config-name "train copy.yaml" env=dubins frameskip=5 num_hist=3
#WANDB_MODE=disabled python train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3
#WANDB_MODE=disabled accelerate launch train.py --config-name "train copy.yaml" env=maniskill frameskip=5 num_hist=3

# bsub -q gpu-compute < /storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/script.sh

# NVIDIAA40
# NVIDIAA10080GBPCIe
# NVIDIAA100_SXM4_80GB
