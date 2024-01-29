#!/bin/bash -x


#SBATCH --gres=gpu:1
#SBATCH --array=0-399%400
#SBATCH --cpus-per-task=10
#SBATCH --job-name=build_imagenet
#SBATCH --partition midas
#SBATCH -o .onager/logs/slurm/%x_%A_%a.o
#SBATCH -e .onager/logs/slurm/%x_%A_%a.e

eval "$(/data/home/pteterwak/miniconda3/bin/conda shell.bash hook)"
conda activate llava_retrieve

#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
export MASTER_ADDR='localhost'

cd /data/home/pteterwak/LLaVaRetrieve/CLIP_benchmark/clip_benchmark
#export PYTHONPATH="$PYTHONPATH:$PWD/src"
#export NCCL_DEBUG=INFO


srun python  imagenet21k_builder.py  --split "train" --output /fsx/pteterwak/data/imagenet-21k_wds --num-chunks 400  --chunk-idx $SLURM_ARRAY_TASK_ID
