#!/bin/bash -x


#SBATCH --gres=gpu:1
#SBATCH --array=0-99%100
#SBATCH --cpus-per-task=10
#SBATCH --job-name=build_imagenet
#SBATCH --partition tier0
#SBATCH -o .onager/logs/slurm/%x_%A_%a.o
#SBATCH -e .onager/logs/slurm/%x_%A_%a.e

eval "$(/data/home/pteterwak/miniconda3/bin/conda shell.bash hook)"
conda activate clip_benchmark

#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
export MASTER_ADDR='localhost'

cd /data/home/pteterwak/OpenClip/open_clip/benchmark/CLIP_benchmark/clip_benchmark 
#export PYTHONPATH="$PYTHONPATH:$PWD/src"
#export NCCL_DEBUG=INFO


#srun python  imagenet22k_builder.py  --split "train" --output /fsx/pteterwak/data/imagenet-22k_wds --num-chunks 100  --chunk-idx $SLURM_ARRAY_TASK_ID
srun python  llava_eval_builder.py  --dataset wds/imagenet1k --output /fsx/pteterwak/data/clip_benchmark/imagenet --dataset-root "https://huggingface.co/datasets/clip-benchmark/wds_imagenet1k/tree/main" --model-path /fsx/pteterwak/llava_weights/llava-llama-2-13b-chat-lightning-preview --query "Please describe this image" --num-chunks 100 --chunk-idx $SLURM_ARRAY_TASK_ID
