#!/bin/bash -x

#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=vicuna_ablate_rpo
#SBATCH --partition midas
#SBATCH -o /fsx/youngkyun/clamp/logs/slurm/%x_%A.o
#SBATCH -e /fsx/youngkyun/clamp/logs/slurm/%x_%A.e

eval "$(/data/home/youngkyun/miniconda3/bin/conda shell.bash hook)"
conda activate VLM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=57129

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
#export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1


cd /data/home/youngkyun/piotr/open_clip
export PYTHONPATH="$PYTHONPATH:$PWD/src"
srun --cpu_bind=v --accel-bind=gn python -u  src/training/main.py --train-data '/datasets01/img2dataset/laion-400m/data_partitioned/0/{00000..01066}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/1/{00000..01067}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/2/{00000..01014}.tar::/datasets01/img2dataset/laion-400m/data_partitioned/3/{00000..01085}.tar::/fsx/pteterwak/data/imagenet-21k_wds/train/{0..399}/{000000..000003}.tar'  --train-num-samples 10000000 --dataset-type webdataset_double_tokenizer   --batch-size 64 --zeroshot-frequency 1  --precision amp --workers 5  --dataset-resampled --model rebuttal  --gather-with-grad --epochs 6 --lr 0.00005 --wd 0.5 --warmup 1220 --eps 1e-08 --local-loss --lock-image   --pretrained laion400m_e32   --wrap-caption-long-list --logs /fsx/youngkyun/clamp/logs/ --grad-checkpointing --grad-clip-norm 1.0 --distill-model ViT-L-14 --distill-pretrained datacomp_xl_s13b_b90k

