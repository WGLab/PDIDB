#!/bin/bash

main_dir=/mnt/isilon/wang_lab/chenh7/projects/gestaltgan
device=h100

module load CUDA/11.8.0
source activate gestaltgan

log_dir="${main_dir}/logs"
runs_dir="${main_dir}/runs"

mkdir -p "${log_dir}"
mkdir -p "${runs_dir}"

echo "Submitting training job..."

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gestalt_train
#SBATCH --output=${log_dir}/%j.stdout
#SBATCH --error=${log_dir}/%j.stderr
#SBATCH -N 1
#SBATCH --gres=gpu:${device}:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu-xe9680q
#SBATCH --account=hpcusers
#SBATCH --mail-user=hongzhuo@sas.upenn.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=100G

cd ${main_dir}

python train.py \
    --use-gestalt-loss=False \
    --data=/mnt/isilon/wang_lab/chenh7/projects/gestaltgan_training_healthy_balanced \
    --cond=True \
    --cfg=stylegan3-t \
    --gpus=8 \
    --batch=32 \
    --gamma=2 \
    --mirror=1 \
    --kimg=5000 \
    --snap=20 \
    --dlr=0.0002 \
    --glr=0.00025 \
    --outdir=${runs_dir} \
    --lr-scheduler=step \
    --lr-decay-steps=1500 \
    --lr-decay-rate=0.5 

EOF

echo "Job submitted!"