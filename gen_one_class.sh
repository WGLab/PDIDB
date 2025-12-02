#!/bin/bash

main_dir=/mnt/isilon/wang_lab/chenh7/projects/gestaltgan
device=h100


module load CUDA/11.8.0
source activate gestaltgan
#module load CUDA/11.1.1-GCC-10.2.0

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
#SBATCH --gres=gpu:${device}:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-xe9680q
#SBATCH --account=hpcusers
#SBATCH --mail-user=hongzhuo@sas.upenn.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=40G

cd ${main_dir}

python gen_images.py \
  --network=/home/chenh7/projects/gestaltgan/runs/00034-stylegan3-t-gestaltgan_training_healthy_balanced-gpus8-batch32-gamma2/network-snapshot-003840.pkl \
  --outdir=out_3840_10_100 \
  --samples=100 \
  --trunc=1 \
  --class=2

EOF

