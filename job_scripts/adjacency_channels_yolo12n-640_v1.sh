#!/bin/bash
#SBATCH --job-name=adjacency_channels_yolo12n-640_v1
#SBATCH --output=run_logs/adjacency_channels/%j/output_train.log
#SBATCH --error=run_logs/adjacency_channels/%j/error_train.err
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

module purge
module load cuda/11.8
module load python/3.11.0--gcc--11.2.0

cd /leonardo_work/EUHPC_D18_074/Project/yolo-superpixel-framework

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting training script..."
python run_training.py --config configs/adjacency_channels_yolo12n-640_v1.yaml

echo "Job finished at $(date)"