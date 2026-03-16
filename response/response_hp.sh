#!/usr/bin/bash
##SBATCH -J jys_psi3b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-r5
#SBATCH -t 1-00:00:00
#SBATCH -o /data/yoonsuh0615/repos/patientv3/response/logs/slurm-%A_jys_psi0.5b.out

echo "📁 Working dir: $(pwd)"
echo "🐍 Python executable: $(which python)"
echo "📦 Running on host: $(hostname)"
python generate_response.py

exit 0