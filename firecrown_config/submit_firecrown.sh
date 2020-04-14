#!/bin/bash -l

#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -N 4
#SBATCH -t 20:00:00
#SBATCH -A m1727
#SBATCH -J firecrown_3x2
#SBATCH --mail-user=chihway@kicp.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=3x2.out
#SBATCH --error=3x2.err

export OMP_NUM_THREADS=1

cd /global/homes/c/chihway
source initialize_conda.sh
conda activate myenv

cd /global/homes/c/chihway/txpipe-cosmodc2/firecrown_config
mpirun -n 48 firecrown run-cosmosis cosmodc2_firecrown_real_3x2_chain.yaml


