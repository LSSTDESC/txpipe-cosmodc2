#!/bin/bash   
#SBATCH -N 4
#SBATCH --qos=regular
#SBATCH --time=10:00:00
#SBATCH --job-name=cosmodc2_test
##SBATCH --image=docker:joezuntz/txpipe_cosmosis_firecrown:latest
#SBATCH --license=SCRATCH
#SBATCH --constraint=haswell
## knl
#SBATCH --mail-user=chihway@kicp.uchicago.edu

export OMP_NUM_THREADS=1

#source setup-firecrown

module load python
pip uninstall parsl
pip uninstall ceci
pip uninstall pyccl

#export PYTHONPATH=$PYTHONPATH:/global/homes/c/chihway/TXPipe:/global/homes/c/chihway/TJPCov
#shifter --volume=/global/homes/c/chihway/cosmosis:/opt/cosmosis --volume=/global/homes/c/chihway/FireCrown/:/opt/firecrown --volume=/global/homes/c/chihway/TXPipe:/opt/txpipe --image=joezuntz/txpipe_cosmosis_firecrown -- bash


#firecrown compute /global/homes/c/chihway/txpipe-cosmodc2/firecrown_config/cosmodc2_firecrown_real.yaml

#srun -n  64 shifter --volume=/global/homes/c/chihway/cosmosis:/opt/cosmosis --volume /global/homes/c/chihway/FireCrown/:/opt/firecrown --volume /global/homes/c/chihway/TXPipe:/opt/txpipe firecrown run-emcee /global/homes/c/chihway/txpipe-cosmodc2/firecrown_config/cosmodc2_firecrown_real.yaml

srun -n  48 shifter --volume=/global/homes/c/chihway/cosmosis:/opt/cosmosis --volume=/global/homes/c/chihway/FireCrown/:/opt/firecrown   --volume=/global/homes/c/chihway/TXPipe:/opt/txpipe --image=joezuntz/txpipe_cosmosis_firecrown firecrown run-emcee /global/homes/c/chihway/txpipe-cosmodc2/firecrown_config/cosmodc2_firecrown_real.yaml 


