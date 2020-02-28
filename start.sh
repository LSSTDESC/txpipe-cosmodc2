#! /bin/sh

module load python
pip uninstall parsl
pip uninstall ceci
pip uninstall pyccl

export PYTHONPATH=$PYTHONPATH:/global/homes/c/chihway/TXPipe_dc2/TXPipe:/global/homes/c/chihway/TJPCov
#shifter --image docker:joezuntz/txpipe bash
shifter --volume=$PWD/cosmosis:/opt/cosmosis --volume $PWD/FireCrown/:/opt/firecrown --volume $PWD/TXPipe:/opt/txpipe --image=joezuntz/txpipe_cosmosis_firecrown -- bash

#module load python
#pip uninstall parsl
#pip uninstall ceci

