#! /bin/sh

module load python
pip uninstall parsl
pip uninstall ceci
pip uninstall pyccl

#export PYTHONPATH=$PYTHONPATH:/global/homes/c/chihway/TXPipe:/global/homes/c/chihway/TJPCov
#shifter --image docker:joezuntz/txpipe bash
#shifter --volume=$PWD/cosmosis:/opt/cosmosis --volume $PWD/FireCrown/:/opt/firecrown --volume $PWD/TXPipe:/opt/txpipe --image=joezuntz/txpipe_cosmosis_firecrown -- bash

shifter --image=joezuntz/txpipe-firecrown -V $PWD/FireCrown:/opt/firecrown -V $PWD/TXPipe:/opt/txpipe bash

export PYTHONPATH=/global/homes/c/chihway/TJPCov:$PYTHONPATH 



