#!/bin/bash

CONDA=$1
if [ "$CONDA" = "" ]; then
    CONDA="/home/$USER/conda.sh"
fi

if ! test -f $CONDA; then
  echo "Conda environmens must be installed.  If conda is already active, use an empty file"
  exit
fi

source $CONDA

conda create --yes --name waymo python=3.9
conda install --yes --name waymo -c anaconda ipykernel
conda activate waymo
pip install -r requirements.txt
python -m ipykernel install --user --name=waymo

