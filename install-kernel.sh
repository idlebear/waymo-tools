#!/bin/bash

conda create --yes --name waymo python=3.9
conda install --yes -c anaconda ipykernel
pip install -r requirements.txt
ipykernel install --user --name=waymo

