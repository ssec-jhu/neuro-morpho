#!/bin/bash

source activate /home/idies/mambaforge/envs/py312
source /home/idies/workspace/Storage/ryanhausen/persistent/.bashrc
cd /home/idies/workspace/ssec_neural_morphology/ryanhausen/neuro-morpho
python -m pip install -r requirements/dev.txt
python -m neuro_morpho.cli unet.config.gin
