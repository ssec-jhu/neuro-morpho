#!/bin/bash

source activate py312
pip install comet_ml fire gin-config matplotlib opencv-python-headless pandas scikit-image scikit-learn skan --upgrade
source ~/.bashrc
python -m neuro_morpho.cli unet.config.gin
