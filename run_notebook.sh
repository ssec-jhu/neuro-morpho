#!/bin/bash

source activate py312
pip install pathlib tqdm concurrent papermill --upgrade
papermill neuro_morpho/notebooks/data_organizer.ipynb neuro_morpho/notebooks/output.ipynb
