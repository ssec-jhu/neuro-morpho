# Development


## Getting/Processing the Data

This project uses [DVC](https://dvc.org) to manage the data. The data is on SciServer 
in the `ssec-neural-morphology` data volume. In the root folder of this repository,
run the following command:

`dvc pull`

This will pull the raw image data into `data/raw`

