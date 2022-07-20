# RealPatch
The supplementary matherial file is written in ``RealPatch_supplementary.pdf``.

## 1. Create Environment with necessary dependencies
### Install Poetry (if not already installed)
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
# To configure your current shell run
source $HOME/.poetry/env

poetry --version
# Update poetry to the latest version
poetry self update
```

### Install Dependencies with Poetry
```
# Enter code directory
cd code 
# Install dependecies
poetry install
```

### Activating poetry environment
```
# Check the virtual environment created and activated
poetry env list
<env_name>

# If not activated, activate the virtual environment (entering the output of previous command)
source `poetry config virtualenvs.path`/<env_name>/bin/activate
```

## 2. Data
The data are hosted in a [google drive folder](https://drive.google.com/drive/folders/15co1MTrVPzzzt4tRPGK8zTLzH4kFqkXs?usp=sharing). 

To run the code you can either manually download the data and place them in the ``data`` folder here. Be sure all the required files are present:
- features_file.npz
- labels_train.csv
- labels_val.csv
- labels_test.csv

Or as an alternative, you can run the following:
```
python download_data.py
```
which will download the four files and place them in the ``data`` folder for you.

## 3. Run RealPatch
```
python realpatch.py data=celeba
```

## 4. Results
The script will produce multiple outputs in the ``outputs`` folder for evaluating the matching results. The ``Evaluation`` folder will contain 3 ``.csv`` files. One saving the number of training examples before and after matching (similar to Table 10 in Appendix), and two to assess the achieved covariate balanced in terms of *SMD* and *VR* (consistent to Table 3).

An example of results for a single run of CelebA with selected RealPatch hyperparameters is in the folder ``output/2022-03-14/16-13-25``.
