# SSEC-JHU neuro_morpho

[![CI](https://github.com/ssec-jhu/neuro-morpho/actions/workflows/ci.yml/badge.svg)](https://github.com/ssec-jhu/neuro-morpho/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/neuro-morpho/badge/?version=latest)](https://neuro-morpho.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/ssec-jhu/neuro-morpho/graph/badge.svg?token=nO3cCBglK2)](https://codecov.io/gh/ssec-jhu/neuro-morpho)
[![Security](https://github.com/ssec-jhu/neuro-morpho/actions/workflows/security.yml/badge.svg)](https://github.com/ssec-jhu/neuro-morpho/actions/workflows/security.yml)
<!---[![DOI](https://zenodo.org/badge/<insert_ID_number>.svg)](https://zenodo.org/badge/latestdoi/<insert_ID_number>) --->


![SSEC-JHU Logo](docs/_static/SSEC_logo_horiz_blue_1152x263.png)

B# Neural Morphology (Neuro-Morpho)

Neurons form complex dendritic arbors to integrate signals from many sources at once. The
structure of a neuron is so essential to its function that classes of neuron can be identified by
their structure alone. Additionally, the morphology of a neuron gives important insights into the
mechanisms of nervous system development and disfunction. Therefore, software that can
accurately trace the structure of the dendritic arbor is essential. This software shouldn’t require human supervision, which is time-consuming and introduces biases and inconsistencies, and should keep pace with modern imaging techniques that can rapidly generate large datasets.
To address these issues, we propose developing open-source software based on convolutional
neural networks (CNNs - specifically Unet) to segment/skeletonize neural dendrites.


# Quickstart

```term
pip install git+https://github.com/ssec-jhu/neuro-morpho.git
```

See [Usage](#usage) for quick and easy usage instructions for this Python package.

To start using the application for training/ testing purposes run:
```term
pip install -r requirements/all.txt
pip install -e .
python -m  neuro_morpho.cli 
```

# Installation, Build, & Run instructions


### Conda:

For additional cmds see the [Conda cheat-sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf).

 * Download and install either [miniconda](https://docs.conda.io/en/latest/miniconda.html#installing) or [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).
 * Create new environment (env) and install ``conda create -n <environment_name>``
 * Activate/switch to new env ``conda activate <environment_name>``
 * ``cd`` into repo dir.
 * Install ``python`` and ``pip`` ``conda install python=3.11 pip``
 * Install all required dependencies (assuming local dev work), there are two ways to do this
   * If working with tox (recommended) ``pip install -r requirements/dev.txt``.
   * If you would like to setup an environment with all requirements to run outside of tox ``pip install -r requirements/all.txt``.

### Build:

  #### with Python ecosystem:
  * ``cd`` into repo dir.
  * ``conda activate <environment_name>``
  * Build and install package in <environment_name> conda env: ``pip install .``
  * Do the same but in dev/editable mode (changes to repo will be reflected in env installation upon python kernel restart)
    _NOTE: This is the preferred installation method for dev work._
    ``pip install -e .``.
    _NOTE: If you didn't install dependencies from ``requirements/dev.txt``, you can install
    a looser constrained set of deps using: ``pip install -e .[dev]``._
    _NOTE:
    For GPU acceleration PyTorch can be re-installed with their accelerator options.
    For PyTorch see the [PyTorch installation docs](https://pytorch.org/get-started/locally/).
    E.g., ``pip install --force -r requirements/pytorch.txt --index-url https://download.pytorch.org/whl/cu126``.
    Since it's installed via ``requirements/prd.txt``, ``--force``or ``--upgrade`` must  be used to re-install
    the accelerator versions.  ``--force`` is preferable as it will error if the distribution is not available
    at the given url index, however ``--upgrade`` may not.

### Usage

Follow the above [Quickstart](#quickstart) or [Build with Python ecosystem instructions](#with-python-ecosystem).

# Preprocessing
Run neuro_morpho/notebooks/data_organizer notebook to partition the data to three disjoint groups: training, validation and test sets.
The partition ratios are hardocded in notebook and set currently on 60% of data going to training, 20% to validation and 20% to testing.

# Main pipeline
Pipeline configuration is maintained in ``unet.config.gin`` file.
Using the command line interface (i.e., from a terminal prompt):
```term
python -m neuro_morpho.cli
```
command runs the pipeline that consists of 4 separate modules:
Each one of them can be run separately, or alternatively, all 4 can be run one afrter another.

# Training.
The relevant params in config file are:
```run.train = True
run.training_x_dir = "/Path/to/training/images"
run.training_y_dir = "/Path/to/training/labels"
run.validating_x_dir = "/Path/to/validation/images"
run.validating_y_dir = "/Path/to/validation/labels"
run.logger = @CometLogger()
```

# Threshold calculation.
The relevant param in config file is:
```run.get_threshold = True```

# Testing.
The relevant params in config file are:
```run.test = True
run.testing_x_dir = "/Path/to/testing/images"
run.testing_y_dir = "/Path/to/testing/labels"
```

# Image inference.
The relevant params in config file are:
```run.infer = True```
and the same paths to use as in case of testing


# Testing
_NOTE: The following steps require ``pip install -r requirements/dev.txt``._

## Using tox

* Run tox ``tox``. This will run all of linting, security, test, docs and package building within tox virtual environments.
* To run an individual step, use ``tox -e {step}`` for example, ``tox -e test``, ``tox -e build-docs``, etc.

Typically, the CI tests run in github actions will use tox to run as above. See also [ci.yml](https://github.com/ssec-jhu/neuro-morpho/blob/main/.github/workflows/ci.yml).

## Outside of tox:

The below assume you are running steps without tox, and that all requirements are installed into a conda environment, e.g. with ``pip install -r requirements/all.txt``.

_NOTE: Tox will run these for you, this is specifically if there is a requirement to setup environment and run these outside the purview of tox._

### Linting:
Facilitates in testing typos, syntax, style, and other simple code analysis tests.
  * ``cd`` into repo dir.
  * Switch/activate correct environment: ``conda activate <environment_name>``
  * Run ``ruff .``
  * This can be automatically run (recommended for devs) every time you ``git push`` by installing the provided
    ``pre-push`` git hook available in ``./githooks``.
    Instructions are in that file - just ``cp ./githooks/pre-push .git/hooks/;chmod +x .git/hooks/pre-push``.

### Security Checks:
Facilitates in checking for security concerns using [Bandit](https://bandit.readthedocs.io/en/latest/index.html).
 * ``cd`` into repo dir.
 * ``bandit --severity-level=medium -r neuro_morpho``

### Unit Tests:
Facilitates in testing core package functionality at a modular level.
  * ``cd`` into repo dir.
  * Run all available tests: ``pytest .``
  * Run specific test: ``pytest tests/test_util.py::test_base_dummy``.

### Regression tests:
Facilitates in testing whether core data results differ during development.
  * WIP

### Smoke Tests:
Facilitates in testing at the application and infrastructure level.
  * WIP

### Build Docs:
Facilitates in building, testing & viewing the docs.
 * ``cd`` into repo dir.
 * ``pip install -r requirements/docs.txt``
 * ``cd docs``
 * ``make clean``
 * ``make html``
 * To view the docs in your default browser run ``open docs/_build/html/index.html``.