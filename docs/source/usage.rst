Usage
=====

.. _installation:

Installation
------------

To install the project, clone the repository and install the dependencies using pip:

.. code-block:: bash

    git clone https://github.com/ssec-jhu/neuro-morpho.git
    cd neuro-morpho
    pip install -e .


Training a Model
----------------

To train a model, you can use the command-line interface. You will need to
provide a gin configuration file to specify the model, data, and training
parameters.

.. code-block:: bash

    python neuro_morpho.cli unet.config.gin

A sample gin configuration file is provided in the root of the repository.
This file can be modified to change the training parameters.
