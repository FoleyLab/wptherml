Quickstart Guide
=============================

Once all dependencies are installed, you may perform a developmental install by typing the following
command from the top-level wptherml directory:

`pip install -e .`

Dependencies
**************
You need to install `numpy`, `scipy`, and `pip`.
We recommend `matplotlib`, `jupyter`, and `pytest`.

You may use the `wptherml.yml <https://github.com/FoleyLab/wptherml/edit/main/docs/quickstart.rst>`_ 
to create a conda environment will these dependencies.

name: wptherml

dependencies:
    - python>=3.7
    - pip>=19.0
    - numpy
    - scipy
    - matplotlib 
    - jupyter 
    - pytest

To create your conda environment (called `wptherml` based on the name above), type

`conda env create -f wptherml.yml`

Activate this environment *before* performing performing the development install by typing
`conda activate wptherml`
 
Examples
**********
Check out the example scripts and notebooks in the /examples/ folder.
