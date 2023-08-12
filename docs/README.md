# Compiling wptherml's Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).
To compile the docs, first ensure that Sphinx and the ReadTheDocs theme are installed.


```bash
conda install sphinx sphinx_rtd_theme 
```


Once installed, you can use the `Makefile` in this directory to compile static HTML pages by
```bash
make html
```

The compiled docs will be in the `_build` directory and can be viewed by opening `index.html` (which may itself 
be inside a directory called `html/` depending on what version of Sphinx is installed).


A configuration file for [Read The Docs](https://readthedocs.org/) (readthedocs.yaml) is included in the top level of the repository. To use Read the Docs to host your documentation, go to https://readthedocs.org/ and connect this repository. You may need to change your default branch to `main` under Advanced Settings for the project.

If you would like to use Read The Docs with `autodoc` (included automatically) and your package has dependencies, you will need to include those dependencies in your documentation yaml file (`docs/requirements.yaml`).

# Original Release of WPTherml
Version 2.0.0 of WPTherml features several syntax changes in addition to new features and restructuring of the underlying code.  
Users desiring version 1.0.0 may still install the original release using pip as follows:

`pip install wptherml==1.0.0`

Basic usage of version 1.0.0 is described below.

## Overview
WPTherml stands for **W**icked **P**ackage for **Th**ermal **E**nergy and **R**adiation management with **M**ulti **L**ayer nanostructures.
The vision of this software package is to provide an easy-to-use platform for the design of materials with tailored optical and thermal properties for
the vast number of energy applications where control of absorption and emission of radiation, or conversion of heat to radiation or vice versa, is paramount.
The optical properties are treated within classical electrodynamics, and the current version has a Transfer Matrix Method solver to rigorously solve Maxwell's equations
for layered isotropic media, and a Mie Theory solver for spherical nanoparticles.  WPTherml was conceived and developed by the [Foley Lab](https://foleylab.github.io).  Analytic gradients of the transfer matrix equations also facilitate optimization of high-level figures of merit for multilayered structures as described [here](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013018).

**Citation Instructions** The following citation should be used in any publication utilizing the WPTherml program: "WPTherml: A Python Package for the Design
of Materials for Harnessing Heat", J. F. Varner, N. Eldabagh, D. Volta, R. Eldabagh, J. J. Foley IV, *Journal of Open Research Software*, **7**, 28 (2019).  The open-access software papaer can be accessed [here](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.271/)!

Use of analytic gradient technology should cite "Accelerating the discovery of multilayer nanostructures with analytic differentiation of the transfer matrix equations",
James F. Varner, Dayanara Wert, Aya Matari, Raghad Nofal, and Jonathan J. Foley, IV
*Phys. Rev. Research* **2**, 013018 (2020).

More details of the Transfer Matrix equations, along will the full mathematical formulation currently implemented in WPTherml, can be found in
the [documentation](https://github.com/FoleyLab/wptherml/blob/main/docs/Equations.pdf).

## Quick Start
Quickstart Guide
=============================

This is a quick guide for setting up a conda environment with all dependencies necessary and recommended for WPTherml,
and for a development install of WPTherml.  If you are unfamiliar with conda and/or conda environments, take a few minutes to watch 
`this short video for Windows users <https://youtu.be/XCvgyvBFjyM>`_, `this short video for Mac users <https://youtu.be/OH0E7FIHyQo>`_, 
or `this short video for Linux users <https://youtu.be/Avx_FYdFBcc>`_.  You can find miniconda installers and instructions for Mac (intel and M1), Windows, and Linux `here <https://docs.conda.io/en/latest/miniconda.html>`_.

Dependencies
**************
You need to install `numpy`, `scipy`, and `pip`.
We recommend `matplotlib`, `jupyter`, and `pytest`.

You may use the `wptherml_env.yml <https://github.com/FoleyLab/wptherml/blob/main/wptherml_env.yml>`_ 
to create a conda environment with these dependencies.  This yaml file is located in the top-level wptherml directory,
so to create your conda environment (called `wptherml` based on the name above), you can type the following command from the top-level wptherml directory:

`conda env create -f wptherml_env.yml`

Activate this environment by typing

`conda activate wptherml`

Development Install
*******************
After you have created and activated your wptherml environment, you can perform a ddevelopment install from the top-level wptherml directory
by typing:

`pip install -e .`
 
Examples
**********
Check out the example scripts and notebooks in the /examples/ folder `here <https://github.com/FoleyLab/wptherml/tree/main/examples>`_.
## Playlist
The developers of WPTherml compiled a thematic [Spotify Playlist called "Everything Thermal"](https://open.spotify.com/playlist/1Vb7MV4WwjOMMHLbrX4TNN); we hope it will inspire you to imagine new possibilities for
harnessing heat and thermal radiation!

## Podcast
One of the developers co-hosts a light-matter themed [podcast](https://open.spotify.com/show/7hh6eZ3TLxJFwuWnLAXM6L?si=qp-GQc4ZSXGgvbvyYtUi2w) called [The Goeppert Mayer Gauge](https://foleylab.github.io/gmgauge/) along with [Prof. Dugan Hayes](http://www.chm.uri.edu/hayesgroup/) at University of Rhode Island.

## Feature List


