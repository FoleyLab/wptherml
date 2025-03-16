Quickstart Guide
=============================

This is a quick guide for setting up a conda environment with all dependencies necessary and recommended for WPTherml,
and for a development install of WPTherml.  If you are unfamiliar with conda and/or conda environments, take a few minutes to watch 
`this short video for Windows users <https://youtu.be/XCvgyvBFjyM>`_, `this short video for Mac users <https://youtu.be/OH0E7FIHyQo>`_, 
or `this short video for Linux users <https://youtu.be/Avx_FYdFBcc>`_.  You can find miniconda installers and instructions for Mac (intel and M1), Windows, and Linux `here <https://docs.conda.io/en/latest/miniconda.html>`_.

Dependencies
**************
You need to install `numpy`, `scipy`, and `pip`, `pytest` for basic 
operations, and for optimization features you need `pytorch`, `scikit-learn`, `pandas`, `pyqubo`, and `tqdm`.  We also recommend `matplotlib` and `jupyter`.

You may use the `wptherml_env.yml <https://github.com/FoleyLab/wptherml/blob/optdriver/wptherml_env.yml>`_ 
to create a conda environment with most these dependencies; pip is required .  This yaml file is located in the top-level wptherml directory,
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
