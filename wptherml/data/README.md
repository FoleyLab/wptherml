# Material Data

This directory contains optical constants, solar spectrum data, atmospheric transmissivity data, and color-matching
functions used by WPTherml's electrodynamics and thermal-radiation workflows.

Please note that it is not recommended to place large files in your git directory. If your project requires files larger
than a few megabytes in size it is recommended to host these files elsewhere. This is especially true for binary files
as the `git` structure is unable to correctly take updates to these files and will store a complete copy of every version
in your `git` history which can quickly add up. As a note most `git` hosting services like GitHub have a 1 GB per repository
cap.

:)

## Including package data

Modify your package's `setup.py` file and the `setup()` command. Include the 
[`package_data`](http://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use) keyword and point it at the 
correct files.

## Manifest

See `material_references.md` for optical-constant provenance.
