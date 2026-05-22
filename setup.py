"""
wptherml
A python package for modeling light-matter interactions!
"""
from pathlib import Path
from setuptools import setup, find_packages
import versioneer

short_description = __doc__.split("\n")

try:
    long_description = Path("readme.md").read_text(encoding="utf-8")
except OSError:
    long_description = "\n".join(short_description[2:])


setup(
    # Self-descriptive entries which should always be present
    name='wptherml',
    author='Foley Lab',
    author_email='jfoley19@uncc.edu',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='LGPLv3',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "matplotlib",
        "numpy>=2.0",
        "scipy",
    ],

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    zip_safe=False,

)
