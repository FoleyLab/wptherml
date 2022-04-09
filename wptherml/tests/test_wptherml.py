"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wptherml
import numpy as np
import pytest
import sys


def test_wptherml_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "wptherml" in sys.modules
