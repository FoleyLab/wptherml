"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wpspecdev
import pytest
import sys

def test_wpspecdev_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "wpspecdev" in sys.modules
