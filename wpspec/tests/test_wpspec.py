"""
Unit and regression test for the wpspec package.
"""

# Import package, test suite, and other packages as needed
import wpspec
import pytest
import sys

def test_wpspec_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "wpspec" in sys.modules
