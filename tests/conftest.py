# -*- coding: utf-8 -*-
import sys
import os
import pytest

# Make helper utilities importable from all tests.
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        'helpers'
    )
)

test_data_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "./test_data"
    )
)
assert os.path.isdir(test_data_dir)


@pytest.fixture(scope="module")
def crystal_file_dir():
    return test_data_dir


@pytest.fixture(
    scope="module",
    params=[
        os.path.join(test_data_dir, "1511812_butane.cif"),
        os.path.join(test_data_dir, "4501704_benzene.cif"),
        os.path.join(test_data_dir, "4503066_methanol.cif"),
    ]
)
def cif_path(request):
    """PyTest fixture that creates a test for every cif found in './data'
    """
    assert os.path.exists(request.param)
    return request.param
