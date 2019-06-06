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


@pytest.fixture(
    scope="module",
    params=[
        "test_data/1511812_butane.cif",
        "test_data/4501704_benzene.cif",
        "test_data/4503066_methanol.cif",
    ]
)
def cif_path(request):
    """PyTest fixture that creates a test for every cif found in './data'
    """
    # print(request.param)
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            request.param
        )
    )
    assert os.path.exists(request.param)
    return request.param
