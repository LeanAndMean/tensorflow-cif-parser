# -*- coding: utf-8 -*-
import os
import glob
import pytest


@pytest.fixture(
    scope="module",
    params=[
        "data/1511812_butane.cif",
        "data/4501704_benzene.cif",
        "data/4503066_methanol.cif",
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