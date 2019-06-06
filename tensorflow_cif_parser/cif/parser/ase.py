# -*- coding: utf-8 -*-
from ase.io import read as ase_reader


def parse_file(filepath):
    """Parses a CIF.
    Parameters
    ----------
    filepath: str
        Path to CIF.
    Returns
    -------
    ase.Atoms
        Parsed CIF as an ASE Atoms object.
    """
    return ase_reader(filepath)
