# -*- coding: utf-8 -*-
from ase.io import read as ase_reader

def parse_file(filepath):
    return ase_reader(filepath)