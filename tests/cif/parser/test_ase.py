# -*- coding: utf-8 -*-
import functools
from pyinstrument import Profiler
from tensorflow_cif_parser.cif.parser.ase import parse_file as ase_parser


def profile_with_pytinstrument(func):
    @functools.wraps(func)
    def pytinstrument_decorator(*args, **kwargs):
        # Before function is called.
        profiler = Profiler()
        profiler.start()
        # Call function.
        func_return = func(*args, **kwargs)
        # After function is called.
        profiler.stop()
        print(profiler.output_text(unicode=False, color=True))
        return func_return
    return pytinstrument_decorator


@profile_with_pytinstrument
def test_ase_parser(cif_path):
    print("Path of test CIF:", str(cif_path))
    ase_atoms = ase_parser(cif_path)
    print("ASE atoms object:")
    print(ase_atoms)
    print("Unit Cell:")
    print(ase_atoms.get_cell())
    print("Sites:")
    for site_idx, site in enumerate(ase_atoms):
        print("  {}  {}".format(
            site_idx,
            str(site)
        ))