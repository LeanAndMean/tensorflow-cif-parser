# -*- coding: utf-8 -*-
from utils import profile_with_pytinstrument
from tensorflow_cif_parser.cif.parser.ase import parse_file as ase_parser


@profile_with_pytinstrument
def test_ase_parser(cif_path):
    """Loads crystal structure and prints some information.
    Parameters
    ----------
    cif_path: str
        Path to Crystal Information File (CIF).
    """
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
            )
        )
