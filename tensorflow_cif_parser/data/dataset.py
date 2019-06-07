# -*- coding: utf-8 -*-
"""Contains functions for generating an input pipeline for crystal structures
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow_cif_parser.cif.parser.ase import parse_file as ase_parser


def glob_crystal_files(path, extension=".cif"):
    """Discovers and returns all CIFs in the specified directory.
    Parameters
    ----------
    path: str
        Path to a directory containing Crystal Information Files (CIFs).
    extension: str
        Extension of CIFs. Defaults to the stardard (".cif").
    Returns
    -------
    list
        Set of paths to CIFs found in input path.
    """
    assert os.path.isdir(path)
    return tf.gfile.Glob(
        os.path.join(path, "*" + str(extension))
    )


def create_atom_dataset(paths, ignore_file_errors=False):
    """Flattens a set of crystals into atoms.
    Parameters
    ----------
    paths: list
        Paths to Crystal Information Files (CIFs).
    ignore_file_errors: bool
        If True, files that fail to load are ignored. Useful for automatically
        filtering out bad files in a dataset.
    Returns
    -------
    tf.data.Dataset
        A dataset of atoms. Each row has the following 3 entries:
            (
                tf.string,  # Path to crystal structure.
                tf.int64,  # Atomic number.
                tf.int64  # Atom index in crystal structure.
            )
    """
    def crystal_to_element_rows(path):
        """Extract atomic numbers and pair them with the filename."""
        crystal = ase_parser(path.numpy().decode())
        # Assert that crystal contains atoms.
        tf.debugging.assert_positive(len(crystal))
        crystal_paths = tf.tile(tf.expand_dims(path, 0), [len(crystal)])
        atomic_numbers = np.asarray(
            crystal.get_atomic_numbers(),
            dtype=np.int64
        )
        # Assert tile operation created the correct number of path duplicates.
        tf.debugging.assert_equal(
            crystal_paths.shape[0],
            atomic_numbers.shape[0]
        )
        # Add atom index. This is necessary to retain information about an
        # atom's location in the crystal file after flattening.
        atom_index = np.arange(len(crystal), dtype=np.int64)
        return crystal_paths, atomic_numbers, atom_index
    atom_dataset = tf.data.Dataset.from_tensor_slices((paths,)).map(
        lambda x: tuple(
            tf.py_function(
                crystal_to_element_rows,
                [x],
                [tf.string, tf.int64, tf.int64]
            )
        )
    )
    if ignore_file_errors:
        atom_dataset = atom_dataset.apply(tf.data.experimental.ignore_errors())
    # Flatten data from crystals so each row is an atom.
    atom_dataset = atom_dataset.flat_map(
        lambda *x: tf.data.Dataset.from_tensor_slices(
            tuple(x)
        )
    )
    return atom_dataset


def unique_atomic_numbers(atom_dataset, atomic_number_index=1):
    """Identify unique atomic_numbers in atom dataset.
    Atomic numbers are used to establish the set of unique atomic_numbers.
    Parameters
    ----------
    atom_dataset: tf.data.Dataset
        Each row should corresond to an atom.
    atomic_number_index: int
        Column index of the atomic number in a given row.
    Returns
    -------
    tf.data.Dataset
        Unique atomic numbers.
    """
    return atom_dataset.map(
        lambda *x: x[atomic_number_index]
    ).apply(tf.data.experimental.unique())


def get_dataset_cardinality(dataset):
    """Counts the number of rows in a dataset.
    Parameters
    ----------
    dataset: tf.data.Dataset
        Input dataset.
    Returns
    -------
    int
        Dataset cardinality.
    """
    row_itr = tf.data.make_initializable_iterator(dataset)
    next_example = row_itr.get_next()
    # Track appearance and count of each atomic number.
    row_count = 0
    with tf.Session() as sess:
        try:
            sess.run(row_itr.initializer)
            while True:
                sess.run(next_example)
                row_count += 1
        except tf.errors.OutOfRangeError:
            pass
    return row_count


def create_equal_sampling_atom_dataset(
        paths,
        ignore_file_errors=False,
        repeat=None):
    """Uniform random sampling of atomic_numbers from a large set of
    crystal structures.
    Flattens a set of crystals into atoms.
    Parameters
    ----------
    paths: list
        Paths to Crystal Information Files (CIFs).
    ignore_file_errors: bool
        If True, files that fail to load are ignored. Useful for automatically
        filtering out bad files in a dataset.
    repeat: None or int
        Number of times to repeat each unique element's dataset. Defaults to
        None, which corresponds to infinite repeats.
        NOTE: Using a positive integer with an unbalanced dataset will result
        in less common elements appearing only at the start of the dataset.
    Returns
    -------
    tf.data.Dataset
        A dataset of atoms. Each row has the following 3 entries:
            (
                tf.string,  # Path to crystal structure.
                tf.int64,  # Atomic number.
                tf.int64  # Atom index in crystal structure.
            )
    """
    atom_dataset = create_atom_dataset(
        paths,
        ignore_file_errors=ignore_file_errors
    )
    unique_atomic_number_dataset = unique_atomic_numbers(atom_dataset)
    atomic_number_cardinality = get_dataset_cardinality(
        unique_atomic_number_dataset
    )
    assert atomic_number_cardinality > 0
    # Create a separate dataset for each atomic number.
    equal_sampling_atom_dataset = unique_atomic_number_dataset.interleave(
        lambda atomic_number: atom_dataset.filter(
            lambda *x: tf.equal(x[1], atomic_number)
        ).repeat(repeat),
        cycle_length=atomic_number_cardinality,
        block_length=1
    )
    return equal_sampling_atom_dataset
