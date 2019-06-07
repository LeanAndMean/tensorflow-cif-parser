# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from utils import profile_with_pytinstrument
from tensorflow_cif_parser.data.dataset import glob_crystal_files
from tensorflow_cif_parser.data.dataset import create_atom_dataset
from tensorflow_cif_parser.data.dataset import unique_atomic_numbers
from tensorflow_cif_parser.data.dataset import get_dataset_cardinality
from tensorflow_cif_parser.data.dataset\
    import create_equal_sampling_atom_dataset


@profile_with_pytinstrument
def test_glob_crystal_files(crystal_file_dir):
    """Tests that all the CIFs in the test data directory are discovered.
    Parameters
    ----------
    crystal_file_dir: str
        Path to a directory containing Crystal Information Files (CIFs).
    """
    print(crystal_file_dir)
    assert os.path.exists(crystal_file_dir)
    assert os.path.isdir(crystal_file_dir)
    crystal_files = glob_crystal_files(crystal_file_dir)
    print("{} files found.".format(len(crystal_files)))
    for filepath in crystal_files:
        print(crystal_files)
    assert len(crystal_files) == 3


@profile_with_pytinstrument
def test_create_atom_dataset(crystal_file_dir):
    """Tests flattening of crystal structure files into an atom dataset.
    Parameters
    ----------
    crystal_file_dir: str
        Path to a directory containing Crystal Information Files (CIFs).
    """
    crystal_files = glob_crystal_files(crystal_file_dir)
    atom_dataset = create_atom_dataset(crystal_files)
    dataset_itr = tf.data.make_one_shot_iterator(atom_dataset)
    next_example = dataset_itr.get_next()
    total_atoms = 0
    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(next_example))
                total_atoms += 1
        except tf.errors.OutOfRangeError:
            pass
    # Hardcoded expectations (modification will be needed if test data files
    # are updated).
    print(
        "Total atoms found:",
        total_atoms
    )
    assert total_atoms == 76


@profile_with_pytinstrument
def test_unique_atomic_numbers(crystal_file_dir):
    """Tests identification of the set of atomic numbers in a dataset.
    Parameters
    ----------
    crystal_file_dir: str
        Path to a directory containing Crystal Information Files (CIFs).
    """
    crystal_files = glob_crystal_files(crystal_file_dir)
    atom_dataset = create_atom_dataset(crystal_files)
    unique_atomic_numbers_dataset = unique_atomic_numbers(atom_dataset)
    atomic_number_itr = tf.data.make_one_shot_iterator(
        unique_atomic_numbers_dataset
    )
    next_example = atomic_number_itr.get_next()
    unique_atomic_number_set = set()
    with tf.Session() as sess:
        try:
            while True:
                unique_atomic_number = sess.run(next_example)
                assert unique_atomic_number not in unique_atomic_number_set
                print(unique_atomic_number)
                unique_atomic_number_set.add(unique_atomic_number)
        except tf.errors.OutOfRangeError:
            pass
    # Hardcoded expectations (modification will be needed if test data files
    # are updated).
    expected_atomic_numbers = set(
        (
            1,
            6,
            8,
        )
    )
    print("Unique atomic_number set:")
    print(unique_atomic_number_set)
    print("Expected atomic_number set:")
    print(expected_atomic_numbers)
    assert expected_atomic_numbers == unique_atomic_number_set


@profile_with_pytinstrument
def test_get_dataset_cardinality(crystal_file_dir):
    """Tests identification of the total number of rows in a dataset.
    Parameters
    ----------
    crystal_file_dir: str
        Path to a directory containing Crystal Information Files (CIFs).
    """
    crystal_files = glob_crystal_files(crystal_file_dir)
    atom_dataset = create_atom_dataset(crystal_files)
    atom_dataset_cardinality = get_dataset_cardinality(atom_dataset)
    print("Atom dataset cardinality:", atom_dataset_cardinality)
    assert atom_dataset_cardinality == 76
    unique_atomic_numbers_dataset = unique_atomic_numbers(atom_dataset)
    atomic_numbers_cardinality = get_dataset_cardinality(
        unique_atomic_numbers_dataset
    )
    print("Atomic numbers cardinality:", atomic_numbers_cardinality)
    assert atomic_numbers_cardinality == 3


@profile_with_pytinstrument
def test_create_equal_sampling_atom_dataset(crystal_file_dir):
    """Tests that elements are sampled uniformly by atomic number.
    Parameters
    ----------
    crystal_file_dir: str
        Path to a directory containing Crystal Information Files (CIFs).
    """
    crystal_files = glob_crystal_files(crystal_file_dir)
    # Create an infinite dataset (repeat == None).
    equal_sampling_atom_dataset = create_equal_sampling_atom_dataset(
        crystal_files,
        repeat=None
    )
    atom_itr = tf.data.make_initializable_iterator(equal_sampling_atom_dataset)
    next_example = atom_itr.get_next()
    # Track appearance and count of each atomic number.
    atomic_number_counts = {}
    # Expected number of unique atomic numbers:
    unique_atomic_numbers = 3
    with tf.Session() as sess:
        try:
            sess.run(atom_itr.initializer)
            # Exit on a multiple of the number of atomic numbers.
            for _ in range(unique_atomic_numbers * 10):
                atom_example = sess.run(next_example)
                atomic_number = atom_example[1]
                # Count atomic number appearance.
                if atomic_number not in atomic_number_counts:
                    atomic_number_counts[atomic_number] = 1
                else:
                    atomic_number_counts[atomic_number] += 1
                print(atom_example)
        except tf.errors.OutOfRangeError:
            pass
    for atomic_number, count in atomic_number_counts.items():
        print("atomic number: {}, count: {}".format(atomic_number, count))
    assert len(atomic_number_counts.keys()) == unique_atomic_numbers
    # Check that all counts are the same.
    atomic_number_counts_values = np.asarray(atomic_number_counts.values())
    assert (
        atomic_number_counts_values.max() == atomic_number_counts_values.min()
    )
