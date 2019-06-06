# -*- coding: utf-8 -*-
from utils import profile_with_pytinstrument
import pytest
import numpy as np
import tensorflow as tf
from tensorflow_cif_parser.data.augmentation.rotation\
    import random_uniform_to_3d_rotation_mtx
from tensorflow_cif_parser.data.augmentation.rotation\
    import random_3d_rotation


@pytest.mark.parametrize(
    "test_loops",
    [
        10,
    ]
)
@profile_with_pytinstrument
def test_random_uniform_to_3d_rotation_mtx(test_loops):
    """Generates random rotation matrices and tests that they are all
    different. Test complexity is O(N^2) due to the pairwise comparisons.
    Parameters
    ----------
    test_loops: int
        Number of rotation matrices to generate and compare. Must be > 1.
    """
    assert test_loops > 1
    # Create random uniform values.
    uniform_distribution = tf.distributions.Uniform()
    theta_phi_z = uniform_distribution.sample([3, 1])
    rot_mtx = random_uniform_to_3d_rotation_mtx(theta_phi_z)
    print(theta_phi_z)
    print(rot_mtx)
    with tf.Session() as sess:
        generated_rotations = []
        # Generate 10 rotation matrices and test that they are all different.
        for _ in range(test_loops):
            generated_rotations.append(
                sess.run(rot_mtx)
            )
            print(generated_rotations[-1])
            for rotation_matrix in generated_rotations[:-1]:
                assert not np.allclose(
                    rotation_matrix,
                    generated_rotations[-1]
                )


@pytest.mark.parametrize(
    "coordinate_count",
    [
        1,
        100,
        1000
    ]
)
@profile_with_pytinstrument
def test_random_3d_rotation(coordinate_count):
    """Generates a random set of coordinates and applies a random rotation.
    Test passes if coordinates are not in the same location after rotation.
    Parameters
    ----------
    coordinate_count: int
        Number of coordinates to generate in test. Must be > 0.
    """
    assert coordinate_count > 0
    RandomNormal = tf.distributions.Normal(0.0, 50.0)
    coordinates = RandomNormal.sample([coordinate_count, 3])
    rotated_coordinates = random_3d_rotation(coordinates)
    with tf.Session() as sess:
        # Get coordinates and their rotations.
        coords, rotated_coords = sess.run(
            (
                coordinates,
                rotated_coordinates
            )
        )
        print("Coordinates:")
        print(coords)
        print("Rotated Coordinates:")
        print(rotated_coords)
        assert not np.allclose(
            coords,
            rotated_coords
        )
