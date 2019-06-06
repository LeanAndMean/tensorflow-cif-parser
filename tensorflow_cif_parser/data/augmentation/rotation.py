# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def random_uniform_to_3d_rotation_mtx(theta_phi_z, deflection=1.0):
    """Generates a 3D random rotation matrix from a uniform distribution [0,1).
    Parameters
    ----------
    theta_phi_z: tf.Tensor
        Random numbers used to generate the rotation matrix.
            shape: (3, 1)
            dtype: tf.float32
    deflection: float or tf.Tensor
        A scalar in the range [0, 1] that constrains the rotation as a
        function of the deflection magnitude. The default value of 1
        corresponds to no constraint (full deflection is possible).
    Returns
    -------
    tf.Tensor
        Rotation matrix.
            shape: (3, 3)
            dtype: tf.float32
    """
    theta, phi, z = tf.unstack(theta_phi_z)
    pi_const = tf.constant(np.pi, dtype=tf.float32)
    theta = theta * 2.0 * deflection * pi_const
    phi = phi * 2.0 * pi_const
    z = z * 2.0 * deflection
    r = tf.sqrt(z)
    V = tf.stack([tf.sin(phi)*r, tf.cos(phi)*r, tf.sqrt(2.0-z)])
    st = tf.sin(theta)
    ct = tf.cos(theta)
    R = tf.stack(
        [
            ct,
            st,
            tf.zeros(
                [1],
                dtype=tf.float32
            ),
            -st,
            ct,
            tf.zeros(
                [1],
                dtype=tf.float32
            ),
            tf.zeros(
                [1],
                dtype=tf.float32
            ),
            tf.zeros(
                [1],
                dtype=tf.float32
            ),
            tf.ones(
                [1],
                dtype=tf.float32
            )
        ]
    )
    R = tf.reshape(R, [3, 3])
    a = tf.reshape(V, [3, 1]) * tf.reshape(V, [1, 3]) - tf.eye(3)
    M = tf.matmul(a, R)
    return M


def random_3d_rotation(coords):
    """Generate a random rotation and apply to all coordinates.
    Parameters
    ----------
    coords: tf.Tensor
        Coorinates to rotate.
            shape: (None, 3)
            dtype: tf.float32 .
    Returns
    -------
    tf.Tensor
        Rotated coorinates.
            shape: (None, 3)
            dtype: tf.float32 .
    """
    uniform_distribution = tf.distributions.Uniform()
    theta_phi_z = uniform_distribution.sample([3, 1])
    rot_mtx = random_uniform_to_3d_rotation_mtx(theta_phi_z)
    # The same rotation matrix must be applied to all points. The number of
    # points (x,y,z) varies. In order to have the tf.matmul work properly,
    # the rotation matrix must be duplicated so that there is one copy per
    # point.
    # The batch size is not known a priori, the tile function requires
    # a multiple to be specified.
    rot_mtx = tf.multiply(
        tf.tile(
            tf.expand_dims(tf.ones_like(coords), axis=-1),
            [1, 1, 3]
        ),
        tf.expand_dims(rot_mtx, axis=0)
    )
    rotated_coords = tf.matmul(
        rot_mtx,
        tf.expand_dims(coords, axis=-1)
    )
    return tf.reshape(
        rotated_coords,
        tf.shape(coords)
    )
