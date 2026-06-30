#!/usr/bin/env python3
"""Pose interpolation between two frames.

Given a target time and two frames (each a timestamp, translation, and
quaternion), compute the pose at the target time.  Translation is linearly
interpolated; rotation uses SLERP.
"""

import numpy as np


def interpolate_pose(t1, p1, q1, t2, p2, q2, target_t):
    """Interpolate the pose at *target_t* between two frames.

    Args:
        t1:       timestamp of the first frame.
        p1:       (3,) translation of the first frame.
        q1:       (x, y, z, w) quaternion of the first frame.
        t2:       timestamp of the second frame.
        p2:       (3,) translation of the second frame.
        q2:       (x, y, z, w) quaternion of the second frame.
        target_t: target timestamp (must be between t1 and t2).

    Returns:
        (translation, quaternion) — interpolated (3,) translation and
        (4,) quaternion [x, y, z, w].
    """
    alpha = (target_t - t1) / (t2 - t1)

    # Linear interpolation for translation
    p_interp = p1 + alpha * (p2 - p1)

    # SLERP for rotation
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        q_interp = q1 + alpha * (q2 - q1)
        q_interp /= np.linalg.norm(q_interp)
    else:
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * alpha
        s1 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
        s2 = np.sin(theta) / sin_theta_0
        q_interp = s1 * q1 + s2 * q2

    return p_interp, q_interp