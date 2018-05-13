#!/usr/bin/env python
# coding=utf-8

import PyKDL
from math import pi


def quat_to_angle(quat):
    # R: roll(回转角，绕X轴转)
    # P: pitch(俯仰角，绕Y轴转)
    # Y: yaw(偏转角，绕Z轴转)
    rot = PyKDL.Rotation.Quaternion(quat.x, quat.y, quat.z, quat.w)
    return rot.GetRPY()[2]


def normalize_angle(angle):
    res = angle

    while res > pi:
        res -= 2.0 * pi
    while res < -pi:
        res += 2.0 * pi

    return res
