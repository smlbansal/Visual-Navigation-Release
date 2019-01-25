import os
import json
import copy
import numpy as np
import tensorflow as tf
import dotmap
import shutil
from dotmap import DotMap


"""
This file contains some hacky ways to convert the waypoints trained on one set of camera parameters to be executed a
different set of camera parameters.
"""

params = DotMap()
params.nn = DotMap(f=0.01, h=0.8, t=45.0*np.pi/180.0, half_fov_x=45.0*np.pi/180.0, half_fov_y=45.0*np.pi/180.0)
params.robot = DotMap(f=0.01, h=0.8, t=45.0*np.pi/180.0, half_fov_x=25.0*np.pi/180.0, half_fov_y=25.0*np.pi/180.0)


def project_to_image_plane_as_per_nn_parameters(wx_n, wy_n):
    den = wx_n * np.cos(params.nn.t) + params.nn.h * np.sin(params.nn.t)
    wx_image_n = -params.nn.f * wy_n / den
    wy_image_n = -params.nn.f * (wx_n * np.sin(params.nn.t) - params.nn.h * np.cos(params.nn.t)) / den
    return wx_image_n, wy_image_n


def project_to_world_frame_as_per_robot_parameters(wx_n, wy_n):
    den = wy_n * np.cos(params.robot.t) + params.robot.f * np.sin(params.robot.t)
    wx_world_n = params.robot.h * (params.robot.f * np.cos(params.robot.t) - wy_n * np.sin(params.robot.t)) / den
    wy_world_n = -wx_n * params.robot.h / den
    return wx_world_n, wy_world_n


def convert_waypoint_from_nn_to_robot(wx_n, wy_n, wtheta_n=None):
    # Convert the waypoint to the image space according to NN camera parameters
    wx_image_n, wy_image_n = project_to_image_plane_as_per_nn_parameters(wx_n, wy_n)
    
    # Normalize as per the image space size
    wx_image_n = wx_image_n / (params.nn.f * np.tan(params.nn.half_fov_x))
    wy_image_n = wy_image_n / (params.nn.f * np.tan(params.nn.half_fov_y))
    
    # Convert back in the image coordinates of the robot
    wx_image_n = wx_image_n * (params.robot.f * np.tan(params.robot.half_fov_x))
    wy_image_n = wy_image_n * (params.robot.f * np.tan(params.robot.half_fov_y))
    
    # Project the waypoint according to the robot camera parameters
    wx_world_n, wy_world_n = project_to_world_frame_as_per_robot_parameters(wx_image_n, wy_image_n)
    
    return wx_world_n, wy_world_n, wtheta_n
