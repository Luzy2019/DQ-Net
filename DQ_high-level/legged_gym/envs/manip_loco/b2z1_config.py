# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""
B2-Z1 configuration for the DQ high-level environment.

This is used by the DQ high-level env (B2Z1Base) for the terrain config only.
It inherits the B1Z1 terrain from DQ_high-level's legged_gym, keeping the
DQ_Bench task setup unchanged.
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

from .b1z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO


class B2Z1RoughCfg(B1Z1RoughCfg):
    class my_train_name:
        train_name = 'b2_z1'

    class goal_ee(B1Z1RoughCfg.goal_ee):
        class sphere_center(B1Z1RoughCfg.goal_ee.sphere_center):
            x_offset = 0.18  # Relative to base
            z_invariant_offset = 0.8

        class ranges(B1Z1RoughCfg.goal_ee.ranges):
            pos_l = [0.45, 0.95]
            pos_y = [-0.75, 0.75]

    class init_state(B1Z1RoughCfg.init_state):
        pos = [0.0, 0.0, 0.55]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,

            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,

            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,

            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,

            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "jointGripper": -1.57,
        }

    class control(B1Z1RoughCfg.control):
        stiffness = {
            "_joint": 250.0,
            "joint1": 50.0,
            "joint2": 50.0,
            "joint3": 80.0,
            "joint4": 30.0,
            "joint5": 30.0,
            "joint6": 20.0,
        }
        damping = {
            "_joint": 5.0,
            "joint1": 3.0,
            "joint2": 2.0,
            "joint3": 3.0,
            "joint4": 3.0,
            "joint5": 2.5,
            "joint6": 1.0,
        }
        action_scale = [0.35] * 12 + [0.25] * 6

    class asset(B1Z1RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b2_z1/urdf/b2_z1.urdf'
        base_name = "base_link"
        foot_name = "foot"
        gripper_name = "gripperMover"
        penalize_contacts_on = ["thigh", "base_link", "calf"]
        collapse_fixed_joints = False

    class arm(B1Z1RoughCfg.arm):
        base_offset = [0.0, 0.0, 0.09]

    class rewards(B1Z1RoughCfg.rewards):
        base_height_target = 0.48


class B2Z1RoughCfgPPO(B1Z1RoughCfgPPO):
    class runner(B1Z1RoughCfgPPO.runner):
        experiment_name = 'b2z1_v2'