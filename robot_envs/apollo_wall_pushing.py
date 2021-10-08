import os
import pybullet_utils.bullet_client as bc
import pybullet
import time
import pybullet_data
import numpy as np
import gym
import gym.spaces
from gym.spaces import Box
from utils.rewards import *
from utils.data_logging import Log, ListOfLogs, DataCollector
import os
from matplotlib import animation, rc
import matplotlib.pyplot as plt
from IPython.display import HTML
import yaml
import json
from utils.plotting import prepare_plot
from scipy.stats import norm
import uuid

import json
import argparse

from robot_envs.position_gain_controller import PositionGainController
from robot_envs.hopper.hopping_rewards import TrajectoryTrackingReward

class Quaternion:

    @staticmethod
    def norm(v):
        return np.sqrt(np.sum(np.square(v)))
    
    @staticmethod
    def quat_prod(q, r):
        t = np.zeros(4)
        t[0] = q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3]
        t[1] = q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2]
        t[2] = q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1]
        t[3] = q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0]
        if Quaternion.norm(t) == 0:
            return np.array([0, 0, 0, 0])
        return t / Quaternion.norm(t)

    @staticmethod
    def quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def quat_log(q):
        if Quaternion.norm(q[1:]) == 0:
            return np.array([0, 0, 0])
        return 2 * np.arccos(q[0]) / Quaternion.norm(q[1:]) * q[1:]

    @staticmethod
    def quat_exp(v):
        t = np.sin(0.5 * Quaternion.norm(v)) / Quaternion.norm(v) * v
        return np.array([np.cos(0.5 * Quaternion.norm(v)), t[0], t[1], t[2]])

    @staticmethod
    def quat_dist(q, r):
        return 2 * Quaternion.quat_log(Quaternion.quat_prod(q, Quaternion.quat_conj(r)))


class VelocityReward():

    def __init__(self, reacher, k_v, square=True):
        self.reacher = reacher
        self.k_v = k_v
        self.square = square

    def get_reward(self):
        _, joint_vel = self.reacher.get_arm_state()
        value = np.linalg.norm(joint_vel) ** 2
        if self.square:
            return -self.k_v * value
        else:
            # Multiplying with 0.1 to stay 100% consistent with
            # the roboschool environment
            return self.k_v * 1.0 / np.exp(0.05 * value)

    def is_done(self):
        joint_pos, joint_vel = self.reacher.get_arm_state()
        # Multiplying with 0.1 to stay 100% consistent with
        # the roboschool environment
        #return np.linalg.norm(0.1 * joint_vel) ** 2 < 1e-3
        return False


class ActionPenalty():

    def __init__(self, get_action, k_a):
        self.get_action = get_action
        self.k_a = k_a

    def get_reward(self):
        action = self.get_action()
        return -self.k_a * np.sum(np.absolute(action)) / action.shape[0]
    
    def is_done(self):
        return True


class ElbowPenalty():

    def __init__(self, get_elbow_force, k_e):
        self.get_elbow_force = get_elbow_force
        self.k_e = k_e

    def get_reward(self):
        f = np.linalg.norm(self.get_elbow_force())
        if f > 1e-6:
            if f > 100.0:
                f = 100.0
            return -self.k_e * 0.5 * (2.0 - 1.0 / np.exp(0.2 * f))
        else:
            return 0.0

    def is_done(self):
        return False


class WrongSidePenalty():

    def __init__(self, get_endeff_force_z, k_w):
        self.get_endeff_force_z = get_endeff_force_z
        self.k_w = k_w

    def get_reward(self):
        f = self.get_endeff_force_z()
        if f < -1e-6:
            if f > 100.0:
                f = 100.0
            return -self.k_w * 0.5 * (2.0 - 1.0 / np.exp(-0.2 * f))
        else:
            return 0.0

    def is_done(self):
        return False


def exp_rew(x, x_range, y_range, curve, flipped=False):
    def f(x):
        return np.exp(curve * x)

    def g(x):
        return (f(x) - f(0)) / (f(1) - f(0))

    if flipped:
        return y_range[1] - g((x_range[1] - x) / (x_range[1] - x_range[0])) * (y_range[1] - y_range[0])
    else:
        return y_range[0] + g((x - x_range[0]) / (x_range[1] - x_range[0])) * (y_range[1] - y_range[0])

class DesiredForceReward():

    def __init__(self, get_endeff_force_z, get_endeff_vel, k_f, goal_force, clipping_vel, alt_reward=False,
        version='old'):
        self.get_endeff_force_z = get_endeff_force_z
        self.get_endeff_vel = get_endeff_vel
        self.k_f = k_f
        self.goal_force = goal_force
        self.clipping_vel = clipping_vel
        self.alt_reward = alt_reward
        self.version = version

    def get_reward(self):
        f = self.get_endeff_force_z()
        vz = self.get_endeff_vel()[2]
        if f > 1e-6:
            if self.clipping_vel is not None and np.absolute(vz) > self.clipping_vel:
                return 0.0
            if f > 300.0:
                f = 300.0
            if self.version == 'v1' or self.version == 'v2' or self.version == 'v3':
                normalized_diff = np.absolute(self.goal_force - f) / self.goal_force
                if self.version == 'v1':
                    return 2.0 + 6.0 * exp_rew(normalized_diff, [0, 1], [1, 0], 2, True)
                elif self.version == 'v2':
                    return 2.0 + 14.0 * exp_rew(normalized_diff, [0, 1], [1, 0], 5, True)
                else:
                    return 1.0 + 7.0 * exp_rew(normalized_diff, [0, 1], [1, 0], 5, True)
            else:
                if self.alt_reward:
                    if f < self.goal_force:
                        scaling = 3.0
                    else:
                        scaling = 0.75
                    return self.k_f * (0.4 + 0.6 * np.exp(-np.abs((f - self.goal_force) / self.goal_force) * scaling))
                else:
                    f_error = np.abs(f - self.goal_force) / self.goal_force
                    if self.version == 'new':
                        if f < self.goal_force:
                            scaling = 2.0
                            return self.k_f * (0.5 + 0.5 * (np.exp(scaling * (1.0 - f_error)) - 1.0) / (np.exp(scaling) - 1.0))
                        else:
                            scaling = 1.0
                            return self.k_f * (0.5 + 0.5 * np.exp(-f_error * scaling))
                    else:
                        if f < self.goal_force:
                            scaling = 2.0
                        else:
                            scaling = 0.2
                        return self.k_f * (0.5 + 0.5 * np.exp(-f_error * scaling))
        else:
            return 0.0


    def is_done(self):
        return False


class VelocityPenalty():

    def __init__(self, robot, k_v, square=True):
        self.robot = robot
        self.k_v = k_v
        self.square = square

    def get_reward(self):
        _, joint_vel = self.robot.get_arm_state()
        joint_vel = np.array(joint_vel)
        return -self.k_v * np.square(np.linalg.norm(joint_vel))
        #return -self.k_v * np.sum(np.square(joint_vel)) / joint_vel.shape[0]
        #normed_joint_vel = joint_vel / np.array(self.robot.vel_limit)
        #mean_v_square = np.linalg.norm(joint_vel) ** 2 / self.robot.NUM_JOINTS
        #if self.square:
        #    return -self.k_v * mean_v_square
        #else:
        #    # Multiplying with 0.1 to stay 100% consistent with
        #    # the roboschool environment
        #    #return self.k_v * 1.0 / np.exp(0.05 * value)
        #    assert False, 'Not using this reward type any more.'

    def is_done(self):
        joint_pos, joint_vel = self.robot.get_arm_state()
        # Multiplying with 0.1 to stay 100% consistent with
        # the roboschool environment
        #return np.linalg.norm(0.1 * joint_vel) ** 2 < 1e-3
        return False


class TablePointReward():

    def __init__(self, get_endeff_pos, k_p, goal_pos, table_height, extended=False, full_dist=False,
        table_thickness=0.05, scale=10.0):
        self.get_endeff_pos = get_endeff_pos
        self.k_p = k_p
        self.goal_pos = goal_pos
        self.table_height = table_height
        self.extended = extended
        self.full_dist = full_dist
        self.table_thickness = table_thickness
        self.scale = scale

    def get_reward(self):
        endeff_pos = self.get_endeff_pos()
        if endeff_pos[2] < self.table_height:
            return 0.0
        else:
            if self.full_dist:
                goal_point = np.array([self.goal_pos[0], self.goal_pos[1], self.table_height + 0.13])
                dist = np.linalg.norm(goal_point - endeff_pos)
            else:
                dist = np.linalg.norm(self.goal_pos - endeff_pos[:2])
            if dist > 3.0:
                dist = 3.0
            value = 1.0 - dist / 3.0
            if self.extended:
                return self.k_p * (np.exp(self.scale * value) / np.exp(self.scale) +  + np.exp(100.0 * value) / np.exp(100.0))
            else:
                return self.k_p * np.exp(self.scale * value) / np.exp(self.scale)

    def is_done(self):
        return False


class OrientationReward():

    def __init__(self, get_endeff_xy_angle, k_o, scale=3.0):
        self.get_endeff_xy_angle = get_endeff_xy_angle
        self.k_o = k_o
        self.scale = scale

    def get_reward(self):
        # Gets an angle in range [-pi/2, pi/2]
        x = 1.0 - self.get_endeff_xy_angle() / (np.pi / 2.0)
        return self.k_o * np.exp(self.scale * x) / np.exp(self.scale * 2.0)

    def is_done(self):
        return False


class TorquePenalty():

    def __init__(self, robot, k_t):
        self.robot = robot
        self.k_t = k_t

    def get_reward(self):
        max_torque = np.array(self.robot.max_torque)
        torque = self.robot.get_torque()
        normed_torque = torque / (self.robot.max_torque_percent * max_torque)
        return -self.k_t * np.sum(np.absolute(normed_torque)) / self.robot.NUM_JOINTS

    def is_done(self):
        return True


class Circle():

    def __init__(self, circle_params):
        self.center = circle_params['center']
        self.radius = circle_params['radius']
        self.tan_velocity = circle_params['tan_velocity']
        self.clockwise = circle_params['clockwise']
        self.height = circle_params['height']

    def des_pos_vel(self, point):
        point_2d = point[:2]
        des_pos = self.center + (point_2d - self.center) / np.linalg.norm(point_2d - self.center) * self.radius
        v = (des_pos - self.center) / np.linalg.norm(des_pos - self.center)
        if self.clockwise:
            v = np.array([v[1], -v[0]])
        else:
            v = np.array([-v[1], v[0]])
        des_vel = self.tan_velocity * v
        des_pos_3d = np.array(des_pos.tolist() + [self.height])
        des_vel_3d = np.array(des_vel.tolist() + [0.0])
        return des_pos_3d, des_vel_3d


class CircleReward():

    def __init__(self, circle, get_state, k_t, log=None, use_alt_vel_reward=False, vel_rew_scale_with_dist=None, vel_mult=1.0):
        self.circle = circle
        self.get_state = get_state
        self.k_t = k_t
        self.log = log
        self.use_alt_vel_reward = use_alt_vel_reward
        self.vel_rew_scale_with_dist = vel_rew_scale_with_dist
        self.vel_mult = vel_mult

    def pos_reward(self, pos_diff, scale=10.0):
        max_diff = 2 * np.sqrt(2)
        value = 1.0 - pos_diff / max_diff
        return np.exp(scale * value) / np.exp(scale)

    def vel_reward(self, vel, des_vel):
        des_vel_S = np.linalg.norm(des_vel)
        diff = des_vel - vel
        diff_S = np.linalg.norm(diff)
        if self.log is not None:
            self.log.add('vel_diff', diff.tolist())
            self.log.add('des_vel_norm', des_vel_S)
        return 1.0 / np.exp(2.0 * diff_S / des_vel_S)

    def error_reward(self, x, k=5.0):
        return 1.0 / np.exp(k * x)

    def alt_vel_reward(self, vel, des_vel):
        des_vel_2D = des_vel[:2]
        des_vel_2D_S = np.linalg.norm(des_vel_2D)
        des_vel_2D_N = des_vel_2D / des_vel_2D_S

        vel_2D = vel[:2]

        vel_2D_TAN = np.dot(vel_2D, des_vel_2D_N) * des_vel_2D_N
        vel_2D_TAN_S = np.linalg.norm(vel_2D_TAN)
        vel_2D_NORM = vel_2D - vel_2D_TAN

        vel_TAN = np.array(vel_2D_TAN.tolist() + [0.0])
        vel_NORM = np.array(vel_2D_NORM.tolist() + [vel[2]])
        #print(vel, vel_NORM + vel_TAN)

        normal_reward = self.error_reward(np.linalg.norm(vel_NORM))
        tangential_reward = self.error_reward(np.abs(des_vel_2D_S - vel_2D_TAN_S))

        if self.log is not None:
            self.log.add('normal_reward', normal_reward)
            self.log.add('tangential_reward', tangential_reward)

        return normal_reward + tangential_reward

    def get_reward(self):
        pos, vel = self.get_state()

        des_pos, des_vel = self.circle.des_pos_vel(pos)

        dist_to_des = np.linalg.norm(des_pos - pos)
        pos_reward = self.pos_reward(dist_to_des)
        if self.use_alt_vel_reward:
            vel_reward = self.alt_vel_reward(vel, des_vel)
        else:
            vel_reward = self.vel_reward(vel, des_vel)
        if self.vel_rew_scale_with_dist is not None:
            vel_reward *= (np.tanh(100.0 * (-dist_to_des + self.vel_rew_scale_with_dist * self.circle.radius)) + 1.0) / 2.0
        vel_reward *= self.vel_mult

        if self.log is not None:
            self.log.add('pos_reward', pos_reward)
            self.log.add('vel_reward', vel_reward)
            self.log.add('des_pos', des_pos.tolist())
            self.log.add('des_vel', des_vel.tolist())

        return self.k_t * (pos_reward + vel_reward)

    def is_done(self):
        return False


class ConfigurationPenalty():

    def __init__(self, get_joint_pos, joint_limits, k_c):
        self.get_joint_pos = get_joint_pos
        self.joint_limits = joint_limits
        self.k_c = k_c
        self.num_joints = len(joint_limits)

    def get_reward(self):
        penalty = 0.0
        joint_pos = self.get_joint_pos()
        for i in range(self.num_joints):
            normalized = (joint_pos[i] - joint_limits[i][0]) / (joint_limits[i][1] - joint_limits[i][0]) * 2.0 - 1.0
            penalty += (norm.pdf(normalized) - norm.pdf(0.0)) / (norm.pdf(0.0) - norm.pdf(1.0))
        return self.k_c / self.num_joints * penalty

    def is_done(self):
        return False


class AccControl():

    def __init__(self, robot, k_p, k_v):
        self.robot = robot
        self.k_p = k_p
        self.k_v = k_v

        self.robot.init_torque_control()

    def reset(self):
        self.des_pos = 0
        self.des_vel = 0

    def clip_torque(self, torque, percent):
        max_torque = np.array(self.robot.max_torque)
        return np.clip(torque, -percent * max_torque, percent * max_torque)

    def act(self, action):
        curr_pos, curr_vel = self.robot.get_arm_state()
        des_acc = action * np.array(self.robot.vel_limit) / self.robot.dt / 10.0

        self.des_vel += des_acc * self.robot.dt
        self.des_pos += self.des_vel * self.robot.dt
        torque = self.k_p * (self.des_pos - curr_pos) \
            + self.k_v * (self.des_vel - curr_vel)

        torque = self.clip_torque(torque, 0.1) + self.robot.inv_dyn(des_acc)

        self.robot.torque_control(torque)


class VelControl():

    def __init__(self, robot, k_p, k_v):
        self.robot = robot
        self.k_p = k_p
        self.k_v = k_v

        self.robot.init_torque_control()

    def reset(self):
        self.des_pos = 0

    def clip_torque(self, torque, percent):
        max_torque = np.array(self.robot.max_torque)
        return np.clip(torque, -percent * max_torque, percent * max_torque)

    def act(self, action):
        curr_pos, curr_vel = self.robot.get_arm_state()
        des_vel = action * np.array(self.robot.vel_limit)

        self.des_pos += des_vel * self.robot.dt
        torque = self.k_p * (self.des_pos - curr_pos) \
            + self.k_v * (des_vel - curr_vel)

        torque = self.clip_torque(torque, 0.1) + self.robot.inv_dyn(np.zeros(curr_pos.shape))

        self.robot.torque_control(torque)


class PosControl():

    def __init__(self, robot, k_p, k_v, diff_control):
        self.robot = robot
        self.k_p = k_p
        self.k_v = k_v
        self.diff_control = diff_control

        self.robot.init_torque_control()

    def reset(self):
        pass

    def clip_torque(self, torque, percent):
        max_torque = np.array(self.robot.max_torque)
        return np.clip(torque, -percent * max_torque, percent * max_torque)

    def act(self, action):
        if not self.diff_control:
            des_pos = np.array([self.robot.joint_limits[i][0] + ((action[i] + 1.0) / 2.0 * (self.robot.joint_limits[i][1] - self.robot.joint_limits[i][0])) for i in range(self.robot.NUM_JOINTS)])
        else:
            joint_pos, joint_vel = self.robot.get_arm_state()
            des_pos = joint_pos + action
        curr_pos, curr_vel = self.robot.get_arm_state()

        torque = self.k_p * (des_pos - curr_pos) \
            + self.k_v * (-curr_vel)

        torque = self.clip_torque(torque, 0.1) + self.robot.inv_dyn(np.zeros(curr_pos.shape))

        self.robot.torque_control(torque)


class TorqueControl():

    def __init__(self, robot, grav_comp=True, clip_based_on_velocity=False):
        self.robot = robot
        self.grav_comp = grav_comp
        self.clip_based_on_velocity = clip_based_on_velocity
        #print('VEL CLIPPING:', self.clip_based_on_velocity)

        self.robot.init_torque_control()

    def get_control_space(self):
        return Box(-1.0, 1.0, (self.robot.NUM_JOINTS,))

    def reset(self):
        pass

    def act(self, action):
        torque = np.array([self.robot.max_torque_percent * self.robot.max_torque[i] * action[i] for i in range(action.shape[0])])

        if self.grav_comp:
            torque += self.robot.inv_dyn(np.zeros(torque.shape))

        if self.clip_based_on_velocity:
            _, joint_vel = self.robot.get_arm_state()
            for i in range(self.robot.NUM_JOINTS):
                if joint_vel[i] < -self.robot.vel_limit[i] and torque[i] < 0.0:
                    torque[i] = 0.0
                if joint_vel[i] > self.robot.vel_limit[i] and torque[i] > 0.0:
                    torque[i] = 0.0
        
        self.robot.torque_control(torque)


class PositionControl():

    def __init__(self, robot, diff_control=False):
        self.robot = robot
        self.diff_control = diff_control

    def reset(self):
        pass

    def act(self, action):
        if not self.diff_control:
            des_pos = np.array([self.robot.joint_limits[i][0] + ((action[i] + 1.0) / 2.0 * (self.robot.joint_limits[i][1] - self.robot.joint_limits[i][0])) for i in range(self.robot.NUM_JOINTS)])
        else:
            joint_pos, joint_vel = self.robot.get_arm_state()
            des_pos = joint_pos + action

        self.robot.position_control(des_pos)





class ApolloWallPushing(gym.Env):

    def __init__(self, table_height=None, reward_specs={}, initial_pushee_pos=None,
        observable=[], visualize=False, exp_name=None, output_dir='',
        initial_joint_state=None, fixed_timestep=None, full_log=False,
        max_velocity=None, log_file=None, controller_params={'type': 'torque'}, reduced_init=False,
        sim_steps_per_action=1, max_torque_percent=0.1,
        joint_friction_percent=0.0, joint_damping_percent=0.0, joint_5_torque_mult=1.0,
        endeff_link_id=9, collect_data=False, pack_size=100, packs_to_keep=20, lateral_friction=None,
        contact_stiffness=None, contact_damping=None, contact_damping_multiplier=None,
        cont_timestep_mult=1, logging=False):

        self.observable = observable
        self.visualize = visualize
        self.initial_pushee_pos = initial_pushee_pos
        self.initial_joint_state = initial_joint_state
        self.full_log = full_log

        if isinstance(table_height, list):
            self.random_table_height = True
            self.table_height_range = table_height
            self.table_height = np.random.uniform(self.table_height_range[0], self.table_height_range[1])
        else:
            self.random_table_height = False
            self.table_height = table_height

        if isinstance(lateral_friction, list):
            self.random_lateral_friction = True
            self.lateral_friction_range = lateral_friction
            self.lateral_friction = np.random.uniform(self.lateral_friction_range[0], self.lateral_friction_range[1])
        else:
            self.random_lateral_friction = False
            self.lateral_friction = lateral_friction

        if isinstance(contact_stiffness, list):
            self.random_contact_stiffness = True
            self.contact_stiffness_range = contact_stiffness
            self.contact_stiffness = np.random.uniform(self.contact_stiffness_range[0], self.contact_stiffness_range[1])
        else:
            self.random_contact_stiffness = False
            self.contact_stiffness = contact_stiffness

        self.fixed_timestep = fixed_timestep
        self.max_velocity = max_velocity
        self.reduced_init = reduced_init
        self.exp_name = exp_name
        self.logging = logging and ((exp_name is not None) or (log_file is not None))
        self.sim_steps_per_action = sim_steps_per_action
        self.max_torque_percent = max_torque_percent
        self.joint_friction_percent = joint_friction_percent
        self.joint_damping_percent = joint_damping_percent
        self.endeff_link_id = endeff_link_id

        self.collect_data = collect_data
        if self.collect_data:
            self.data_collector = DataCollector(self.exp_name + '_packs', pack_size, packs_to_keep)

        if self.fixed_timestep is not None:
            self.dt = self.fixed_timestep
        else:
            self.dt = 1./240.

        if self.visualize:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.p.setGravity(0,0,-10)
        if fixed_timestep is not None:
            self.p.setPhysicsEngineParameter(fixedTimeStep=fixed_timestep)
        self.generate_urdf()
        #print('\n\n\n' + self.urdf_file + '\n\n\n')
        #robot = self.p.loadURDF(self.urdf_file)
        #pybullet.setAdditionalSearchPath('/Users/miroslav/work/experiments_local_runs/117_ppo_reaching_baseline/100/')
        #path2 = os.path.join(os.path.dirname(__file__), 'apollo_minimal.urdf')
        #path1 = '/Users/miroslav/work/code/robot_envs/robot_envs/apollo_minimal.urdf'
        #self.p.setAdditionalSearchPath('/Users/miroslav/work/experiments_local_runs/117_ppo_reaching_baseline/100/')
        #path3 = '/Users/miroslav/work/experiments_local_runs/117_ppo_reaching_baseline/100/apollo_minimal.urdf'
        robot = self.p.loadURDF(self.urdf_file)
        #print(path1)
        #print(path2)
        #assert False
        #print('TABLE', self.table_height)
        if self.table_height is not None:
            self.p.loadURDF(os.path.join(os.path.dirname(__file__), 'wall.urdf'))
            self.p.resetBasePositionAndOrientation(
                1, (0.0, -0.9, self.table_height), (0.0, 0.0, 0.0, 1.0))

        self.ARM_ID = 0
        self.ARM_JOINTS = [2, 3, 4, 5, 6, 7, 8]
        self.ENDEFF_ID = 12
        self.NUM_JOINTS = len(self.ARM_JOINTS)
        
        for joint_id in self.ARM_JOINTS:
            self.p.enableJointForceTorqueSensor(bodyUniqueId=self.ARM_ID,
                jointIndex=joint_id, enableSensor=1)
        
        self.joint_limits = []
        self.max_torque = []
        self.vel_limit = []

        for i in range(len(self.ARM_JOINTS)):
            joint_info = self.p.getJointInfo(self.ARM_ID, self.ARM_JOINTS[i])
            self.joint_limits.append([joint_info[8], joint_info[9]])
            self.max_torque.append(joint_info[10])
            self.vel_limit.append(joint_info[11])
            #print(joint_info[6], joint_info[7])

        self.max_torque[5] *= joint_5_torque_mult

        self.init_controller(controller_params)

        self.action_space = self.controller.get_control_space()
        obs_dim = len(self.get_state())
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

        if self.logging:
            if self.full_log:
                self.log = ListOfLogs(exp_name + '_episodes', separate_files=True)
            else:
                if log_file is not None:
                    self.log = Log(log_file)
                else:
                    self.log = Log(exp_name + '_episodes')
        else:
            self.log = None

        self.init_reward(reward_specs)

        self.n_dof = 3

        self.start_time = time.time()
        self.timestep_times = []

        if self.lateral_friction is not None:
            for i in range(self.p.getNumJoints(0)):
                self.p.changeDynamics(0, i, lateralFriction=self.lateral_friction)
            self.p.changeDynamics(1, -1, lateralFriction=self.lateral_friction)

        self.contact_damping = contact_damping
        self.contact_damping_multiplier = contact_damping_multiplier

        if self.random_contact_stiffness:
            self.contact_stiffness = np.random.uniform(self.contact_stiffness_range[0], self.contact_stiffness_range[1])

        if self.contact_stiffness is not None:
            if self.contact_damping_multiplier is not None:
                contact_damping = self.contact_damping_multiplier * 2.0 * np.sqrt(self.contact_stiffness)
            else:
                if self.contact_damping is None:
                    contact_damping = 2.0 * np.sqrt(self.contact_stiffness)
                else:
                    contact_damping = self.contact_damping

            self.p.changeDynamics(1, -1, contactStiffness=self.contact_stiffness, contactDamping=contact_damping)
            for i in range(self.p.getNumJoints(0)):
                self.p.changeDynamics(0, i, contactStiffness=self.contact_stiffness, contactDamping=contact_damping)

        self.cont_timestep_mult = cont_timestep_mult


    def generate_urdf(self):
        path = os.path.join(os.path.dirname(__file__), 'apollo_minimal_template.urdf')
        #print(path)
        with open(path, 'r') as f:
            data = f.read()

        max_torque = self.max_torque_percent * np.array([200.0, 200.0, 100.0, 100.0, 100.0, 30.0, 30.0])
        max_velocity = np.array([1.9634, 1.9634, 1.9634, 1.9634, 3.1415926, 1.9634, 1.9634])

        for i in range(7):
            data = data.replace('<DAMPING' + str(i) + '>', str(self.joint_damping_percent * max_torque[i] / max_velocity[i]))
            data = data.replace('<FRICTION' + str(i) + '>', str(self.joint_friction_percent * max_torque[i]))

        self.urdf_file = os.path.join(os.path.dirname(__file__), 'generated_urdf_files/' + uuid.uuid4().hex)
        with open(self.urdf_file, 'w') as f:
            f.write(data)

    def get_state(self):
        state = []
        joint_pos, joint_vel = self.get_arm_state()

        state += joint_pos.tolist()
        state += joint_vel.tolist()
        if 'endeff_pos' in self.observable:
            state += self.get_endeff_pos().tolist()
        if 'endeff_vel' in self.observable:
            state += self.get_endeff_vel().tolist()
        if 'endeff_force_z' in self.observable:
            state += [self.get_endeff_force_z()]
        if 'elbow_force_norm' in self.observable:
            state += [np.linalg.norm(self.get_elbow_force())]
        if 'pushee_state' in self.observable:
            pushee_pos, pushee_vel = self.get_pushee_state()
            state += pushee_pos.tolist()
            state += pushee_vel.tolist()
        if 'joint_loads' in self.observable:
            state += self.get_joint_loads().tolist()
        if 'endeff_force' in self.observable:
            state += self.get_endeff_force().tolist()
        return np.array(state)

    def get_arm_state(self):
        joint_pos = np.zeros(len(self.ARM_JOINTS))
        joint_vel = np.zeros(len(self.ARM_JOINTS))
        for i in range(len(self.ARM_JOINTS)):
            joint_pos[i], joint_vel[i], _, _ = self.p.getJointState(self.ARM_ID, self.ARM_JOINTS[i])
        return joint_pos, joint_vel

    def get_joint_pos(self):
        joint_pos, joint_vel = self.get_arm_state()
        return joint_pos

    def get_pushee_state(self):
        pushee_state = self.p.getLinkState(self.ARM_ID, 6, computeLinkVelocity=1)
        pushee_pos = np.array(self.p.getBasePositionAndOrientation(8)[0][:2])
        pushee_vel = np.array(self.p.getBaseVelocity(8)[0][:2])
        return pushee_pos, pushee_vel

    def get_endeff_state(self):
        endeff_state = self.p.getLinkState(self.ARM_ID, self.endeff_link_id, computeLinkVelocity=1)
        endeff_pos = np.array(endeff_state[0])
        endeff_vel = np.array(endeff_state[6])
        return endeff_pos, endeff_vel

    def get_joint_loads(self):
        joint_load = np.zeros(len(self.ARM_JOINTS))
        for i in range(len(self.ARM_JOINTS)):
            _, _, joint_force_torque, _ = self.p.getJointState(self.ARM_ID, self.ARM_JOINTS[i])
            joint_load[i] = joint_force_torque[5]
        return joint_load

    def get_total_force(self, links):
        contacts = self.p.getContactPoints(bodyA=1, bodyB=0)
        total_force = np.zeros(3)
        for contact in contacts:
            if contact[4] in links:
                contact_normal = np.array(contact[7])
                normal_force = contact[9]
                total_force += normal_force * contact_normal
        return total_force

    def get_endeff_force(self):
        return self.get_total_force([self.endeff_link_id])

    def get_endeff_force_z(self):
        return self.get_endeff_force()[2]

    def get_elbow_force(self):
        return self.get_total_force(self.ARM_JOINTS)

    def check_contacts(self):
        links_a = []
        links_b = []
        contacts = self.p.getContactPoints(bodyA=1, bodyB=0)
        for contact in contacts:
            links_a.append(contact[3])
            links_b.append(contact[4])
        #print(links_a, links_b)

    def get_pushee_pos(self):
        pushee_pos, pushee_vel = self.get_pushee_state()
        return pushee_pos

    def get_endeff_pos(self):
        endeff_pos, endeff_vel = self.get_endeff_state()
        return endeff_pos

    def get_endeff_vel(self):
        endeff_pos, endeff_vel = self.get_endeff_state()
        return endeff_vel

    def get_endeff_orient(self):
        #print('endeff_link_id:', self.endeff_link_id)
        return np.array(self.p.getLinkState(self.ARM_ID, self.endeff_link_id, computeLinkVelocity=1)[1])

    def get_endeff_xy_angle(self):
        rot = np.array(self.p.getMatrixFromQuaternion(self.get_endeff_orient())).reshape((3, 3))
        z = np.dot(rot, np.array([[0], [0], [1]])).reshape((3,))
        return np.arcsin(z[2])

    def get_action(self):
        return self.latest_action

    def get_torque(self):
        return self.latest_torque


    def init_reward(self, rewards_config):
        self.reward_parts = {}
        for reward_type, reward_spec in rewards_config.items():
            if reward_type == 'desired_force':
                alt_reward = False
                if 'alt_reward' in reward_spec:
                    alt_reward = reward_spec['alt_reward']
                version = 'old'
                if 'version' in reward_spec:
                    version = reward_spec['version']
                self.reward_parts[reward_type] = DesiredForceReward(self.get_endeff_force_z, self.get_endeff_vel, reward_spec['k_f'], reward_spec['goal_force'], reward_spec['clipping_vel'], alt_reward, version=version)
            elif reward_type == 'elbow_penalty':
                self.reward_parts[reward_type] = ElbowPenalty(self.get_elbow_force, reward_spec['k_e'])
            elif reward_type == 'wrong_side_penalty':
                self.reward_parts[reward_type] = ElbowPenalty(self.get_endeff_force_z, reward_spec['k_w'])
            elif reward_type == 'velocity_penalty':
                self.reward_parts[reward_type] = VelocityPenalty(self, reward_spec['k_v'])
            elif reward_type == 'table_point':
                full_dist = False
                if 'full_dist' in reward_spec:
                    full_dist = reward_spec['full_dist']
                scale = 10.0
                if 'scale' in reward_spec:
                    scale = reward_spec['scale']
                self.reward_parts[reward_type] = TablePointReward(self.get_endeff_pos, reward_spec['k_p'], reward_spec['goal_pos'], self.table_height, full_dist=full_dist, scale=scale)
            elif reward_type == 'orientation':
                scale = 3.0
                if 'scale' in reward_spec:
                    scale = reward_spec['scale']
                self.reward_parts[reward_type] = OrientationReward(self.get_endeff_xy_angle, reward_spec['k_o'], scale)
            elif reward_type == 'torque_penalty':
                self.reward_parts[reward_type] = TorquePenalty(self, reward_spec['k_t'])
            elif reward_type == 'circle':
                circle = Circle(reward_spec)
                use_alt_vel_reward = False
                if 'use_alt_vel_reward' in reward_spec:
                    use_alt_vel_reward = reward_spec['use_alt_vel_reward']
                vel_rew_scale_with_dist = None
                if 'vel_rew_scale_with_dist' in reward_spec:
                    vel_rew_scale_with_dist = reward_spec['vel_rew_scale_with_dist']
                vel_mult = 1.0
                if 'vel_mult' in reward_spec:
                    vel_mult = reward_spec['vel_mult']
                self.reward_parts[reward_type] = CircleReward(circle, self.get_endeff_state, reward_spec['k_t'], self.log, use_alt_vel_reward, vel_rew_scale_with_dist, vel_mult)
            # DEPRECATED REWARD TYPES FOR COMPATIBILITY
            elif reward_type == 'velocity':
                self.reward_parts[reward_type] = VelocityReward(self, reward_spec['k_v'])
            elif reward_type == 'action_penalty':
                self.reward_parts[reward_type] = ActionPenalty(self.get_action, reward_spec['k_a'])
            elif reward_type == 'configuration_penalty':
                self.reward_parts[reward_type] = ConfigurationPenalty(self.get_joint_pos, self.joint_limits, reward_spec['k_c'])
            elif reward_type == 'trajectory_tracking_reward':
                self.reward_parts[reward_type] = TrajectoryTrackingReward(self, reward_spec)
            else:
                assert False, 'Unknown reward type: ' + str(reward_type)


    def init_controller(self, controller_params):
        controller_type = controller_params['type']

        baseline = np.array([200.0, 200.0, 100.0, 100.0, 100.0, 30.0, 30.0])
        k_p = baseline / 2.0
        k_v = baseline / 5.0

        if controller_type == 'torque':
            clip_based_on_velocity = self.max_velocity is not None
            self.controller = TorqueControl(self, grav_comp=False, clip_based_on_velocity=clip_based_on_velocity)
        elif controller_type == 'torque_gc':
            clip_based_on_velocity = self.max_velocity is not None
            self.controller = TorqueControl(self, grav_comp=True, clip_based_on_velocity=clip_based_on_velocity)
        elif controller_type == 'position':
            self.controller = PositionControl(self, diff_control=False)
        elif controller_type == 'position_diff':
            self.controller = PositionControl(self, diff_control=True)
        elif controller_type == 'acc_control':
            self.controller = AccControl(self, k_p=k_p, k_v=k_v)
        elif controller_type == 'vel_control':
            self.controller = VelControl(self, k_p=k_p, k_v=k_v)
        elif controller_type == 'pos_control':
            self.controller = PosControl(self, k_p=k_p, k_v=k_v, diff_control=False)
        elif controller_type == 'pos_control_diff':
            self.controller = PosControl(self, k_p=k_p, k_v=k_v, diff_control=True)
        elif controller_type == 'position_gain':
            self.controller = PositionGainController(robot=self, params=controller_params, grav_comp=True)
        else:
            assert False, 'Unknown controller type: ' + controller_type


    def update_log(self):
        joint_pos, joint_vel = self.get_arm_state()
        self.log.add('joint_pos', joint_pos.tolist())
        self.log.add('joint_vel', joint_vel.tolist())

        endeff_pos, endeff_vel = self.get_endeff_state()
        self.log.add('endeff_pos', endeff_pos.tolist())
        self.log.add('endeff_vel', endeff_vel.tolist())

        # for (reward_type, reward_part) in self.reward_parts.items():
        #     self.log.add(reward_type, reward_part.get_reward())

        self.log.add('latest_action', self.latest_action.tolist())

        self.log.add('endeff_force_z', self.get_endeff_force_z())

        normalized_reward = 0.0
        for reward_type, reward_class in self.reward_parts.items():
            if reward_type == 'circle':
                normalized_reward += reward_class.get_reward() / reward_class.k_t
            else:
                normalized_reward += reward_class.get_reward()
        self.log.add('normalized_reward', normalized_reward)

        state = self.get_state()
        self.log.add('state', state.tolist())

        action = self.latest_action
        self.log.add('action', action.tolist())

    def ok_configuration(self):
        contacts = self.p.getContactPoints(bodyA=1, bodyB=0)
        for contact in contacts:
            if contact[4] in self.ARM_JOINTS + [9]:
                return False
        return True

    def ok_joint_pos(self, joint_pos):
        for i in range(len(self.ARM_JOINTS)):
            self.p.resetJointState(bodyUniqueId=self.ARM_ID, jointIndex=self.ARM_JOINTS[i], targetValue=joint_pos[i], targetVelocity=0.0)
        self.p.stepSimulation()
        contacts = self.p.getContactPoints(bodyA=1, bodyB=0)
        for contact in contacts:
            if contact[4] in self.ARM_JOINTS + [9]:
                return False
        if self.reduced_init:
            endeff_pos = self.get_endeff_pos()
            if endeff_pos[1] > -0.3 or endeff_pos[2] < 1.03 or endeff_pos[2] > 1.33:
                return False
        return True

    def get_ok_joint_pos(self):
        joint_pos = np.zeros(len(self.ARM_JOINTS))
        for i in range(len(self.ARM_JOINTS)):
            joint_pos[i] = np.random.uniform(low=self.joint_limits[i][0], high=self.joint_limits[i][1])
        while not self.ok_joint_pos(joint_pos):
            for i in range(len(self.ARM_JOINTS)):
                joint_pos[i] = np.random.uniform(low=self.joint_limits[i][0], high=self.joint_limits[i][1])
        return joint_pos

    def inv_dyn(self, des_acc):
        joint_pos, joint_vel = self.get_arm_state()
        torques = self.p.calculateInverseDynamics(
            bodyUniqueId=self.ARM_ID,
            objPositions=joint_pos.tolist(),
            objVelocities=joint_vel.tolist(),
            objAccelerations=des_acc.tolist())
        return np.array(torques)

    def init_torque_control(self, enabled_joints=range(7)):
        for joint_id in [self.ARM_JOINTS[i] for i in enabled_joints]:
            # As per PyBullet manual, this has to be done to be able to do
            # torque control later
            self.p.setJointMotorControl2(self.ARM_ID, joint_id,
                controlMode=self.p.VELOCITY_CONTROL, force=0)


    def torque_control(self, des_torque):
        max_torque = np.array(self.max_torque)
        des_torque = np.clip(des_torque, -max_torque, max_torque)

        for i in range(self.NUM_JOINTS):
            self.p.setJointMotorControl2(
                bodyUniqueId=self.ARM_ID,
                jointIndex=self.ARM_JOINTS[i],
                controlMode=self.p.TORQUE_CONTROL,
                force=des_torque[i])

        self.p.stepSimulation()

        self.latest_torque = des_torque


    def position_control(self, des_pos):
        for i in range(self.NUM_JOINTS):
            self.p.setJointMotorControl2(
                bodyUniqueId=self.ARM_ID,
                jointIndex=self.ARM_JOINTS[i],
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=des_pos[i],
                maxVelocity=self.vel_limit[i],
                force=self.max_torque_percent*self.max_torque[i])

        self.p.stepSimulation()

        self.latest_torque = np.zeros(self.NUM_JOINTS)
        for i in range(self.NUM_JOINTS):
            _, _, _, self.latest_torque[i] = self.p.getJointState(self.ARM_ID, self.ARM_JOINTS[i])


    def _reset(self):
        #if 'endeff_vel' in self.log.d:
        #    plt.plot(np.linalg.norm(np.array(self.log.d['endeff_vel']), axis=1))
        #    plt.show()
        #print(self.log)

        if self.collect_data:
            self.data_collector.ep_done()

        if self.logging:
            if self.full_log:
                self.log.finish_log()
            else:
                self.log.save()
                self.log.clear()

        self.controller.reset()

        if self.random_table_height:
            self.table_height = np.random.uniform(self.table_height_range[0], self.table_height_range[1])
            self.p.resetBasePositionAndOrientation(1, (0.0, -0.9, self.table_height), (0.0, 0.0, 0.0, 1.0))

        if self.random_lateral_friction:
            self.lateral_friction = np.random.uniform(self.lateral_friction_range[0], self.lateral_friction_range[1])
            for i in range(self.p.getNumJoints(0)):
                self.p.changeDynamics(0, i, lateralFriction=self.lateral_friction)
            self.p.changeDynamics(1, -1, lateralFriction=self.lateral_friction)

        if self.random_contact_stiffness:
            self.contact_stiffness = np.random.uniform(self.contact_stiffness_range[0], self.contact_stiffness_range[1])

            if self.contact_damping_multiplier is not None:
                contact_damping = self.contact_damping_multiplier * 2.0 * np.sqrt(self.contact_stiffness)
            else:
                if self.contact_damping is None:
                    contact_damping = 2.0 * np.sqrt(self.contact_stiffness)
                else:
                    contact_damping = self.contact_damping

            self.p.changeDynamics(1, -1, contactStiffness=self.contact_stiffness, contactDamping=contact_damping)
            for i in range(self.p.getNumJoints(0)):
                self.p.changeDynamics(0, i, contactStiffness=self.contact_stiffness, contactDamping=contact_damping)

        # Setting initial joint configuration to be random
        if self.initial_joint_state is None:
            joint_pos = self.get_ok_joint_pos()
            for i in range(len(self.ARM_JOINTS)):
                self.p.resetJointState(
                    bodyUniqueId=self.ARM_ID,
                    jointIndex=self.ARM_JOINTS[i],
                    targetValue=joint_pos[i],
                    targetVelocity=0.0)
        else:
            for i in range(len(self.ARM_JOINTS)):
                self.p.resetJointState(
                    bodyUniqueId=self.ARM_ID,
                    jointIndex=self.ARM_JOINTS[i],
                    targetValue=self.initial_joint_state[i],
                    targetVelocity=0.0)

        if self.logging:
            joint_pos, joint_vel = self.get_arm_state()
            self.log.add('joint_pos', joint_pos.tolist())
            self.log.add('joint_vel', joint_vel.tolist())

            endeff_pos, endeff_vel = self.get_endeff_state()
            self.log.add('endeff_pos', endeff_pos.tolist())
            self.log.add('endeff_vel', endeff_vel.tolist())

            state = self.get_state()
            self.log.add('state', state.tolist())

        state = self.get_state()
        if self.collect_data:
            self.data_collector.new_state(state)
        return state


    def _step(self, action):
        action = np.clip(action, -1.0, 1.0)
        if self.collect_data:
            self.data_collector.new_action(action)
        self.latest_action = action

        sim_step_rewards = {}
        calc_at_sim_step = ['circle', 'desired_force', 'elbow_penalty', 'orientation']
        for r in calc_at_sim_step:
            sim_step_rewards[r] = 0.0

        for i in range(self.cont_timestep_mult):
            self.controller.act(action)

            for r in calc_at_sim_step:
                if r in self.reward_parts:
                    sim_step_rewards[r] += self.reward_parts[r].get_reward() / 16.5

        if self.visualize:
            if len(self.timestep_times) > 0:
                curr_time = time.time()
                self.timestep_times.append(time.time() - self.start_time)
                since_last = curr_time - self.timestep_times[-1]
                if since_last < self.dt:
                    time.sleep(self.dt - since_last)
            else:
                self.timestep_times.append(time.time() - self.start_time)

            #if len(self.timestep_times) > 0:
            #    curr_time = time.time()
            #    self.timestep_times.append(time.time() - self.start_time)
            #    since_first = curr_time - self.timestep_times[0]
            #    if since_first < (len(self.timestep_times) - 1) * self.dt:
            #        time.sleep((len(self.timestep_times) - 1) * self.dt - since_first)
            #else:
            #    self.timestep_times.append(time.time() - self.start_time)

        #if self.visualize:
        #    time.sleep(self.dt)

        if self.logging:
            self.update_log()

        state = self.get_state()
        if self.collect_data:
            self.data_collector.new_state(state)

        reward = 0.0
        for r, f in self.reward_parts.items():
            if r in calc_at_sim_step:
                reward += sim_step_rewards[r]
                if self.log is not None:
                    self.log.add(r, sim_step_rewards[r])
            else:
                reward_value = f.get_reward()
                reward += reward_value
                if self.log is not None:
                    self.log.add(r, reward_value)

        # reward = sum([reward_part.get_reward() for reward_part in self.reward_parts.values()])
        # TODO: I don't think I am using is_done() anywhere any more
        # done = all([reward_part.is_done() for reward_part in self.reward_parts.values()])
        done = False
        return state, reward, done, {}

    def _render(self, mode, close):
        pass

    def _seed(self, seed):
        pass

    def get_cont_joint_state(self):
        return self.get_arm_state()

def log_to_video(log, fixed_timestep, reward_specs, table_height, video_file, observable, max_velocity, control, log_file, validation_plot_file):
    with open(log) as f:    
        data = json.load(f)
    initial_joint_state = data['joint_pos'][0]
    apollo_reaching = ApolloWallPushing(initial_joint_state=initial_joint_state, fixed_timestep=fixed_timestep, reward_specs=reward_specs, visualize=True, table_height=table_height, initial_pushee_pos=None, max_velocity=max_velocity, control=control, log_file=log_file, exp_name='tmp/')
    pybullet.setRealTimeSimulation(0)
    apollo_reaching.p.setRealTimeSimulation(0)
    apollo_reaching._reset()
    time.sleep(1.0)
    logging_id = apollo_reaching.p.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, video_file)
    time.sleep(1.0)
    for action in data['latest_action']:
        #print(action)
        apollo_reaching._step(np.array(action))
    apollo_reaching.p.stopStateLogging(logging_id)

    apollo_reaching.log.save()


    plt.plot(np.array(apollo_reaching.timestep_times) - apollo_reaching.timestep_times[0], 'b')
    plt.plot(np.arange(len(apollo_reaching.timestep_times)) * apollo_reaching.dt, 'r--')
    plt.savefig(validation_plot_file)
    plt.close()

    del apollo_reaching.p


    #print(video_file)
    #assert False

def specific(exp_folder, minus, log_file):
    with open(exp_folder + 'conf.yaml') as f:
        conf = yaml.load(f)

    if conf['alg'] == 'ddpg':
        max_num = max(os.listdir(exp_folder + 'val_env_episodes/'))
    else:
        max_num = max(os.listdir(exp_folder + 'env_episodes/'))
    max_num = int(max_num.split('.')[0])

    fixed_timestep = conf['env_params'][0]['env_specific_params']['fixed_timestep']
    reward_specs = conf['env_params'][0]['env_specific_params']['reward_specs']
    table_height = conf['env_params'][0]['env_specific_params']['table_height']
    observable = conf['env_params'][0]['env_specific_params']['observable']
    control = conf['env_params'][0]['env_specific_params']['control']
    max_velocity = None
    if 'max_velocity' in conf['env_params'][0]['env_specific_params']:
        max_velocity = conf['env_params'][0]['env_specific_params']['max_velocity']

    if conf['alg'] == 'ddpg':
        ep_log = exp_folder + 'val_env_episodes/' + str(max_num - minus).zfill(6) + '.json'
    else:
        ep_log = exp_folder + 'env_episodes/' + str(max_num - minus).zfill(6) + '.json'

    video_file = exp_folder + str(max_num - minus).zfill(6) + '.mp4'
    tmp_video_file = exp_folder + str(max_num - minus).zfill(6) + '_tmp.mp4'
    validation_plot_file = exp_folder + str(max_num - minus).zfill(6) + '.png'

    log_to_video(ep_log, fixed_timestep, reward_specs, table_height, tmp_video_file, observable, max_velocity, control, log_file, validation_plot_file)

    os.system('ffmpeg -y -i ' + tmp_video_file + ' -filter:v "setpts=2.0*PTS" ' + video_file)

def visualize(exp_folder, eps_folder, ep_num, output_log):
    with open(exp_folder + 'conf.yaml') as f:
        conf = yaml.load(f)

    fixed_timestep = conf['env_params'][0]['env_specific_params']['fixed_timestep']
    reward_specs = conf['env_params'][0]['env_specific_params']['reward_specs']
    table_height = conf['env_params'][0]['env_specific_params']['table_height']
    observable = conf['env_params'][0]['env_specific_params']['observable']
    control = conf['env_params'][0]['env_specific_params']['control']
    max_velocity = None
    if 'max_velocity' in conf['env_params'][0]['env_specific_params']:
        max_velocity = conf['env_params'][0]['env_specific_params']['max_velocity']

    ep_log = exp_folder + eps_folder + str(ep_num).zfill(6) + '.json'
    video_file = exp_folder + str(ep_num).zfill(6) + '.mp4'

    log_to_video(ep_log, fixed_timestep, reward_specs, table_height, video_file, observable, max_velocity, control, output_log)

def speed_test(num_steps=1000):
    apollo_reaching = ApolloWallPushing(control='torque_gc', reward_specs={'orientation': {'k_o': 1.0}}, reduced_init=True)
    apollo_reaching._reset()
    start = time.time()
    for i in range(num_steps):
        action = np.random.uniform(low=-1.0, high=1.0, size=7)
        apollo_reaching._step(action)
    end = time.time()
    print()
    print('Time for', num_steps, 'steps:', end - start)

if __name__ == "__main__":
    #speed_test()
    #assert False
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vis', help='')
    parser.add_argument('--o', help='')
    parser.add_argument('--mp4', help='')
    parser.add_argument('--video', help='')
    parser.add_argument('--how_many', help='')
    parser.add_argument('--minus', help='')
    parser.add_argument('--log', help='')
    parser.add_argument('--folder')
    parser.add_argument('--control')
    args = parser.parse_args()

    if args.vis is not None:
        assert args.o is not None, 'You have to provide output path'
        with open(args.vis) as f:    
            data = json.load(f)

        initial_joint_state = data['joint_pos'][0]
        apollo_reaching = ApolloWallPushing(reward_specs={}, initial_pushee_pos=[-0.15, 0.15], observable=[], visualize=True, initial_joint_state=initial_joint_state, exp_name=args.o, full_log=True, fixed_timestep=0.0165, table_height=0.9)
        while True:
            apollo_reaching._reset()
            for action in data['latest_action']:
                apollo_reaching._step(np.array(action))
    elif args.video is not None:
        if args.minus is not None:
            specific(args.video, int(args.minus), args.log)
    elif args.folder is not None:
        exps = os.listdir(args.folder)
        for exp in exps:
            exp_folder = args.folder + exp + '/'
            if os.path.isdir(exp_folder):
                try:
                    specific(exp_folder, 0, exp_folder + 'video_log.json')
                except:
                    pass
    else:
        control = args.control
        if control is None:
            control = 'torque_gc'
        #initial_joint_state = [1.1, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0]
        apollo_reaching = ApolloWallPushing(visualize=True, control=control, reward_specs={'orientation': {'k_o': 1.0}}, reduced_init=True, max_torque_percent=0.1, log_file='/Users/miroslav/Desktop/tmpfile.json')
        while True:
            apollo_reaching._reset()
            #apollo_reaching.check_contacts()
            #print(apollo_reaching.ok_configuration())
            for i in range(500):
                action = np.random.uniform(low=-1.0, high=1.0, size=7)
                apollo_reaching._step(action)
