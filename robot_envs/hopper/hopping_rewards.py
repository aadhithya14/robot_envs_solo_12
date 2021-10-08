import numpy as np
from scipy.stats import norm
from robot_envs.controllers.position_gain_controller import angle_diff, PositionGainController


def exp_rew(x, x_range, y_range, curve, flipped=False):

    def f(x):
        return np.exp(curve * x)

    def g(x):
        return (f(x) - f(0)) / (f(1) - f(0))

    if flipped:
        return y_range[1] - g((x_range[1] - x) / (x_range[1] - x_range[0])) * (y_range[1] - y_range[0])
    else:
        return y_range[0] + g((x - x_range[0]) / (x_range[1] - x_range[0])) * (y_range[1] - y_range[0])


def select_rew(robot, ground_reward, air_reward, selection_criteria):
    if selection_criteria is None:
        return air_reward
    elif selection_criteria == 'height':
        if robot.get_base_height() < robot.get_upright_height():
            return ground_reward
        else:
            return air_reward
    else:
        assert selection_criteria == 'force'
        if robot.get_endeff_force()[2] == 0.0:
            return ground_reward
        else:
            return air_reward


def base_rew(robot, rew_params):
    height = robot.get_base_height()

    ground_reward = exp_rew(height, *rew_params['ground_reward_params'])
    air_reward = exp_rew(height, *rew_params['air_reward_params'])

    return select_rew(robot, ground_reward, air_reward, rew_params['selection_criteria'])


class HoppingReward():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        if self.params['type'] == 'max':
            self.max_value = None

    def get_reward(self):
        base_value = base_rew(self.robot, self.params)

        if self.params['type'] == 'integral':
            return base_value
        else:
            if self.max_value is None:
                self.max_value = base_value
                return 0.0
            else:
                if base_value > self.max_value:
                    diff = base_value - self.max_value
                    self.max_value = base_value
                    return diff
                else:
                    return 0.0


class BaseAccPenalty():

    def __init__(self, robot, params, initial_delay_in_timesteps=5):
        self.robot = robot
        self.params = params
        self.initial_delay_in_timesteps = initial_delay_in_timesteps
        self.version = 'sum'
        if 'version' in self.params:
            self.version = self.params['version']

    def reset(self):
        self.prev_vel = None
        self.timestep = 0
        self.abs_acc = 0.0
        if self.version == 'max' or self.version == 'sum_max':
            self.max_abs_acc = 0.0

    def get_reward(self):
        curr_vel = self.robot.get_base_vel_z()

        reward = 0.0
        if self.timestep > self.initial_delay_in_timesteps:
            self.abs_acc = np.absolute(curr_vel - self.prev_vel) / self.robot.sim_timestep
            if self.version == 'sum':
                if self.abs_acc > self.params['acc_limit']:
                    reward = -1.0 * self.params['k'] * self.abs_acc
            elif self.version == 'max':
                if self.abs_acc > self.max_abs_acc:
                    reward = -1.0 * self.params['k'] * (self.abs_acc - self.max_abs_acc)
                    self.max_abs_acc = self.abs_acc
            elif self.version == 'sum_max':
                if self.abs_acc > self.max_abs_acc:
                    self.max_abs_acc = self.abs_acc
                reward = -1.0 * self.params['k'] * self.max_abs_acc
            else:
                assert False, 'Unknown BaseAccPenalty version: ' + self.version

        self.timestep += 1
        self.prev_vel = curr_vel

        return reward


class BaseStabilityPenalty():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        pass

    def get_reward(self):
        pos, orient = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)

        pos_xy = np.array(pos[:2])
        base_pos_xy = np.array([0, 0])
        dist = np.linalg.norm(pos_xy - base_pos_xy)

        base_orient = np.array([0, 0, 0, 1])
        orient = np.array(orient)
        # [0, 1]
        angle_dist = 1 - np.square(np.dot(orient, base_orient))

        return self.params['k'] * (exp_rew(dist, [0, 1], [0, -1], 5, True) + exp_rew(angle_dist, [0, 1], [0, -1], 5, True))


class TrajectoryTrackingReward():

    def __init__(self, robot, params):
        self.robot = robot
        #TODO: Check that the controller is the appropriate class
        #      (has the des_pos variable)
        self.params = params

    def reset(self):
        pass

    def get_reward(self):
        pos, _ = self.robot.get_cont_joint_state()
        diff = PositionGainController.pos_diff(self.robot.controller.des_pos, pos, self.robot.cont_joint_type)

        exp_penalty = False
        if 'exp_penalty' in self.params:
            exp_penalty = self.params['exp_penalty']

        if exp_penalty:
            negative = False
            if 'negative' in self.params:
                negative = self.params['negative']

            if negative:
                return self.params['k'] * (np.exp(-self.params['c'] * np.linalg.norm(diff)) - 1.0)
            else:
                return self.params['k'] * np.exp(-self.params['c'] * np.linalg.norm(diff))
        else:
            diff = np.sum(np.absolute(diff))
            normalized_diff = diff / np.pi / pos.shape[0]
            squared_penalty = False
            if 'squared_penalty' in self.params:
                squared_penalty = self.params['squared_penalty']
            if squared_penalty:
                normalized_diff = normalized_diff ** 2
            return -1.0 * self.params['k'] * normalized_diff


class VelocityTrackingReward():

    def __init__(self, robot, params):
        self.robot  = robot
        self.params = params

    def reset(self):
        pass

    def get_reward(self):
        _, qdot = self.robot.get_cont_joint_state()
        qdotdes = self.robot.controller.qdotdes
        diff    = qdotdes - qdot

        assert not 'exp_penalty' in self.params
        assert 'squared_penalty' in self.params
        assert self.params['squared_penalty'] == True
        assert 'max_velocity' in self.params

        diff            = np.sum(np.absolute(diff))
        normalized_diff = diff / (self.params['max_velocity'] * 2) / qdot.shape[0]
        normalized_diff = normalized_diff ** 2

        return -1.0 * self.params['k'] * normalized_diff


class ShakingPenalty():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        self.hist = []

    def get_reward(self, update=True):
        if update:
            self.hist.append(self.robot.controller.des_pos)
        penalty = 0.0
        if len(self.hist) > 2:
            for i in range(2):
                # TODO: Doesn't work 100% correctly for circular joints
                # TODO: Only work for hopper right now
                if (self.hist[-2][i] - self.hist[-3][i]) * (self.hist[-1][i] - self.hist[-2][i]) < 0:
                    penalty += 1.0
        return -1.0 * self.params['k'] * penalty


class ForwardMotionReward():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        pass

    def get_reward(self):
        pos, _ = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)
        return self.params['k'] * pos[0]


class DesiredEndeffectorPositionReward():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        pass

    def get_reward(self):
        endeff_pos, _ = self.robot.get_endeff_state()
        dist = np.linalg.norm(self.params['des_pos'] - endeff_pos)
        return -self.params['k'] * exp_rew(dist, [0, 1], [1, 0], 10, True)
        return 0.0

class DesiredEndeffectorForceReward():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        pass

    def calc_force(self):
        contacts = self.robot.p.getContactPoints(bodyA=self.robot.robot_id,
                                           linkIndexA=self.robot.get_endeff_link_id(),
                                           bodyB=self.robot.vertical_surface_id)
        total_force = np.zeros(3)
        for contact in contacts:
            contact_normal = np.array(contact[7])
            normal_force = contact[9]
            total_force += normal_force * contact_normal
        return total_force

    def force_reward_function(self, desired_force, actual_force):
        if actual_force < 1e-9:
            return 0.0
        else:
            normalized_diff = np.absolute(desired_force - actual_force) / desired_force
            return 0.4 + 0.6 * exp_rew(normalized_diff, [0, 1], [1, 0], 5, True)

    def get_reward(self):
        # Assuming vertical surface and force in X direction
        normal_force = -self.calc_force()[0]
        force_reward = self.force_reward_function(self.params['des_force'], normal_force)
        if 'des_height' in self.params:
            endeff_pos, _ = self.robot.get_endeff_state()
            height_error = np.absolute(self.params['des_height'] - endeff_pos[2])
            height_error_multiplier = norm.pdf(height_error, scale=self.params['des_height_sigma']) / norm.pdf(0, scale=self.params['des_height_sigma'])
            force_reward *= height_error_multiplier
        return self.params['k'] * force_reward

class StateDifferenceNormReward():

    def __init__(self, robot, params):
        self.robot = robot
        self.des_pos = np.array(params['des_pos'])
        if isinstance(params['k'], list):
            self.k = np.array(params['k'])
        else:
            self.k = params['k'] * np.ones(self.robot.num_obs_joints)

    def reset(self):
        pass

    def get_reward(self):
        pos, _ = self.robot.get_obs_joint_state()
        return -1.0 * np.linalg.norm(np.multiply(self.k, self.des_pos - pos))
