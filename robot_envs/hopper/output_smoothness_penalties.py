import numpy as np


class DesiredPositionSmoothnessPenalty():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        self.last_des_pos = None

    def get_reward(self):
        new_des_pos = self.robot.controller.des_pos

        if self.last_des_pos is None:
            diff = np.zeros(new_des_pos.shape)
        else:
            diff = new_des_pos - self.last_des_pos

        self.last_des_pos = new_des_pos.copy()

        k = self.params['k']
        return -k * np.sum(np.square(diff))


class GainSmoothnessPenalty():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        self.last_gain = None

    def get_reward(self):
        new_gain = self.robot.controller.kp

        if self.last_gain is None:
            diff = np.zeros(new_gain.shape)
        else:
            diff = new_gain - self.last_gain

        self.last_gain = new_gain.copy()

        k = self.params['k']
        return -k * np.sum(np.square(diff))
