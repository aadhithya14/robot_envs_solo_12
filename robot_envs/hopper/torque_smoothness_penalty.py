import numpy as np

class TorqueSmoothnessPenalty():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        self.last_torque = np.zeros(self.robot.num_cont_joints)

    def get_reward(self):
        new_torque = self.robot.des_torque
        
        diff = new_torque - self.last_torque
        self.last_torque = new_torque.copy()

        k = self.params['k']

        return -k * np.sum(np.square(diff))