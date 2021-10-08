import numpy as np

class TorqueSmoothnessPenalty():

    def __init__(self, robot, k=1.0, version='v0', exp_coeff=5.0, calc_at_sim_step=False):
        self.robot = robot
        self.k = k
        self.version = version
        self.exp_coeff = exp_coeff
        self.calc_at_sim_step = calc_at_sim_step

    def step(self):
        self.last_torque = self.robot.des_torque.copy()

    def reset(self):
        self.last_torque = np.zeros(self.robot.num_cont_joints)

    def get_reward(self):
        diff = self.robot.des_torque - self.last_torque

        if self.version == 'v0':
            return -self.k * np.sum(np.square(diff))
        else:
            assert self.version == 'v1'
            x = np.linalg.norm(diff)
            x = np.clip(x, 0, np.log(np.power(1000, 0.2)))
            x = np.exp(self.exp_coeff * x)
            return -self.k * x
