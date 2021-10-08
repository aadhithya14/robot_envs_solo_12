import numpy as np
from gym.spaces import Box
from utils.my_math import scale

class TorqueController():

    def __init__(self, robot, grav_comp=True):
        self.robot = robot
        self.grav_comp = grav_comp

        self.robot.init_torque_control()

    def get_control_space(self):
        return Box(-1.0, 1.0, (self.robot.num_cont_joints,))

    def reset(self):
        pass

    def act(self, action, raw_torque_input=False):
        if not raw_torque_input:
            torque = np.zeros(action.shape[0])
            for i in range(action.shape[0]):
                torque[i] = scale(action[i], [-1, 1], [-self.robot.max_torque[i], self.robot.max_torque[i]])
        else:
            torque = action

        if self.grav_comp:
            torque += self.robot.inv_dyn(np.zeros(torque.shape))

        self.robot.torque_control(torque)
