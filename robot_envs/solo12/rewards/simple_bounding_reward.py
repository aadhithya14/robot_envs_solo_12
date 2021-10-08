import os
import numpy as np
import matplotlib.pyplot as plt


class SimpleBoundingReward():

    def __init__(self,
                 robot,
                 k=1.0,
                 min_value=None,
                 max_value=None,
                 calc_at_sim_step=True):
        self.robot = robot
        self.k = k
        self.min_value = min_value
        self.max_value = max_value
        self.calc_at_sim_step = calc_at_sim_step

    def reset(self):
        pass

    def step(self):
        pass

    def get_reward(self):
        base_pos, base_orient = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)
        base_ang = self.robot.p.getEulerFromQuaternion(base_orient)

        if self.min_value is not None and np.abs(base_ang[1]) < self.min_value:
            return 0.0

        if self.max_value is not None and np.abs(base_ang[1]) > self.max_value:
            return 0.0

        return self.k * np.abs(base_ang[1])
