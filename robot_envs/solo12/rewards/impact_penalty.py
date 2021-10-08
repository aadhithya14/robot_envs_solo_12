import numpy as np


class ImpactPenalty():

    def __init__(self, robot, k, max_allowed_force, calc_at_sim_step=True):
        self.robot = robot
        self.calc_at_sim_step = calc_at_sim_step
        self.k = k
        self.max_allowed_force = max_allowed_force

    def step(self):
        pass

    def reset(self):
        pass

    def get_reward(self):
        endeff_forces = self.robot.get_endeff_forces()
        reward = 0.0
        for i in range(4):
            if abs(endeff_forces[i, 2]) > self.max_allowed_force:
                reward += -1.0 * self.k * abs(endeff_forces[i, 2])
        self.robot.log.add("impact_penalty",reward)
        return reward
