import numpy as np


class BaseStaticReward():

    def __init__(self,
                 robot,
                 k_vel=1.0,
                 k_angvel=1.0,
                 calc_at_sim_step=True):
        self.robot = robot
        self.k_vel = k_vel
        self.k_angvel = k_angvel
        self.calc_at_sim_step = calc_at_sim_step

    def reset(self):
        pass

    def step(self):
        pass

    def get_reward(self):
        base_vel, base_angvel = self.robot.p.getBaseVelocity(self.robot.robot_id)

        self.robot.log.add('base_vel_norm', np.linalg.norm(base_vel))
        self.robot.log.add('base_angvel_norm', np.linalg.norm(base_angvel))

        return -1.0 * (self.k_vel * np.linalg.norm(base_vel) + self.k_angvel * np.linalg.norm(base_angvel))
