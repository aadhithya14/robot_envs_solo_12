import numpy as np


class ForwardMotionReward():

    def __init__(self, robot, k=0.5, desired_velocity=None, desired_velocity_range=None, calc_at_sim_step=True):
        self.robot = robot
        self.calc_at_sim_step = calc_at_sim_step
        self.k = k
        self.desired_velocity = desired_velocity
        self.desired_velocity_range = desired_velocity_range

    def step(self):
        pass

    def reset(self):
        pass

    def get_reward(self):
        base_vel, _ = self.robot.p.getBaseVelocity(self.robot.robot_id)

        if self.desired_velocity is not None:
            diff = np.abs(self.desired_velocity - base_vel[0])
            return -1.0 * self.k * diff
        elif self.desired_velocity_range is not None:
            if base_vel[0] < self.desired_velocity_range[0]:
                return -1.0 * self.k * (self.desired_velocity_range[0] - base_vel[0])
            elif base_vel[0] > self.desired_velocity_range[1]:
                return -1.0 * self.k * (base_vel[0] - self.desired_velocity_range[1])
            else:
                return 0.0
        else:
            return self.k * base_vel[0]
