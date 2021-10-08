import numpy as np


class ContactCountReward():

    def __init__(self,
                 robot,
                 version,
                 k,
                 positive_reward=False,
                 calc_at_sim_step=True):
        self.robot = robot
        self.version = version
        self.k = k
        self.positive_reward = positive_reward
        self.calc_at_sim_step = calc_at_sim_step

    def reset(self):
        pass

    def step(self):
        pass

    def get_reward(self):
        if self.robot.with_motor_rotor:
            endeff_ids = [4, 9, 14, 19]
        else:
            endeff_ids = [2, 5, 8, 11]

        active_contacts = [0] * 4

        contacts = self.robot.p.getContactPoints(bodyA=self.robot.surface_id, bodyB=self.robot.robot_id)
        if contacts is not None:
            for contact in contacts:
                for i in range(4):
                    if contact[4] == endeff_ids[i]:
                        active_contacts[i] = 1

        # print(active_contacts)

        if self.positive_reward:
            good_reward = self.k
            bad_reward  = 0.0
        else:
            good_reward = 0.0
            bad_reward  = -1.0 * self.k

        if active_contacts == [0, 0, 0, 0]:
            return good_reward

        if self.version == 'binary_front_back':
            if active_contacts == [1, 1, 0, 0] or active_contacts == [0, 0, 1, 1]:
                return good_reward
            else:
                return bad_reward

        elif self.version == 'binary_diagonal':
            if active_contacts == [1, 0, 0, 1] or active_contacts == [0, 1, 1, 0]:
                return good_reward
            else:
                return bad_reward

        return endeff_forces
