import numpy as np


class DesiredJointPosReward():

    def __init__(self,
                 robot,
                 des_joint_pos,
                 k=1.0,
                 c=0.5,
                 calc_at_sim_step=True):
        self.robot = robot
        self.des_joint_pos = np.array(des_joint_pos)
        self.k = k
        self.c = c
        self.calc_at_sim_step = calc_at_sim_step

    def reset(self):
        pass

    def step(self):
        pass

    def get_reward(self):
        joint_pos, _  = self.robot.get_obs_joint_state()
        reward = self.k * np.exp(-self.c  * np.linalg.norm(self.des_joint_pos - joint_pos) ** 2)
        # print(joint_pos)

        return reward
