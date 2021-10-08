import numpy as np


class QuadrupedHoppingReward():

    def __init__(self, robot, base_pos_2_k=1.0, base_ang_0_k=0.0, base_ang_1_k=0.0, calc_at_sim_step=False, version=None, max_height=None, min_height=None):
        self.robot = robot
        self.calc_at_sim_step = calc_at_sim_step
        self.base_pos_2_k = base_pos_2_k
        self.base_ang_0_k = base_ang_0_k
        self.base_ang_1_k = base_ang_1_k
        self.version = version
        self.max_height = max_height
        self.min_height = min_height

    def step(self):
        pass

    def reset(self):
        pass

    def get_reward(self):
        base_pos, base_orient = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)
        base_ang = self.robot.p.getEulerFromQuaternion(base_orient)

        # base_pos_2_rew = np.exp(-5.0 * np.abs(1.0 - min(base_pos[2], 1.0)))
        if base_pos[2] < 0.35:
            base_pos_2_rew = 0.0
        elif self.max_height is not None and base_pos[2] > self.max_height:
            base_pos_2_rew = 0.0
        elif self.min_height is not None and base_pos[2] < self.min_height:
            base_pos_2_rew = 0.0
        else:
            if self.version == 'v1':
                base_pos_2_rew = 4.0 * base_pos[2] / 0.8 - 3.0 * 0.35
            elif self.version == 'v2':
                base_pos_2_rew = 2.0 * base_pos[2] / 0.8 - 0.35
            elif self.version == 'v3':
                base_pos_2_rew = 2.0 * base_pos[2] / 0.8
            elif self.version == 'v4':
                base_pos_2_rew = 4.0 * base_pos[2] / 0.8
            else:
                base_pos_2_rew = base_pos[2] / 0.8
            
        base_ang_0_rew = np.exp(-5.0 * np.abs(base_ang[0]))
        base_ang_1_rew = np.exp(-5.0 * np.abs(base_ang[1]))

        self.robot.log.add('base_pos_2_rew', base_pos_2_rew)
        self.robot.log.add('base_ang_0_rew', base_ang_0_rew)
        self.robot.log.add('base_ang_1_rew', base_ang_1_rew)

        reward =   self.base_pos_2_k * base_pos_2_rew \
                 + self.base_ang_0_k * base_ang_0_rew \
                 + self.base_ang_1_k * base_ang_1_rew

        return reward