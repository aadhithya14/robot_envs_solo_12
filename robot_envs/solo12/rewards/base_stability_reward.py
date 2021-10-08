import numpy as np


class BaseStabilityReward():

    def __init__(self,
                 robot,
                 base_pos_0_sigma=0.02,
                 base_pos_1_sigma=0.02,
                 base_ang_0_sigma=0.05,
                 base_ang_1_sigma=0.05,
                 base_ang_2_sigma=0.05,
                 allow_forward_motion=False,
                 base_pos_0_k=1.0,
                 base_pos_1_k=1.0,
                 base_ang_0_k=1.0,
                 base_ang_1_k=1.0,
                 base_ang_2_k=1.0,
                 version='v0',
                 k=1.0,
                 calc_at_sim_step=True):
        self.robot = robot
        self.calc_at_sim_step = calc_at_sim_step
        self.base_pos_0_sigma = base_pos_0_sigma
        self.base_pos_1_sigma = base_pos_1_sigma
        self.base_ang_0_sigma = base_ang_0_sigma
        self.base_ang_1_sigma = base_ang_1_sigma
        self.base_ang_2_sigma = base_ang_2_sigma
        self.allow_forward_motion = allow_forward_motion
        self.base_pos_0_k = base_pos_0_k
        self.base_pos_1_k = base_pos_1_k
        self.base_ang_0_k = base_ang_0_k
        self.base_ang_1_k = base_ang_1_k
        self.base_ang_2_k = base_ang_2_k
        self.k = k
        self.version = version


    def reset(self):
        pass

    def step(self):
        pass

    def get_reward(self):
        base_pos, base_orient = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)
        base_ang = self.robot.p.getEulerFromQuaternion(base_orient)
        base_ang = np.array(base_ang)

        old_base_stability_reward = np.exp(-200.0  * np.square(np.linalg.norm(base_ang[:2])))
        self.robot.log.add('old_base_stability_reward', old_base_stability_reward)

        base_pos_0_rew = np.exp(-0.5 * (base_pos[0] / self.base_pos_0_sigma) ** 2)
        base_pos_1_rew = np.exp(-0.5 * (base_pos[1] / self.base_pos_1_sigma) ** 2)
        base_ang_0_rew = np.exp(-0.5 * (base_ang[0] / self.base_ang_0_sigma) ** 2)
        base_ang_1_rew = np.exp(-0.5 * (base_ang[1] / self.base_ang_1_sigma) ** 2)
        base_ang_2_rew = np.exp(-0.5 * (base_ang[2] / self.base_ang_2_sigma) ** 2)

        self.robot.log.add('base_pos_0_sigma', self.base_pos_0_sigma)
        self.robot.log.add('base_pos_1_sigma', self.base_pos_1_sigma)
        self.robot.log.add('base_ang_0_sigma', self.base_ang_0_sigma)
        self.robot.log.add('base_ang_1_sigma', self.base_ang_1_sigma)
        self.robot.log.add('base_ang_2_sigma', self.base_ang_2_sigma)

        self.robot.log.add('base_pos_0_rew', base_pos_0_rew)
        self.robot.log.add('base_pos_1_rew', base_pos_1_rew)
        self.robot.log.add('base_ang_0_rew', base_ang_0_rew)
        self.robot.log.add('base_ang_1_rew', base_ang_1_rew)
        self.robot.log.add('base_ang_2_rew', base_ang_2_rew)

        if self.version == 'v0':
            if not self.allow_forward_motion:
                reward = base_pos_0_rew * base_pos_1_rew * base_ang_0_rew * base_ang_1_rew * base_ang_2_rew
            else:
                reward = base_pos_1_rew * base_ang_0_rew * base_ang_1_rew * base_ang_2_rew
            self.robot.log.add('new_base_stability_reward', reward)
        elif self.version == 'v1':
            reward = self.k * base_ang_0_rew * base_ang_1_rew
        elif self.version == 'v2':
            reward =   self.base_pos_0_k * base_pos_0_rew \
                     + self.base_pos_1_k * base_pos_1_rew \
                     + self.base_ang_0_k * base_ang_0_rew \
                     + self.base_ang_1_k * base_ang_1_rew \
                     + self.base_ang_2_k * base_ang_2_rew
        elif self.version == 'v3':
            reward = base_pos_1_rew * base_ang_0_rew * base_ang_2_rew
        else:
            assert self.version == 'v4'
            reward = base_pos_0_rew * base_pos_1_rew * base_ang_0_rew * base_ang_2_rew

        return reward
