import numpy as np


class VelocityTrackingReward():

    def __init__(self, robot, params, calc_at_sim_step=False):
        self.robot  = robot
        self.params = params
        if 'calc_at_sim_step' in self.params:
            self.calc_at_sim_step = self.params['calc_at_sim_step']
        else:
            self.calc_at_sim_step = calc_at_sim_step

        self.lookback = None
        if 'lookback' in self.params:
            self.lookback = self.params['lookback']

    def step(self):
        if self.lookback is not None:
            self.qdotdes_hist.append(self.robot.controller.qdotdes_for_ttr)

    def reset(self):
        if self.lookback is not None:
            self.qdotdes_hist = []

    def get_reward(self):
        _, qdot = self.robot.get_cont_joint_state()
        if 'compare_to_current' in self.params and self.params['compare_to_current']:
            qdot = self.robot.controller.qdot

        qdotdes = self.robot.controller.qdotdes_for_ttr
        if self.lookback is not None:
            if len(self.qdotdes_hist) < self.lookback - 1:
                return 0.0
            qdotdes = self.qdotdes_hist[-(self.lookback - 1)]

        diff    = qdotdes - qdot

        if 'reward_type' in self.params:
            if self.params['reward_type'] == 'l1':
                return -1.0 * self.params['k'] * np.linalg.norm(diff, ord=1)
            elif self.params['reward_type'] == 'l2':
                return -1.0 * self.params['k'] * np.linalg.norm(diff)
            elif self.params['reward_type'] == 'max_abs':
                return -1.0 * self.params['k'] * np.max(np.abs(diff))
            else:
                assert False, 'Unknown reward type for TTR: ' + self.params['reward_type']

        assert not 'exp_penalty' in self.params
        assert 'squared_penalty' in self.params
        assert self.params['squared_penalty'] == True
        assert 'max_velocity' in self.params

        diff            = np.sum(np.absolute(diff))
        normalized_diff = diff / (self.params['max_velocity'] * 2) / qdot.shape[0]
        normalized_diff = normalized_diff ** 2

        return -1.0 * self.params['k'] * normalized_diff
