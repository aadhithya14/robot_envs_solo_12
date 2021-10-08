import numpy as np
from robot_envs.controllers.position_gain_controller import PositionGainController


class TrajectoryTrackingReward():

    def __init__(self, robot, params, calc_at_sim_step=False):
        self.robot = robot
        #TODO: Check that the controller is the appropriate class
        #      (has the des_pos variable)
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
            self.qdes_hist.append(self.robot.controller.qdes_for_ttr)

    def reset(self):
        if self.lookback is not None:
            self.qdes_hist = []

    def get_reward(self):
        pos, _ = self.robot.get_cont_joint_state()
        if 'compare_to_current' in self.params and self.params['compare_to_current']:
            pos = self.robot.controller.q

        qdes = self.robot.controller.qdes_for_ttr
        if self.lookback is not None:
            if len(self.qdes_hist) < self.lookback - 1:
                return 0.0
            qdes = self.qdes_hist[-(self.lookback - 1)]

        diff = PositionGainController.pos_diff(qdes, pos, self.robot.cont_joint_type)

        if 'reward_type' in self.params:
            if self.params['reward_type'] == 'l1':
                return -1.0 * self.params['k'] * np.linalg.norm(diff, ord=1)
            elif self.params['reward_type'] == 'l2':
                return -1.0 * self.params['k'] * np.linalg.norm(diff)
            elif self.params['reward_type'] == 'max_abs':
                return -1.0 * self.params['k'] * np.max(np.abs(diff))
            else:
                assert False, 'Unknown reward type for TTR: ' + self.params['reward_type']

        else:
            exp_penalty = False
            if 'exp_penalty' in self.params:
                exp_penalty = self.params['exp_penalty']

            if exp_penalty:
                negative = False
                if 'negative' in self.params:
                    negative = self.params['negative']

                if negative:
                    return self.params['k'] * (np.exp(-self.params['c'] * np.linalg.norm(diff)) - 1.0)
                else:
                    return self.params['k'] * np.exp(-self.params['c'] * np.linalg.norm(diff))
            else:
                # self.robot.log.add('ttr_l2', np.linalg.norm(diff))
                # self.robot.log.add('ttr_l1', np.linalg.norm(diff, ord=1))

                # self.robot.log.add('ttr_ord_inf', np.linalg.norm(diff, ord='inf'))
                # self.robot.log.add('ttr_max_abs', np.max(np.abs(diff)))
                # self.robot.log.add('ttr_max_sqr', np.max(np.square(diff)))

                diff = np.sum(np.absolute(diff))

                # self.robot.log.add('ttr_diff_sum_abs', diff)

                normalized_diff = diff / np.pi / pos.shape[0]

                # self.robot.log.add('ttr_diff_normalized', normalized_diff)

                squared_penalty = False
                if 'squared_penalty' in self.params:
                    squared_penalty = self.params['squared_penalty']
                if squared_penalty:
                    normalized_diff = normalized_diff ** 2
                    # self.robot.log.add('ttr_diff_squared', normalized_diff)
                return -1.0 * self.params['k'] * normalized_diff
