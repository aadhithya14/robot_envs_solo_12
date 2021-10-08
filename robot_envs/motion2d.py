import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import prepare_plot
from robot_envs.base_env import BaseEnv
from robot_envs.motion2d_rewards import GoalPositionReward
from gym.spaces import Box
import time
from utils.data_logging import ListOfLogs, DataCollector


class Motion2D(BaseEnv):

    def __init__(self, box_size, radius, visualize=False, exp_name=None, logging=False, reward_specs={}, output_dir=None, action_type='circle', collect_data=False, pack_size=100, packs_to_keep=20, max_log_files=20, initial_state=None):
        self.box_size = box_size
        self.box = Box(-box_size, box_size, (2,))
        self.radius = radius
        self.visualize = visualize
        self.log_file = None
        self.exp_name = exp_name
        self.action_type = action_type
        self.initial_state = initial_state
        if logging:
            self.log_file = self.exp_name + '_log'
        self.collect_data = collect_data
        if self.collect_data:
            self.data_collector = DataCollector(self.exp_name + '_packs', pack_size, packs_to_keep)

        #todo: standardize action space
        self.action_space = Box(-1.0, 1.0, (2,))
        self.observation_space = self.box

        if self.visualize:
            prepare_plot(wide=False)
            plt.grid(True)
            plt.axis([self.box.low[0], self.box.high[0], self.box.low[1], self.box.high[1]])
            self.plot = plt.plot([], [])[0]
            plt.show(block=False)

        if self.log_file is not None:
            self.log = ListOfLogs(self.log_file, separate_files=True, max_files=max_log_files)

        super(Motion2D, self).__init__(reward_specs)

    def get_state(self):
        return self.state

    def init_reward(self, reward_specs):
        reward_parts = {}
        for reward_type, reward_spec in reward_specs.items():
            if reward_type == 'goal_position':
                radius = None
                if radius in reward_spec:
                    radius = reward_spec['radius']
                value = None
                if 'value' in reward_spec:
                    value = reward_spec['value']
                reward_parts[reward_type] = GoalPositionReward(self.get_state, reward_spec['type'], reward_spec['goal'], \
                                                               radius=radius, value=value, box_size=self.box_size)
            else:
                assert False, 'Unknown reward type: ' + str(reward_type)
        return reward_parts

    def reset_internal(self):
        if self.collect_data:
            self.data_collector.ep_done()

        if self.log_file is not None:
            self.log.finish_log()

        if self.initial_state is None:
            self.state = self.box.sample()
        else:
            self.state = self.initial_state.copy()

        if self.visualize:
            self.ep_states = [self.state.copy()]
            ep_states = np.array(self.ep_states)
            self.plot.set_data(ep_states[:, 0], ep_states[:, 1])
            #plt.plot(ep_states[:, 0], ep_states[:, 1])
            #plt.draw()
            plt.pause(0.01)
            #time.sleep(0.01)

        if self.log_file is not None:
            self.log.add('state', self.state.tolist())

        if self.collect_data:
            self.data_collector.new_state(self.state)

        return self.state

    def step_internal(self, action):
        action = np.clip(action, -1.0, 1.0)
        if self.collect_data:
            self.data_collector.new_action(action)

        if self.action_type == 'circle':
            angle = np.interp(action[0], (-1.0, 1.0), (0.0, 2 * np.pi))
            distance = np.interp(action[1], (-1.0, 1.0), (0.0, self.radius))
            self.state += distance * np.array([np.cos(angle), np.sin(angle)])
        elif self.action_type == 'square':
            motion = np.interp(action, (-1.0, 1.0), (-self.radius, self.radius))
            self.state += motion

        if self.visualize:
            self.ep_states.append(self.state.copy())
            ep_states = np.array(self.ep_states)
            self.plot.set_data(ep_states[:, 0], ep_states[:, 1])
            #plt.plot(ep_states[:, 0], ep_states[:, 1])
            #plt.draw()
            #time.sleep(0.01)
            plt.pause(0.003)

        if self.log_file is not None:
            self.log.add('action', action.tolist())
            self.log.add('state', self.state.tolist())

        if self.collect_data:
            self.data_collector.new_state(self.state)

        return self.state

    def _render(self, mode, close):
        pass

    def _seed(self, seed):
        pass

if __name__ == '__main__':
    motion2d = Motion2D(10.0, 0.1, True)
    for i in range(1):
        motion2d._reset()
        for j in range(300):
            motion2d._step(np.random.uniform(-1.0, 1.0, 2))
