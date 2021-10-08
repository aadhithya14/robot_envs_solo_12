import numpy as np


class ActionFilter():

    def __init__(self, sim_timestep_length):
        self.sim_timestep_length = sim_timestep_length

    def reset(self):
        self.history = []

    def add_action(self, action):
        self.history.append(action)

    def get_filtered_action(self):
        return np.mean(np.array(self.history[-self.sim_timestep_length:]), axis=0)
