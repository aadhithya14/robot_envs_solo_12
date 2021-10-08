import numpy as np

class GoalPositionReward():

    def __init__(self, get_state, version, goal, value=None, radius=None, k_p=1.0, box_size=None):
        self.get_state = get_state
        self.version = version
        self.goal = goal
        self.value = value
        self.radius = radius
        self.k_p = k_p
        self.box_size = box_size


    def get_reward(self):
        state = self.get_state()
        dist = np.linalg.norm(self.goal - state)

        if self.version == 'discrete':
            if dist <= self.radius:
                return self.value
            else:
                return 0.0
        elif self.version == 'linear':
            return self.k_p * (self.box_size * np.sqrt(2) - dist)
        elif self.version == 'continous':
            return 1.0 / (1.0 + dist)
