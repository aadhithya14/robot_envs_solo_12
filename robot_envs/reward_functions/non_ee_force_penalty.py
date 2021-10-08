import numpy as np

class NonEEForcePenalty():

    def __init__(self, robot, params):
        self.robot = robot
        self.params = params

    def reset(self):
        pass

    def get_reward(self):
        non_ee_force = self.robot.get_non_endeff_ground_force_scalar()

        if self.params['variant'] == 'tanh':
            penalty = np.tanh(non_ee_force)
        elif self.params['variant'] == 'linear': 
            penalty = non_ee_force
        else:
            assert False, 'Unknown NonEEForcePenalty variant: ' + self.params['variant']

        k = self.params['k']

        return -k * penalty