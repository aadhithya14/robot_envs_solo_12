import numpy as np


class GroundImpactTermination():

    def __init__(self, robot, max_allowed_force):
        self.robot = robot
        self.max_allowed_force = max_allowed_force

    def reset(self):
        pass

    def step(self):
        pass

    def done(self):
        force = self.robot.get_total_ground_force(with_friction=True)
        # force_norm = np.linalg.norm(force)
        force_z = force[2]

        if force_z > self.max_allowed_force:
            return True
        else:
            return False
