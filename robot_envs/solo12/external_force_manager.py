import numpy as np


class ExternalForceManager():

    def __init__(self, robot, xforce_range=None, yforce_range=None, xmoment_range=None, ymoment_range=None):
        self.robot = robot
        self.xforce_range = xforce_range
        self.yforce_range = yforce_range
        self.xmoment_range = xmoment_range
        self.ymoment_range = ymoment_range

    def apply(self):
        self.robot.p.applyExternalForce(self.robot.robot_id, -1, [self.xforce, self.yforce, 0], [0, 0, 0], self.robot.p.LINK_FRAME)

        self.robot.p.applyExternalForce(self.robot.robot_id, -1, [0, 0, self.xmoment / 2.0], [1.0, 0, 0], self.robot.p.LINK_FRAME)
        self.robot.p.applyExternalForce(self.robot.robot_id, -1, [0, 0, -self.xmoment / 2.0], [-1.0, 0, 0], self.robot.p.LINK_FRAME)

        self.robot.p.applyExternalForce(self.robot.robot_id, -1, [0, 0, self.ymoment / 2.0], [0, 1.0, 0], self.robot.p.LINK_FRAME)
        self.robot.p.applyExternalForce(self.robot.robot_id, -1, [0, 0, -self.ymoment / 2.0], [0, -1.0, 0], self.robot.p.LINK_FRAME)

    def reset(self):
        self.xforce = 0.0
        if self.xforce_range is not None:
            self.xforce = np.random.uniform(self.xforce_range[0], self.xforce_range[1])

        self.yforce = 0.0
        if self.yforce_range is not None:
            self.yforce = np.random.uniform(self.yforce_range[0], self.yforce_range[1])

        self.xmoment = 0.0
        if self.xmoment_range is not None:
            self.xmoment = np.random.uniform(self.xmoment_range[0], self.xmoment_range[1])

        self.ymoment = 0.0
        if self.ymoment_range is not None:
            self.ymoment = np.random.uniform(self.ymoment_range[0], self.ymoment_range[1])
