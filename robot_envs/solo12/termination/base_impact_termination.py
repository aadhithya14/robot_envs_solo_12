import numpy as np


class BaseImpactTermination():

    def __init__(self, robot):
        self.robot = robot

    def reset(self):
        pass

    def step(self):
        pass

    def done(self):
        # contacts = self.robot.p.getContactPoints(
        #                bodyA=self.robot.robot_id,
        #                bodyB=self.robot.surface_id)
        # print([contact[3] for contact in contacts])
        # return False

        contacts = self.robot.p.getContactPoints(
                       bodyA=self.robot.robot_id,
                       bodyB=self.robot.surface_id,
                       linkIndexA=-1)
        if len(contacts) > 0:
            return True
        else:
            return False
