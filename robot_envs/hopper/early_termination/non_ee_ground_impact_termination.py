class NonEEGroundImpactTermination():

    def __init__(self, robot):
        self.robot = robot

    def reset(self):
        pass

    def step(self):
        pass

    def done(self):
        contacts = self.robot.p.getContactPoints(bodyA=self.robot.surface_id, bodyB=self.robot.robot_id)
        endeff_link_id = self.robot.get_endeff_link_id()
        for contact in contacts:
            if contact[4] != endeff_link_id:
                return True
        return False
