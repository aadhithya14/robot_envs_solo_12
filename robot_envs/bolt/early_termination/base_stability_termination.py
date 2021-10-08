class BaseStabilityTermination():

    def __init__(self,
                 robot,
                 base_ang_0_range=None,
                 base_ang_1_range=None,
                 base_pos_2_range=None):
        self.robot = robot
        self.base_ang_0_range = base_ang_0_range
        self.base_ang_1_range = base_ang_1_range
        self.base_pos_2_range = base_pos_2_range

    def reset(self):
        pass

    def step(self):
        pass

    def done(self):
        base_pos, base_orient = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)
        base_ang = self.robot.p.getEulerFromQuaternion(base_orient)

        if self.base_ang_0_range is not None:
            if base_ang[0] < self.base_ang_0_range[0] or base_ang[0] > self.base_ang_0_range[1]:
                return True

        if self.base_ang_1_range is not None:
            if base_ang[1] < self.base_ang_1_range[0] or base_ang[1] > self.base_ang_1_range[1]:
                return True

        if self.base_pos_2_range is not None:
            if base_pos[2] < self.base_pos_2_range[0] or base_pos[2] > self.base_pos_2_range[1]:
                return True

        return False
