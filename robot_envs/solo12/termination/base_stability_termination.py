import numpy as np


class BaseStabilityTermination():

    def __init__(self,
                 robot,
                 max_angle=0.2,
                 start_timestep=0,
                 check_x_angle=True,
                 check_y_angle=True,
                 allowed_range_x=None,
                 allowed_range_y=None,
                 base_pos_0_range=None,
                 base_pos_1_range=None,
                 base_pos_2_range=None,
                 base_ang_0_range=None,
                 base_ang_1_range=None,
                 base_ang_2_range=None):
        self.robot = robot
        self.max_angle = max_angle
        self.start_timestep = start_timestep
        self.check_x_angle = check_x_angle
        self.check_y_angle = check_y_angle
        self.allowed_range_x = allowed_range_x
        self.allowed_range_y = allowed_range_y
        self.base_pos_0_range = base_pos_0_range
        self.base_pos_1_range = base_pos_1_range
        self.base_pos_2_range = base_pos_2_range
        self.base_ang_0_range = base_ang_0_range
        self.base_ang_1_range = base_ang_1_range
        self.base_ang_2_range = base_ang_2_range

    def reset(self):
        self.current_timestep = 0

    def step(self):
        self.current_timestep += 1

    def done(self):
        if self.current_timestep < self.start_timestep:
            return False

        base_pos, base_orient = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)
        base_ang = self.robot.p.getEulerFromQuaternion(base_orient)

        if self.check_x_angle:
            if self.allowed_range_x is not None:
                if base_ang[0] < self.allowed_range_x[0] or base_ang[0] > self.allowed_range_x[1]:
                    return True
            elif np.abs(base_ang[0]) > self.max_angle:
                return True
        if self.check_y_angle:
            if self.allowed_range_y is not None:
                if base_ang[1] < self.allowed_range_y[0] or base_ang[1] > self.allowed_range_y[1]:
                    return True
            elif np.abs(base_ang[1]) > self.max_angle:
                return True

        if self.base_pos_0_range is not None:
            if base_pos[0] < self.base_pos_0_range[0] or base_pos[0] > self.base_pos_0_range[1]:
                return True
        if self.base_pos_1_range is not None:
            if base_pos[1] < self.base_pos_1_range[0] or base_pos[1] > self.base_pos_1_range[1]:
                return True
        if self.base_pos_2_range is not None:
            if base_pos[2] < self.base_pos_2_range[0] or base_pos[2] > self.base_pos_2_range[1]:
                return True
        if self.base_ang_0_range is not None:
            if base_ang[0] < self.base_ang_0_range[0] or base_ang[0] > self.base_ang_0_range[1]:
                return True
        if self.base_ang_1_range is not None:
            if base_ang[1] < self.base_ang_1_range[0] or base_ang[1] > self.base_ang_1_range[1]:
                return True
        if self.base_ang_2_range is not None:
            if base_ang[2] < self.base_ang_2_range[0] or base_ang[2] > self.base_ang_2_range[1]:
                return True

        return False
