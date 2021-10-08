import numpy as np


class BoxesOnGround():

    def __init__(self, robot, active=False, height_range=[0.01, 0.03], single_box=None, four_boxes=False, one_of_four=False):
        self.robot = robot
        self.active = active
        self.height_range = height_range
        self.single_box = single_box
        self.four_boxes = four_boxes
        self.one_of_four = one_of_four

        if self.single_box is not None:
            dx = self.single_box['dx']
            dy = self.single_box['dy']
            dz = self.single_box['dz']
            cuid = self.robot.p.createCollisionShape(self.robot.p.GEOM_BOX, halfExtents = [dx, dy, dz])

            x = self.single_box['x']
            y = self.single_box['y']
            object_id = self.robot.p.createMultiBody(0.0, cuid, basePosition=[x, y, dz / 2.0])

        self.box_ids = []


    def reset(self):
        if self.active:
            for old_box_id in self.box_ids:
                self.robot.p.removeBody(old_box_id)

            self.box_ids = []

            if self.four_boxes:
                x = [0.19, 0.19, -0.19, -0.19]
                y = [0.1046, -0.1046, 0.1046, -0.1046]
                dx = 0.1
                dy = 0.1
                for i in range(4):
                    dz = np.random.uniform(self.height_range[0], self.height_range[1])
                    cuid = self.robot.p.createCollisionShape(self.robot.p.GEOM_BOX, halfExtents = [dx, dy, dz])
                    mass = 0 # Static box
                    object_id = self.robot.p.createMultiBody(mass, cuid, basePosition=[x[i], y[i], dz / 2.0])
                    if self.robot.lateral_friction is not None and not isinstance(self.robot.lateral_friction, list):
                        self.robot.p.changeDynamics(object_id, -1, lateralFriction=self.robot.lateral_friction)
                    if self.robot.lateral_friction is not None and isinstance(self.robot.lateral_friction, list):
                        self.robot.p.changeDynamics(object_id, -1, lateralFriction=self.robot.episode_friction)
                    self.box_ids.append(object_id)
            elif self.one_of_four:
                x = [0.19, 0.19, -0.19, -0.19]
                y = [0.1046, -0.1046, 0.1046, -0.1046]
                dx = 0.1
                dy = 0.1
                i = np.random.randint(4)
                dz = np.random.uniform(self.height_range[0], self.height_range[1])
                cuid = self.robot.p.createCollisionShape(self.robot.p.GEOM_BOX, halfExtents = [dx, dy, dz])
                mass = 0 # Static box
                object_id = self.robot.p.createMultiBody(mass, cuid, basePosition=[x[i], y[i], dz / 2.0])
                if self.robot.lateral_friction is not None and not isinstance(self.robot.lateral_friction, list):
                    self.robot.p.changeDynamics(object_id, -1, lateralFriction=self.robot.lateral_friction)
                if self.robot.lateral_friction is not None and isinstance(self.robot.lateral_friction, list):
                    self.robot.p.changeDynamics(object_id, -1, lateralFriction=self.robot.episode_friction)
                self.box_ids.append(object_id)
            else:
                for i in range(20):
                    dx = np.random.uniform(0.05, 0.1)
                    dy = np.random.uniform(0.1, 0.2)
                    h = np.random.uniform(self.height_range[0], self.height_range[1])
                    cuid = self.robot.p.createCollisionShape(self.robot.p.GEOM_BOX, halfExtents = [dx, dy, h])
                    mass = 0 # Static box
                    x = np.random.uniform(-0.5, 0.5)
                    y = np.random.uniform(-0.5, 0.5)
                    angle = np.random.uniform(-np.pi, np.pi)
                    object_id = self.robot.p.createMultiBody(mass, cuid, basePosition=[x, y, h / 2.0], baseOrientation=self.robot.p.getQuaternionFromEuler([0.0, 0.0, angle]))
                    if self.robot.lateral_friction is not None and not isinstance(self.robot.lateral_friction, list):
                        self.robot.p.changeDynamics(object_id, -1, lateralFriction=self.robot.lateral_friction)
                    if self.robot.lateral_friction is not None and isinstance(self.robot.lateral_friction, list):
                        self.robot.p.changeDynamics(object_id, -1, lateralFriction=self.robot.episode_friction)
                    self.box_ids.append(object_id)
