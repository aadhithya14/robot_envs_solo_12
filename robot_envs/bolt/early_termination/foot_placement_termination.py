import os
import numpy as np
from robot_envs.bolt.bolt_util import get_foot_locations, get_foot_location_clusters


class FootPlacementTermination():

    def __init__(self, robot, circle_size=0.025):
        self.robot = robot
        self.circle_size = circle_size

        foot_locations = get_foot_locations(self.robot, self.robot.demo_traj)
        self.clusters = get_foot_location_clusters(foot_locations, self.circle_size)

    def reset(self):
        pass

    def step(self):
        pass

    def done(self):
        contact_points = self.robot.p.getContactPoints(bodyA=self.robot.robot_id, bodyB=self.robot.surface_id)
        for cp in contact_points:
            fl = np.array(cp[6])
            inside = False
            for c in self.clusters:
                dist = np.linalg.norm(c - fl)

                if dist < self.circle_size:
                    inside = True
            if not inside:
                return True
        return False
