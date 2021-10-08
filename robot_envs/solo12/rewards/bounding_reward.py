import os
import numpy as np
import matplotlib.pyplot as plt


class BoundingReward():

    def __init__(self,
                 robot,
                 traj_file,
                 point_select,
                 k=1.0,
                 demonstration_z_correction = 0.016832177805689,
                 ignore_x=False,
                 calc_at_sim_step=True):
        self.robot = robot
        self.calc_at_sim_step = calc_at_sim_step

        traj_directory = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) \
                         + '/trajectories/'
        pos_file = traj_directory + traj_file + '_positions.dat'
        pos = np.loadtxt(pos_file)[:, 1:]
        pos[:, 2] += demonstration_z_correction

        self.pos = pos[point_select[0]:point_select[1]:point_select[2]]

        self.ang = []
        for i in range(self.pos.shape[0]):
            self.ang.append(self.robot.p.getEulerFromQuaternion(self.pos[i, 3:7]))
        self.ang = np.array(self.ang)

        # print(self.pos.shape, self.ang.shape)
        #
        # plt.rcParams["figure.figsize"] = (10, 4)
        # fig, axes = plt.subplots(1, 2, sharex=True)
        # axes[0].scatter(self.pos[:, 0], self.pos[:, 2])
        # axes[1].scatter(self.pos[:, 0], self.ang[:, 1])
        # axes[0].grid(True)
        # axes[1].grid(True)
        # plt.tight_layout()
        # plt.show()

        self.k = k
        self.ignore_x = ignore_x





    def reset(self):
        pass

    def step(self):
        pass

    def get_reward(self):
        base_pos, base_orient = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)
        base_ang = self.robot.p.getEulerFromQuaternion(base_orient)

        def dist(pos1, ang1, pos2, ang2):
            dist_vec = np.zeros(3)
            if self.ignore_x:
                dist_vec[0] = 0.0
            else:
                dist_vec[0] = pos2[0] - pos1[0]
            dist_vec[1] = pos2[2] - pos1[2]
            dist_vec[2] = ang2[1] - ang2[1]
            return np.linalg.norm(dist_vec)

        min_dist = dist(base_pos, base_ang, self.pos[0], self.ang[0])
        for i in range(1, self.pos.shape[0]):
            new_dist = dist(base_pos, base_ang, self.pos[i], self.ang[i])
            if new_dist < min_dist:
                min_dist = new_dist

        # return min_dist
        if min_dist > 0.5 / 8.0:
            return 0.0
        else:
            # return min_dist
            return self.k * 2.0 * (0.5 / 8.0 - min_dist)
