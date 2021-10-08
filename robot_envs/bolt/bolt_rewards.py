import os
import numpy as np


class BaseStabilityPenalty():

    def __init__(self, robot, mean, sigma, k=1.0, calc_at_sim_step=False):
        self.robot = robot
        self.mean = mean
        self.sigma = sigma
        self.k = k
        self.calc_at_sim_step = calc_at_sim_step

    def step(self):
        pass

    def reset(self):
        pass

    def penalty(self, value, mean, sigma):
        return np.exp(-np.square((value - mean) / sigma))

    def get_reward(self):
        pos, orient = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)
        base_angles = self.robot.p.getEulerFromQuaternion(orient)

        reward = self.k
        reward *= self.penalty(pos[2], self.mean[0], self.sigma[0])
        reward *= self.penalty(base_angles[0], self.mean[1], self.sigma[1])
        reward *= self.penalty(base_angles[1], self.mean[2], self.sigma[2])

        return reward


class Quaternion:

    @staticmethod
    def norm(v):
        return np.sqrt(np.sum(np.square(v)))

    @staticmethod
    def quat_prod(q, r):
        t = np.zeros(4)
        t[0] = q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3]
        t[1] = q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2]
        t[2] = q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1]
        t[3] = q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0]
        if Quaternion.norm(t) == 0:
            return np.array([0, 0, 0, 0])
        return t / Quaternion.norm(t)

    @staticmethod
    def quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def quat_log(q):
        if Quaternion.norm(q[1:]) == 0:
            return np.array([0, 0, 0])
        return 2 * np.arccos(q[0]) / Quaternion.norm(q[1:]) * q[1:]

    @staticmethod
    def quat_exp(v):
        t = np.sin(0.5 * Quaternion.norm(v)) / Quaternion.norm(v) * v
        return np.array([np.cos(0.5 * Quaternion.norm(v)), t[0], t[1], t[2]])

    @staticmethod
    def quat_dist(q, r):
        return 2 * Quaternion.quat_log(Quaternion.quat_prod(q, Quaternion.quat_conj(r)))


class TrajectoryImitationReward():

    def __init__(self, robot, scaling_variant='default', ratios_variant='default', calc_at_sim_step=False):
        self.robot = robot
        self.scaling_variant = scaling_variant
        self.ratios_variant = ratios_variant
        self.calc_at_sim_step = calc_at_sim_step
        self.calc_endeff_positions()

    def calc_endeff_positions(self):
        endeff_pos_file = self.robot.demo_traj_file[:-5] + 'endeff_pos'
        traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                         + '/trajectories/'
        if os.path.exists(traj_directory + endeff_pos_file):
            self.endeff_pos = np.loadtxt(traj_directory + endeff_pos_file)
        else:
            self.endeff_pos = []
            traj = self.robot.demo_traj
            for i in range(traj.shape[0]):
                self.robot.p.resetBasePositionAndOrientation(
                    bodyUniqueId=self.robot.robot_id,
                    posObj=traj[i, 0:3],
                    ornObj=traj[i, 3:7])
                for j in range(self.robot.num_obs_joints):
                    self.robot.p.resetJointState(
                        bodyUniqueId=self.robot.robot_id,
                        jointIndex=self.robot.obs_joint_ids[j],
                        targetValue=traj[i, j + 7],
                        targetVelocity=0.0)

                self.endeff_pos.append(self.robot.get_endeff_state()[0])
            self.endeff_pos = np.array(self.endeff_pos)
            np.savetxt(traj_directory + endeff_pos_file, self.endeff_pos)

    def step(self):
        self.current_timestep += self.robot.cont_timestep_mult

    def reset(self):
        self.current_timestep = self.robot.demo_traj_start_timestep
        # self.current_timestep = 0

    def get_reward(self):
        base_pos, base_orient = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)
        base_vel, base_angvel = self.robot.p.getBaseVelocity(self.robot.robot_id)
        joint_pos, joint_vel  = self.robot.get_obs_joint_state()
        endeff_pos, _         = self.robot.get_endeff_state()

        # print('base_pos', base_pos)
        # print('base_orient', base_orient)

        # print('base_vel', base_vel)
        # print('base_angvel', base_angvel)

        des_index = self.current_timestep + self.robot.cont_timestep_mult
        if des_index >= self.robot.demo_traj.shape[0]:
            return 0.0

        # self.robot.log.add('joint_pos', joint_pos)
        # self.robot.log.add('joint_vel', joint_vel)
        # self.robot.log.add('endeff_pos', endeff_pos)
        # self.robot.log.add('base_pos', base_pos)
        # self.robot.log.add('base_orient', base_orient)
        # self.robot.log.add('base_vel', base_vel)
        # self.robot.log.add('base_angvel', base_angvel)



        # print('des_index', des_index)

        des_joint_pos   = self.robot.demo_traj[des_index, 7:13]
        des_joint_vel   = self.robot.demo_traj[des_index, 19:25]
        des_endeff_pos  = self.endeff_pos[des_index]
        des_base_pos    = self.robot.demo_traj[des_index, 0:3]
        des_base_orient = self.robot.demo_traj[des_index, 3:7]
        des_base_vel    = self.robot.demo_traj[des_index, 13:16]
        des_base_angvel = self.robot.demo_traj[des_index, 16:19]

        # self.robot.log.add('des_joint_pos', des_joint_pos)
        # self.robot.log.add('des_joint_vel', des_joint_vel)
        # self.robot.log.add('des_endeff_pos', des_endeff_pos)
        # self.robot.log.add('des_base_pos', des_base_pos)
        # self.robot.log.add('des_base_orient', des_base_orient)
        # self.robot.log.add('des_base_vel', des_base_vel)
        # self.robot.log.add('des_base_angvel', des_base_angvel)

        # print('des_base_vel', des_base_vel)
        # print('des_base_angvel', des_base_angvel)

        # print('des_base_pos', des_base_pos)
        # print('des_base_orient', des_base_orient)

        if self.scaling_variant == 'default':
            joint_pos_rew       = np.exp( -5.0 * np.linalg.norm(des_joint_pos - joint_pos))
            joint_vel_rew       = np.exp( -0.1 * np.linalg.norm(des_joint_vel - joint_vel))
            endeff_pos_rew      = np.exp(-40.0 * np.linalg.norm(des_endeff_pos - endeff_pos))
            base_pos_orient_rew = np.exp(-20.0 * np.linalg.norm(des_base_pos - base_pos)
                                         -10.0 * np.linalg.norm(Quaternion.quat_dist(des_base_orient, base_orient)))
            base_vel_angvel_rew = np.exp( -2.0 * np.linalg.norm(des_base_vel - base_vel)
                                          -0.2 * np.linalg.norm(des_base_angvel - base_angvel))
        elif self.scaling_variant == 'less_severe':
            joint_pos_rew       = np.exp( -1.0 * np.linalg.norm(des_joint_pos - joint_pos))
            joint_vel_rew       = np.exp( -0.05 * np.linalg.norm(des_joint_vel - joint_vel))
            endeff_pos_rew      = np.exp(-5.0 * np.linalg.norm(des_endeff_pos - endeff_pos))
            base_pos_orient_rew = np.exp(-10.0 * np.linalg.norm(des_base_pos - base_pos)
                                         -5.0 * np.linalg.norm(Quaternion.quat_dist(des_base_orient, base_orient)))
            base_vel_angvel_rew = np.exp( -2.0 * np.linalg.norm(des_base_vel - base_vel)
                                          -0.2 * np.linalg.norm(des_base_angvel - base_angvel))
        else:
            assert False, 'Unknown TrajectoryImitationReward scaling_variant: ' + self.scaling_variant

        # print(joint_pos_rew)
        # print(joint_vel_rew)
        # print(endeff_pos_rew)
        # print(base_pos_orient_rew)
        # print(base_vel_angvel_rew)
        # assert False
        # self.robot.log.add('joint_pos_rew', joint_pos_rew)
        # self.robot.log.add('joint_vel_rew', joint_vel_rew)
        # self.robot.log.add('endeff_pos_rew', endeff_pos_rew)
        # self.robot.log.add('base_pos_orient_rew', base_pos_orient_rew)
        # self.robot.log.add('base_vel_angvel_rew', base_vel_angvel_rew)

        # print(self.variant)
        if self.ratios_variant == 'default':
            reward =    0.5 * joint_pos_rew           \
                     + 0.05 * joint_vel_rew           \
                     + 0.2  * endeff_pos_rew          \
                     + 0.15 * base_pos_orient_rew     \
                     + 0.1  * base_vel_angvel_rew
        elif self.ratios_variant == 'base_biased':
            reward =   0.25 * joint_pos_rew           \
                     + 0.05 * joint_vel_rew           \
                     + 0.2  * endeff_pos_rew          \
                     + 0.3  * base_pos_orient_rew     \
                     + 0.2  * base_vel_angvel_rew
        elif self.ratios_variant == 'pos_only':
            reward =   0.3  * joint_pos_rew           \
                     + 0.0  * joint_vel_rew           \
                     + 0.3  * endeff_pos_rew          \
                     + 0.4  * base_pos_orient_rew     \
                     + 0.0  * base_vel_angvel_rew
        else:
            assert False, 'Unknown TrajectoryImitationReward ratios_variant: ' + self.ratios_variant

        return reward


class TorqueSmoothnessPenalty():

    def __init__(self, robot, sigma, k=1.0, env_log=None, calc_at_sim_step=False):
        self.robot = robot
        self.sigma = sigma
        self.k = k
        self.env_log = env_log
        self.calc_at_sim_step = calc_at_sim_step

    def step(self):
        self.last_torque = self.robot.des_torque.copy()

    def reset(self):
        self.last_torque = np.zeros(self.robot.num_cont_joints)

    def penalty(self, value, mean, sigma):
        return np.exp(-np.square((value - mean) / sigma))

    def get_reward(self):
        diff = self.robot.des_torque - self.last_torque
        rewards = self.penalty(diff, 0.0, self.sigma)
        if self.env_log is not None:
            self.env_log.add('torque_smoothness_diff', diff.tolist())
            self.env_log.add('torque_smoothness_penalties', rewards.tolist())
        return self.k * np.sum(rewards) / rewards.shape[0]


# DONE: Add forward motion reward.
# DONE: Test LinearDistanceToGoalReward.
class LinearDistanceToGoalReward():

    def __init__(self, robot, des_pos=1.75, max_dist=1.75, k=1.0, calc_at_sim_step=False):
        self.robot = robot
        self.des_pos = des_pos
        self.max_dist = max_dist
        self.k = k
        self.calc_at_sim_step = calc_at_sim_step

    def step(self):
        pass

    def reset(self):
        pass

    def get_reward(self):
        pos, _ = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)
        dist = abs(self.des_pos - pos[0])
        reward = 1.0 - dist / self.max_dist
        if reward < 0.0:
            reward = 0.0
        return self.k * reward
