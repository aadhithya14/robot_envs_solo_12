import os
import numpy as np
from pyquaternion import Quaternion


class TrajectoryImitationReward():

    def __init__(self,
                 robot,
                 traj_file,
                 k=1.0,
                 random_point_init=False,
                 fixed_init_point=None,
                 calc_at_sim_step=False,
                 joint_pos_rew_k   = 0.2,
                 joint_vel_rew_k   = 0.1,
                 base_pos_rew_k    = 0.25,
                 base_orient_rew_k = 0.25,
                 base_vel_rew_k    = 0.1,
                 base_angvel_rew_k = 0.1,
                 # =========================
                 joint_pos_exp_coeff = -0.5,
                 joint_vel_exp_coeff = -0.05,
                 base_pos_exp_coeff = -5.0,
                 base_orient_exp_coeff = -5.0,
                 base_vel_exp_coeff = -0.5,
                 base_angvel_exp_coeff = -0.2,
                 demonstration_z_correction = 0.04334521635,#0.016832177805689
                 ignore_z = False,
                 ignore_xyplane = False):

        self.robot = robot
        self.k = k
        self.random_point_init = random_point_init
        self.fixed_init_point = fixed_init_point
        self.calc_at_sim_step = calc_at_sim_step

        self.joint_pos_rew_k   = joint_pos_rew_k
        self.joint_vel_rew_k   = joint_vel_rew_k
        self.base_pos_rew_k    = base_pos_rew_k
        self.base_orient_rew_k = base_orient_rew_k
        self.base_vel_rew_k    = base_vel_rew_k
        self.base_angvel_rew_k = base_angvel_rew_k

        self.joint_pos_exp_coeff = joint_pos_exp_coeff
        self.joint_vel_exp_coeff = joint_vel_exp_coeff
        self.base_pos_exp_coeff = base_pos_exp_coeff
        self.base_orient_exp_coeff = base_orient_exp_coeff
        self.base_vel_exp_coeff = base_vel_exp_coeff
        self.base_angvel_exp_coeff = base_angvel_exp_coeff

        self.ignore_z = ignore_z
        self.ignore_xyplane = ignore_xyplane

        traj_directory = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) \
                         + '/trajectories/'

        pos_file = traj_directory + traj_file + '_positions.dat'
        vel_file = traj_directory + traj_file + '_velocities.dat'

        
        if traj_file == "solo12_jump_two_jumps":
            self.pos_f = np.loadtxt(pos_file)[:, 1:]
            self.vel_f = np.loadtxt(vel_file)[:, 1:]
            self.pos_l = self.pos_f.tolist()
            self.vel_l = self.vel_f.tolist()
            self.pos=[]
            self.vel=[]
            """print(self.pos_l[0,0:7])
            print(self.pos_l[0,7:9])
            print(self.pos_l[0,9:11])
            print(self.pos_l[0,11:13])
            print(self.pos_l[0,13:15])"""
        
            for i in range(self.pos_f.shape[0]):
                self.pos_val  = self.pos_l[i][0:7]+[0]+self.pos_l[i][7:9]+[0]+self.pos_l[i][9:11]+[0]+self.pos_l[i][11:13]+[0]+self.pos_l[i][13:15]
                self.pos.append(self.pos_val)
                self.vel_val = self.vel_l[i][0:7]+[0]+self.vel_l[i][7:9]+[0]+self.vel_l[i][9:11]+[0]+self.vel_l[i][11:13]+[0]+self.vel_l[i][13:15]
                self.vel.append(self.vel_val)
            self.pos=np.array(self.pos)
            self.vel=np.array(self.vel)
        else:
            self.pos=np.loadtxt(pos_file)[:, 1:]
            self.vel=np.loadtxt(vel_file)[:, 1:]
        self.pos[:, 2] += demonstration_z_correction

        self.robot.demo_traj_start_timestep = 0
        self.robot.demo_traj = self.pos

    def step(self):
        self.current_timestep += self.robot.cont_timestep_mult

    def reset(self):
        # self.current_timestep = 0
        # Setting this in Solo8 init for now
        pass

    def get_reward(self):
        base_pos, base_orient = self.robot.p.getBasePositionAndOrientation(self.robot.robot_id)
        base_vel, base_angvel = self.robot.p.getBaseVelocity(self.robot.robot_id)
        joint_pos, joint_vel  = self.robot.get_obs_joint_state()

        # self.robot.log.add('joint_pos', joint_pos)
        # self.robot.log.add('joint_vel', joint_vel)
        # self.robot.log.add('base_pos', base_pos)
        # self.robot.log.add('base_orient', base_orient)
        # self.robot.log.add('base_vel', base_vel)
        # self.robot.log.add('base_angvel', base_angvel)

        des_index = self.current_timestep + self.robot.cont_timestep_mult
        # print(des_index)
        if des_index >= self.pos.shape[0]:
            return 0.0

        des_joint_pos   = self.pos[des_index, 7:19]
        des_joint_vel   = self.vel[des_index, 6:18]
        des_base_pos    = self.pos[des_index, 0:3]
        des_base_orient = self.pos[des_index, 3:7]
        des_base_vel    = self.vel[des_index, 0:3]
        des_base_angvel = self.vel[des_index, 3:6]

        # self.robot.log.add('des_joint_pos', des_joint_pos)
        # self.robot.log.add('des_joint_vel', des_joint_vel)
        # self.robot.log.add('des_base_pos', des_base_pos)
        # self.robot.log.add('des_base_orient', des_base_orient)
        # self.robot.log.add('des_base_vel', des_base_vel)
        # self.robot.log.add('des_base_angvel', des_base_angvel)

        if self.ignore_xyplane:
            des_base_pos = np.array([0.0, 0.0, des_base_pos[2]])
            base_pos = np.array([0.0, 0.0, base_pos[2]])

            des_base_vel = np.array([0.0, 0.0, des_base_vel[2]])
            base_vel = np.array([0.0, 0.0, base_vel[2]])

            ea = self.robot.p.getEulerFromQuaternion(des_base_orient)
            des_base_orient = self.robot.p.getQuaternionFromEuler((ea[0], ea[1], 0.0))
            ea = self.robot.p.getEulerFromQuaternion(base_orient)
            base_orient = self.robot.p.getQuaternionFromEuler((ea[0], ea[1], 0.0))

            des_base_angvel = np.array([des_base_angvel[0], des_base_angvel[1], 0.0])
            base_angvel = np.array([base_angvel[0], base_angvel[1], 0.0])

        des_base_orient = Quaternion(x=des_base_orient[0], y=des_base_orient[1], z=des_base_orient[2], w=des_base_orient[3])
        base_orient = Quaternion(x=base_orient[0], y=base_orient[1], z=base_orient[2], w=base_orient[3])

        joint_pos_rew   = np.exp(self.joint_pos_exp_coeff   * np.linalg.norm(des_joint_pos - joint_pos))
        joint_vel_rew   = np.exp(self.joint_vel_exp_coeff   * np.linalg.norm(des_joint_vel - joint_vel))
        if self.ignore_z:
            base_pos_rew = np.exp(self.base_pos_exp_coeff    * np.linalg.norm(des_base_pos[:2] - base_pos[:2]))
        else:
            base_pos_rew = np.exp(self.base_pos_exp_coeff    * np.linalg.norm(des_base_pos - base_pos))
        base_orient_rew = np.exp(self.base_orient_exp_coeff * Quaternion.absolute_distance(des_base_orient, base_orient))
        base_vel_rew    = np.exp(self.base_vel_exp_coeff    * np.linalg.norm(des_base_vel - base_vel))
        base_angvel_rew = np.exp(self.base_angvel_exp_coeff * np.linalg.norm(des_base_angvel - base_angvel))

        self.robot.log.add('joint_pos_rew', joint_pos_rew)
        self.robot.log.add('joint_vel_rew', joint_vel_rew)
        self.robot.log.add('base_pos_rew', base_pos_rew)
        self.robot.log.add('base_orient_rew', base_orient_rew)
        self.robot.log.add('base_vel_rew', base_vel_rew)
        self.robot.log.add('base_angvel_rew', base_angvel_rew)

        reward =   self.joint_pos_rew_k   * joint_pos_rew   \
                 + self.joint_vel_rew_k   * joint_vel_rew   \
                 + self.base_pos_rew_k    * base_pos_rew    \
                 + self.base_orient_rew_k * base_orient_rew \
                 + self.base_vel_rew_k    * base_vel_rew    \
                 + self.base_angvel_rew_k * base_angvel_rew

        return self.k * reward
