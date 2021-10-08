import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet_data
from robot_envs.bolt.robot import Robot
from robot_envs.bolt.movable_disks_surface.movable_disks_surface import MovableDisksSurface


class Bolt(Robot):

    def __init__(self, demo_traj_file=None, **kwargs):
        self.demo_traj_file = demo_traj_file
        if self.demo_traj_file is not None:
            traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                             + '/trajectories/'
            self.demo_traj = np.loadtxt(traj_directory + self.demo_traj_file)

        super(Bolt, self).__init__(**kwargs)


    def robot_specific_init(self, movable_disks_surface_conf):
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        urdf_base_string = str(os.path.dirname(os.path.abspath(__file__)))


        if movable_disks_surface_conf is not None:
            self.movable_disks_surface = MovableDisksSurface(self, **movable_disks_surface_conf)
        else:
            self.surface_id = self.p.loadURDF(urdf_base_string + "/plane_with_restitution.urdf")
            self.movable_disks_surface = None

        self.robot_id = self.p.loadURDF(urdf_base_string + "/bolt.urdf", [0, 0, 1], flags=self.p.URDF_USE_INERTIA_FROM_FILE)

        for joint_id in self.get_cont_joint_ids():
            self.p.changeDynamics(self.robot_id,
                                  joint_id,
                                  linearDamping=0.04,
                                  angularDamping=0.04,
                                  restitution=0.0,
                                  lateralFriction=1.0,
                                  maxJointVelocity=1000)

    def get_obs_joint_ids(self):
        return self.get_cont_joint_ids()

    def get_cont_joint_ids(self):
        return [0, 1, 2, 4, 5, 6]

    def get_cont_joint_type(self):
        return ['limited', 'limited', 'limited', 'limited', 'limited', 'limited']

    def get_endeff_link_ids(self):
        return [3, 7]

    def set_initial_configuration(self, on_demo_traj=False, use_demo_vel=False, fixed_start_timestep=None):
        self.initial_base_vel = np.zeros(3)
        self.initial_base_ang_vel = np.zeros(3)
        self.initial_joint_vel = np.zeros(6)

        if on_demo_traj:
            if fixed_start_timestep is None:
                self.demo_traj_start_timestep = np.random.randint(0, self.demo_traj.shape[0])
            else:
                self.demo_traj_start_timestep = fixed_start_timestep
            self.initial_base_pos = self.demo_traj[self.demo_traj_start_timestep, 0:3]
            self.initial_base_orn = self.demo_traj[self.demo_traj_start_timestep, 3:7]
            self.initial_joint_pos = self.demo_traj[self.demo_traj_start_timestep, 7:13]
            if use_demo_vel:
                self.initial_base_vel = self.demo_traj[self.demo_traj_start_timestep, 13:16]
                self.initial_base_ang_vel = self.demo_traj[self.demo_traj_start_timestep, 16:19]
                self.initial_joint_vel = self.demo_traj[self.demo_traj_start_timestep, 19:25]
        else:
            self.initial_base_pos = np.array([0.0, 0.0, 1.0])
            self.initial_base_orn = np.array(self.p.getQuaternionFromEuler([0.0, 0.0, 0.0]))
            self.initial_joint_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # print(self.initial_base_pos)
        # print(self.initial_base_orn)
        # print(self.initial_joint_pos )
        # print(self.initial_base_vel)
        # print(self.initial_base_ang_vel)
        # print(self.initial_joint_vel)

        self.p.resetBasePositionAndOrientation(
            bodyUniqueId=self.robot_id,
            posObj=self.initial_base_pos,
            ornObj=self.initial_base_orn)

        self.p.resetBaseVelocity(
            objectUniqueId=self.robot_id,
            linearVelocity=self.initial_base_vel,
            angularVelocity=self.initial_base_ang_vel)

        for i in range(self.num_obs_joints):
            self.p.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=self.obs_joint_ids[i],
                targetValue=self.initial_joint_pos[i],
                targetVelocity=self.initial_joint_vel[i])
