import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pybullet_data
from robot_envs.solo12.robot import Robot
from robot_envs.bolt.movable_disks_surface.movable_disks_surface import MovableDisksSurface
from robot_envs.solo12.surface_control import SurfaceControl


class Solo12(Robot):

    def __init__(self, demo_traj_file=None, **kwargs):
        self.demo_traj_file = demo_traj_file
        if self.demo_traj_file is not None:
            traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                             + '/trajectories/'
            self.demo_traj_f = np.loadtxt(traj_directory + self.demo_traj_file)

        super(Solo12, self).__init__(**kwargs)


    def robot_specific_init(self, movable_disks_surface_conf, environment_conf, ground_conf={}, lateral_friction=None, special_setup=None):
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        urdf_base_string = str(os.path.dirname(os.path.abspath(__file__))) + '/urdf/'
        #print(urdf_base_string)

        self.surface_control = SurfaceControl(self, **ground_conf)

        self.movable_disks_surface = None

        if self.with_motor_rotor:
            self.robot_id = self.p.loadURDF(urdf_base_string + "solo12_with_motor_rotor.urdf", [0, 0, 1], flags=self.p.URDF_USE_INERTIA_FROM_FILE)
        else:
            self.robot_id = self.p.loadURDF(urdf_base_string + "solo12.urdf", [0, 0, 1], flags=self.p.URDF_USE_INERTIA_FROM_FILE)

        # MOVING BASE UP AND DOWN
        if special_setup == 'base_locked_z':
            self.p.createConstraint(self.robot_id, -1, -1, -1, self.p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [0, 0, 1])

        if self.with_motor_rotor:
            c = self.p.createConstraint(self.robot_id, 0, self.robot_id, 1, jointType=self.p.JOINT_GEAR, jointAxis=[1, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)
            c = self.p.createConstraint(self.robot_id, 2, self.robot_id, 3, jointType=self.p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)
            c = self.p.createConstraint(self.robot_id, 4, self.robot_id, 5, jointType=self.p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)
            c = self.p.createConstraint(self.robot_id, 7, self.robot_id, 8, jointType=self.p.JOINT_GEAR, jointAxis=[1, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)
            c = self.p.createConstraint(self.robot_id, 9, self.robot_id, 10, jointType=self.p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)
            c = self.p.createConstraint(self.robot_id, 11, self.robot_id, 12, jointType=self.p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)
            c = self.p.createConstraint(self.robot_id, 14, self.robot_id, 15, jointType=self.p.JOINT_GEAR, jointAxis=[1, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)
            c = self.p.createConstraint(self.robot_id, 16, self.robot_id, 17, jointType=self.p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)
            c = self.p.createConstraint(self.robot_id, 18, self.robot_id, 19, jointType=self.p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)
            c = self.p.createConstraint(self.robot_id, 21, self.robot_id, 22, jointType=self.p.JOINT_GEAR, jointAxis=[1, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)
            c = self.p.createConstraint(self.robot_id, 23, self.robot_id, 24, jointType=self.p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)
            c = self.p.createConstraint(self.robot_id, 25, self.robot_id, 26, jointType=self.p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
            self.p.changeConstraint(c, gearRatio=-9, maxForce=10000)

        

        for joint_id in self.get_cont_joint_ids():
            self.p.changeDynamics(self.robot_id,
                                  joint_id,
                                  linearDamping=0.04,
                                  angularDamping=0.04,
                                  restitution=0.0,
                                  lateralFriction=0.0,
                                  maxJointVelocity=1000)

        self.lateral_friction = lateral_friction
        if self.lateral_friction is not None and not isinstance(self.lateral_friction, list):
            for link_id in range(-1, self.p.getNumJoints(self.robot_id)):
                self.p.changeDynamics(self.robot_id,
                                      link_id,
                                      lateralFriction=self.lateral_friction)
            for link_id in range(-1, self.p.getNumJoints(self.surface_id)):
                self.p.changeDynamics(self.surface_id,
                                      link_id,
                                      lateralFriction=self.lateral_friction)

    def get_obs_joint_ids(self):
        return self.get_cont_joint_ids()

    def get_cont_joint_ids(self):
        if self.with_motor_rotor:
            return [1, 3, 5, 8, 10, 12, 15, 17, 19, 22, 24, 26]
        else:
            return [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

    def get_upper_leg_link_ids(self):
        if self.with_motor_rotor:
            return [3, 10 ,17 , 24]
        else:
            return [1,5,9,13]

    def get_cont_joint_type(self):
        return ['circular'] * 12

    def get_endeff_link_ids(self):
        if self.with_motor_rotor:
            return [6, 13, 20, 27]
        else:
            return [3, 7, 11, 15]

    def air_and_ground_init(self,
                            hip_range=[0.5, 1.3],
                            knee_variance=[-0.1, 0.1],
                            height_range=[0.0, 0.8],
                            timesteps_to_stabilize=2000):

        hip    = np.random.uniform(hip_range[0], hip_range[1])
        knee   = 2.0 * hip + np.random.uniform(knee_variance[0], knee_variance[1])
        height = np.random.uniform(height_range[0], height_range[1])

        ndof = 12
        kp   = 5.0
        kd   = 0.1

        qdes    = np.array([hip, -knee,  hip, -knee, -hip, knee, -hip, knee])
        qdotdes = np.zeros(ndof)

        constraint_id = self.p.createConstraint(self.robot_id, -1, -1, -1, self.p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], [0, 0, 1])

        self.p.resetBasePositionAndOrientation(
            bodyUniqueId=self.robot_id,
            posObj=[0, 0, height_range[1]],
            ornObj=[0, 0, 0, 1])

        for i in range(timesteps_to_stabilize):
            q, qdot = self.get_cont_joint_state()
            tau = kp * (qdes - q) + kd * (qdotdes - qdot)
            self.torque_control(tau)

        pos, _ = self.p.getBasePositionAndOrientation(self.robot_id)
        if pos[2] < height:
            self.p.resetBasePositionAndOrientation(
                    bodyUniqueId=self.robot_id,
                    posObj=[0, 0, height],
                    ornObj=[0, 0, 0, 1])

        self.p.removeConstraint(constraint_id)
    
    def set_initial_configuration(self,
                                  on_demo_traj=False,
                                  use_demo_vel=False,
                                  fixed_start_timestep=None,
                                  throw=None,
                                  box_in_air=None,
                                  air_and_ground=None):

        if 'trajectory_imitation_reward' in self.reward_parts:
            pos = self.reward_parts['trajectory_imitation_reward'].pos
            vel = self.reward_parts['trajectory_imitation_reward'].vel

            init_point = 0
            if self.reward_parts['trajectory_imitation_reward'].random_point_init:
                #print(pos.shape[0])
                init_point = np.random.randint(pos.shape[0])

            self.reward_parts['trajectory_imitation_reward'].current_timestep = init_point
            if 'imitation_length_termination' in self.termination_dict:
                self.termination_dict['imitation_length_termination'].current_timestep = init_point
            
            #print(self.termination_dict['imitation_length_termination'].current_timestep)
            #print("***")
            #print(self.termination_dict['imitation_length_termination'].done())
            self.p.resetBasePositionAndOrientation(
                bodyUniqueId=self.robot_id,
                posObj=pos[init_point, 0:3],
                ornObj=pos[init_point, 3:7])

            #print(vel[init_point, 0:3])   
            #print(vel[init_point, 3:6])
            #print("-----")

            self.p.resetBaseVelocity(
                objectUniqueId=self.robot_id,
                linearVelocity=vel[init_point, 0:3],
                angularVelocity=vel[init_point, 3:6])
                
            for i in range(self.num_obs_joints):
                #print(pos[init_point, i + 7])
                #print(vel[init_point, i + 6])
                #print("*****")
                self.p.resetJointState(
                    bodyUniqueId=self.robot_id,
                    jointIndex=self.obs_joint_ids[i],
                    targetValue=pos[init_point, i + 7],
                    targetVelocity=vel[init_point, i + 6])

        elif air_and_ground is not None:
            self.air_and_ground_init(**air_and_ground)

        elif box_in_air is not None:

            self.initial_base_vel = np.zeros(3)
            self.initial_base_ang_vel = np.zeros(3)
            self.initial_joint_vel = np.zeros(12)

            joint_ranges = [[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1],[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[-0.1,0.1]]
            if 'joint_ranges' in box_in_air:
                joint_ranges = box_in_air['joint_ranges']

            base_x_pos_range = [0.0, 0.0]
            if 'base_x_pos_range' in box_in_air:
                base_x_pos_range = box_in_air['base_x_pos_range']

            base_y_pos_range = [0.0, 0.0]
            if 'base_y_pos_range' in box_in_air:
                base_y_pos_range = box_in_air['base_y_pos_range']

            base_z_pos_range = [0.2, 0.2]
            if 'base_z_pos_range' in box_in_air:
                base_z_pos_range = box_in_air['base_z_pos_range']

            base_x_ang_range = [0.0, 0.0]
            if 'base_x_ang_range' in box_in_air:
                base_x_ang_range = box_in_air['base_x_ang_range']

            base_y_ang_range = [0.0, 0.0]
            if 'base_y_ang_range' in box_in_air:
                base_y_ang_range = box_in_air['base_y_ang_range']

            base_z_ang_range = [0.0, 0.0]
            if 'base_z_ang_range' in box_in_air:
                base_z_ang_range = box_in_air['base_z_ang_range']

            self.initial_base_pos = np.array([
                np.random.uniform(base_x_pos_range[0], base_x_pos_range[1]),
                np.random.uniform(base_y_pos_range[0], base_y_pos_range[1]),
                np.random.uniform(base_z_pos_range[0], base_z_pos_range[1])])

            self.initial_base_orn = np.array(self.p.getQuaternionFromEuler([
                np.random.uniform(base_x_ang_range[0], base_x_ang_range[1]),
                np.random.uniform(base_y_ang_range[0], base_y_ang_range[1]),
                np.random.uniform(base_z_ang_range[0], base_z_ang_range[1])]))

            self.initial_joint_pos = np.zeros(self.num_obs_joints)
            for i in range(self.num_obs_joints):
                self.initial_joint_pos[i] = np.random.uniform(joint_ranges[i][0], joint_ranges[i][1])

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

        else:

            self.initial_base_vel = np.zeros(3)
            self.initial_base_ang_vel = np.zeros(3)
            self.initial_joint_vel = np.zeros(12)

            self.initial_base_pos = np.array([0.0, 0.0, 1.0])
            self.initial_base_orn = np.array(self.p.getQuaternionFromEuler([0.0, 0.0, 0.0]))
            self.initial_joint_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0])

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
