import os
import yaml
import json
import time
import numpy as np
from itertools import repeat

import pybullet_utils.bullet_client as bc
import pybullet
from robot_envs.hopper.robot import Robot

from robot_envs.hopper.moving_surface import MovingSurface




class Hopper(Robot, object):

    def __init__(self, **kwargs):
        super(Hopper, self).__init__(**kwargs)

    def init_simulation(
        self,
        floor_height=0.3,
        visualize=True,
        sim_timestep=0.001,
        cont_timestep_mult=16,
        lateral_friction=1.0,
        joint_damping=0.0,
        contact_stiffness=10000.0,
        contact_damping=200.0,
        contact_damping_multiplier=None,
        restitution=None,
        initial_joint_pos=[0.0499923, 1.41840898, 3.29190477],
        #initial_joint_pos=[0.3051549395776893, 0.5, -1.0],
        use_moving_surface=False,
        moving_surface_amplitude=0.1,
        small_moving_surface=False,
        real_system_init=False,
        big_init=False,
        big_init_base_vel=False,
        load_vertical_surface=None,
        vertical_surface_friction=None,
        base_damping=None,
        parkour_surface=False,
        init_traj_file=None,
        init_traj_file_use_velocity=False,
        parkour_random_init=False,
        joint_type='circular'):

        self.visualize = visualize
        if self.visualize:
            p = bc.BulletClient(connection_mode=pybullet.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.moving_surface = None
        urdf_base_string = os.path.dirname(os.path.abspath(__file__))
        if use_moving_surface:
            self.moving_surface = MovingSurface(p, sim_timestep, amplitude=moving_surface_amplitude, small_moving_surface=small_moving_surface)
            planeId = self.moving_surface.surface_id
        else:
            print(urdf_base_string)
            planeId = p.loadURDF(urdf_base_string + "/urdf/plane_with_restitution.urdf")

        print(urdf_base_string)
        cubeStartPos = [0, 0, 0]
        cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
        self.robotId = p.loadURDF(urdf_base_string + "/urdf/teststand.urdf",cubeStartPos, cubeStartOrientation, flags=p.URDF_USE_INERTIA_FROM_FILE)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.robotId)

        if load_vertical_surface == 'small':
            self.vertical_surface_id = p.loadURDF(urdf_base_string + "/vertical_surface_small.urdf")
        elif load_vertical_surface == 'medium':
            self.vertical_surface_id = p.loadURDF(urdf_base_string + "/vertical_surface_medium.urdf")
        elif load_vertical_surface == 'large':
            self.vertical_surface_id = p.loadURDF(urdf_base_string + "/vertical_surface_large.urdf")

        if vertical_surface_friction is not None:
            p.changeDynamics(self.vertical_surface_id, -1, lateralFriction=vertical_surface_friction)

        if isinstance(lateral_friction, list):
            self.random_lateral_friction = True
            self.lateral_friction_range = lateral_friction
            self.lateral_friction = np.random.uniform(self.lateral_friction_range[0], self.lateral_friction_range[1])
        else:
            self.random_lateral_friction = False
            self.lateral_friction = lateral_friction

        if isinstance(contact_stiffness, list):
            self.random_contact_stiffness = True
            self.contact_stiffness_range = contact_stiffness
            self.contact_stiffness = np.random.uniform(self.contact_stiffness_range[0], self.contact_stiffness_range[1])
        else:
            self.random_contact_stiffness = False
            self.contact_stiffness = contact_stiffness

        useRealTimeSimulation = False

        # Query all the joints.
        num_joints = p.getNumJoints(self.robotId)
        print("Number of joints={}".format(num_joints))

        for ji in range(num_joints):
            p.changeDynamics(self.robotId, ji, linearDamping=.04, angularDamping=0.04, restitution=0.0, lateralFriction=self.lateral_friction, maxJointVelocity=1000)
            p.changeDynamics(self.robotId, ji, jointDamping=joint_damping)

        if base_damping is not None:
            p.changeDynamics(self.robotId, 1, jointDamping=base_damping)


        self.contact_damping = contact_damping
        self.contact_damping_multiplier = contact_damping_multiplier

        if self.random_contact_stiffness:
            self.contact_stiffness = np.random.uniform(self.contact_stiffness_range[0], self.contact_stiffness_range[1])

        if self.contact_damping_multiplier is not None:
            contact_damping = self.contact_damping_multiplier * 2.0 * np.sqrt(self.contact_stiffness)
        else:
            if self.contact_damping is None:
                contact_damping = 2.0 * np.sqrt(self.contact_stiffness)
            else:
                contact_damping = self.contact_damping

        if use_moving_surface:
            p.changeDynamics(planeId, 1, lateralFriction=self.lateral_friction)
            p.changeDynamics(planeId, 1, contactStiffness=self.contact_stiffness, contactDamping=contact_damping)
        else:
            p.changeDynamics(planeId, -1, lateralFriction=self.lateral_friction)
            p.changeDynamics(planeId, -1, contactStiffness=self.contact_stiffness, contactDamping=contact_damping)

        p.setGravity(0.0, 0.0, -9.81)
        #p.setPhysicsEngineParameter(1e-3, numSubSteps=1)

        self.sim_timestep = sim_timestep
        self.cont_timestep_mult = cont_timestep_mult
        self.dt = self.cont_timestep_mult * self.sim_timestep
        p.setPhysicsEngineParameter(fixedTimeStep=self.sim_timestep)

        print(p.getPhysicsEngineParameters())

        # Create the pinocchio robot.
        #urdf = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/urdf/solo.urdf')
        #self.robot = QuadrupedWrapper(urdf)

        #self.controlled_joints = ['BL_HFE', 'BL_KFE', 'BR_HFE', 'BR_KFE', 'FL_HFE', 'FL_KFE', 'FR_HFE', 'FR_KFE']

        # Create the simulator for easier mapping between
        #self.sim = Simulator(self.robotId, self.robot,
        #    self.controlled_joints,
        #    ['BL_END', 'BR_END', 'FL_END', 'FR_END', ]
        #)

        self.initial_joint_pos = initial_joint_pos
        self.real_system_init = real_system_init
        self.big_init = big_init
        self.big_init_base_vel = big_init_base_vel
        self.init_traj_file = init_traj_file
        self.init_traj_file_use_velocity = init_traj_file_use_velocity
        self.parkour_random_init = parkour_random_init

        if parkour_surface:
            p.loadURDF(urdf_base_string + "/parkour_surface.urdf", [0.15, 0.0, 0.156 - 0.0125])
            p.loadURDF(urdf_base_string + "/parkour_surface.urdf", [-0.125, 0.0, 0.358 - 0.0125])
            p.loadURDF(urdf_base_string + "/parkour_surface.urdf", [0.125, 0.0, 0.593 - 0.0125])
        
        self.joint_type = joint_type

        return p, self.robotId, planeId, [1, 2, 3], [2, 3]

    def set_initial_configuration(self):
        if self.init_traj_file is not None:
            traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                             + '/trajectories/'
            traj = np.loadtxt(traj_directory + self.init_traj_file)
            random_timestep = np.random.randint(0, traj.shape[0] - 1)
            self.initial_joint_pos = np.array(traj[random_timestep][1:4])
            if self.init_traj_file_use_velocity:
                self.initial_joint_vel = np.array(traj[random_timestep][4:7])
            else:
                self.initial_joint_vel = np.zeros(3)
            self.initial_joint_pos[0] *= -1.0
            self.initial_joint_vel[0] *= -1.0
            for i in range(self.num_obs_joints):
                self.p.resetJointState(
                    bodyUniqueId=self.robot_id,
                    jointIndex=self.obs_joint_ids[i],
                    targetValue=self.initial_joint_pos[i],
                    targetVelocity=self.initial_joint_vel[i])
        elif self.parkour_random_init:
            init_done = False
            while not init_done:
                hip_angle = np.random.uniform(-np.pi / 4.0, np.pi / 4.0)
                knee_angle = np.random.uniform(-np.pi / 4.0, np.pi / 4.0) - hip_angle
                self.initial_joint_pos = [
                    # np.random.uniform(0.0, 1.0),
                    # np.random.uniform(-np.pi / 2.0, np.pi / 2.0),
                    # np.random.uniform(-np.pi, np.pi)]
                    np.random.uniform(0.25, 0.75),
                    hip_angle,
                    knee_angle]
                self.initial_joint_vel = [
                    np.random.uniform(-1.0, 2.0),
                    0.0,
                    0.0]
                for i in range(self.num_obs_joints):
                    self.p.resetJointState(
                        bodyUniqueId=self.robot_id,
                        jointIndex=self.obs_joint_ids[i],
                        targetValue=self.initial_joint_pos[i],
                        targetVelocity=self.initial_joint_vel[i])
                self.p.stepSimulation()
                contacts = self.p.getContactPoints()
                if len(contacts) == 0:
                    init_done = True
        else:
            if self.real_system_init:
                self.initial_joint_pos = [
                    np.random.uniform(0.4, 0.6),
                    np.random.uniform(-0.1, 0.1),
                    np.random.uniform(-0.1, 0.1)]

            if self.big_init:
                init_done = False
                des_vel = [0.0, 0.0, 0.0]

                while not init_done:
                    if self.big_init_base_vel:
                        self.initial_joint_pos = [
                            np.random.uniform(0.2, 0.6),
                            np.random.uniform(-np.pi / 2.0, np.pi / 2.0),
                            np.random.uniform(-np.pi, np.pi)]
                        base_vel = np.random.uniform(-3.0, 3.0)
                        des_vel = [base_vel, 0.0, 0.0]
                    else:
                        self.initial_joint_pos = [
                            np.random.uniform(0.0, 0.6),
                            np.random.uniform(-np.pi / 2.0, np.pi / 2.0),
                            np.random.uniform(-np.pi, np.pi)]

                    for i in range(self.num_obs_joints):
                        self.p.resetJointState(
                            bodyUniqueId=self.robot_id,
                            jointIndex=self.obs_joint_ids[i],
                            targetValue=self.initial_joint_pos[i],
                            targetVelocity=des_vel[i])

                    self.p.stepSimulation()
                    contacts = self.p.getContactPoints()
                    if len(contacts) == 0:
                        init_done = True

                for i in range(self.num_obs_joints):
                    self.p.resetJointState(
                        bodyUniqueId=self.robot_id,
                        jointIndex=self.obs_joint_ids[i],
                        targetValue=self.initial_joint_pos[i],
                        targetVelocity=des_vel[i])

            else:
                for i in range(self.num_obs_joints):
                    self.p.resetJointState(
                        bodyUniqueId=self.robot_id,
                        jointIndex=self.obs_joint_ids[i],
                        targetValue=self.initial_joint_pos[i],
                        targetVelocity=0.0)

    def get_endeff_link_id(self):
        return 4

    def get_base_link_id(self):
        return 0

    def get_cont_joint_type(self):
        return [self.joint_type, self.joint_type]

    def get_default_pd_params(self):
        #return  np.array([1.0, 1.0]), np.array([0.1, 0.1])
        #return np.array([1.0, 1.0]), np.array([0.1, 0.1])
        return np.array([1.0, 0.1]), np.array([0.1, 0.01])

    def get_base_height(self):
        return self.get_obs_joint_state()[0][0]

    def get_upright_height(self):
        return 0.345

    def inv_dyn(self, des_acc):
        return np.zeros(des_acc.shape)

    def get_base_vel_z(self):
        _, joint_vel = self.get_obs_joint_state()
        return joint_vel[0]


if __name__ == '__main__':
    #hopper = Hopper(controller_params={'type': 'torque'}, reward_specs={})
    #hopper = Hopper(controller_params={'type': 'position_gain', 'variant': 'fixed'}, reward_specs={}, visualize=True)
    upright_height = 0.345
    max_height = 1.0
    hopper = Hopper(controller_params={'type': 'position_gain', 'variant': 'fixed'}, \
                    reward_specs={'hopping_reward': {'type': 'max', \
                                                     'selection_criteria': 'height', \
                                                     'ground_reward_params': [[0.0, upright_height], [0.0, 0.2], 0.001],\
                                                     'air_reward_params': [[upright_height, max_height], [0.75, 4.0], 5, True]}}, \
                    visualize=True,
                    observable=['endeff_force'],
                    exp_name='~/Desktop/tmp/')
    hopper._reset()
    a = np.random.uniform(-1.0, 1.0, 2)
    i = 0
    while True:
        i += 1
        #if i % 12 == 0:
        a = np.random.uniform(-1.0, 1.0, 2)
        _, r, _, _ = hopper._step(a)
        #print(r)
        #hopper.p.stepSimulation()
        #print(hopper.get_obs_joint_state())
        #time.sleep(hopper.dt)

    #timesteps = 250
    #goal = np.array([-1.0, 0.0])

    #hopper.controller.base_kp = np.array([0.0, 0.1])
    #hopper.controller.base_kv = np.array([0.0, 0.01])

    #while True:
    #    hopper._reset()

    #    for t in range(timesteps):
    #        hopper._step(goal)
    #        time.sleep(hopper.dt)
