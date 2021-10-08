import os
import yaml
import json
import time
import numpy as np
from itertools import repeat

import pybullet_utils.bullet_client as bc
import pybullet
from robot_envs.hopper.robot import Robot

from scipy.spatial.transform import Rotation as R



class Quadruped(Robot, object):

    def __init__(self, **kwargs):
        super(Quadruped, self).__init__(**kwargs)

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
        #initial_joint_pos=[1.53242382, -3.08782035,  1.53222638, -3.08777907, -1.53246088, 3.08786258, -1.53241491,  3.08782701],
        initial_joint_pos=[0.5, -1.0,  0.5, -1.0, -0.5, 1.0, -0.5, 1.0],
        initial_base_pos=[0.0, 0.0, 0.2972777446745981]):

        self.visualize = visualize
        if self.visualize:
            p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        urdf_base_string = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + '/robot_properties_teststand'
        print(urdf_base_string)
        planeId = p.loadURDF(urdf_base_string + "/urdf/plane_with_restitution.urdf")

        urdf_base_string = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + '/robot_properties_solo'
        print(urdf_base_string)
        #self.start_pos = [0, 0, 0.026]
        self.start_pos = initial_base_pos
        self.start_orn = p.getQuaternionFromEuler([0,0,0])
        self.robotId = p.loadURDF(urdf_base_string + "/urdf/solo.urdf", self.start_pos, self.start_orn, flags=p.URDF_USE_INERTIA_FROM_FILE)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(self.robotId)

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


        print(lateral_friction, joint_damping, contact_stiffness, contact_damping)

        useRealTimeSimulation = False

        # Query all the joints.
        num_joints = p.getNumJoints(self.robotId)
        print("Number of joints={}".format(num_joints))


        print('FRICTION:', self.lateral_friction)

        for ji in range(num_joints):
            p.changeDynamics(self.robotId, ji, linearDamping=.04, angularDamping=0.04, restitution=0.0, lateralFriction=self.lateral_friction, maxJointVelocity=1000)
            p.changeDynamics(self.robotId, ji, jointDamping=joint_damping)

        p.changeDynamics(planeId, -1, lateralFriction=self.lateral_friction)

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

        print('STIFFNESS:', self.contact_stiffness, 'DAMPING:', contact_damping)
        p.changeDynamics(planeId, -1, contactStiffness=self.contact_stiffness, contactDamping=contact_damping)

        p.setGravity(0.0, 0.0, -9.81)

        self.sim_timestep = sim_timestep
        self.cont_timestep_mult = cont_timestep_mult
        self.dt = self.cont_timestep_mult * self.sim_timestep
        p.setPhysicsEngineParameter(fixedTimeStep=self.sim_timestep)

        print(p.getPhysicsEngineParameters())

        self.initial_joint_pos = initial_joint_pos

        self.moving_surface = None

        return p, self.robotId, planeId, self.get_obs_joint_ids(), self.get_cont_joint_ids()

    def set_initial_configuration(self):
        for i in range(self.num_obs_joints):
            self.p.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=self.obs_joint_ids[i],
                targetValue=self.initial_joint_pos[i],
                targetVelocity=0.0)

        self.p.resetBasePositionAndOrientation(
            bodyUniqueId=self.robot_id,
            posObj=self.start_pos,
            ornObj=self.start_orn)

    def get_endeff_link_id(self):
        return 4

    def get_base_link_id(self):
        return 0

    def get_cont_joint_type(self):
        return ['circular'] * 8

    def get_default_pd_params(self):
        return np.array([1.0] * 8), np.array([0.0] * 8)

    def get_base_height(self):
        pos, _ = self.p.getBasePositionAndOrientation(self.robot_id)
        return pos[2]

    def get_upright_height(self):
        return 0.336

    def inv_dyn(self, des_acc):
        return np.zeros(des_acc.shape)

    def get_obs_joint_ids(self):
        return [0, 1, 3, 4, 6, 7, 9, 10]

    def get_cont_joint_ids(self):
        return self.get_obs_joint_ids()

    def get_base_vel_z(self):
        vel, _ = self.p.getBaseVelocity(self.robot_id)
        return vel[2]




def visualize(conf_file, log_files):
    with open(conf_file) as f:
        conf = yaml.load(f)

    env_params = conf['env_params'][0]['env_specific_params']
    env_params['visualize'] = True
    env_params['full_log'] = False
    env_params['exp_name'] = 'tmp/'
    if not os.path.exists(env_params['exp_name']):
        os.makedirs(env_params['exp_name'])
    env_params['log_file'] = 'tmp/log.json'
    if 'control' in env_params:
        del env_params['control']

    env = Hopper(**env_params)
    env._reset()

    #joint_pos[log_num][timestep][joint]
    joint_pos = []
    loaded = list(repeat(False, len(log_files)))

    step_duration = env_params['sim_timestep'] * env_params['cont_timestep_mult']
    while True:
        time.sleep(2.0)
        for log_num in [0]:
            if not loaded[log_num]:
                with open(log_files[log_num]) as f:
                    log = json.load(f)
                    joint_pos.append(np.array(log['joint_pos']))
            start_time = time.time()
            endeff_orient = []
            for i in range(joint_pos[log_num].shape[0]):
                step_start = time.time()
                for j in range(joint_pos[log_num].shape[1]):
                    env.p.resetJointState(
                        bodyUniqueId=env.robot_id,
                        jointIndex=env.obs_joint_ids[j],
                        targetValue=joint_pos[log_num][i, j],
                        targetVelocity=0.0)
                env.p.stepSimulation()
                #endeff_orient.append(env.get_endeff_xy_angle())
                expected_time = (i + 1) * step_duration
                sleep_duration = expected_time - (time.time() - start_time)
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
            duration = time.time() - start_time
            print('real time constant:', step_duration * joint_pos[log_num].shape[0] / duration)
            #prepare_plot()
            #plt.grid(True)
            #plt.plot(endeff_orient)
            #plt.show()




if __name__ == '__main__':
    #hopper = Hopper(controller_params={'type': 'torque'}, reward_specs={})
    #hopper = Hopper(controller_params={'type': 'position_gain', 'variant': 'fixed'}, reward_specs={}, visualize=True)
    upright_height = 0.336
    max_height = 1.65
    quadruped = Quadruped(controller_params={'type': 'torque'}, \
                    reward_specs={'hopping_reward': {'type': 'max', \
                                                     'selection_criteria': 'height', \
                                                     'ground_reward_params': [[0.0, upright_height], [0.0, 0.2], 0.001],\
                                                     'air_reward_params': [[upright_height, max_height], [0.75, 4.0], 5, True]}, \
                                  'base_acc_penalty': {'acc_limit': 100.0, 'k': 0.01}, \
                                  'base_stability_penalty': {'k': 1.0}}, \
                    visualize=True,
                    observable=['endeff_force'],
                    exp_name='~/Desktop/tmp/')
    quadruped._reset()
    a = np.random.uniform(-1.0, 1.0, 2)
    i = 0
    while True:
        quadruped.p.removeAllUserDebugItems()
        i += 1
        #if i % 12 == 0:
        a = np.random.uniform(-1.0, 1.0, 8)
        #a = np.array([0, 0])
        s, r, _, _ = quadruped._step(np.zeros(8))
        #print(hopper.get_total_ground_force())
        #print(-quadruped.get_total_ground_force()[2])
        #print(s[0])
        #print(r)
        #hopper.p.stepSimulation()
        #print(quadruped.get_obs_joint_state())
        #print(quadruped.p.getBasePositionAndOrientation(quadruped.robot_id))
        #print(quadruped.get_base_height())
        pos, orient = quadruped.p.getBasePositionAndOrientation(quadruped.robot_id)
        r = R.from_quat(orient)
        v = [0.19, 0.1046, 0]

        
        quadruped.add_user_debug_point(pos + r.apply(v))

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
