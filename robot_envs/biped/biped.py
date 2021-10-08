import time
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
import os
from robot_envs.biped.robot import Robot
import numpy as np

class Biped(Robot, object):

    def __init__(self, **kwargs):
        super(Biped, self).__init__(**kwargs)

    def init_simulation(
        self,
        visualize=True,
        sim_timestep=0.001,
        cont_timestep_mult=16):

        self.visualize = visualize
        if self.visualize:
            p = bc.BulletClient(connection_mode=pybullet.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        p.setGravity(0,0,-10)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        biped = p.loadURDF(os.path.dirname(os.path.abspath(__file__)) + '/urdf/biped_locked_on_z.urdf', [0, 0, 0])
        surface = p.loadURDF("plane.urdf")

        p.setJointMotorControl2(biped, 1, controlMode=p.VELOCITY_CONTROL, force=0)

        self.moving_surface = None

        self.sim_timestep = sim_timestep
        self.cont_timestep_mult = cont_timestep_mult
        self.dt = self.cont_timestep_mult * self.sim_timestep
        p.setPhysicsEngineParameter(fixedTimeStep=self.sim_timestep)

        return p, biped, surface

    def set_initial_configuration(self):
        self.initial_joint_pos = np.array([np.random.uniform(0.2, 0.6),
                                           np.random.uniform(-np.pi / 4.0, np.pi / 4.0),
                                           np.random.uniform(-np.pi / 2.0, np.pi / 2.0),
                                           np.random.uniform(-np.pi, np.pi),
                                           np.random.uniform(-np.pi / 4.0, np.pi / 4.0),
                                           np.random.uniform(-np.pi / 2.0, np.pi / 2.0),
                                           np.random.uniform(-np.pi, np.pi)])
        for i in range(self.num_obs_joints):
            self.p.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=self.obs_joint_ids[i],
                targetValue=self.initial_joint_pos[i],
                targetVelocity=0.0)

    def get_endeff_link_id(self):
        return 0

    def get_base_link_id(self):
        return 0

    def get_cont_joint_type(self):
        return ['limited'] * 6

    def get_default_pd_params(self):
        return  np.array([1.0, 1.0]), np.array([0.1, 0.1])

    def get_base_height(self):
        return self.get_obs_joint_state()[0][0]

    def get_upright_height(self):
        return 0.345

    def inv_dyn(self, des_acc):
        return np.zeros(des_acc.shape)

    def get_base_vel_z(self):
        _, joint_vel = self.get_obs_joint_state()
        return joint_vel[0]

    def get_obs_joint_ids(self):
        return [1, 2, 3, 4, 6, 7, 8]

    def get_cont_joint_ids(self):
        return [2, 3, 4, 6, 7, 8]


if __name__ == '__main__':
    biped = Biped(controller_params={'type': 'torque_gc'}, reward_specs={}, physics_parameters={'jointDamping': 0.05, 'contactStiffness': 1000000.0, 'contactDamping': 0.0})
    while True:
        biped._reset()
        for i in range(100):
            eps = 1.0
            biped._step(np.random.uniform(-eps, eps, len(biped.cont_joint_ids)))
            time.sleep(biped.dt)
