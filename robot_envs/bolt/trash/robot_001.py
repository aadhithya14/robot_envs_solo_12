import numpy as np
import gym
from gym.spaces import Box
import pybullet
import pybullet_utils.bullet_client as bc
from utils.data_logging import Log, ListOfLogs, NoLog
from robot_envs.controllers.torque_controller import TorqueController
from robot_envs.controllers.position_gain_controller import PositionGainController
from robot_envs.bolt.bolt_rewards import BaseStabilityPenalty
from robot_envs.bolt.early_termination import EarlyTermination


class Robot(gym.Env):

    def __init__(self,
                 controller_params={'type': 'torque'},
                 reward_specs={},
                 visualize=False,
                 sim_timestep=0.001,
                 cont_timestep_mult=8,
                 enable_gravity=True,
                 exp_name=None,
                 log_file=None,
                 full_log=True,
                 early_termination_params=None,
                 output_dir=None,
                 joint_limits=None):
        self.init_bullet(visualize, enable_gravity)
        self.robot_specific_init()
        self.init_joints(joint_limits)
        self.init_controller(controller_params)
        self.init_reward(reward_specs)
        self.init_log(exp_name, log_file, full_log)
        self.init_spaces()
        self.init_time_params(sim_timestep, cont_timestep_mult)

        self.early_termination = None
        if early_termination_params is not None:
            self.early_termination = EarlyTermination(**self.early_termination_params)

    def init_bullet(self, visualize, enable_gravity):
        self.visualize = visualize
        if self.visualize:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        if enable_gravity:
            self.p.setGravity(0.0, 0.0, -9.81)

    def init_joints(self, joint_limits):
        self.obs_joint_ids = self.get_obs_joint_ids()
        self.cont_joint_ids = self.get_cont_joint_ids()
        self.num_obs_joints = len(self.obs_joint_ids)
        self.num_cont_joints = len(self.cont_joint_ids)
        self.max_torque = np.zeros(self.num_cont_joints)
        self.joint_limits = np.zeros((self.num_cont_joints, 2))
        for i in range(self.num_cont_joints):
            joint_info = self.p.getJointInfo(self.robot_id, self.cont_joint_ids[i])
            self.max_torque[i] = joint_info[10]
            self.joint_limits[i][0] = joint_info[8]
            self.joint_limits[i][1] = joint_info[9]
        if joint_limits is not None:
            self.joint_limits = np.array(joint_limits)

    def init_controller(self, controller_params):
        controller_type = controller_params['type']
        if controller_type == 'torque':
            self.controller = TorqueController(self, grav_comp=False)
        elif controller_type == 'position_gain':
            self.controller = PositionGainController(robot=self, params=controller_params)
        else:
            assert False, 'Unknown controller type: ' + controller_type

    def init_reward(self, reward_params):
        self.reward_parts = {}
        for reward_type, reward_spec in reward_params.items():
            if reward_type == 'base_stability_penalty':
                self.reward_parts[reward_type] = BaseStabilityPenalty(self, **reward_spec)
            else:
                assert False, 'Unknown reward type: ' + reward_type

    def init_log(self, env_path, log_file, make_full_log):
        if (env_path is not None) or (log_file is not None):
            if make_full_log:
                self.log = ListOfLogs(env_path + '_episodes', separate_files=True)
            else:
                if log_file is not None:
                    self.log = Log(log_file)
                else:
                    self.log = Log(env_path + '_episodes')
        else:
            self.log = NoLog()

    def update_log(self):
        self.log.add('state', self.get_state().tolist())
        for r, f in self.reward_parts.items():
            self.log.add(r, f.get_reward())

    def init_spaces(self):
        self.action_space = self.controller.get_control_space()
        obs_dim = len(self.get_state())
        high = np.inf * np.ones([obs_dim])
        self.observation_space = Box(-high, high)

    def init_time_params(self, sim_timestep, cont_timestep_mult):
        self.sim_timestep = sim_timestep
        self.cont_timestep_mult = cont_timestep_mult
        self.p.setPhysicsEngineParameter(fixedTimeStep=self.sim_timestep)

    def init_torque_control(self):
        for joint_id in self.cont_joint_ids:
            self.p.setJointMotorControl2(self.robot_id, joint_id,
                controlMode=self.p.VELOCITY_CONTROL, force=0)

    def torque_control(self, des_torque, clip_to_limits=True):
        if clip_to_limits:
            self.des_torque = np.clip(des_torque, -self.max_torque, self.max_torque)
        else:
            self.des_torque = des_torque
        for i in range(self.num_cont_joints):
            self.p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=self.cont_joint_ids[i],
                controlMode=self.p.TORQUE_CONTROL,
                force=self.des_torque[i])
        self.p.stepSimulation()

    def get_obs_joint_state(self):
        joint_pos = np.zeros(self.num_obs_joints)
        joint_vel = np.zeros(self.num_obs_joints)
        for i in range(self.num_obs_joints):
            joint_pos[i], joint_vel[i], _, _ = self.p.getJointState(self.robot_id, self.obs_joint_ids[i])
        return joint_pos, joint_vel


    def get_cont_joint_state(self):
        joint_pos = np.zeros(self.num_cont_joints)
        joint_vel = np.zeros(self.num_cont_joints)
        for i in range(self.num_cont_joints):
            joint_pos[i], joint_vel[i], _, _ = self.p.getJointState(self.robot_id, self.cont_joint_ids[i])
        return joint_pos, joint_vel

    def get_state(self):
        state = []

        pos, orient = self.p.getBasePositionAndOrientation(self.robot_id)
        state += pos
        state += orient
        vel, angvel = self.p.getBaseVelocity(self.robot_id)
        state += vel
        state += angvel

        joint_pos, joint_vel = self.get_obs_joint_state()
        state += joint_pos.tolist()
        state += joint_vel.tolist()

        return np.array(state)

    def _reset(self):
        if self.log is not None:
            if isinstance(self.log, ListOfLogs):
                self.log.finish_log()
            else:
                self.log.save()
                self.log.clear()

        self.controller.reset()
        self.set_initial_configuration()
        state = self.get_state()

        if self.log is not None:
            self.update_log()

        if self.early_termination is not None:
            self.early_termination.reset()

        return state

    def _step(self, action):
        sim_step_rewards = {}
        calc_at_sim_step = []
        for r in calc_at_sim_step:
            sim_step_rewards[r] = 0.0

        for i in range(self.cont_timestep_mult):
            self.controller.act(action)

            for r in calc_at_sim_step:
                if r in self.reward_parts:
                    sim_step_rewards[r] += self.reward_parts[r].get_reward()

        state = self.get_state()

        reward = 0.0
        for r, f in self.reward_parts.items():
            if r in calc_at_sim_step:
                reward += sim_step_rewards[r]
            else:
                reward += f.get_reward()

        done = False

        if self.early_termination is not None:
            reward, done = self.early_termination.step(reward, done)

        if self.log is not None:
            self.update_log()

        return state, reward, done, {}

    def _render(self, mode, close):
        pass

    def _seed(self, seed):
        pass
