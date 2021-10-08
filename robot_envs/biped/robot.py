import time
import numpy as np
import gym
from gym.spaces import Box
from utils.my_math import scale
from utils.data_logging import Log, ListOfLogs, NoLog
from robot_envs.controllers.torque_controller import TorqueController
from robot_envs.controllers.position_gain_controller import PositionGainController
from robot_envs.hopper.hopping_rewards import HoppingReward, BaseAccPenalty, BaseStabilityPenalty, TrajectoryTrackingReward, ShakingPenalty, ForwardMotionReward
from robot_envs.hopper.impact_penalty import ImpactPenalty
from robot_envs.hopper.force_penalty import ForcePenalty
from robot_envs.hopper.torque_smoothness_penalty import TorqueSmoothnessPenalty
from robot_envs.reward_functions.non_ee_force_penalty import NonEEForcePenalty

from scipy.spatial.transform import Rotation as R


class Robot(gym.Env):


    def __init__(self,
                 controller_params,
                 reward_specs,
                 exp_name=None,
                 log_file=None,
                 full_log=True,
                 observable=[],
                 output_dir=None,
                 physics_parameters={},
                 **kwargs):
        self.p, self.robot_id, self.surface_id = self.init_simulation(**kwargs)
        self.init_joints()
        self.init_controller(controller_params)
        self.init_reward(reward_specs)
        self.init_log(exp_name, log_file, full_log)
        self.init_state(observable)
        self.init_spaces()
        self.init_physics(physics_parameters)


    def init_joints(self):
        self.obs_joint_ids = self.get_obs_joint_ids()
        self.cont_joint_ids = self.get_cont_joint_ids()
        self.num_obs_joints = len(self.obs_joint_ids)
        self.num_cont_joints = len(self.cont_joint_ids)
        self.cont_joint_type = self.get_cont_joint_type()
        self.max_torque = np.zeros(self.num_cont_joints)
        self.joint_limits = np.zeros((self.num_cont_joints, 2))
        for i in range(self.num_cont_joints):
            joint_info = self.p.getJointInfo(self.robot_id, self.cont_joint_ids[i])
            self.max_torque[i] = joint_info[10]
            if self.cont_joint_type[i] == 'limited':
                self.joint_limits[i][0] = joint_info[8]
                self.joint_limits[i][1] = joint_info[9]
            else:
                self.joint_limits[i][0] = 0.0
                self.joint_limits[i][1] = 2 * np.pi


    def init_controller(self, controller_params):
        controller_type = controller_params['type']
        if controller_type == 'torque':
            self.controller = TorqueController(self, grav_comp=False)
        elif controller_type == 'torque_gc':
            self.controller = TorqueController(self, grav_comp=True)
        elif controller_type == 'position_gain':
            self.controller = PositionGainController(robot=self, params=controller_params)
        elif controller_type == 'locked_joints':
            self.controller = LockedJointsController(self)
        else:
            assert False, 'Unknown controller type: ' + controller_type


    def init_reward(self, reward_params):
        self.reward_parts = {}
        for reward_type, reward_spec in reward_params.items():
            if reward_type == 'hopping_reward':
                self.reward_parts[reward_type] = HoppingReward(self, reward_spec)
            elif reward_type == 'impact_penalty':
                self.reward_parts[reward_type] = ImpactPenalty(self, reward_spec)
            elif reward_type == 'base_acc_penalty':
                self.reward_parts[reward_type] = BaseAccPenalty(self, reward_spec)
            elif reward_type == 'base_stability_penalty':
                self.reward_parts[reward_type] = BaseStabilityPenalty(self, reward_spec)
            elif reward_type == 'trajectory_tracking_reward':
                self.reward_parts[reward_type] = TrajectoryTrackingReward(self, reward_spec)
            elif reward_type == 'shaking_penalty':
                self.reward_parts[reward_type] = ShakingPenalty(self, reward_spec)
            elif reward_type == 'forward_motion_reward':
                self.reward_parts[reward_type] = ForwardMotionReward(self, reward_spec)
            elif reward_type == 'force_penalty':
                self.reward_parts[reward_type] = ForcePenalty(self, reward_spec)
            elif reward_type == 'torque_smoothness_penalty':
                self.reward_parts[reward_type] = TorqueSmoothnessPenalty(self, reward_spec)
            elif reward_type == 'non_ee_force_penalty':
                self.reward_parts[reward_type] = NonEEForcePenalty(self, reward_spec)
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


    def init_state(self, observable):
        self.observable = observable


    def init_spaces(self):
        self.action_space = self.controller.get_control_space()
        obs_dim = len(self.get_state())
        high = np.inf * np.ones([obs_dim])
        self.observation_space = Box(-high, high)


    def init_physics(self, physics_parameters):
        if len(physics_parameters) > 0:
            for body_id in [self.robot_id, self.surface_id]:
                for i in range(self.p.getNumJoints(body_id)):
                    self.p.changeDynamics(body_id, i, **physics_parameters)


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


    def get_total_force(self, links):
        contacts = self.p.getContactPoints(bodyA=self.surface_id, bodyB=self.robot_id)
        total_force = np.zeros(3)
        for contact in contacts:
            if contact[4] in links:
                contact_normal = np.array(contact[7])
                normal_force = contact[9]
                total_force += normal_force * contact_normal
        return total_force


    def get_total_ground_force(self):
        contacts = self.p.getContactPoints(bodyA=self.surface_id, bodyB=self.robot_id)
        total_force = np.zeros(3)
        for contact in contacts:
            contact_normal = np.array(contact[7])
            normal_force = contact[9]
            total_force += normal_force * contact_normal
        return total_force


    def get_endeff_force(self):
        return self.get_total_force([self.get_endeff_link_id()])


    def get_non_endeff_ground_force_scalar(self):
        contacts = self.p.getContactPoints(bodyA=self.surface_id, bodyB=self.robot_id)
        total_scalar = 0.0
        endeff_link_id = self.get_endeff_link_id()
        for contact in contacts:
            if contact[4] != endeff_link_id:
                force_scalar = np.absolute(contact[9])
                total_scalar += force_scalar
        return total_scalar


    def get_endeff_state(self):
        full_state = self.p.getLinkState(self.robot_id, self.get_endeff_link_id(), computeLinkVelocity=True)
        endeff_pos = full_state[0]
        endeff_vel = full_state[6]
        return np.array(endeff_pos), np.array(endeff_vel)


    def get_state(self):
        state = []

        joint_pos, joint_vel = self.get_obs_joint_state()

        if 'hopper_no_joint_vel' in self.observable:
            state += joint_pos.tolist()
            state += joint_vel.tolist()[0:1]
        else:
            state += joint_pos.tolist()
            state += joint_vel.tolist()

        if 'endeff_force' in self.observable:
            state += self.get_endeff_force().tolist()

        if 'base_state' in self.observable:
            pos, orient = self.p.getBasePositionAndOrientation(self.robot_id)
            vel, angvel = self.p.getBaseVelocity(self.robot_id)
            state += pos
            state += orient
            state += vel
            state += angvel

        return np.array(state)


    def _reset(self):
        self.log.add('timer', (time.time(), 'episode_done'))

        if self.log is not None:
            if isinstance(self.log, ListOfLogs):
                self.log.finish_log()
            else:
                self.log.save()
                self.log.clear()

        self.log.add('timer', (time.time(), 'episode'))

        if self.moving_surface is not None:
            self.moving_surface.reset()

        self.controller.reset()

        for reward_part in self.reward_parts.values():
            reward_part.reset()

        self.set_initial_configuration()

        state = self.get_state()
        if self.log is not None:
            self.log.add('state', state.tolist())
            joint_pos, joint_vel = self.get_obs_joint_state()
            self.log.add('joint_pos', joint_pos.tolist())
            self.log.add('joint_vel', joint_vel.tolist())
            self.log.add('endeff_force', self.get_endeff_force().tolist())
            self.log.add('non_ee_force', self.get_non_endeff_ground_force_scalar())
            endeff_pos, endeff_vel = self.get_endeff_state()
            self.log.add('endeff_pos', endeff_pos.tolist())
            self.log.add('endeff_vel', endeff_vel.tolist())
            self.log.add('total_force', self.get_total_ground_force().tolist())
            self.log.add('full_contact_info', self.p.getContactPoints(bodyA=self.surface_id, bodyB=self.robot_id))

        return state


    def _step(self, action):
        self.log.add('timer', (time.time(), 'env_step'))
        self.log.add('action', action.tolist())

        sim_step_rewards = {}
        calc_at_sim_step = ['base_acc_penalty', 'force_penalty', 'torque_smoothness_penalty', 'non_ee_force_penalty']
        for r in calc_at_sim_step:
            sim_step_rewards[r] = 0.0

        for i in range(self.cont_timestep_mult):

            self.log.add('timer', (time.time(), 'controller_call'))
            self.controller.act(action)
            self.log.add('timer', (time.time(), 'controller_call_done'))

            if self.moving_surface is not None:
                self.moving_surface.step()


            if self.log is not None:
                joint_pos, joint_vel = self.get_obs_joint_state()
                self.log.add('joint_pos', joint_pos.tolist())
                self.log.add('joint_vel', joint_vel.tolist())
                self.log.add('endeff_force', self.get_endeff_force().tolist())
                self.log.add('non_ee_force', self.get_non_endeff_ground_force_scalar())
                endeff_pos, endeff_vel = self.get_endeff_state()
                self.log.add('endeff_pos', endeff_pos.tolist())
                self.log.add('endeff_vel', endeff_vel.tolist())
                self.log.add('total_force', self.get_total_ground_force().tolist())
                self.log.add('full_contact_info', self.p.getContactPoints(bodyA=self.surface_id, bodyB=self.robot_id))
                if self.moving_surface is not None:
                    self.log.add('moving_surface_pos', self.p.getJointState(self.surface_id, self.moving_surface.joint_index)[0])
                self.log.add('base_height', self.get_base_height())

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


        if self.log is not None:
            self.log.add('state', self.get_state().tolist())

            for r, f in self.reward_parts.items():
                if r in calc_at_sim_step:
                    self.log.add(r, sim_step_rewards[r])
                else:
                    if r == 'shaking_penalty':
                        self.log.add(r, f.get_reward(update=False))
                    else:
                        self.log.add(r, f.get_reward())

            self.log.add('reward', reward)

        self.log.add('timer', (time.time(), 'env_step_done'))

        return state, reward, False, {}


    def init_torque_control(self):
        for joint_id in self.cont_joint_ids:
            self.p.setJointMotorControl2(self.robot_id, joint_id,
                controlMode=self.p.VELOCITY_CONTROL, force=0)


    def torque_control(self, des_torque):
        self.des_torque = np.clip(des_torque, -self.max_torque, self.max_torque)
        if self.log is not None:
                self.log.add('des_torque', self.des_torque.tolist())

        for i in range(self.num_cont_joints):
            self.p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=self.cont_joint_ids[i],
                controlMode=self.p.TORQUE_CONTROL,
                force=self.des_torque[i])

        self.log.add('timer', (time.time(), 'simulation_call'))
        self.p.stepSimulation()
        self.log.add('timer', (time.time(), 'simulation_call_done'))


    def _render(self, mode, close):
        pass


    def _seed(self, seed):
        pass
