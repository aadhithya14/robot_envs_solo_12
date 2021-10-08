import numpy as np
import gym
from gym.spaces import Box
import pybullet
import pybullet_utils.bullet_client as bc
from utils.data_logging import Log, ListOfLogs, NoLog, SimpleLog
from robot_envs.controllers.torque_controller import TorqueController
from robot_envs.controllers.position_gain_controller import PositionGainController
from robot_envs.bolt.bolt_rewards import BaseStabilityPenalty, TorqueSmoothnessPenalty, LinearDistanceToGoalReward, TrajectoryImitationReward
from robot_envs.solo12.rewards.trajectory_tracking_reward import TrajectoryTrackingReward
from robot_envs.bolt.early_termination.reward_threshold_termination import RewardThresholdTermination
from robot_envs.bolt.early_termination.imitation_length_termination import ImitationLengthTermination
from robot_envs.bolt.early_termination.foot_placement_termination import FootPlacementTermination
from robot_envs.bolt.early_termination.base_stability_termination import BaseStabilityTermination
from robot_envs.bolt.early_termination.ground_impact_termination import GroundImpactTermination
from robot_envs.bolt.action_filter import ActionFilter


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
                 termination_conf={},
                 output_dir=None,
                 joint_limits=None,
                 initialization_conf={},
                 action_filter_conf=None,
                 movable_disks_surface_conf=None,
                 log_conf={}):
        self.exp_name = exp_name
        self.init_bullet(visualize, enable_gravity)
        self.robot_specific_init(movable_disks_surface_conf)
        self.init_joints(joint_limits)
        self.init_log(exp_name, log_file, full_log, log_conf)
        self.init_reward(reward_specs)
        self.init_controller(controller_params)
        self.init_spaces()
        self.init_time_params(sim_timestep, cont_timestep_mult)
        self.initialization_conf = initialization_conf
        self.init_termination(termination_conf)
        self.init_action_filter(action_filter_conf)


    def init_bullet(self, visualize, enable_gravity):
        self.visualize = visualize
        if self.visualize:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_SHADOWS, 0)
            self.p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[1.0,0,0])
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
            self.controller = PositionGainController(robot=self, params=controller_params, robot_log=self.log)
        else:
            assert False, 'Unknown controller type: ' + controller_type

    def init_reward(self, reward_params):
        self.reward_parts = {}
        for reward_type, reward_spec in reward_params.items():
            if reward_type == 'base_stability_penalty':
                self.reward_parts[reward_type] = BaseStabilityPenalty(self, **reward_spec)
            elif reward_type == 'torque_smoothness_penalty':
                self.reward_parts[reward_type] = TorqueSmoothnessPenalty(self, **reward_spec, env_log=self.log)
            elif reward_type == 'linear_distance_to_goal':
                self.reward_parts[reward_type] = LinearDistanceToGoalReward(self, **reward_spec)
            elif reward_type == 'trajectory_imitation_reward':
                self.reward_parts[reward_type] = TrajectoryImitationReward(self, **reward_spec)
            elif reward_type == 'trajectory_tracking_reward':
                self.reward_parts[reward_type] = TrajectoryTrackingReward(self, reward_spec)
            else:
                assert False, 'Unknown reward type: ' + reward_type

    def init_termination(self, termination_conf):
        self.termination_dict = {}
        # DONE: Test ImitationLengthTermination.
        if termination_conf is not None:
            for termination_name, termination_params in termination_conf.items():
                if termination_name == 'reward_threshold_termination':
                    self.termination_dict[termination_name] = RewardThresholdTermination(self, **termination_params)
                elif termination_name == 'imitation_length_termination':
                    self.termination_dict[termination_name] = ImitationLengthTermination(self)
                elif termination_name == 'foot_placement_termination':
                    self.termination_dict[termination_name] = FootPlacementTermination(self, **termination_params)
                elif termination_name == 'base_stability_termination':
                    self.termination_dict[termination_name] = BaseStabilityTermination(self, **termination_params)
                elif termination_name == 'ground_impact_termination':
                    self.termination_dict[termination_name] = GroundImpactTermination(self, **termination_params)
                else:
                    assert False, 'Unknown termination type: ' + termination_name

    def init_log(self, env_path, log_file, make_full_log, log_conf):
        if (env_path is not None) or (log_file is not None):
            if make_full_log:
                self.log = ListOfLogs(env_path + '_episodes', separate_files=True)
            else:
                if log_file is not None:
                    self.log = Log(log_file)
                else:
                    self.log = Log(env_path + '_episodes')
        else:
            self.log = SimpleLog()
        self.log_conf = log_conf

    def update_log(self):
        self.log.add('state', self.get_state().tolist())
        for reward_type, reward_part in self.rewards.items():
            self.log.add(reward_type, reward_part)
        self.log.add('total_ground_force', self.get_total_ground_force().tolist())

        if 'log_all_contact_points' in self.log_conf:
            if self.log_conf['log_all_contact_points']:
                contact_points = self.p.getContactPoints(bodyA=self.robot_id, bodyB=self.surface_id)
                for cp in contact_points:
                    self.log.add('all_contact_points', cp[6])

        endeff_pos, endeff_vel = self.get_endeff_state()
        self.log.add('endeff_pos', endeff_pos.tolist())
        self.log.add('endeff_vel', endeff_vel.tolist())
        #
        # base_pos, base_orient = self.p.getBasePositionAndOrientation(self.robot_id)
        # base_ang = self.p.getEulerFromQuaternion(base_orient)
        # self.log.add('base_pos', base_pos)
        # self.log.add('base_ang', base_ang)

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

    def init_action_filter(self, action_filter_conf):
        if action_filter_conf is not None:
            self.action_filter = ActionFilter(**action_filter_conf)
        else:
            self.action_filter = None

    def torque_control(self, des_torque, no_clipping=False):
        if not no_clipping:
            self.des_torque = np.clip(des_torque, -self.max_torque, self.max_torque)
        else:
            self.des_torque = des_torque
        if self.log is not None:
            self.log.add('torque', self.des_torque.tolist())
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

    def get_total_ground_force(self, with_friction=True):
        contacts = self.p.getContactPoints(bodyA=self.surface_id, bodyB=self.robot_id)
        total_force = np.zeros(3)
        for contact in contacts:
            contact_normal = np.array(contact[7])
            normal_force = contact[9]
            total_force += normal_force * contact_normal

            if with_friction:
                contact_normal = np.array(contact[11])
                normal_force = contact[10]
                total_force += normal_force * contact_normal

                contact_normal = np.array(contact[13])
                normal_force = contact[12]
                total_force += normal_force * contact_normal
        return total_force

    def get_link_z(self, link_id):
        state = self.p.getLinkState(self.robot_id, link_id)
        pos = state[0]
        return pos[2]

    def get_endeff_state(self):
        endeff_pos = []
        endeff_vel = []
        for link_id in self.get_endeff_link_ids():
            full_state = self.p.getLinkState(self.robot_id, link_id, computeLinkVelocity=True)
            endeff_pos += full_state[0]
            endeff_vel += full_state[6]
        return np.array(endeff_pos), np.array(endeff_vel)

    def _reset(self):
        if self.log is not None:
            if isinstance(self.log, ListOfLogs):
                self.log.finish_log()
            else:
                self.log.save()
                self.log.clear()

        if self.action_filter is not None:
            self.action_filter.reset()
        self.controller.reset()
        
        if self.movable_disks_surface is not None:
            self.movable_disks_surface.reset()
        self.set_initial_configuration(**self.initialization_conf)
        
        state = self.get_state()
        for _, f in self.reward_parts.items():
            f.reset()
        for termination_object in self.termination_dict.values():
            termination_object.reset()

        if self.log is not None:
            self.log.add('state', self.get_state().tolist())

        return state

    def _step(self, action):
        self.rewards = {}
        for reward_name in self.reward_parts.keys():
            self.rewards[reward_name] = 0.0

        for i in range(self.cont_timestep_mult):
            # self.log.add('total_ground_force', self.get_total_ground_force().tolist())
            # pos, orient = self.p.getBasePositionAndOrientation(self.robot_id)
            # self.log.add('pos', pos)
            # self.log.add('orient', orient)
            # vel, angvel = self.p.getBaseVelocity(self.robot_id)
            # self.log.add('vel', vel)
            # self.log.add('angvel', angvel)
            # joint_pos, joint_vel = self.get_obs_joint_state()
            # self.log.add('joint_pos', joint_pos)
            # self.log.add('joint_vel', joint_vel)
            if self.action_filter is not None:
                self.action_filter.add_action(action)
                self.controller.act(self.action_filter.get_filtered_action())
            else:
                self.controller.act(action)
            for reward_name, reward_object in self.reward_parts.items():
                if reward_object.calc_at_sim_step:
                    self.rewards[reward_name] += reward_object.get_reward()
                    reward_object.step()

        for reward_name, reward_object in self.reward_parts.items():
            if not reward_object.calc_at_sim_step:
                self.rewards[reward_name] = reward_object.get_reward()
                reward_object.step()

        self.reward = 0.0
        for reward_value in self.rewards.values():
            self.reward += reward_value

        done = False
        for termination_object in self.termination_dict.values():
            if termination_object.done():
                done = True
            termination_object.step()

        if self.log is not None:
            self.update_log()

        state = self.get_state()

        return state, self.reward, done, {}

    def _render(self, mode, close):
        pass

    def _seed(self, seed):
        pass
