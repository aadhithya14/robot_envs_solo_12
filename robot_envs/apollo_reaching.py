import pybullet_utils.bullet_client as bc
import pybullet
import time
import pybullet_data
import numpy as np
import gym
import gym.spaces
from utils.rewards import *
from utils.data_logging import Log, ListOfLogs
import os
from matplotlib import animation, rc
import matplotlib.pyplot as plt
from IPython.display import HTML
import json
from utils.plotting import prepare_plot
from scipy.stats import norm

import yaml
import json
import argparse

class ConfigurationReward():

    def __init__(self, reacher, configuration_score_scale):
        #assert(reacher.with_joint_limits)
        self.reacher = reacher
        self.configuration_score_scale = configuration_score_scale
        self.old_score = None

    def get_configuration_score(self):
        configuration_score = 0.0
        joint_pos, joint_vel = self.reacher.get_arm_state()
        for i in range(1, self.reacher.n_dof):
            # Dividing with 3.0 to stay 100% consistent with
            # the roboschool environment
            configuration_score += self.configuration_score_scale * (norm(0.3).pdf(joint_pos[i] / 3.0) - norm(0.3).pdf(0))
        return configuration_score

    def get_reward(self):
        new_score = self.get_configuration_score()
        if self.old_score == None:
            diff = [0.0]
        else:
            diff = [new_score - self.old_score]
        self.old_score = new_score
        return new_score

    def is_done(self):
        return True


class TargetReward():

    def __init__(self, reacher, k_p, func, goal_pos):
        self.reacher = reacher
        self.k_p = k_p
        self.func = func
        self.goal_pos = goal_pos
        self.old_potential = None

    def calc_dist(self):
        #print(np.linalg.norm(self.reacher.get_endeff_pos() - self.goal_pos))
        return np.linalg.norm(self.reacher.get_endeff_pos() - self.goal_pos)

    def get_potential(self):
        dist = self.calc_dist()
        if self.func == 'linear':
            return self.k_p * (1.0 - dist / 3.0)
        if self.func == 'square':
            return -self.k_p * dist ** 2
        if self.func == 'cube':
            return -self.k_p * dist ** 3
        if self.func == 'exp':
            value = 1.0 - dist / 3.0
            return self.k_p * np.exp(10.0 * value) / np.exp(10.0)
        if self.func == 'extended':
            value = 1.0 - dist / 3.0
            return self.k_p * (np.exp(10.0 * value) / np.exp(10.0) +  + np.exp(100.0 * value) / np.exp(100.0))
        assert False

    def get_reward(self):
        new_potential = self.get_potential()

        if self.old_potential == None:
            reward = [0.0]
        else:
            reward = [new_potential - self.old_potential]

        self.old_potential = new_potential

        return new_potential

    def is_done(self):
        #return self.calc_dist() < 0.0025
        return False


class VelocityReward():

    def __init__(self, reacher, k_v, square=False):
        self.reacher = reacher
        self.k_v = k_v
        self.square = square

    def get_reward(self):
        _, joint_vel = self.reacher.get_arm_state()
        value = np.linalg.norm(joint_vel) ** 2
        if self.square:
            return -self.k_v * value
        else:
            # Multiplying with 0.1 to stay 100% consistent with
            # the roboschool environment
            return self.k_v * 1.0 / np.exp(0.05 * value)

    def is_done(self):
        joint_pos, joint_vel = self.reacher.get_arm_state()
        # Multiplying with 0.1 to stay 100% consistent with
        # the roboschool environment
        #return np.linalg.norm(0.1 * joint_vel) ** 2 < 1e-3
        return False


class TorqueReward():

    def __init__(self, reacher, k_t, linear=False, max_torque=None):
        self.reacher = reacher
        self.k_t = k_t
        self.linear = linear
        if max_torque is not None:
            self.max_norm = np.linalg.norm(0.1 * np.array(max_torque))

    def get_reward(self):
        if self.linear:
            return -self.k_t * (self.max_norm - np.linalg.norm(self.reacher.latest_torque)) / self.max_norm
        else:
            value = np.linalg.norm(self.reacher.latest_torque)
            return self.k_t * (1.0 / np.exp(0.2 * value))

    def is_done(self):
        return True


class ApolloReaching(gym.Env):

    def __init__(self, reward_specs={}, initial_pushee_pos=None, observable=[], visualize=False, exp_name=None, output_dir='', initial_joint_state=None, fixed_timestep=None, full_log=False, max_velocity=None):
        self.observable = observable
        self.visualize = visualize
        self.initial_pushee_pos = initial_pushee_pos
        self.initial_joint_state = initial_joint_state
        self.full_log = full_log
        self.logging = exp_name is not None
        self.fixed_timestep = fixed_timestep
        self.max_velocity = max_velocity

        if self.visualize:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        #self.p.setGravity(0,0,-10)
        if fixed_timestep is not None:
            self.p.setPhysicsEngineParameter(fixedTimeStep=fixed_timestep)
        robot = self.p.loadURDF(os.path.join(os.path.dirname(__file__), 'apollo_only_left_arm_without_hand.urdf'))
        if self.visualize:
            self.p.loadURDF(os.path.join(os.path.dirname(__file__), 'ball.urdf'))
        self.ARM_ID = 0
        self.ARM_JOINTS = [2, 3, 4, 5, 6, 7, 8]
        self.ENDEFF_ID = 12

        
        for joint_id in self.ARM_JOINTS:
            # As per PyBullet manual, this has to be done to be able to do torque control later
            self.p.setJointMotorControl2(self.ARM_ID, joint_id, controlMode=self.p.VELOCITY_CONTROL, force=0)

        for joint_id in self.ARM_JOINTS:
            self.p.enableJointForceTorqueSensor(bodyUniqueId=self.ARM_ID, jointIndex=joint_id, enableSensor=1)
        '''for i in range(len(self.ARM_JOINTS)):
            self.p.setJointMotorControl2(bodyUniqueId=self.ARM_ID, jointIndex=self.ARM_JOINTS[i], controlMode=self.p.TORQUE_CONTROL, force=25.0)
        '''
        self.joint_limits = []
        self.max_torque = []

        for i in range(len(self.ARM_JOINTS)):
            joint_info = self.p.getJointInfo(self.ARM_ID, self.ARM_JOINTS[i])
            self.joint_limits.append([joint_info[8], joint_info[9]])
            self.max_torque.append(joint_info[10])

        for i in range(self.p.getNumJoints(0)):
            if self.p.getJointState(0, i)[0] < self.p.getJointInfo(0, i)[8] - 1e-6 or self.p.getJointState(0, i)[0] > self.p.getJointInfo(0, i)[9] + 1e-6:
                print(i, self.p.getJointState(0, i)[0], self.p.getJointInfo(0, i)[8], self.p.getJointInfo(0, i)[9])

        #self.p.resetJointState(0, 3, targetValue=-1.5, targetVelocity=0.0)
        #assert False
        #print(i, self.p.getLinkState(0, i)[0])
        #print(self.p.getJointState(0, 61))
        #self.p.getJointState(0, i)

        action_dim = len(self.ARM_JOINTS)
        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        obs_dim = len(self.get_state())
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

        self.init_reward(reward_specs)

        if self.logging:
            if self.full_log:
                self.log = ListOfLogs(exp_name + '_episodes', separate_files=True)
            else:
                self.log = Log(exp_name + '_episodes')

        print(self.p.getPhysicsEngineParameters())

        self.n_dof = 3
        self.debug_log = []


    def get_state(self):
        state = []
        joint_pos, joint_vel = self.get_arm_state()

        state += joint_pos.tolist()
        state += joint_vel.tolist()
        if 'pushee_state' in self.observable:
            pushee_pos, pushee_vel = self.get_pushee_state()
            state += pushee_pos.tolist()
            state += pushee_vel.tolist()
        if 'joint_loads' in self.observable:
            state += self.get_joint_loads().tolist()
        if 'endeff_force' in self.observable:
            state += self.get_endeff_force().tolist()
        return np.array(state)

    def get_arm_state(self):
        joint_pos = np.zeros(len(self.ARM_JOINTS))
        joint_vel = np.zeros(len(self.ARM_JOINTS))
        for i in range(len(self.ARM_JOINTS)):
            joint_pos[i], joint_vel[i], _, _ = self.p.getJointState(self.ARM_ID, self.ARM_JOINTS[i])
        return joint_pos, joint_vel

    def get_pushee_state(self):
        pushee_state = self.p.getLinkState(self.ARM_ID, 6, computeLinkVelocity=1)
        pushee_pos = np.array(self.p.getBasePositionAndOrientation(8)[0][:2])
        pushee_vel = np.array(self.p.getBaseVelocity(8)[0][:2])
        return pushee_pos, pushee_vel

    def get_endeff_state(self):
        endeff_state = self.p.getLinkState(self.ARM_ID, self.ENDEFF_ID, computeLinkVelocity=1)
        endeff_pos = np.array(endeff_state[0])
        endeff_vel = np.array(endeff_state[6])
        return endeff_pos, endeff_vel

    def get_joint_loads(self):
        joint_load = np.zeros(len(self.ARM_JOINTS))
        for i in range(len(self.ARM_JOINTS)):
            _, _, joint_force_torque, _ = self.p.getJointState(self.ARM_ID, self.ARM_JOINTS[i])
            joint_load[i] = joint_force_torque[5]
        return joint_load

    def get_endeff_force(self):
        endeff_force = np.zeros(2)
        contacts = self.p.getContactPoints(bodyA=self.PUSHEE_ID, bodyB=self.ARM_ID)
        for contact in contacts:
            if contact[4] == self.ENDEFF_ID:
                #print('index on arm:', contact[4])
                contact_normal = np.array(contact[7][:2])
                normal_force = contact[9]
                #print('contact normal:', contact[7])
                #print('normal force:', contact[9])
                endeff_force = normal_force * contact_normal
        #print(endeff_force)
        return endeff_force

    def get_pushee_pos(self):
        pushee_pos, pushee_vel = self.get_pushee_state()
        return pushee_pos

    def get_endeff_pos(self):
        endeff_pos, endeff_vel = self.get_endeff_state()
        return endeff_pos


    def init_reward(self, rewards_config):
        self.reward_parts = {}
        for reward_type, reward_spec in rewards_config.items():
            if reward_type == 'configuration':
                self.reward_parts[reward_type] = ConfigurationReward(self, reward_spec['k_c'])
            elif reward_type == 'position':
                self.reward_parts[reward_type] = TargetReward(self, reward_spec['k_p'], reward_spec['func'], np.array(reward_spec['goal_pos']))
                if self.visualize:
                    self.p.resetBasePositionAndOrientation(1, reward_spec['goal_pos'], (0.0, 0.0, 0.0, 1.0))
            elif reward_type == 'velocity':
                self.reward_parts[reward_type] = VelocityReward(self, reward_spec['k_v'], reward_spec['square'])
            elif reward_type == 'torque':
                self.reward_parts[reward_type] = TorqueReward(self, reward_spec['k_t'], reward_spec['linear'], self.max_torque)
            else:
                assert False, 'Unknown reward type: ' + str(reward_type)


    def update_log(self):
        joint_pos, joint_vel = self.get_arm_state()
        self.log.add('joint_pos', joint_pos.tolist())
        self.log.add('joint_vel', joint_vel.tolist())

        endeff_pos, endeff_vel = self.get_endeff_state()
        self.log.add('endeff_pos', endeff_pos.tolist())
        self.log.add('endeff_vel', endeff_vel.tolist())

        for (reward_type, reward_part) in self.reward_parts.items():
            self.log.add(reward_type, reward_part.get_reward())
            #self.log.add(reward_type + '_is_done', reward_part.is_done())

        self.log.add('latest_action', self.latest_action.tolist())


    def _reset(self):
        if self.logging:
            if self.full_log:
                self.log.finish_log()
            else:
                self.log.save()
                self.log.clear()

        # Setting initial joint configuration to be random
        if self.initial_joint_state is None:
            for i in range(len(self.ARM_JOINTS)):
                self.p.resetJointState(bodyUniqueId=self.ARM_ID, jointIndex=self.ARM_JOINTS[i], targetValue=np.random.uniform(low=self.joint_limits[i][0], high=self.joint_limits[i][1]), targetVelocity=0.0)
        else:
            for i in range(len(self.ARM_JOINTS)):
                self.p.resetJointState(bodyUniqueId=self.ARM_ID, jointIndex=self.ARM_JOINTS[i], targetValue=self.initial_joint_state[i], targetVelocity=0.0)
        
        for i in range(self.p.getNumJoints(0)):
            print(i, self.p.getJointState(0, i)[0])

        if self.logging:
            joint_pos, joint_vel = self.get_arm_state()
            self.log.add('joint_pos', joint_pos.tolist())
            self.log.add('joint_vel', joint_vel.tolist())

            endeff_pos, endeff_vel = self.get_endeff_state()
            self.log.add('endeff_pos', endeff_pos.tolist())
            self.log.add('endeff_vel', endeff_vel.tolist())


        return self.get_state()

    def _step(self, action):
        '''self.debug_log.append([])
        joint_loads = self.get_joint_loads()
        for i in range(p.getNumJoints(0)):
            if i in [2]:
                jl = joint_loads[i - 2]
            else:
                jl = 0.0
            self.debug_log[-1].append([i, p.getJointState(0, i)[0], p.getJointInfo(0, i)[8], p.getJointInfo(0, i)[9], jl, p.getJointState(0, i)[3]])
            print(i, p.getJointState(0, i)[0], p.getJointInfo(0, i)[8], p.getJointInfo(0, i)[9])
        print('-------------------')
        with open('/home/miroslav/Desktop/debug_log.json', 'w') as f:
            json.dump(self.debug_log, f)'''
        self.latest_action = action
        scaled_action = np.array([0.1 * self.max_torque[i] * action[i] for i in range(action.shape[0])])
        self.latest_torque = scaled_action
        #print(scaled_action)
        
        if self.max_velocity is not None:
            _, joint_vel = self.get_arm_state()
            for i in range(len(self.ARM_JOINTS)):
                if joint_vel[i] < -self.max_velocity and scaled_action[i] < 0.0:
                    scaled_action[i] = 0.0
                if joint_vel[i] > self.max_velocity and scaled_action[i] > 0.0:
                    scaled_action[i] = 0.0
        
        for i in range(len(self.ARM_JOINTS)):
            self.p.setJointMotorControl2(bodyUniqueId=self.ARM_ID, jointIndex=self.ARM_JOINTS[i], controlMode=self.p.TORQUE_CONTROL, force=scaled_action[i])
        
        self.p.stepSimulation()
        if self.visualize:
            if self.fixed_timestep is not None:
                time.sleep(self.fixed_timestep)
            else:
                time.sleep(1./240.)

        if self.logging:
            self.update_log()

        state = self.get_state()
        reward = sum([reward_part.get_reward() for reward_part in self.reward_parts.values()])
        done = all([reward_part.is_done() for reward_part in self.reward_parts.values()])
        #print(reward)
        return state, reward, done, {}

    def _render(self, mode, close):
        pass

    def _seed(self, seed):
        pass

def log_to_video(log, fixed_timestep, reward_specs, video_file):
    with open(log) as f:    
        data = json.load(f)
    initial_joint_state = data['joint_pos'][0]
    apollo_reaching = ApolloReaching(initial_joint_state=initial_joint_state, fixed_timestep=fixed_timestep, reward_specs=reward_specs, visualize=True)
    pybullet.setRealTimeSimulation(0)
    apollo_reaching.p.setRealTimeSimulation(0)
    apollo_reaching._reset()
    time.sleep(1.0)
    logging_id = apollo_reaching.p.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, video_file)
    time.sleep(1.0)
    for action in data['latest_action']:
        #print(action)
        apollo_reaching._step(np.array(action))
    apollo_reaching.p.stopStateLogging(logging_id)
    del apollo_reaching.p
    #print(video_file)
    #assert False

def generate_videos(folder):

    exp_num = 0
    exp_folder = folder + str(exp_num).zfill(3) + '/'
    while (os.path.isdir(exp_folder)):
        if conf['alg'] == 'ddpg':
            ep_log = exp_folder + 'val_env_episodes/' + max(os.listdir(exp_folder + 'val_env_episodes/'))
        else:
            ep_log = exp_folder + 'env_episodes/' + max(os.listdir(exp_folder + 'env_episodes/'))

        with open(exp_folder + 'conf.yaml') as f:
            conf = yaml.load(f)
        fixed_timestep = conf['env_params'][0]['env_specific_params']['fixed_timestep']

        video_file = exp_folder + 'final.mp4'

        log_to_video(ep_log, fixed_timestep, video_file)

        exp_num += 1
        exp_folder = folder + str(exp_num).zfill(3) + '/'

def generate_video(exp_folder):
    with open(exp_folder + 'conf.yaml') as f:
        conf = yaml.load(f)
    if conf['alg'] == 'ddpg':
        ep_log = exp_folder + 'val_env_episodes/' + max(os.listdir(exp_folder + 'val_env_episodes/'))
    else:
        ep_log = exp_folder + 'env_episodes/' + max(os.listdir(exp_folder + 'env_episodes/'))
    with open(exp_folder + 'conf.yaml') as f:
        conf = yaml.load(f)
    fixed_timestep = conf['env_params'][0]['env_specific_params']['fixed_timestep']
    reward_specs = conf['env_params'][0]['env_specific_params']['reward_specs']
    video_file = exp_folder + 'final.mp4'

    log_to_video(ep_log, fixed_timestep, reward_specs, video_file)

def many_videos(exp_folder, num_videos):
    with open(exp_folder + 'conf.yaml') as f:
        conf = yaml.load(f)

    if conf['alg'] == 'ddpg':
        max_num = max(os.listdir(exp_folder + 'val_env_episodes/'))
    else:
        max_num = max(os.listdir(exp_folder + 'env_episodes/'))
    max_num = int(max_num.split('.')[0])

    fixed_timestep = conf['env_params'][0]['env_specific_params']['fixed_timestep']
    reward_specs = conf['env_params'][0]['env_specific_params']['reward_specs']

    for i in range(num_videos):
        if conf['alg'] == 'ddpg':
            ep_log = exp_folder + 'val_env_episodes/' + str(max_num - i).zfill(6) + '.json'
        else:
            ep_log = exp_folder + 'env_episodes/' + str(max_num - i).zfill(6) + '.json'

        video_file = exp_folder + str(max_num - i).zfill(6) + '.mp4'

        log_to_video(ep_log, fixed_timestep, reward_specs, video_file)

def specific(exp_folder, minus):
    with open(exp_folder + 'conf.yaml') as f:
        conf = yaml.load(f)

    if conf['alg'] == 'ddpg':
        max_num = max(os.listdir(exp_folder + 'val_env_episodes/'))
    else:
        max_num = max(os.listdir(exp_folder + 'env_episodes/'))
    max_num = int(max_num.split('.')[0])

    fixed_timestep = conf['env_params'][0]['env_specific_params']['fixed_timestep']
    reward_specs = conf['env_params'][0]['env_specific_params']['reward_specs']

    if conf['alg'] == 'ddpg':
        ep_log = exp_folder + 'val_env_episodes/' + str(max_num - minus).zfill(6) + '.json'
    else:
        ep_log = exp_folder + 'env_episodes/' + str(max_num - minus).zfill(6) + '.json'

    video_file = exp_folder + str(max_num - minus).zfill(6) + '.mp4'

    log_to_video(ep_log, fixed_timestep, reward_specs, video_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vis', help='')
    parser.add_argument('--o', help='')
    parser.add_argument('--mp4', help='')
    parser.add_argument('--video', help='')
    parser.add_argument('--how_many', help='')
    parser.add_argument('--minus', help='')

    args = parser.parse_args()

    if args.vis is not None:
        assert args.o is not None, 'You have to provide output path'
        with open(args.vis) as f:    
            data = json.load(f)

        initial_joint_state = data['joint_pos'][0]
        apollo_reaching = ApolloReaching(reward_specs={}, initial_pushee_pos=[-0.15, 0.15], observable=[], visualize=True, initial_joint_state=initial_joint_state, exp_name=args.o, full_log=True, fixed_timestep=0.0165)
        #while True:
        for i in range(1):
            apollo_reaching._reset()
            logging_id = apollo_reaching.p.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, args.o + '.mp4')
            for action in data['latest_action']:
                apollo_reaching._step(np.array(action))
            apollo_reaching.p.stopStateLogging(logging_id)
    elif args.mp4 is not None:
        generate_videos(args.mp4)
    elif args.video is not None:
        if args.minus is not None:
            specific(args.video, int(args.minus))
        elif args.how_many is None:
            generate_video(args.video)
        else:
            many_videos(args.video, int(args.how_many))
    else:

        apollo_reaching = ApolloReaching(reward_specs={'position':{'k_p': 1.0, 'func': 'extended', 'goal_pos': [0.0, -0.3, 0.6]}}, initial_pushee_pos=[-0.15, 0.15], observable=[], visualize=True)
        while True:
            apollo_reaching._reset()
            for i in range(10000):
                action = np.random.uniform(low=-1.0, high=1.0, size=7)
                apollo_reaching._step(action)
