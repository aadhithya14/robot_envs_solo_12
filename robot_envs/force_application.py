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

def visualize(log):
    pushee_pos = log['pushee_pos']
    endeff_pos = log['endeff_pos']
    endeff_pos = np.array(endeff_pos)
    joint_pos = log['joint_pos']

    def add_circle(ax, data, color='b'):
        circle = plt.Circle((0.0, 0.0), 0.05, color=color)
        ax.add_artist(circle)
        
        def animate_circle(i):
            circle.center = data[i]
            
        return circle, animate_circle

    def add_traj(ax, data, color='b'):
        traj = ax.plot([], [], color=color)[0]
        
        def animate_traj(i):
            traj.set_data(data[:i, 0], data[:i, 1])
            
        return traj, animate_traj

    def add_arm(ax, link_lengths, angles):
        links = []
        for i in range(len(link_lengths)):
            links.append(ax.plot([], [])[0])
            
        def animate_arm(i):
            x = 0.0
            y = 0.0
            angle = 0.0

            for j in range(len(link_lengths)):
                angle += angles[i][j]
                new_x = x + link_lengths[j] * np.cos(angle)
                new_y = y + link_lengths[j] * np.sin(angle)
                links[j].set_data([x, new_x], [y, new_y])
                x = new_x
                y = new_y
                
        return links, animate_arm

    prepare_plot(wide=False)
    rc('animation', html='html5')
    fig, ax = plt.subplots()
    plt.grid(False)
    ax.set_xlim((-0.3, 0.3))
    ax.set_ylim((-0.3, 0.3))
    ax.set_aspect('equal', adjustable='box')
        
    circle_a, animate_circle_a = add_circle(ax, pushee_pos, 'b')
    endeff_traj, animate_endeff_traj = add_traj(ax, endeff_pos, 'r')
    links, animate_arm = add_arm(ax, [0.1, 0.1, 0.11], joint_pos)

    def init():
        return circle_a, endeff_traj, links[0], links[1], links[2]

    def animate(i):
        animate_circle_a(i)
        animate_endeff_traj(i)
        animate_arm(i)
        return circle_a, endeff_traj, links[0], links[1], links[2]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(pushee_pos), interval=20, blit=True)

    HTML(anim.to_html5_video())

    plt.close()

    return anim


class DesiredForceVectorReward():

    def __init__(self, get_endeff_force, k_f, goal_force):
        self.get_endeff_force = get_endeff_force
        self.k_f = k_f
        self.goal_force = goal_force

    def get_reward(self):
        return -self.k_f * np.linalg.norm(self.goal_force - self.get_endeff_force())

    def is_done(self):
        return False


class IncentiveReward():

    def __init__(self, get_endeff_pos, get_pushee_pos, k_i, goal_force, incentive_type, critical_zone, box_dim=0.6):
        self.get_endeff_pos = get_endeff_pos
        self.get_pushee_pos = get_pushee_pos
        self.k_i = k_i
        self.goal_force = goal_force
        self.incentive_type = incentive_type
        self.critical_zone = critical_zone
        self.box_dim = box_dim

    def get_reward(self):
        pusher_pos = self.get_endeff_pos()
        pushee_pos = self.get_pushee_pos()
        pusher_r, pusher_theta = polar_coord(pusher_pos - pushee_pos)
        force_r, force_theta = polar_coord(self.goal_force)

        if pusher_r > self.critical_zone:
            incentive = 0.1 * (1.0 - pusher_r / (self.box_dim * np.sqrt(2)))
        elif self.incentive_type == 1:
            incentive = 1.0
        else:
            off_angle = ang_dist(pusher_theta, force_theta + np.pi)
            incentive = 1.0 - 0.6 * off_angle / np.pi
            if self.incentive_type == 3:
                incentive -= 0.4 * (pusher_r / self.critical_zone)

        return self.k_i * incentive

    def is_done(self):
        return False


class ForceApplication(gym.Env):

    def __init__(self, reward_specs, initial_pushee_pos, max_torque=0.1, observable=[], visualize=False, exp_name='', output_dir='', initial_joint_state=None, fixed_timestep=None, full_log=False, clip_force=None):
        self.observable = observable
        self.visualize = visualize
        self.initial_pushee_pos = initial_pushee_pos
        self.max_torque = max_torque
        self.initial_joint_state = initial_joint_state
        self.full_log = full_log
        self.clip_force = clip_force

        if self.visualize:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        self.p.setGravity(0,0,-10)
        if fixed_timestep is not None:
            self.p.setPhysicsEngineParameter(fixedTimeStep=fixed_timestep)
        robot = self.p.loadMJCF(os.path.join(os.path.dirname(__file__), 'force_application.xml'))

        self.ARM_ID = 6
        self.ARM_JOINTS = [0, 2, 4]
        self.ENDEFF_ID = 6
        self.PUSHEE_ID = 8

        for joint_id in self.ARM_JOINTS:
            # As per PyBullet manual, this has to be done to be able to do torque control later
            self.p.setJointMotorControl2(self.ARM_ID, joint_id, controlMode=self.p.VELOCITY_CONTROL, force=0)

        for joint_id in self.ARM_JOINTS:
            self.p.enableJointForceTorqueSensor(bodyUniqueId=self.ARM_ID, jointIndex=joint_id, enableSensor=1)

        action_dim = 3
        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        obs_dim = len(self.get_state())
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

        self.init_reward(reward_specs)

        if self.full_log:
            self.log = ListOfLogs(exp_name + '_episodes', separate_files=True)
        else:
            self.log = Log(exp_name + '_episodes')

        print(self.p.getPhysicsEngineParameters())


    def get_state(self):
        state = []
        joint_pos, joint_vel = self.get_arm_state()
        # Reproducing roboschool way of encoding arm state
        # TODO: Consider switching to a simpler way 
        encoded_joint_state = [
            np.sin(joint_pos[0]),
            np.cos(joint_pos[0]),
            joint_vel[0] * 0.1,
            joint_pos[1] / 3.0,
            joint_vel[1] * 0.1,
            joint_pos[2] / 3.0,
            joint_vel[2] * 0.1]
        state += encoded_joint_state
        if 'pushee_state' in self.observable:
            pushee_pos, pushee_vel = self.get_pushee_state()
            state += pushee_pos.tolist()
            state += pushee_vel.tolist()
        if 'joint_loads' in self.observable:
            state += self.get_joint_loads().tolist()
        if 'endeff_force' in self.observable:
            state += self.get_endeff_force().tolist()
        if 'endeff_force_intensity' in self.observable:
            state += [self.get_endeff_force_intensity()]
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
        endeff_state = self.p.getLinkState(self.ARM_ID, 6, computeLinkVelocity=1)
        endeff_pos = np.array(endeff_state[0][:2])
        endeff_vel = np.array(endeff_state[6][:2])
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
                if self.clip_force is not None:
                    if normal_force > self.clip_force:
                        normal_force = self.clip_force
                    if normal_force < -self.clip_force:
                        normal_force = -self.clip_force
                #print('contact normal:', contact[7])
                #print('normal force:', contact[9])
                endeff_force = normal_force * contact_normal
        #print(endeff_force)
        return endeff_force

    def get_endeff_force_intensity(self):
        return np.linalg.norm(self.get_endeff_force())

    def get_pushee_pos(self):
        pushee_pos, pushee_vel = self.get_pushee_state()
        return pushee_pos

    def get_endeff_pos(self):
        endeff_pos, endeff_vel = self.get_endeff_state()
        return endeff_pos

    def get_pushee_force(self):
        pushee_force = np.zeros(2)
        contacts = self.p.getContactPoints(bodyA=self.PUSHEE_ID, bodyB=self.ARM_ID)
        for contact in contacts:
            if contact[4] == self.ENDEFF_ID:
                contact_normal = -1 * np.array(contact[7][:2])
                normal_force = contact[9]
                if self.clip_force is not None:
                    if normal_force > self.clip_force:
                        normal_force = self.clip_force
                    if normal_force < -self.clip_force:
                        normal_force = -self.clip_force
                pushee_force += normal_force * contact_normal
        return pushee_force

    def get_pushee_force_intensity(self):
        return np.linalg.norm(self.get_pushee_force())

    def get_pushee_to_endeff(self):
        return self.get_pushee_pos() - self.get_endeff_pos()


    def init_reward(self, rewards_config):
        self.reward_parts = {}
        for reward_type, reward_spec in rewards_config.items():
            if reward_type == 'desired_force':
                self.reward_parts[reward_type] = DesiredForceVectorReward(self.get_pushee_force, reward_spec['k_f'], reward_spec['goal_force'])
            elif reward_type == 'incentive_reward':
                self.reward_parts[reward_type] = IncentiveReward(self.get_endeff_pos, self.get_pushee_pos, reward_spec['k_i'], reward_spec['goal_force'], reward_spec['incentive_type'], reward_spec['critical_zone'])
            elif reward_type == 'desired_force_intensity':
                self.reward_parts[reward_type] = DesiredForceIntensityReward(self.get_pushee_force_intensity, reward_spec['k_f'], reward_spec['goal_force'])
            elif reward_type == 'desired_force_direction':
                self.reward_parts[reward_type] = DesiredForceDirectionReward(self.get_pushee_force, reward_spec['k_d'], reward_spec['goal_direction'])
            elif reward_type == 'desired_push_direction':
                self.reward_parts[reward_type] = DesiredForceDirectionReward(self.get_pushee_to_endeff, reward_spec['k_d'], reward_spec['goal_direction'])
            else:
                assert False, 'Unknown reward type: ' + str(reward_type)


    def update_log(self):
        joint_pos, joint_vel = self.get_arm_state()
        self.log.add('joint_pos', joint_pos.tolist())
        self.log.add('joint_vel', joint_vel.tolist())

        pushee_pos, pushee_vel = self.get_pushee_state()
        self.log.add('pushee_pos', pushee_pos.tolist())
        self.log.add('pushee_vel', pushee_vel.tolist())

        endeff_pos, endeff_vel = self.get_endeff_state()
        self.log.add('endeff_pos', endeff_pos.tolist())
        self.log.add('endeff_vel', endeff_vel.tolist())

        self.log.add('pushee_force', self.get_pushee_force().tolist())
        self.log.add('pushee_force_intensity', self.get_pushee_force_intensity())

        for (reward_type, reward_part) in self.reward_parts.items():
            self.log.add(reward_type, reward_part.get_reward())
            #self.log.add(reward_type + '_is_done', reward_part.is_done())


    def _reset(self):
        if self.full_log:
            self.log.finish_log()
        else:
            self.log.save()
            self.log.clear()

        #pushee_state = p.getBasePositionAndOrientation(8)
        #print(pushee_state)
        self.p.resetBasePositionAndOrientation(self.PUSHEE_ID, (self.initial_pushee_pos[0], self.initial_pushee_pos[1], 0.01), (0.0, 0.0, 0.0, 1.0))

        # Setting initial joint configuration to be random
        if self.initial_joint_state is None:
            for joint_id in self.ARM_JOINTS:
                self.p.resetJointState(bodyUniqueId=self.ARM_ID, jointIndex=joint_id, targetValue=np.random.uniform(low=-np.pi, high=np.pi), targetVelocity=0.0)
        else:
            for i in range(len(self.ARM_JOINTS)):
                self.p.resetJointState(bodyUniqueId=self.ARM_ID, jointIndex=self.ARM_JOINTS[i], targetValue=self.initial_joint_state[i], targetVelocity=0.0)

        return self.get_state()

    def _step(self, action):
        scaled_action = self.max_torque * action

        for i in range(len(self.ARM_JOINTS)):
            self.p.setJointMotorControl2(bodyUniqueId=self.ARM_ID, jointIndex=self.ARM_JOINTS[i], controlMode=self.p.TORQUE_CONTROL, force=scaled_action[i])

        self.p.stepSimulation()
        if self.visualize:
            time.sleep(1./240.)

        self.update_log()

        state = self.get_state()
        reward = sum([reward_part.get_reward() for reward_part in self.reward_parts.values()])
        done = all([reward_part.is_done() for reward_part in self.reward_parts.values()])

        #print(self.get_pushee_force())
        #print(self.get_endeff_force())
        #print(self.reward_parts['incentive_reward'].get_reward())

        return state, reward, done, {}

    def _render(self, mode, close):
        pass

    def _seed(self, seed):
        pass

if __name__ == "__main__":
    pushing_arm = ForceApplication(reward_specs={'incentive_reward': {'k_i': 1.0, 'critical_zone': 0.1, 'goal_force': [-0.5, -0.5], 'incentive_type': 2}}, initial_pushee_pos=[0.07, 0.07], observable=['joint_loads', 'endeff_force', 'pushee_state'], visualize=True, max_torque=0.05, fixed_timestep=0.0165)
    while True:
        pushing_arm._reset()
        for i in range(10000):
            action = np.random.uniform(low=-1.0, high=1.0, size=3)
            pushing_arm._step(action)
