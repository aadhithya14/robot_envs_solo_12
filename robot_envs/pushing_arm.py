import pybullet as p
import time
import pybullet_data
import numpy as np
import gym
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

class PushingArm(gym.Env):

    def __init__(self, reward_specs, initial_pushee_pos, max_torque=0.1, observable=[], visualize=False, exp_name='', output_dir='', initial_joint_state=None, fixed_timestep=None, full_log=False):
        self.observable = observable
        self.visualize = visualize
        self.initial_pushee_pos = initial_pushee_pos
        self.max_torque = max_torque
        self.initial_joint_state = initial_joint_state
        self.full_log = full_log

        if self.visualize:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        if fixed_timestep is not None:
            p.setPhysicsEngineParameter(fixedTimeStep=fixed_timestep)
        robot = p.loadMJCF(os.path.join(os.path.dirname(__file__), 'pushing_arm.xml'))

        self.ARM_ID = 6
        self.ARM_JOINTS = [0, 2, 4]
        self.ENDEFF_ID = 6
        self.PUSHEE_ID = 8

        for joint_id in self.ARM_JOINTS:
            # As per PyBullet manual, this has to be done to be able to do torque control later
            p.setJointMotorControl2(self.ARM_ID, joint_id, controlMode=p.VELOCITY_CONTROL, force=0)

        for joint_id in self.ARM_JOINTS:
            p.enableJointForceTorqueSensor(bodyUniqueId=self.ARM_ID, jointIndex=joint_id, enableSensor=1)

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

        print(p.getPhysicsEngineParameters())


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
        return np.array(state)

    def get_arm_state(self):
        joint_pos = np.zeros(len(self.ARM_JOINTS))
        joint_vel = np.zeros(len(self.ARM_JOINTS))
        for i in range(len(self.ARM_JOINTS)):
            joint_pos[i], joint_vel[i], _, _ = p.getJointState(self.ARM_ID, self.ARM_JOINTS[i])
        return joint_pos, joint_vel

    def get_pushee_state(self):
        pushee_state = p.getLinkState(self.ARM_ID, 6, computeLinkVelocity=1)
        pushee_pos = np.array(p.getBasePositionAndOrientation(8)[0][:2])
        pushee_vel = np.array(p.getBaseVelocity(8)[0][:2])
        return pushee_pos, pushee_vel

    def get_endeff_state(self):
        endeff_state = p.getLinkState(self.ARM_ID, 6, computeLinkVelocity=1)
        endeff_pos = np.array(endeff_state[0][:2])
        endeff_vel = np.array(endeff_state[6][:2])
        return endeff_pos, endeff_vel

    def get_joint_loads(self):
        joint_load = np.zeros(len(self.ARM_JOINTS))
        for i in range(len(self.ARM_JOINTS)):
            _, _, joint_force_torque, _ = p.getJointState(self.ARM_ID, self.ARM_JOINTS[i])
            joint_load[i] = joint_force_torque[5]
        return joint_load

    def get_endeff_force(self):
        endeff_force = np.zeros(2)
        contacts = p.getContactPoints(bodyA=self.PUSHEE_ID, bodyB=self.ARM_ID)
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
            if reward_type == 'pushee_location':
                self.reward_parts[reward_type] = GoalPositionReward(self.get_pushee_pos, reward_spec['k_p'], reward_spec['goal_pos'], reward_spec['req_dist'])
            elif reward_type == 'endeff_to_pushee':
                self.reward_parts[reward_type] = RelativePositionReward(self.get_endeff_pos, self.get_pushee_pos, reward_spec['k_p'])
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
        p.resetBasePositionAndOrientation(self.PUSHEE_ID, (self.initial_pushee_pos[0], self.initial_pushee_pos[1], 0.01), (0.0, 0.0, 0.0, 1.0))

        # Setting initial joint configuration to be random
        if self.initial_joint_state is None:
            for joint_id in self.ARM_JOINTS:
                p.resetJointState(bodyUniqueId=self.ARM_ID, jointIndex=joint_id, targetValue=np.random.uniform(low=-np.pi, high=np.pi), targetVelocity=0.0)
        else:
            for i in range(len(self.ARM_JOINTS)):
                p.resetJointState(bodyUniqueId=self.ARM_ID, jointIndex=self.ARM_JOINTS[i], targetValue=self.initial_joint_state[i], targetVelocity=0.0)

        return self.get_state()

    def _step(self, action):
        scaled_action = self.max_torque * action

        for i in range(len(self.ARM_JOINTS)):
            p.setJointMotorControl2(bodyUniqueId=self.ARM_ID, jointIndex=self.ARM_JOINTS[i], controlMode=p.TORQUE_CONTROL, force=scaled_action[i])

        p.stepSimulation()
        if self.visualize:
            time.sleep(1./240.)

        self.update_log()

        state = self.get_state()
        reward = sum([reward_part.get_reward() for reward_part in self.reward_parts.values()])
        done = all([reward_part.is_done() for reward_part in self.reward_parts.values()])

        return state, reward, done, {}

    def _render(self, mode, close):
        pass

    def _seed(self, seed):
        pass

if __name__ == "__main__":
    pushing_arm = PushingArm(reward_specs={}, initial_pushee_pos=[-0.15, 0.15], observable=['joint_loads', 'endeff_force', 'pushee_state'], visualize=True)
    while True:
        pushing_arm._reset()
        for i in range(10000):
            action = np.random.uniform(low=-1.0, high=1.0, size=3)
            pushing_arm._step(action)
