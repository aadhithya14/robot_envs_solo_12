import json
import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Box

from utils.my_math import scale
from utils.data_logging import SimpleLog
from utils.plotting import prepare_plot
import robot_envs.apollo_wall_pushing

class PositionGainController():

    def __init__(self, robot, params, grav_comp=True, enabled_joints=range(7)):
        self.robot = robot
        self.base_kp = np.array([33.8, 66.7, 16.9, 40.0, 5.0, 7.2, 0.05])
        self.base_kv = np.array([5.0, 20.0, 5.0, 20.0, 1.0, 1.0, 0.02])
        self.grav_comp = grav_comp
        self.variant = params['variant']
        self.sqrt_kv_scaling = True
        if 'sqrt_kv_scaling' in params:
            self.sqrt_kv_scaling = params['sqrt_kv_scaling']

        if self.variant in ['vary_single', 'vary_all']:
            self.scaling_base = 10.0
            if 'max_scale' in params:
                self.scaling_base = params['max_scale']
            if 'scaling_range' in params:
                self.scaling_base = np.sqrt(params['scaling_range'])

        self.ndof = self.robot.NUM_JOINTS
        self.max_torque = np.array([self.robot.max_torque_percent * torque for torque in self.robot.max_torque])
        self.joint_limits = self.robot.joint_limits

        self.log = SimpleLog()

        self.robot.init_torque_control(enabled_joints=enabled_joints)

        self.params = params

    def get_control_space(self):
        if self.variant == 'fixed':
            return Box(-1.0, 1.0, (self.ndof,))
        elif self.variant == 'vary_single':
            return Box(-1.0, 1.0, (self.ndof + 1,))
        elif self.variant == 'vary_all':
            return Box(-1.0, 1.0, (2 * self.ndof,))
        else:
            assert False, 'Unknown PG controller variant: ' + self.variant

    def reset(self):
        self.log.clear()

    def act(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.des_pos = scale(action[:self.ndof], [-1.0, 1.0], self.joint_limits)

        new_parametrization = False
        if 'new_parametrization' in self.params:
            new_parametrization = self.params['new_parametrization']
        if new_parametrization:
            if self.variant == 'fixed':
                scaling_factor = np.ones(self.ndof)
                if 'gain_multiplier' in self.params:
                    scaling_factor *= self.params['gain_multiplier']
            elif self.variant == 'vary_all':
                scaling_range = self.params['gain_multiplier_range']
                scaling_factor = scale(action[self.ndof:], [-1.0, 1.0], scaling_range)
        else:
            if self.variant == 'fixed':
                scaling_factor = np.ones(self.ndof)

            elif self.variant == 'vary_single':
                scaling_factor = np.power(self.scaling_base, action[self.ndof]) * np.ones(self.ndof)
            elif self.variant == 'vary_all':
                scaling_factor = np.power(self.scaling_base, action[self.ndof:])
            else:
                assert False, 'Unknown PositionGainController variant: ' + self.variant

        kp = np.multiply(scaling_factor, self.base_kp)
        if self.sqrt_kv_scaling:
            kv = np.multiply(np.sqrt(scaling_factor), self.base_kv)
        else:
            kv = np.multiply(scaling_factor, self.base_kv)

        #if self.variant != 'fixed':
        #    if self.variant == 'vary_single':
        #        self.k_p = np.power(self.max_scale, action[self.ndof]) * self.base_kp
        #    else:
        #        self.k_p = np.multiply(np.power(self.max_scale, action[self.ndof:]), self.base_kp)

        pos, vel = self.robot.get_arm_state()
        pd_torque = np.multiply(kp, self.des_pos - pos) - np.multiply(kv, vel)

        gc_torque = 0.0
        if self.grav_comp:
            gc_torque = self.robot.inv_dyn(np.zeros(self.ndof))

        self.log.add('des_pos', self.des_pos)
        self.log.add('pos', pos)
        self.log.add('vel', vel)
        self.log.add('p_torque', np.multiply(kp, self.des_pos - pos))
        self.log.add('d_torque', -np.multiply(kv, vel))
        self.log.add('pd_torque', pd_torque)
        self.log.add('gc_torque', gc_torque)

        torque = pd_torque + gc_torque
        self.robot.torque_control(torque)

    def visualize(self):
        prepare_plot()
        plt.rcParams["figure.figsize"] = (24, 8)

        fig, axes = plt.subplots(2, self.ndof)

        pos = np.array(self.log.d['pos'])
        des_pos = np.array(self.log.d['des_pos'])
        pd_torque = np.array(self.log.d['pd_torque'])
        gc_torque = np.array(self.log.d['gc_torque'])

        for i in range(self.ndof):
            axes[0, i].set_title('pos[' + str(i) + ']')
            axes[0, i].grid(True)
            axes[0, i].plot(pos[:, i])
            axes[0, i].plot(des_pos[:, i], '--')

            axes[1, i].set_title('torque[' + str(i) + ']')
            axes[1, i].grid(True)
            axes[1, i].plot(pd_torque[:, i])
            axes[1, i].plot(gc_torque[:, i], '--')

        plt.tight_layout()
        plt.show()


def tune(joint):
    apollo_env = robot_envs.apollo_wall_pushing.ApolloWallPushing(visualize=False, control='position_gain', max_torque_percent=0.1, log_file='/Users/miroslav/Desktop/tmpfile.json', exp_name='/Users/miroslav/Desktop/', table_height=None)

    #6: k_ps = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    #6: k_vs = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    k_ps = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    k_vs = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]


    prepare_plot()
    plt.rcParams["figure.figsize"] = (24, 16)

    fig, axes = plt.subplots(len(k_ps), len(k_vs))

    for i in range(len(k_ps)):
        k_p = k_ps[i]
        apollo_env.controller.k_p[joint] = k_p * 10.0
        for j in range(len(k_vs)):
            k_v = k_vs[j]
            apollo_env.controller.k_v[joint] = k_v * 0.1
            error = 0.0
            for l in range(1):
                apollo_env._reset()
                for k in range(500):
                    apollo_env._step(np.zeros(7))
                pos = np.array(apollo_env.controller.log.d['pos'])
                error += np.sum(np.absolute(pos[:, joint]))
            pos = np.array(apollo_env.controller.log.d['pos'])
            des_pos = np.array(apollo_env.controller.log.d['des_pos'])
            vel = np.array(apollo_env.controller.log.d['vel'])
            #error = np.sum(np.absolute(pos[:, 6]))
            axes[i, j].set_title('k_p=' + str(apollo_env.controller.k_p[joint]) + ',k_v=' + str(apollo_env.controller.k_v[joint]))
            #axes[i, j].set_title(str(error))
            axes[i, j].grid(True)
            axes[i, j].plot(pos[:, joint])
            axes[i, j].plot(des_pos[:, joint], '--')
            #axes[i, j].plot(vel[:, 6])

    plt.tight_layout()
    plt.show()

def goto():
    apollo_env = robot_envs.apollo_wall_pushing.ApolloWallPushing(visualize=True, control='position_gain', max_torque_percent=0.1, log_file='/Users/miroslav/Desktop/tmpfile.json', exp_name='/Users/miroslav/Desktop/', table_height=None)
    while True:
        apollo_env._reset()
        for i in range(250):
            apollo_env._step(np.zeros(7))
    #apollo_env.controller.visualize()

def visualize_log(log_file):
    prepare_plot()
    plt.rcParams["figure.figsize"] = (24, 18)

    fig, axes = plt.subplots(4, 7)

    with open(log_file, 'r') as f:
        log = json.load(f)
    print(log.keys())

    action = np.array(log['latest_action'])
    for i in range(action.shape[1]):
        axes[0, i].set_title('action[%d]' % i)
        axes[0, i].grid(True)
        axes[0, i].plot(action[:, i])

    joint_pos = np.array(log['joint_pos'])
    for i in range(joint_pos.shape[1]):
        axes[1, i].set_title('joint_pos[%d]' % i)
        axes[1, i].grid(True)
        axes[1, i].plot(joint_pos[:, i])

    joint_vel = np.array(log['joint_vel'])
    for i in range(joint_vel.shape[1]):
        axes[2, i].set_title('joint_vel[%d]' % i)
        axes[2, i].grid(True)
        axes[2, i].plot(joint_vel[:, i])

    endeff_force_z = np.array(log['endeff_force_z'])
    axes[3, 0].set_title('endeff_forcez')
    axes[3, 0].grid(True)
    axes[3, 0].plot(endeff_force_z)

    plt.tight_layout()
    plt.show()


def get_init_position():
    log_file = '/Users/miroslav/work/code/robot_envs/robot_envs/apollo_unit_tests/106_018_009997_log.json'
    with open(log_file, 'r') as f:
        log = json.load(f)
    joint_pos = np.array(log['joint_pos'])
    return joint_pos[100]

def press():
    apollo_env = robot_envs.apollo_wall_pushing.ApolloWallPushing(initial_joint_state=get_init_position(), visualize=False, control='position_gain', max_torque_percent=0.1, log_file='/Users/miroslav/Desktop/tmpfile.json', exp_name='/Users/miroslav/Desktop/', table_height=0.9)
    base = scale(get_init_position(), apollo_env.controller.joint_limits, [-1.0, 1.0])
    force_profiles = []
    W = 5
    H = 5
    N = W * H
    count = 0
    while len(force_profiles) < N:
        #print(len(force_profiles), '/', count)
        apollo_env._reset()
        action = base + np.random.uniform(low=-0.3, high=0.3, size=7)
        for i in range(250):
            apollo_env._step(action)
        #force_profiles.append(apollo_env.log.d['endeff_force_z'].copy())
        endeff_pos = np.array(apollo_env.log.d['endeff_pos'])
        endeff_force_z = np.array(apollo_env.log.d['endeff_force_z'])
        if np.sum(endeff_force_z) > 250.0:
            print(len(force_profiles), ':', action - base)
            force_profiles.append(endeff_force_z)
        count += 1

    prepare_plot()
    plt.rcParams["figure.figsize"] = (24, 18)
    fig, axes = plt.subplots(H, W)

    for i in range(N):
        axes[i // W, i % W].set_title(str(i))
        axes[i // W, i % W].grid(True)
        axes[i // W, i % W].plot(force_profiles[i])

    plt.tight_layout()
    plt.show()

def vary_point(base_change):
    apollo_env = robot_envs.apollo_wall_pushing.ApolloWallPushing(initial_joint_state=get_init_position(), visualize=False, control='position_gain', max_torque_percent=0.1, log_file='/Users/miroslav/Desktop/tmpfile.json', exp_name='/Users/miroslav/Desktop/', table_height=0.9)
    base = scale(get_init_position(), apollo_env.controller.joint_limits, [-1.0, 1.0])
    force_profiles = []
    W = 5
    H = 5
    N = W * H
    count = 0
    for k in np.linspace(0.1, 2.5, 25):
        #print(len(force_profiles), '/', count)
        apollo_env._reset()
        action = base + base_change * k
        for i in range(250):
            apollo_env._step(action)
        #force_profiles.append(apollo_env.log.d['endeff_force_z'].copy())
        endeff_pos = np.array(apollo_env.log.d['endeff_pos'])
        endeff_force_z = np.array(apollo_env.log.d['endeff_force_z'])
        #if np.sum(endeff_force_z) > 250.0:
        #    print(len(force_profiles), ':', action - base)
        force_profiles.append(endeff_force_z)
        count += 1

    prepare_plot()
    plt.rcParams["figure.figsize"] = (24, 18)
    fig, axes = plt.subplots(H, W)

    for i in range(N):
        axes[i // W, i % W].set_title(str(i))
        axes[i // W, i % W].grid(True)
        axes[i // W, i % W].plot(force_profiles[i])

    plt.tight_layout()
    plt.show()

def vary_kp(base_change):
    apollo_env = robot_envs.apollo_wall_pushing.ApolloWallPushing(initial_joint_state=get_init_position(), visualize=False, control='position_gain', max_torque_percent=0.1, log_file='/Users/miroslav/Desktop/tmpfile.json', exp_name='/Users/miroslav/Desktop/', table_height=0.9)
    base = scale(get_init_position(), apollo_env.controller.joint_limits, [-1.0, 1.0])
    force_profiles = []
    W = 5
    H = 5
    N = W * H
    count = 0
    space = np.linspace(0.1, 2.5, 25)
    for k in space:
        #print(len(force_profiles), '/', count)
        apollo_env._reset()
        apollo_env.controller.k_p = k * np.array([100.0, 2000.0, 50.0, 1000.0, 10.0, 10.0, 0.5])
        action = base + base_change
        for i in range(250):
            apollo_env._step(action)
        #force_profiles.append(apollo_env.log.d['endeff_force_z'].copy())
        endeff_pos = np.array(apollo_env.log.d['endeff_pos'])
        endeff_force_z = np.array(apollo_env.log.d['endeff_force_z'])
        #if np.sum(endeff_force_z) > 250.0:
        #    print(len(force_profiles), ':', action - base)
        force_profiles.append(endeff_force_z)
        count += 1

    prepare_plot()
    plt.rcParams["figure.figsize"] = (24, 18)
    fig, axes = plt.subplots(H, W)

    for i in range(N):
        axes[i // W, i % W].set_title(str(space[i]))
        axes[i // W, i % W].grid(True)
        axes[i // W, i % W].plot(force_profiles[i])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #tune(0)
    #goto()
    #visualize_log('/Users/miroslav/work/code/robot_envs/robot_envs/apollo_unit_tests/106_018_009997_log.json')
    #press()
    vary_kp(np.array([-0.18996608, -0.17067587, 0.22522406, 0.29590095, -0.29584202, -0.04026848, -0.22075812]))
