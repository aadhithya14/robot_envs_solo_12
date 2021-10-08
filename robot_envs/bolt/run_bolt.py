import os
import time
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from robot_envs.bolt.bolt import Bolt
from robot_envs.bolt.bolt_initialization import *


def show_trajectory(traj_file, loop_motion=False):
    traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                     + '/trajectories/'
    traj = np.loadtxt(traj_directory + traj_file)

    bolt = Bolt(visualize=True,
                reward_specs={
                    'base_stability_penalty': {
                        # 'mean': [0.3, 0.0, 0.0],
                        # 'sigma': [0.02, 0.0081, 0.0094]}})
                        'mean': [0.3, 0.0, 0.0],
                        'sigma': [0.04, 0.02, 0.02]},
                    'linear_distance_to_goal': {}},
                demo_traj_file=traj_file,
                use_movable_disk_surface=True)
    bolt._reset()
    angles = []
    rewards = []

    def episode():
        for i in range(traj.shape[0]):
            bolt.p.resetBasePositionAndOrientation(
                bodyUniqueId=bolt.robot_id,
                posObj=traj[i, 0:3],
                ornObj=traj[i, 3:7])

            for j in range(bolt.num_obs_joints):
                bolt.p.resetJointState(
                    bodyUniqueId=bolt.robot_id,
                    jointIndex=bolt.obs_joint_ids[j],
                    targetValue=traj[i, j + 7],
                    targetVelocity=0.0)

            angles.append(bolt.p.getEulerFromQuaternion(traj[i, 3:7]))
            rewards.append(bolt.reward_parts['linear_distance_to_goal'].get_reward())

            time.sleep(0.001)

    if loop_motion:
        while True:
            episode()
    else:
        episode()

    angles = np.array(angles)
    # plt.plot(traj[:, 2])
    # plt.plot(angles[:, 0], label='angles[0]')
    # plt.plot(angles[:, 1], label='angles[1]')
    # plt.plot(angles[:, 2], label='angles[2]')
    plt.legend()
    # plt.plot(rewards)
    # plt.plot(traj[:, 0])
    # plt.plot(traj[:, 1])
    plt.plot(traj[:, 2])
    # plt.plot(traj[:, 14])
    # for i in range(7, 13):
    #     plt.plot(traj[:, i], label=str(i))
    # plt.legend()
    # print(np.min(traj[:, 2]), np.max(traj[:, 2]))
    # print(np.min(angles[:, 0]), np.max(angles[:, 0]))
    # print(np.min(angles[:, 1]), np.max(angles[:, 1]))
    plt.show()

def test_trajectory_imitation_reward(traj_file):
    traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                     + '/trajectories/'
    traj = np.loadtxt(traj_directory + traj_file)

    bolt = Bolt(visualize=False,
                reward_specs={'trajectory_imitation_reward': {}},
                demo_traj_file=traj_file,
                initialization_conf={'on_demo_traj': True})
    bolt._reset()
    rewards = []

    for i in range(traj.shape[0]):
        print(i)
        bolt.p.resetBasePositionAndOrientation(
            bodyUniqueId=bolt.robot_id,
            posObj=traj[i, 0:3],
            ornObj=traj[i, 3:7])

        for j in range(bolt.num_obs_joints):
            bolt.p.resetJointState(
                bodyUniqueId=bolt.robot_id,
                jointIndex=bolt.obs_joint_ids[j],
                targetValue=traj[i, j + 7],
                targetVelocity=0.0)

        rewards.append(bolt.reward_parts['trajectory_imitation_reward'].get_reward())

        for reward_object in bolt.reward_parts.values():
            reward_object.step()

    plt.plot(rewards)
    plt.show()

def show_log_file(log_file):
    log = json.load(open(log_file, 'r'))
    state = np.array(log['state'])

    bolt = Bolt(visualize=True)
    bolt._reset()

    print(state.shape)
    for i in range(state.shape[0]):
        bolt.p.resetBasePositionAndOrientation(
            bodyUniqueId=bolt.robot_id,
            posObj=state[i, 0:3],
            ornObj=state[i, 3:7])

        for j in range(bolt.num_obs_joints):
            bolt.p.resetJointState(
                bodyUniqueId=bolt.robot_id,
                jointIndex=bolt.obs_joint_ids[j],
                targetValue=state[i, j + 13],
                targetVelocity=0.0)

        time.sleep(0.008)

def show_all_log_files(log_folder):
    files = os.listdir(log_folder)
    files.sort()
    bolt = Bolt(visualize=True)
    for file in files:
        log_file = log_folder + file

        log = json.load(open(log_file, 'r'))
        state = np.array(log['state'])

        bolt._reset()

        for i in range(state.shape[0]):
            bolt.p.resetBasePositionAndOrientation(
                bodyUniqueId=bolt.robot_id,
                posObj=state[i, 0:3],
                ornObj=state[i, 3:7])

            for j in range(bolt.num_obs_joints):
                bolt.p.resetJointState(
                    bodyUniqueId=bolt.robot_id,
                    jointIndex=bolt.obs_joint_ids[j],
                    targetValue=state[i, j + 13],
                    targetVelocity=0.0)

            time.sleep(0.008)

def show_all_experiments(exp_set_folder):
    bolt = Bolt(visualize=True)

    exp_folders = os.listdir(exp_set_folder)
    exp_folders.sort()
    for exp_folder in exp_folders:
        try:
            print(exp_folder)
            exp_folder_path = exp_set_folder + exp_folder + '/'

            conf_file = exp_folder_path + 'conf.yaml'
            with open(conf_file, 'r') as f:
                conf = yaml.load(f)

            params = conf['env_params'][0]['env_specific_params']
            if 'action_filter_conf' in params:
                num = params['action_filter_conf']['sim_timestep_length'] / 8
            else:
                num = 0

            if params['cont_timestep_mult'] == 1:
                assert False

            text_id = bolt.p.addUserDebugText(exp_folder + ' ' + str(num), [0.5, 0.0, 0.5], textColorRGB=[1, 0, 0])
            if os.path.isdir(exp_folder_path):
                if os.path.exists(exp_folder_path + 'val_env_episodes/'):
                    log_folder = exp_folder_path + 'val_env_episodes/'
                else:
                    log_folder = exp_folder_path + 'env_episodes/'

                files = os.listdir(log_folder)
                files.sort()
                for file in files[:5]:
                    log_file = log_folder + file

                    log = json.load(open(log_file, 'r'))
                    state = np.array(log['state'])

                    bolt._reset()

                    for i in range(state.shape[0]):
                        bolt.p.resetBasePositionAndOrientation(
                            bodyUniqueId=bolt.robot_id,
                            posObj=state[i, 0:3],
                            ornObj=state[i, 3:7])

                        for j in range(bolt.num_obs_joints):
                            bolt.p.resetJointState(
                                bodyUniqueId=bolt.robot_id,
                                jointIndex=bolt.obs_joint_ids[j],
                                targetValue=state[i, j + 13],
                                targetVelocity=0.0)

                        time.sleep(params['cont_timestep_mult'] * 0.001)
            bolt.p.removeUserDebugItem(text_id)
        except Exception as e:
            print(e)



def get_clusters(foot_locations, max_dist):
    clusters = []
    for fl in foot_locations:
        # print(len(clusters))
        found = False
        for c in clusters:
            if not found:
                cluster_center = np.mean(np.array(c), axis=0)

                dist = np.linalg.norm(cluster_center - fl)

                if dist < max_dist:
                    c.append(fl.tolist())
                    found = True
        if not found:
            clusters.append([fl.tolist()])
            print(len(clusters))
    r = [np.mean(np.array(c), axis=0) for c in clusters]
    return np.array(r)

def get_foot_locations(traj_file):
    traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                     + '/trajectories/'
    traj = np.loadtxt(traj_directory + traj_file)

    bolt = Bolt(visualize=False,
                reward_specs={
                    'base_stability_penalty': {
                        # 'mean': [0.3, 0.0, 0.0],
                        # 'sigma': [0.02, 0.0081, 0.0094]}})
                        'mean': [0.3, 0.0, 0.0],
                        'sigma': [0.04, 0.02, 0.02]}})
    bolt._reset()
    angles = []
    rewards = []
    foot_locations = []
    timesteps = []
    for i in range(traj.shape[0]):
        bolt.p.resetBasePositionAndOrientation(
            bodyUniqueId=bolt.robot_id,
            posObj=traj[i, 0:3],
            ornObj=traj[i, 3:7])

        for j in range(bolt.num_obs_joints):
            bolt.p.resetJointState(
                bodyUniqueId=bolt.robot_id,
                jointIndex=bolt.obs_joint_ids[j],
                targetValue=traj[i, j + 7],
                targetVelocity=0.0)

        angles.append(bolt.p.getEulerFromQuaternion(traj[i, 3:7]))
        rewards.append(bolt.reward_parts['base_stability_penalty'].get_reward())

        bolt.p.stepSimulation()

        contact_points = bolt.p.getContactPoints(bodyA=bolt.robot_id, bodyB=bolt.surface_id)
        for cp in contact_points:
            foot_locations.append(cp[6])
            timesteps.append(i)

        # time.sleep(0.001)

    timesteps = np.array(timesteps)
    foot_locations = np.array(foot_locations)
    print(foot_locations.shape)

    plt.scatter(foot_locations[:, 0], foot_locations[:, 1], s=1)
    # plt.show()

    # fig, axes = plt.subplots(3, 1)
    # for i in range(3):
    #     axes[i].plot(timesteps, foot_locations[:, i])

    r = 0.025
    clusters = get_clusters(foot_locations, r)
    print(clusters.shape)
    for c in clusters:
        plt.gca().add_artist(plt.Circle(c, r, color='r', fill=False))
    plt.axis('equal')
    plt.show()

    # angles = np.array(angles)
    # plt.plot(traj[:, 2])
    # plt.plot(angles[:, 0], label='angles[0]')
    # plt.plot(angles[:, 1], label='angles[1]')
    # plt.plot(angles[:, 2], label='angles[2]')
    # plt.legend()
    # plt.plot(rewards)
    # for i in range(7, 13):
    #     plt.plot(traj[:, i], label=str(i))
    # plt.legend()
    # print(np.min(traj[:, 2]), np.max(traj[:, 2]))
    # print(np.min(angles[:, 0]), np.max(angles[:, 0]))
    # print(np.min(angles[:, 1]), np.max(angles[:, 1]))
    # plt.show()

def test_imitation_length_termination(traj_file):
    bolt = Bolt(visualize=True,
                sim_timestep=0.001,
                cont_timestep_mult=8,
                controller_params={'type': 'torque'},
                termination_conf={'imitation_length_termination':{}},
                demo_traj_file=traj_file,
                initialization_conf={'on_demo_traj': True, 'use_demo_vel': True})

    while True:
        bolt._reset()
        done = False
        i = 0
        while not done:
            _, _, done, _ = bolt._step(np.zeros(6))
            i += 1
        print(bolt.demo_traj_start_timestep, i)

def roll_out_pd(traj_file, termination_conf={}, torque_file=None):
    traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                     + '/trajectories/'
    traj = np.loadtxt(traj_directory + traj_file)

    torque = None
    if torque_file is not None:
        torque = np.loadtxt(traj_directory + torque_file)

    bolt = Bolt(visualize=False,
                reward_specs={
                    'trajectory_imitation_reward': {'ratios_variant': 'pos_only', 'scaling_variant': 'less_severe'}},
                sim_timestep=0.001,
                cont_timestep_mult=1,
                controller_params={
                    'type': 'position_gain',
                    'variant': 'fixed',
                    'base_kp': [ 5.0000,  5.0000, 5.0000,  5.0000, 5.0000,  5.0000],
                    'base_kv': [ 0.1000,  0.1000, 0.1000,  0.1000, 0.1000,  0.1000]},
                termination_conf=termination_conf,
                demo_traj_file=traj_file,
                initialization_conf={'on_demo_traj': True, 'use_demo_vel': True})

    bolt._reset()
    bolt.p.resetBasePositionAndOrientation(
        bodyUniqueId=bolt.robot_id,
        posObj=traj[0, 0:3],
        ornObj=traj[0, 3:7])
    for i in range(bolt.num_obs_joints):
        bolt.p.resetJointState(
            bodyUniqueId=bolt.robot_id,
            jointIndex=bolt.obs_joint_ids[i],
            targetValue=traj[0, i + 7],
            targetVelocity=0.0)

    angles = []
    pos = []
    rewards = []
    for i in range(1, traj.shape[0]):
        des_pos = traj[i, 7:13]
        ff_torque = 0.0
        if torque is not None:
            ff_torque = torque[i]
        bolt.controller.act(des_pos, raw_des_pos_input=True, ff_torque=ff_torque, no_torque_clipping=True)
        rewards.append(bolt.reward_parts['trajectory_imitation_reward'].get_reward())

        base_pos, base_orient = bolt.p.getBasePositionAndOrientation(bolt.robot_id)
        base_ang = bolt.p.getEulerFromQuaternion(base_orient)
        angles.append(base_ang)
        pos.append(base_pos)

        bolt.reward_parts['trajectory_imitation_reward'].step()
        # time.sleep(0.01)
        # for termination_object in bolt.termination_dict.values():
        #     print(termination_object.done())

    pos = np.array(pos)
    angles = np.array(angles)
    # plt.plot(traj[:, 2])
    # plt.plot(pos[:, 2], label='pos[2]')
    # plt.plot(angles[:, 0], label='angles[0]')
    # plt.plot(angles[:, 1], label='angles[1]')
    # # # plt.plot(angles[:, 2], label='angles[2]')
    # plt.legend()
    plt.plot(rewards)
    plt.show()
    plt.plot(bolt.log.d['joint_pos_rew'], label='joint_pos_rew')
    plt.plot(bolt.log.d['joint_vel_rew'], label='joint_vel_rew')
    plt.plot(bolt.log.d['endeff_pos_rew'], label='endeff_pos_rew')
    plt.plot(bolt.log.d['base_pos_orient_rew'], label='base_pos_orient_rew')
    plt.plot(bolt.log.d['base_vel_angvel_rew'], label='base_vel_angvel_rew')
    plt.legend()
    # # for i in range(7, 13):
    # #     plt.plot(traj[:, i], label=str(i))
    # # plt.legend()
    # # print(np.min(traj[:, 2]), np.max(traj[:, 2]))
    # # print(np.min(angles[:, 0]), np.max(angles[:, 0]))
    # # print(np.min(angles[:, 1]), np.max(angles[:, 1]))
    plt.show()

    fig, axes = plt.subplots(7, 8)
    parts = ['joint_pos', 'joint_vel', 'endeff_pos', 'base_pos', 'base_orient', 'base_vel', 'base_angvel']
    exp_consts = [-5.0, -0.1, -40.0, -20.0, -10.0, -2.0, -0.2]
    exp_consts = [-1.0, -0.05, -5.0, -10.0, -5.0, -2.0, -0.2]
    for i in range(len(parts)):
        part = parts[i]
        actual = np.array(bolt.log.d[part])
        desired = np.array(bolt.log.d['des_' + part])
        for j in range(actual.shape[1]):
            axes[i, j].plot(actual[:, j], label=part)
            axes[i, j].plot(desired[:, j], label='des_' + part)
            axes[i, j].legend()
        diff = np.linalg.norm(desired - actual, axis=1)
        axes[i, 6].plot(diff)
        axes[i, 7].plot(np.exp(exp_consts[i] * diff))
    plt.show()


def roll_out_torque(traj_file, torque_file):
    traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                     + '/trajectories/'
    torque = np.loadtxt(traj_directory + torque_file)

    # fig, axes = plt.subplots(6, 1)
    # for i in range(6):
    #     axes[i].plot(torque[:, i])
    # plt.show()

    traj = np.loadtxt(traj_directory + traj_file)

    bolt = Bolt(visualize=True,
                reward_specs={
                    'base_stability_penalty': {
                        # 'mean': [0.3, 0.0, 0.0],
                        # 'sigma': [0.02, 0.0081, 0.0094]}})
                        'mean': [0.3, 0.0, 0.0],
                        'sigma': [0.05, 0.05, 0.05]}},
                sim_timestep=0.001,
                cont_timestep_mult=1,
                controller_params={
                    'type': 'torque'})

    bolt._reset()
    bolt.p.resetBasePositionAndOrientation(
        bodyUniqueId=bolt.robot_id,
        posObj=traj[0, 0:3],
        ornObj=traj[0, 3:7])
    for i in range(bolt.num_obs_joints):
        bolt.p.resetJointState(
            bodyUniqueId=bolt.robot_id,
            jointIndex=bolt.obs_joint_ids[i],
            targetValue=traj[0, i + 7],
            targetVelocity=0.0)

    angles = []
    rewards = []
    for i in range(1, traj.shape[0]):
        bolt.controller.act(torque[i], raw_torque_input=True)
        rewards.append(bolt.reward_parts['base_stability_penalty'].get_reward())
        # time.sleep(0.01)

    # angles = np.array(angles)
    # # plt.plot(traj[:, 2])
    # # plt.plot(angles[:, 0], label='angles[0]')
    # # plt.plot(angles[:, 1], label='angles[1]')
    # # plt.plot(angles[:, 2], label='angles[2]')
    # # plt.legend()
    plt.plot(rewards)
    # for i in range(7, 13):
    #     plt.plot(traj[:, i], label=str(i))
    # plt.legend()
    # print(np.min(traj[:, 2]), np.max(traj[:, 2]))
    # print(np.min(angles[:, 0]), np.max(angles[:, 0]))
    # print(np.min(angles[:, 1]), np.max(angles[:, 1]))
    plt.show()



def main():
    bolt = Bolt(visualize=True,
                enable_gravity=True,
                reward_specs={
                    'base_stability_penalty': {
                        'reference_range': [[0.28196055232750367, 0.3018948923871819],
                                            [-0.007891891191439305, 0.008046896075861398],
                                            [-0.009363218820449243, 0.005158881931906215]],
                        'reference_value': [0.3, 0.0, 0.0],
                        'k_inside': 1.0,
                        'k_outside': 100.0}},
                demo_traj_file='bolt_step_adjustment',
                early_termination_reward_threshold=-10.0)
    while True:
        bolt._reset()
        rewards = []
        for i in range(200):
            _, _, done, _ = bolt._step(np.zeros(6))
            time.sleep(0.008)
            rewards.append(bolt.reward_parts['base_stability_penalty'].get_reward())
            if done:
                print(i)
                break
        # plt.plot(rewards)
        # plt.show()

def find_endeff_link_ids():
    bolt = Bolt()
    bolt._reset()
    print(bolt.p.getNumJoints(bolt.robot_id))
    for i in range(8):
        print(i, bolt.get_link_z(i))

def initialization_design():
    bolt = Bolt(visualize=True, enable_gravity=False)
    bolt.p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=75.0, cameraPitch=-25.0, cameraTargetPosition=[0,0,1])
    bolt._reset()



    while True:
        random_joint_configuration = get_random_joint_configuration(bolt)
        set_joint_configuration(bolt, random_joint_configuration)
        for i in range(5000):
            bolt.p.stepSimulation()

if __name__ == '__main__':
    # main()
    # show_trajectory('bolt_step_adjustment')
    # roll_out_pd('bolt_step_adjustment')
    # get_foot_locations('bolt_forward_motion_qqdot')
    # show_trajectory('bolt_forward_motion_qqdot', loop_motion=True)
    # test_imitation_length_termination('bolt_forward_motion_qqdot')
    # TODO: Use this to match RL and traj. opt. simulations.
    # roll_out_pd('bolt_forward_motion_qqdot', termination_conf={'foot_placement_termination': {}},
    # torque_file='bolt_forward_motion_torque')
    # roll_out_torque('bolt_forward_motion_qqdot', 'bolt_forward_motion_torque')
    # DONE: Test foot placement early stopping with rolling out of torques from trajectory.
    # show_log_file('/Users/miroslav/work/experiment_results/181_biped_forward_motion/001/env_episodes/083300.json')
    # show_log_file('/Users/miroslav/work/experiment_results/182_biped_forward_motion/003/env_episodes/091442.json')
    # show_all_log_files('/Users/miroslav/work/experiment_results/182_biped_forward_motion/055/env_episodes/')
    # find_endeff_link_ids()
    # test_trajectory_imitation_reward('bolt_forward_motion_qqdot')
    # show_all_log_files('/Users/miroslav/work/experiment_results/186_biped_forward_motion_action_filtering/141/env_episodes/')
    show_all_log_files('/Users/miroslav/work/experiment_results/191_biped_forward_motion_ppo_base/040/env_episodes/')
    # show_all_experiments('/Users/miroslav/work/experiment_results/190_biped_forward_motion_single_init/')
    # initialization_design()
