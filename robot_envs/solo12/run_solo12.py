import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from robot_envs.solo12.solo12 import Solo12


def show_trajectory(traj_file, loop_motion=True):
    traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                     + '/trajectories/'

    pos_file = traj_directory + traj_file + '_positions.dat'
    vel_file = traj_directory + traj_file + '_velocities.dat'

    pos = np.loadtxt(pos_file)[:, 1:]
    vel = np.loadtxt(vel_file)[:, 1:]

    solo12 = Solo12(visualize=True)
    solo12.p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[-0.5,0,0])
    solo12._reset()

    def episode():
        for i in range(pos.shape[0]):
            solo12.p.resetBasePositionAndOrientation(
                bodyUniqueId=solo8.robot_id,
                posObj=pos[i, 0:3],
                ornObj=pos[i, 3:7])

            for j in range(solo8.num_obs_joints):
                solo12.p.resetJointState(
                    bodyUniqueId=solo8.robot_id,
                    jointIndex=solo8.obs_joint_ids[j],
                    targetValue=pos[i, j + 7],
                    targetVelocity=0.0)

            time.sleep(0.001)

    if loop_motion:
        while True:
            episode()
    else:
        episode()


def init_on_traj(traj_file, loop_motion=True):
    traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                     + '/trajectories/'

    pos_file = traj_directory + traj_file + '_positions.dat'
    vel_file = traj_directory + traj_file + '_velocities.dat'

    pos = np.loadtxt(pos_file)[:, 1:]
    vel = np.loadtxt(vel_file)[:, 1:]
    pos[:, 2] += 0.035

    solo12 = Solo12(visualize=True, enable_gravity=False)
    solo12.p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[-0.5,0,0])
    solo12._reset()

    def episode():
        timestep = np.random.randint(pos.shape[0])

        solo12.p.resetBasePositionAndOrientation(
            bodyUniqueId=solo12.robot_id,
            posObj=pos[timestep, 0:3],
            ornObj=pos[timestep, 3:7])

        # solo8.p.resetBaseVelocity(
        #     objectUniqueId=solo8.robot_id,
        #     linearVelocity=vel[timestep, 0:3],
        #     angularVelocity=vel[timestep, 3:6])

        for j in range(solo12.num_obs_joints):
            solo12.p.resetJointState(
                bodyUniqueId=solo12.robot_id,
                jointIndex=solo12.obs_joint_ids[j],
                targetValue=pos[timestep, j + 7],
                targetVelocity=0.0)

        solo8.p.stepSimulation()
        print(solo12.get_total_ground_force())

        for i in range(200):
            solo12.p.stepSimulation()
            time.sleep(0.001)

    if loop_motion:
        while True:
            episode()
    else:
        episode()


def roll_out_pd(traj_file, termination_conf={}):
    traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                     + '/trajectories/'

    pos_file = traj_directory + traj_file + '_positions.dat'
    vel_file = traj_directory + traj_file + '_velocities.dat'

    pos = np.loadtxt(pos_file)[:, 1:]
    vel = np.loadtxt(vel_file)[:, 1:]
    pos[:, 2] += 0.035

    solo12 = Solo12(visualize=False,
                  sim_timestep=0.001,
                  cont_timestep_mult=1,
                  controller_params={
                      'type': 'position_gain',
                      'variant': 'fixed',
                      'base_kp': [ 20.0000 ] * 12,
                      'base_kv': [ 0.2000 ] * 12})
    solo12.p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[0,0,0])


    while True:
        solo12._reset()
        solo12.p.resetBasePositionAndOrientation(
            bodyUniqueId=solo12.robot_id,
            posObj=pos[0, 0:3],
            ornObj=pos[0, 3:7])
        for i in range(solo12.num_obs_joints):
            solo12.p.resetJointState(
                bodyUniqueId=solo12.robot_id,
                jointIndex=solo12.obs_joint_ids[i],
                targetValue=pos[0, i + 7],
                targetVelocity=0.0)

        for i in range(1000):
            des_pos = pos[0, 7:15]
            solo12.controller.act(des_pos, raw_des_pos_input=True)
            time.sleep(0.001)

        for i in range(1, pos.shape[0]):
            des_pos = pos[i, 7:15]
            solo12.controller.act(des_pos, raw_des_pos_input=True)
            solo12.update_log()
            time.sleep(0.001)

        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 1.2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 1.2
        plt.rcParams['grid.linestyle'] = '-'
        plt.rcParams['grid.linewidth'] = 1
        sns.set_palette("Set1", n_colors=8)


        joint_pos = np.array(solo12.log.d['joint_pos'])
        total_ground_force = np.array(solo12.log.d['total_ground_force'])
        endeff_pos = np.array(solo12.log.d['endeff_pos'])

        plt.rcParams["figure.figsize"] = (15, 10)
        fig, axes = plt.subplots(8, 1)

        # for i in range(8):
        #     axes[i].plot(pos[:, i + 7], '--', label='desired_joint_pos[' + str(i) + ']')
        #     axes[i].plot(joint_pos[:, i], label='actual_joint_pos[' + str(i) + ']')
        #     axes[i].legend(frameon=1, framealpha=1.0, fancybox=False, loc='center right')

        for i in range(3):
            axes[i].plot(total_ground_force[:, i], label='total_ground_force[' + str(i) + ']')
            axes[i].legend(frameon=1, framealpha=1.0, fancybox=False, loc='center right')
            axes[i].set_ylim(-1, 25)

        for i in range(3):
            axes[i + 3].plot(endeff_pos[:, i], label='endeff_pos[' + str(i) + ']')
            axes[i + 3].legend(frameon=1, framealpha=1.0, fancybox=False, loc='center right')

        num_ax = 8
        for i in range(num_ax - 1):
            plt.setp(axes[i].get_xticklabels(), visible=False)
            axes[i].tick_params(axis='x', length=0.0)

        for i in range(num_ax):
            axes[i].grid(True)

        plt.tight_layout(pad=0.3, h_pad=None, w_pad=None, rect=None)
        plt.subplots_adjust(hspace=0.2)

        plt.show()

        # print(solo8.log.d.keys())


def test_trajectory_imitation_reward(traj_file):
    traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                     + '/trajectories/'

    pos_file = traj_directory + traj_file + '_positions.dat'
    vel_file = traj_directory + traj_file + '_velocities.dat'

    pos = np.loadtxt(pos_file)[:, 1:]
    vel = np.loadtxt(vel_file)[:, 1:]
    # pos[:, 2] += 0.035
    pos[:, 2] += 0.016832177805689

    solo12 = Solo12(visualize=True,
                  sim_timestep=0.001,
                  cont_timestep_mult=1,
                  reward_specs={
                      'trajectory_imitation_reward': {
                          'traj_file': traj_file
                      }
                  },
                  controller_params={
                      'type': 'position_gain',
                      'variant': 'fixed',
                      'base_kp': [ 20.0000 ] * 12,
                      'base_kv': [ 0.2000 ] * 12})

    solo12.p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[0,0,0])


    while True:
        solo12._reset()
        solo12.p.resetBasePositionAndOrientation(
            bodyUniqueId=solo12.robot_id,
            posObj=pos[0, 0:3],
            ornObj=pos[0, 3:7])
        for i in range(solo12.num_obs_joints):
            solo12.p.resetJointState(
                bodyUniqueId=solo12.robot_id,
                jointIndex=solo12.obs_joint_ids[i],
                targetValue=pos[0, i + 7],
                targetVelocity=0.0)

        # for i in range(1000):
        #     des_pos = pos[0, 7:15]
        #     solo8.controller.act(des_pos, raw_des_pos_input=True)
        #     time.sleep(0.001)

        # base_pos, base_orient = solo8.p.getBasePositionAndOrientation(solo8.robot_id)
        # print(pos[0, :3])
        # print(base_pos)
        # height_diff = base_pos[2] - pos[0, 2]
        # print(height_diff)
        # pos[:, 2] += height_diff

        for joint_id in solo12.cont_joint_ids:
            solo12.p.setJointMotorControl2(solo12.robot_id, joint_id,
                controlMode=solo12.p.VELOCITY_CONTROL, force=0)

        for i in range(1, pos.shape[0]):
            des_pos = pos[i, 7:15]
            solo12.controller.act(des_pos, raw_des_pos_input=True)
            solo12.update_log()
            solo12.reward_parts['trajectory_imitation_reward'].get_reward()
            time.sleep(0.001)

        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 1.2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 1.2
        plt.rcParams['grid.linestyle'] = '-'
        plt.rcParams['grid.linewidth'] = 1
        sns.set_palette("Set1", n_colors=8)

        joint_pos = np.array(solo12.log.d['joint_pos'])

        plt.rcParams["figure.figsize"] = (15, 10)

        num_ax = 4
        fig, axes = plt.subplots(num_ax, 1)

        # for i in range(8):
        #     axes[i].plot(pos[:, i + 7], '--', label='desired_joint_pos[' + str(i) + ']')
        #     axes[i].plot(joint_pos[:, i], label='actual_joint_pos[' + str(i) + ']')
        #     axes[i].legend(frameon=1, framealpha=1.0, fancybox=False, loc='center right')

        axes[0].plot(solo12.log.d['joint_pos_rew'], label='joint_pos_rew')
        axes[1].plot(solo12.log.d['joint_vel_rew'], label='joint_vel_rew')
        axes[2].plot(solo12.log.d['base_pos_orient_rew'], label='base_pos_orient_rew')
        axes[3].plot(solo12.log.d['base_vel_angvel_rew'], label='base_vel_angvel_rew')
        for i in range(4):
            axes[i].legend(frameon=1, framealpha=1.0, fancybox=False, loc='center right')

        for i in range(num_ax - 1):
            plt.setp(axes[i].get_xticklabels(), visible=False)
            axes[i].tick_params(axis='x', length=0.0)

        for i in range(num_ax):
            axes[i].grid(True)

        plt.tight_layout(pad=0.3, h_pad=None, w_pad=None, rect=None)
        plt.subplots_adjust(hspace=0.2)


        # plt.legend()

        plt.show()

        # print(solo8.log.d.keys())


def test_init(traj_file):
    traj_directory = str(os.path.dirname(os.path.abspath(__file__))) \
                     + '/trajectories/'

    pos_file = traj_directory + traj_file + '_positions.dat'
    vel_file = traj_directory + traj_file + '_velocities.dat'

    pos = np.loadtxt(pos_file)[:, 1:]
    vel = np.loadtxt(vel_file)[:, 1:]
    # pos[:, 2] += 0.035
    pos[:, 2] += 0.016832177805689

    solo12 = Solo12(visualize=True,
                  sim_timestep=0.001,
                  cont_timestep_mult=1,
                  reward_specs={
                      'trajectory_imitation_reward': {
                          'traj_file': traj_file
                      }
                  },
                  controller_params={
                      'type': 'position_gain',
                      'variant': 'fixed',
                      'base_kp': [ 20.0000 ] * 8,
                      'base_kv': [ 0.2000 ] * 8})

    solo12.p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[0,0,0])


    while True:
        solo12._reset()

        for i in range(1, pos.shape[0]):
            des_pos = pos[i, 7:19]
            solo12.controller.act(des_pos, raw_des_pos_input=True)
            solo12.update_log()
            solo12.reward_parts['trajectory_imitation_reward'].get_reward()
            time.sleep(0.001)

        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 1.2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 1.2
        plt.rcParams['grid.linestyle'] = '-'
        plt.rcParams['grid.linewidth'] = 1
        sns.set_palette("Set1", n_colors=8)

        joint_pos = np.array(solo12.log.d['joint_pos'])

        plt.rcParams["figure.figsize"] = (15, 10)

        num_ax = 4
        fig, axes = plt.subplots(num_ax, 1)

        # for i in range(8):
        #     axes[i].plot(pos[:, i + 7], '--', label='desired_joint_pos[' + str(i) + ']')
        #     axes[i].plot(joint_pos[:, i], label='actual_joint_pos[' + str(i) + ']')
        #     axes[i].legend(frameon=1, framealpha=1.0, fancybox=False, loc='center right')

        axes[0].plot(solo12.log.d['joint_pos_rew'], label='joint_pos_rew')
        axes[1].plot(solo12.log.d['joint_vel_rew'], label='joint_vel_rew')
        axes[2].plot(solo12.log.d['base_pos_orient_rew'], label='base_pos_orient_rew')
        axes[3].plot(solo12.log.d['base_vel_angvel_rew'], label='base_vel_angvel_rew')
        for i in range(4):
            axes[i].legend(frameon=1, framealpha=1.0, fancybox=False, loc='center right')

        for i in range(num_ax - 1):
            plt.setp(axes[i].get_xticklabels(), visible=False)
            axes[i].tick_params(axis='x', length=0.0)

        for i in range(num_ax):
            axes[i].grid(True)

        plt.tight_layout(pad=0.3, h_pad=None, w_pad=None, rect=None)
        plt.subplots_adjust(hspace=0.2)


        # plt.legend()

        plt.show()

        # print(solo8.log.d.keys())


def main():
    solo12 = Solo12(visualize=True)
    solo12._reset()
    while True:
        solo12.p.stepSimulation()
        time.sleep(0.001)

if __name__ == '__main__':
    # main()
    # show_trajectory('quadruped_generalized')
    # init_on_traj('quadruped_generalized')
    # roll_out_pd('quadruped_generalized')
    test_trajectory_imitation_reward('quadruped_generalized')

