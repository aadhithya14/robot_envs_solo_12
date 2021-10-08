import numpy as np


def get_random_joint_configuration(robot):
    joint_configuration = np.random.uniform(robot.joint_limits[:, 0], robot.joint_limits[:, 1])
    return joint_configuration

def set_joint_configuration(robot, joint_configuration):
    for i in range(robot.num_obs_joints):
        robot.p.resetJointState(
            bodyUniqueId=robot.robot_id,
            jointIndex=robot.obs_joint_ids[i],
            targetValue=joint_configuration[i],
            targetVelocity=0.0)
