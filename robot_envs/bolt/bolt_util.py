import numpy as np


def get_foot_locations(robot, traj):
    assert False
    robot._reset()
    angles = []
    rewards = []
    foot_locations = []
    timesteps = []
    for i in range(traj.shape[0]):
        robot.p.resetBasePositionAndOrientation(
            bodyUniqueId=robot.robot_id,
            posObj=traj[i, 0:3],
            ornObj=traj[i, 3:7])
        for j in range(robot.num_obs_joints):
            robot.p.resetJointState(
                bodyUniqueId=robot.robot_id,
                jointIndex=robot.obs_joint_ids[j],
                targetValue=traj[i, j + 7],
                targetVelocity=0.0)

        robot.p.stepSimulation()

        contact_points = robot.p.getContactPoints(bodyA=robot.robot_id, bodyB=robot.surface_id)
        for cp in contact_points:
            foot_locations.append(cp[6])

    foot_locations = np.array(foot_locations)
    return foot_locations

def get_foot_location_clusters(foot_locations, circle_size):
    clusters = []
    for fl in foot_locations:
        found = False
        for c in clusters:
            if not found:
                cluster_center = np.mean(np.array(c), axis=0)

                dist = np.linalg.norm(cluster_center - fl)

                if dist < circle_size:
                    c.append(fl.tolist())
                    found = True
        if not found:
            clusters.append([fl.tolist()])
    r = [np.mean(np.array(c), axis=0) for c in clusters]
    return np.array(r)
