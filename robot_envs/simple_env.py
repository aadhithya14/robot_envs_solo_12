import os
import numpy as np
import json
import pybullet_utils.bullet_client as bc
import pybullet
import pybullet_data
import time
import gym
import gym.spaces

from utils.rewards import DesiredForceIntensityReward, GoalPositionReward
from utils.data_logging import Log, ListOfLogs
from utils.my_math import polar_coord, ang_dist


def random_wall():
    angle = np.random.uniform(0.0, 2 * np.pi)
    base_angle = angle % (np.pi / 2.0)
    if base_angle > np.pi / 4.0:
        base_angle = np.pi / 2.0 - base_angle
    max_dist = 1.0 / np.cos(base_angle)
    dist = np.random.uniform(0.0, max_dist)
    return {'dist': dist, 'angle': angle}

def distance_to_wall(point, wall):
    r, theta = polar_coord(point)
    return wall['dist'] - r * np.cos(ang_dist(theta, wall['angle']))


class TrajReward():

    def __init__(self, traj_file, get_state, k_t, log=None, use_alt_vel_reward=False):
        self.traj = Traj(traj_file)
        self.get_state = get_state
        self.k_t = k_t
        self.log = log
        self.use_alt_vel_reward = use_alt_vel_reward

    def pos_reward(self, pos_diff):
        max_diff = 2 * np.sqrt(2)
        value = 1.0 - pos_diff / max_diff
        return np.exp(10.0 * value) / np.exp(10.0)

    def vel_reward(self, vel_diff, des_vel_norm):
        diff = vel_diff / des_vel_norm
        if self.log is not None:
            self.log.add('vel_diff', vel_diff)
            self.log.add('des_vel_norm', des_vel_norm)
        return 1.0 / np.exp(2.0 * diff)

    def error_reward(self, x, k=5.0):
        return 1.0 / np.exp(k * x)

    def alt_vel_reward(self, vel, des_vel_norm, des_vel_scale):
        tangential = np.dot(vel, des_vel_norm)
        normal = vel - tangential
        normal_reward = self.error_reward(normal)
        tangential_reward = self.error_reward(np.abs(des_vel_scale - tangential))
        if self.log is not None:
            self.log.add('normal_reward', normal_reward)
            self.log.add('tangential_reward', tangential_reward)
        return normal_reward + tangential_reward

    def get_reward(self):
        pos, vel = self.get_state()

        des_pos, des_vel = self.traj.closest_point(pos)

        pos_diff = np.linalg.norm(des_pos - pos)
        vel_diff = np.linalg.norm(des_vel - vel)
        des_vel_norm = np.linalg.norm(des_vel)
        des_vel_unit = des_vel / des_vel_norm

        pos_reward = self.pos_reward(pos_diff)
        if self.use_alt_vel_reward:
            vel_reward = self.alt_vel_reward(vel, des_vel_unit, des_vel_norm)
        else:
            vel_reward = self.vel_reward(vel_diff, des_vel_norm)

        if self.log is not None:
            self.log.add('pos_reward', pos_reward)
            self.log.add('vel_reward', vel_reward)
            self.log.add('des_pos', self.points[closest_index])
            self.log.add('des_vel', self.tan_velocity * self.vel_norms[closest_index])

        return self.k_t * (pos_reward + vel_reward)


    def is_done(self):
        return False


class Circle():

    def __init__(self, circle_params):
        self.center = circle_params['center']
        self.radius = circle_params['radius']
        self.tan_velocity = circle_params['tan_velocity']
        self.clockwise = circle_params['clockwise']

    def des_pos_vel(self, point):
        des_pos = self.center + (point - self.center) / np.linalg.norm(point - self.center) * self.radius
        v = (des_pos - self.center) / np.linalg.norm(des_pos - self.center)
        if self.clockwise:
            v = np.array([v[1], -v[0]])
        else:
            v = np.array([-v[1], v[0]])
        des_vel = self.tan_velocity * v
        return des_pos, des_vel


class CircleReward():

    def __init__(self, circle, get_state, k_t, log=None, use_alt_vel_reward=False):
        self.circle = circle
        self.get_state = get_state
        self.k_t = k_t
        self.log = log
        self.use_alt_vel_reward = use_alt_vel_reward

    def pos_reward(self, pos_diff):
        max_diff = 2 * np.sqrt(2)
        value = 1.0 - pos_diff / max_diff
        return np.exp(10.0 * value) / np.exp(10.0)

    def vel_reward(self, vel_diff, des_vel_norm):
        diff = vel_diff / des_vel_norm
        if self.log is not None:
            self.log.add('vel_diff', vel_diff.tolist())
            self.log.add('des_vel_norm', des_vel_norm)
        return 1.0 / np.exp(2.0 * diff)

    def error_reward(self, x, k=5.0):
        return 1.0 / np.exp(k * x)

    def alt_vel_reward(self, vel, des_vel_N, des_vel_S):
        vel_tan = np.dot(vel, des_vel_N) * des_vel_N 
        vel_norm = vel - vel_tan
        #print(vel_norm + vel_tan, vel)
        normal_reward = self.error_reward(np.linalg.norm(vel_norm))
        #print(des_vel_scale, tangential, np.abs(des_vel_scale - tangential))
        vel_tan_S = np.linalg.norm(vel_tan)
        tangential_reward = self.error_reward(np.abs(des_vel_S - vel_tan_S))
        if self.log is not None:
            self.log.add('normal_reward', normal_reward)
            self.log.add('tangential_reward', tangential_reward)
        #print('rewards:', normal_reward, tangential_reward)
        return normal_reward + tangential_reward

    def get_reward(self):
        pos, vel = self.get_state()

        des_pos, des_vel = self.circle.des_pos_vel(pos)

        pos_diff = np.linalg.norm(des_pos - pos)
        vel_diff = np.linalg.norm(des_vel - vel)
        des_vel_norm = np.linalg.norm(des_vel)
        des_vel_unit = des_vel / des_vel_norm

        pos_reward = self.pos_reward(pos_diff)
        if self.use_alt_vel_reward:
            vel_reward = self.alt_vel_reward(vel, des_vel_unit, des_vel_norm)
        else:
            vel_reward = self.vel_reward(vel_diff, des_vel_norm)

        if self.log is not None:
            self.log.add('pos_reward', pos_reward)
            self.log.add('vel_reward', vel_reward)
            self.log.add('des_pos', des_pos.tolist())
            self.log.add('des_vel', des_vel.tolist())

        return self.k_t * (pos_reward + vel_reward)

    def is_done(self):
        return False


class SimpleEnv(gym.Env):

    def __init__(self, reward_specs={}, full_log=False, log_file=None, exp_name='', output_dir='', wall_params=None, pushee_params=None, visualize=False):
        self.full_log = full_log
        self.visualize = visualize
        self.wall_params = wall_params
        self.pushee_params = pushee_params

        if self.visualize:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.loadSDF("stadium.sdf")
        self.p.setGravity(0,0,-10)
        objects = self.p.loadMJCF(os.path.join(os.path.dirname(__file__), "mjcf/sphere.xml"))
        self.sphere = objects[0]
        self.p.resetBasePositionAndOrientation(self.sphere,[0,0,0.01],[0,0,0,1])
        self.p.changeDynamics(self.sphere,-1,linearDamping=0.9)
        self.p.changeVisualShape(self.sphere,-1,rgbaColor=[1,0,0,1])
        self.forward = 0
        self.turn = 0


        forwardVec = [2,0,0]
        self.cameraDistance = 1
        self.cameraYaw = 35
        self.cameraPitch = -35

        if self.wall_params is not None:
            self.wall = self.p.loadMJCF(os.path.join(os.path.dirname(__file__), 'mjcf/wall.xml'))[0]
            quat = self.p.getQuaternionFromEuler([0.0, 0.0, self.wall_params['angle'] - np.pi / 2.0])
            x = self.wall_params['dist'] * np.cos(self.wall_params['angle'])
            y = self.wall_params['dist'] * np.sin(self.wall_params['angle'])
            self.p.resetBasePositionAndOrientation(self.wall, (x, y, 0.01), quat)
        if self.pushee_params is not None:
            if self.pushee_params['fixed']:
                self.p.loadMJCF(os.path.join(os.path.dirname(__file__), 'mjcf/pushee_fixed.xml'))
            else:
                self.p.loadMJCF(os.path.join(os.path.dirname(__file__), 'mjcf/pushee.xml'))

        action_dim = 2
        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        obs_dim = len(self.get_state())
        high = np.inf*np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

        if self.full_log:
            self.log = ListOfLogs(exp_name + '_episodes', separate_files=True)
        else:
            if log_file is not None:
                self.log = Log(log_file)
            else:
                self.log = Log(exp_name + '_episodes')

        self.init_reward(reward_specs)

    def move_wall(self, wall):
        quat = self.p.getQuaternionFromEuler([0.0, 0.0, wall['angle'] - np.pi / 2.0])
        x = wall['dist'] * np.cos(wall['angle'])
        y = wall['dist'] * np.sin(wall['angle'])
        self.p.resetBasePositionAndOrientation(self.wall, (x, y, 0.0), quat)


    def init_reward(self, rewards_config):
        self.reward_parts = {}
        for reward_type, reward_spec in rewards_config.items():
            if reward_type == 'wall_force':
                self.reward_parts[reward_type] = DesiredForceIntensityReward(self.get_wall_normal_force, reward_spec['k_f'], reward_spec['goal_force'])
            elif reward_type == 'position':
                self.reward_parts[reward_type] = GoalPositionReward(self.get_pushee_pos, reward_spec['k_p'], reward_spec['goal_pos'])
            elif reward_type == 'trajectory':
                use_alt_vel_reward = False
                if 'use_alt_vel_reward' in reward_spec:
                    use_alt_vel_reward = reward_spec['use_alt_vel_reward']
                self.reward_parts[reward_type] = TrajReward(reward_spec['traj_file'], reward_spec['tan_velocity'], self.get_pusher_state, reward_spec['k_t'], self.log, use_alt_vel_reward)
            elif reward_type == 'circle':
                circle = Circle(reward_spec)
                use_alt_vel_reward = False
                if 'use_alt_vel_reward' in reward_spec:
                    use_alt_vel_reward = reward_spec['use_alt_vel_reward']
                self.reward_parts[reward_type] = CircleReward(circle, self.get_pusher_state, reward_spec['k_t'], self.log, use_alt_vel_reward)
            else:
                assert False, 'Unknown reward type: ' + str(reward_type)

    def get_pushee_pos(self):
        spherePos, orn = self.p.getBasePositionAndOrientation(self.sphere)
        return np.array(spherePos[:2])

    def get_pusher_state(self):
        spherePos, orn = self.p.getBasePositionAndOrientation(self.sphere)
        linearVelocity, angularVelocity = self.p.getBaseVelocity(self.sphere)
        return np.array(spherePos)[:2], np.array(linearVelocity)[:2]

    def get_state(self):
        spherePos, orn = self.p.getBasePositionAndOrientation(self.sphere)
        linearVelocity, angularVelocity = self.p.getBaseVelocity(self.sphere)
        return np.array(list(spherePos)[:2] + list(linearVelocity)[:2])

    def get_force(self):
        force = np.zeros(2)
        contacts = self.p.getContactPoints(bodyA=self.sphere)
        for contact in contacts:
            contact_normal = np.array(contact[7][:2])
            normal_force = contact[9]
            force += normal_force * contact_normal
        return force

    def update_log(self):
        spherePos, orn = self.p.getBasePositionAndOrientation(self.sphere)
        linearVelocity, angularVelocity = self.p.getBaseVelocity(self.sphere)
        
        self.log.add('pusher_pos', list(spherePos)[:2])
        self.log.add('pusher_vel', list(linearVelocity)[:2])

        for (reward_type, reward_part) in self.reward_parts.items():
            self.log.add(reward_type, reward_part.get_reward())

        self.log.add('latest_action', self.latest_action.tolist())

    def _reset(self):
        if self.full_log:
            self.log.finish_log()
        else:
            self.log.save()
            self.log.clear()

        point = np.random.uniform(-1.0, 1.0, 2)
        if self.wall_params is not None:
            while distance_to_wall(point, self.wall_params) < 0.06:
                point = np.random.uniform(-1.0, 1.0, 2)
            r, theta = polar_coord(point)
            #print(point)
            #print(r, theta)
            #print(self.wall_params)
            #print(ang_dist(theta, self.wall_params['angle']))
            #return wall['dist'] - r * np.cos(ang_dist(theta, wall['angle']))
            #print(distance_to_wall(point, self.wall_params))

        self.p.resetBasePositionAndOrientation(self.sphere,[point[0],point[1],0.01],[0,0,0,1])
        
        spherePos, orn = self.p.getBasePositionAndOrientation(self.sphere)
        linearVelocity, angularVelocity = self.p.getBaseVelocity(self.sphere)
        self.log.add('pusher_pos', list(spherePos)[:2])
        self.log.add('pusher_vel', list(linearVelocity)[:2])

        return self.get_state()

    def _step(self, action=None):

        #print(self.get_force())
        if action is None:

            spherePos, orn = self.p.getBasePositionAndOrientation(self.sphere)
            
            cameraTargetPosition = spherePos
            self.p.resetDebugVisualizerCamera(self.cameraDistance,self.cameraYaw,self.cameraPitch,cameraTargetPosition)
            camInfo = self.p.getDebugVisualizerCamera()
            camForward = camInfo[5]
            
            
            keys = self.p.getKeyboardEvents()
            for k,v in keys.items():
                
                if (k == self.p.B3G_RIGHT_ARROW and (v&self.p.KEY_WAS_TRIGGERED)):
                    self.turn = -0.5
                if (k == self.p.B3G_RIGHT_ARROW and (v&self.p.KEY_WAS_RELEASED)):
                    self.turn = 0
                if (k == self.p.B3G_LEFT_ARROW and (v&self.p.KEY_WAS_TRIGGERED)):
                    self.turn = 0.5
                if (k == self.p.B3G_LEFT_ARROW and (v&self.p.KEY_WAS_RELEASED)):
                    self.turn = 0
                
                if (k == self.p.B3G_UP_ARROW and (v&self.p.KEY_WAS_TRIGGERED)):
                    self.forward=1
                if (k == self.p.B3G_UP_ARROW and (v&self.p.KEY_WAS_RELEASED)):
                    self.forward=0
                if (k == self.p.B3G_DOWN_ARROW and (v&self.p.KEY_WAS_TRIGGERED)):
                    self.forward=-1
                if (k == self.p.B3G_DOWN_ARROW and (v&self.p.KEY_WAS_RELEASED)):
                    self.forward=0
            
            force  = [self.forward*camForward[0],self.forward*camForward[1],0]
            self.cameraYaw = self.cameraYaw+self.turn
            
            if (self.forward):
                self.p.applyExternalForce(self.sphere,-1, force , spherePos, flags = self.p.WORLD_FRAME )
                
            self.p.stepSimulation()
            if self.visualize:
                time.sleep(1./240.)

        else:
            self.latest_action = action

            spherePos, orn = self.p.getBasePositionAndOrientation(self.sphere)
            force = [action[0], action[1], 0.0]
            self.p.applyExternalForce(self.sphere,-1, force , spherePos, flags = self.p.WORLD_FRAME )
                
            self.p.stepSimulation()
            if self.visualize:
                time.sleep(1./240.)

            self.update_log()

            state = self.get_state()
            reward = sum([reward_part.get_reward() for reward_part in self.reward_parts.values()])
            done = False

            return state, reward, done, {}

    def _render(self, mode, close):
        pass

    def _seed(self, seed):
        pass


def speed_test(num_steps=1000):
    simple_env = SimpleEnv(reward_specs={'trajectory': {'traj_file': '/home/miroslav/Desktop/simple_env_test/trajectory.json', 'tan_velocity': 1.0, 'k_t': 1.0}})
    simple_env._reset()
    start = time.time()
    for i in range(num_steps):
        action = np.random.uniform(low=-1.0, high=1.0, size=2)
        simple_env._step(action)
    end = time.time()
    print()
    print('Time for', num_steps, 'steps:', end - start)

def main():
    wall_params = random_wall()
    simple_env = SimpleEnv(wall_params=wall_params, pushee_params={'fixed': True}, visualize=True)
    #while True:
    for i in range(1):
        simple_env._reset()
        #wall = random_wall()
        #simple_env.move_wall(wall)
        #for j in range(1000):
        while True:
            action = np.random.uniform(low=-1.0, high=1.0, size=2)
            action = np.array([0.0, 0.0])
            simple_env._step(None)


if __name__ == "__main__":
    main()
    #speed_test()
    
