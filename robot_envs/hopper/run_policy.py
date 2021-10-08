import sys
import numpy as np
import yaml
import json
import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian
import gym
import time

from baselines.ppo1_primitives.mlp_policy import MlpPolicy
from baselines.ddpg.models import Actor
from utils.experiment_set import Experiment


class ActorWrapper():

    def __init__(self, exp_folder, observation_shape, action_shape, remap_hopper_to_quad=False):
        tf.reset_default_graph()
        self.sess = tf.Session()

        conf_file = exp_folder + 'conf.yaml'
        with open(conf_file, 'r') as f:
            conf = yaml.load(f)

        self.obs = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='obs0')
        observation_range=(-5., 5.)
        normalized_obs0 = tf.clip_by_value(self.obs, observation_range[0], observation_range[1])
        self.actor = Actor(action_shape[0])(self.obs, reuse=False)

        # print('USING OBS CLIPPING, RANGE = [-5, 5]')
        # print('USING OBS CLIPPING, RANGE = [-5, 5]')
        # print('USING OBS CLIPPING, RANGE = [-5, 5]')
        # print('USING OBS CLIPPING, RANGE = [-5, 5]')
        # print('USING OBS CLIPPING, RANGE = [-5, 5]')

        #self.sess.run(tf.global_variables_initializer())

        self.saliency = jacobian(self.actor, self.obs)

        saver = tf.train.Saver()
        saver.restore(self.sess, exp_folder + 'latest_graph')

        self.remap_hopper_to_quad = remap_hopper_to_quad


    def act(self, obs):
        if self.remap_hopper_to_quad:
            a1 = []
            a2 = []
            for i in range(4):
                hopper_action = self.sess.run(self.actor, feed_dict={self.obs: [obs[i * 6:(i + 1) * 6]]})
                a1 += hopper_action[0][:2].tolist()
                a2 += hopper_action[0][2:].tolist()
            return np.array(a1 + a2)

        action, saliency = self.sess.run([self.actor, self.saliency], feed_dict={self.obs: [obs]})

        return action[0], saliency[0]


class PPOActorWrapper():

    @staticmethod
    def policy_fn(name, ob_space, ac_space, exp_config):
        pi_for_shared = MlpPolicy(name + '_for_reuse',
                                  ob_space=ob_space,
                                  ac_space=ac_space,
                                  num_hid_layers=2,
                                  network_structure=exp_config['network_params'],
                                  trainable=exp_config['task_indep_trainable'])
        for_reuse = pi_for_shared.get_variables_dict()
        pi = []
        pi.append(MlpPolicy(name + '0',
                            ob_space=ob_space,
                            ac_space=ac_space,
                            num_hid_layers=2,
                            network_structure=exp_config['network_params'],
                            reuse_from=for_reuse))
        return pi

    def __init__(self, ob_space, ac_space, exp_folder):
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()

        conf_file = exp_folder + 'conf.yaml'
        with open(conf_file, 'r') as f:
            conf = yaml.load(f)

        self.pi = PPOActorWrapper.policy_fn("pi", ob_space, ac_space, conf)[0]

        saver = tf.train.Saver()
        saver.restore(self.sess, exp_folder + 'latest_graph')
        print('SESSION:', tf.get_default_session())

    def act(self, obs):
        return self.pi.act(stochastic=False, ob=obs)[0], None


def run_episode_internal(env,
                         policy,
                         lock_control_on_impact=False,
                         preset_action_start_timestep=None,
                         preset_action_action=None,
                         without_stopping=False,
                         states=None):
    state = env.reset()
    episode_reward = 0.0
    done = False

    i = 0
    saliency_list = []

    control_locked = False

    while not done:
        i += 1

        if preset_action_start_timestep is not None:
            if i == preset_action_start_timestep:
                action = np.array(preset_action_action)
                control_locked = True

        if lock_control_on_impact and not control_locked:
            if env.env.get_endeff_force()[2] != 0.0:
                control_locked = True
                print('LOCK TIMESTEP: ', i)
                print('LAST ACTION: ', action)

        if not control_locked:
            if states is not None:
                action, saliency = policy.act(states[i - 1])
            else:
                action, saliency = policy.act(state)

        state, reward, done, _ = env.step(action)
        saliency_list.append(saliency)

        episode_reward += reward

        if without_stopping:
            done = False

        if states is not None:
            if i < states.shape[0]:
                done = False
            else:
                done = True

    return episode_reward, saliency_list

def run_episode(
    exp_folder,
    num_episodes,
    visualize=False,
    use_moving_surface=False,
    moving_surface_amplitude=0.1,
    initial_joint_pos=None,
    joint_damping=None,
    base_kv=None,
    base_kp=None,
    reward_specs=None,
    contact_stiffness=None,
    contact_damping=None,
    lateral_friction=None,
    ground_offset=None,
    lock_control_on_impact=False,
    preset_action_start_timestep=None,
    preset_action_action=None,
    cont_timestep_mult=None,
    without_stopping=False,
    states=None):

    exp_config = yaml.load(open(exp_folder + 'conf.yaml'))
    env_params = exp_config['env_params'][0]['env_specific_params']
    env_params['output_dir'] = ''
    env_params['visualize'] = visualize
    env_params['use_moving_surface'] = use_moving_surface
    env_params['moving_surface_amplitude'] = moving_surface_amplitude
    env_params['big_init'] = False
    # env_params['base_damping'] = 1.0

    if cont_timestep_mult is not None:
        env_params['cont_timestep_mult'] = cont_timestep_mult

    if joint_damping is not None:
        env_params['joint_damping'] = joint_damping
    if base_kv is not None:
        env_params['controller_params']['base_kv'] = base_kv
    if base_kp is not None:
        env_params['controller_params']['base_kp'] = base_kp
    if reward_specs is not None:
        env_params['reward_specs'] = reward_specs
    if contact_damping is not None:
        env_params['contact_damping'] = contact_damping
    if contact_stiffness is not None:
        env_params['contact_stiffness'] = contact_stiffness
    if lateral_friction is not None:
        env_params['lateral_friction'] = lateral_friction
    if initial_joint_pos is not None:
        env_params['initial_joint_pos'] = initial_joint_pos

    if not 'max_episode_steps' in exp_config['env_params'][0]:
        max_episode_duration = exp_config['env_params'][0]['max_episode_duration']
        sim_timestep = exp_config['env_params'][0]['env_specific_params']['sim_timestep']
        cont_timestep_mult = exp_config['env_params'][0]['env_specific_params']['cont_timestep_mult']

        max_episode_steps = int(max_episode_duration / (cont_timestep_mult * sim_timestep))
        exp_config['env_params'][0]['max_episode_steps'] = max_episode_steps

    env_params['exp_name'] = exp_folder + 'run_env'

    success = False
    env_num = 0

    while not success:
        try:
            env_id = 'RoboschoolReacher3Link-v' + str(env_num)

            gym.envs.register(
                id=env_id,
                entry_point=exp_config['env_params'][0]['entry_point'],
                max_episode_steps=exp_config['env_params'][0]['max_episode_steps'],
                kwargs=env_params
                )
            success = True
        except:
            env_num += 1

    env = gym.make(env_id)

    if exp_config['alg'] == 'ddpg':
        policy = ActorWrapper(exp_folder, env.observation_space.shape, env.action_space.shape)
    else:
        assert exp_config['alg'] == 'ppo'
        policy = PPOActorWrapper(env.observation_space, env.action_space, exp_folder)
        print(tf.get_default_session())

    if ground_offset is not None:
        env.env.p.resetBasePositionAndOrientation(env.env.surface_id, (0.0, 0.0, ground_offset), (0.0, 0.0, 0.0, 1.0))

    episode_rewards = []
    for i in range(num_episodes):
        ep_reward, saliency_list = run_episode_internal(env, policy, lock_control_on_impact,
            preset_action_start_timestep, preset_action_action, without_stopping=without_stopping, states=states)
        episode_rewards.append(ep_reward)
    return episode_rewards, env.env.log, saliency_list, env, policy


if __name__ == '__main__':
    exp_folder = sys.argv[1]
    run_episode(exp_folder, 2, True, False, without_stopping=True)
