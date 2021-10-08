from gym import Env

class BaseEnv(Env):

    def init_reward(self, reward_specs):
        raise NotImplementedError

    def step_internal(self, action):
        raise NotImplementedError

    def reset_internal(self, action):
        raise NotImplementedError

    def __init__(self, reward_specs):
        self.reward_parts = self.init_reward(reward_specs)

    def _reset(self):
        return self.reset_internal()

    def _step(self, action):
        state = self.step_internal(action)
        reward = sum([reward_part.get_reward() for reward_part in self.reward_parts.values()])
        return state, reward, False, {}
