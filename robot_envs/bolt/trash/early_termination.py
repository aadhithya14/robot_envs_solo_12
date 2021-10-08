class EarlyTermination():

    def __init__(self, reward_threshold, termination_penalty=True, max_steps=None, gamma=0.99):
        self.reward_threshold = reward_threshold
        self.termination_penalty = termination_penalty
        if termination_penalty:
            assert max_steps is not None
            self.max_steps = max_steps
            self.gamma = gamma

    def reset(self):
        self.episode_steps = 0

    def step(self, reward, done):
        self.episode_steps += 1
        if reward < self.reward_threshold:
            done = True
            if self.termination_penalty:
                # We give penalty equal to the discounted cummulative reward
                # that would be received if the episode recieved the same minimal
                # reward for all the remaining timesteps.
                reward = 0.0
                for i in range(self.max_steps - self.episode_steps):
                    curr_exp *= self.gamma
                    reward += curr_exp * self.reward_threshold
            else:
                reward = self.reward_threshold
        return reward, done
