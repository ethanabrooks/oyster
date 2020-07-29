import gym

from rlkit.envs import register_env
from trainer_env.main import Trainer


@register_env("l2b")
class L2bEnv(gym.Env, Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, use_tune=False)
        self.iterator = None
        self.t = None
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        s = self.iterator.send(action)
        self.t += 1
        t = self.t == self.max_time_steps
        r = self.eval_policy() if t else 0
        return s, r, t, {}

    def reset(self):
        self.iterator = self.generator()
        self.t = 0
        return next(self.iterator)

    def reset_task(self, idx):
        raise NotImplementedError

    def get_all_task_idx(self):
        raise NotImplementedError

    def render(self, mode="human"):
        pass
