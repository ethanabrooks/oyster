import gym

from rlkit.envs import register_env
from trainer_env.main import Trainer


@register_env('l2b')
class L2bEnv(gym.Env, Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iterator = None
        self.t = None

    def step(self, action):
        s = self.iterator.send(action )
        self.t += 1
        t = self.t == self.max_time_steps
        r = self.eval_policy() if t else 0
        return s, r, t, {}

    def reset(self):
        self.iterator = self.generator()
        self.t = 0
        return next(self.iterator)

    def render(self, mode="human"):
        pass
