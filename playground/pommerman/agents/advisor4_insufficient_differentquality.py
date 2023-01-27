'''An agent that preforms a random action each step'''
from . import BaseAgent


class Advisor4_insufficient_differentquality(BaseAgent):

    def act(self, obs, action_space = 6):
        return action_space.sample()
