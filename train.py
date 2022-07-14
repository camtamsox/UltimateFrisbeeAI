
import gym
from gym import spaces
import numpy as np

from shapely.geometry.polygon import Polygon



class UltimateFrisbee(gym.Env):

    def __init__(self, num_agents=14, characteristics_per_agent=4): # characteristics_per_agent ideas: team, x, y, has frisbee, is throwing, speed, explosiveness, height, throwing ability, disc reading, tiredness...
        self.characteristics_per_agent = characteristics_per_agent
        self.num_agents = num_agents
        self.reward = np.zeros(self.num_agents)
        self.prev_reward = np.zeros(self.num_agents)
        self.field_width = 40.
        self.field_length = 120.

        self.observation_lower_bound = np.array([0, 0., 0., 0]) # team, x, y, has frisbee
        self.observation_upper_bound = np.array([1, self.field_width, self.field_length, 1])
        self.observation_dtype = np.array([np.uint8, np.float32, np.float32, np.uint8])
        
        self.action_space = spaces.Discrete(8) # Box for more realistic changes of direction (would have to incorporate tiredness)
        self.observation_space = spaces.Box(low=self.observation_lower_bound, high=self.observation_upper_bound, shape=(num_agents * self.characteristics_per_agent), dtype=self.observation_dtype)



    def step(self, action):
        step_reward = np.zeros(self.num_agents)
        done = False
        # penalize offense, reward defense
        for i in range(self.num_agents/2):
            step_reward[i] -= 1.
        for i in range(self.num_agents/2., self.num_agents):
            step_reward[i] += 1.
        
        # do actions. Penalize if player out of bounds and return to field

        # check if turnover

        # check if frisbee in endzone




        return self.state, step_reward, done, {}

    def reset(self, action):
        self.reward = np.zeros(self.num_agents)
        self.prev_reward = np.zeros(self.num_agents)

        # reset positions of players
        for i in range(1, self.num_agents/2 * self.characteristics_per_agent, 4): # offense
            # evenly separate so that no player is on sideline
            self.state[i] = self.field_width/(self.num_agents+2) * (i+1) # x
            self.state[i+1] = 25. # y

        for i in range(1 + self.num_agents/2 * self.characteristics_per_agent, self.num_agents * self.characteristics_per_agent, 4): # defense
            # evenly separate so that no player is on sideline
            self.state[i] = self.field_width/(self.num_agents+2) * (i+1) # x
            self.state[i+1] = 95. # y

        # give defense frisbee for pull. make sure to reset all people with frisbee

        return self.state

    def render(self): # pyglet??

        return

    def close(self): # close render

        return