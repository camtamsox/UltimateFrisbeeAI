
import gym
from gym import spaces
import numpy as np
import random
from shapely.geometry.polygon import Polygon


from tf_agents.environments import gym_wrapper, tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
import tf_agents.trajectories.time_step as ts
from tf_agents.utils import common
from tf_agents.policies.policy_saver import PolicySaver


def move_action(x, y, direction, player_step_size):
    if direction == 0: # up
        y+=player_step_size

    if direction == 1: # up right
        y+=player_step_size
        x+=player_step_size

    if direction == 2: # right
        x+=player_step_size

    if direction == 3: # down right
        x+=player_step_size
        y-=player_step_size

    if direction == 4: # down
        y-=player_step_size

    if direction == 5: # down left
        x-=player_step_size
        y-=player_step_size

    if direction == 6: # left
        x-=player_step_size

    if direction == 7: # left up
        x-=player_step_size
        y+=player_step_size
    return x, y

def throw_frisbee_action(state, player_num, target_player_num, num_player_observations):
    state[player_num * num_player_observations + 3] = 0 # player no longer has frisbee

    catch_frisbee_probability = 0.9
    if random.random() < catch_frisbee_probability:
        state[target_player_num * num_player_observations + 3] = 1 # target player caught frisbee
        turnover = False
    else:
        turnover = True

    return state, turnover

class UltimateFrisbee(gym.Env):

    def __init__(self, num_agents=14, characteristics_per_agent=4): # characteristics_per_agent ideas: team, x, y, has frisbee, is throwing, speed, explosiveness, height, throwing ability, disc reading, tiredness...
        self.characteristics_per_agent = characteristics_per_agent
        self.num_agents = num_agents
        self.reward = np.zeros(self.num_agents)
        self.prev_reward = np.zeros(self.num_agents)
        self.field_width = 40.
        self.endzone_length = 25.
        self.field_length = self.endzone_length * 2 + 70
        self.player_step_size = 1.
        
        self.num_player_actions = 3
        self.action_lower_bound = np.tile(np.array([0, 0, 0]), self.num_agents)     # [direction to change, throw frisbee?, who to throw to]
        self.action_upper_bound = np.tile(np.array([7, 1, 1]), self.num_agents)    
        
        self.num_player_observations = 4
        self.observation_lower_bound = np.tile(np.array([0, 0., 0., 0]), self.num_agents) # team, x, y, has frisbee
        self.observation_upper_bound = np.tile(np.array([1, self.field_width, self.field_length, 1]))
        self.observation_dtype = np.array([np.uint8, np.float32, np.float32, np.uint8])

        self.action_space = spaces.Box(low=self.action_lower_bound, high=self.action_upper_bound, shape=(self.num_agents * 3), dtype=np.uint8) 
        self.observation_space = spaces.Box(low=self.observation_lower_bound, high=self.observation_upper_bound, shape=(num_agents * self.characteristics_per_agent), dtype=self.observation_dtype)



    def step(self, action):
        step_reward = np.zeros(self.num_agents)
        done = False
        
        if action is not None:
            # penalize offense, reward defense
            for i in range(self.num_agents/2):
                step_reward[i] -= 1.
            for i in range(self.num_agents/2, self.num_agents):
                step_reward[i] += 1.

            # Do actions
            for player_num in self.num_agents:
                # change x, y
                old_x = self.state[player_num * self.num_player_observations + 1]
                old_y = self.state[player_num * self.num_player_observations + 2]
                direction = action[player_num * self.num_player_actions]
                x, y = move_action(old_x, old_y, direction, self.player_step_size)
                # check if out of bounds
                if x < 0 or x > self.field_width:
                    step_reward[player_num] -= 5
                    x = old_x
                if y < 0 or y > self.field_length:
                    step_reward[player_num] -= 5
                    y = old_y

                self.state[player_num * self.num_player_observations + 1] = x
                self.state[player_num * self.num_player_observations + 2] = y               


                # throw frisbee
                has_frisbee = self.state[player_num * self.num_player_observations + 3]
                throw_frisbee = action[player_num * self.num_player_actions + 1]
                if has_frisbee == 1 and throw_frisbee == 1:

                    target_player_num = action[player_num * self.num_player_actions + 2]
                    self.state, turnover = throw_frisbee_action(self.state, player_num, target_player_num, self.num_player_observations)

                    if turnover:
                        # penalize offense, reward defense
                        for i in range(self.num_agents/2):
                            step_reward[i] -= 1000.
                        for i in range(self.num_agents/2, self.num_agents):
                            step_reward[i] += 1000.
                        done = True

                        return self.state, step_reward, done, {}
                    else:
                        # check if frisbee is in endzone
                        target_player_y = self.state[target_player_num * self.num_player_observations + 2]
                        if target_player_y > (self.field_length - self.endzone_length):
                            # reward offense, penalize defense
                            for i in range(self.num_agents/2):
                                step_reward[i] += 1000.
                            for i in range(self.num_agents/2, self.num_agents):
                                step_reward[i] -= 1000.
                            done = True
                            return self.state, step_reward, done, {}

        return self.state, step_reward, done, {}

    def reset(self, action):
        self.reward = np.zeros(self.num_agents)
        self.prev_reward = np.zeros(self.num_agents)

        # reset positions of players
        for i in range(1, self.num_agents/2 * self.characteristics_per_agent, 4): # offense
            # evenly separate so that no player is on sideline
            self.state[i] = self.field_width/(self.num_agents+2) * (i+1) # x
            self.state[i+1] = self.endzone_length # y

        for i in range(1 + self.num_agents/2 * self.characteristics_per_agent, self.num_agents * self.characteristics_per_agent, 4): # defense
            # evenly separate so that no player is on sideline
            self.state[i] = self.field_width/(self.num_agents+2) * (i+1) # x
            self.state[i+1] = self.field_length - self.endzone_length # y

        # randomly choose who gets frisbee
        for player_num in range(self.num_agents/2): 
            self.state[player_num * self.num_player_observations + 3] = 0
        player_with_frisbee = random.randint(0,self.num_agents-1)
        self.state[player_with_frisbee * self.num_player_observations + 3] = 1


        return self.state

    # def visualize(self):


# For ML part, I was thinking of doing something like this: https://github.com/rmsander/marl_ppo/blob/main/ppo/ppo_marl.py