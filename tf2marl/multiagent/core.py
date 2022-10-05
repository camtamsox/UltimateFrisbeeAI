import numpy as np
from random import random


# state of agents 
class AgentState():
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        # TODO

# action of the agent
class Action(object):
    def __init__(self):
        self.direction_to_change = None
        self.target_player_num = None


# properties of agent entities
class Agent():
    def __init__(self):

        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

        self.name = None
        self.x = None
        self.y = None
        self.has_frisbee = None
        self.is_offense = None
        self.action = None        

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.prev_agent_num = None


    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):

        self.move_agents()
        self.throw_frisbee()
        self.check_in_bounds()

    def move_agents(self):
        for agent in self.agents:
            if agent.has_frisbee == False:
                # do nothing if agent.direction_to_change == 0
                if agent.direction_to_change == 1: # up
                    agent.y += self.player_step_size

                if agent.direction_to_change == 2: # up right
                    agent.y += self.player_step_size
                    agent.x += self.player_step_size

                if agent.direction_to_change == 3: # right
                    agent.x += self.player_step_size

                if agent.direction_to_change == 4: # down right
                    agent.x += self.player_step_size
                    agent.y -= self.player_step_size

                if agent.direction_to_change == 5: # down
                    agent.y -= self.player_step_size

                if agent.direction_to_change == 6: # down left
                    agent.x -= self.player_step_size
                    agent.y -= self.player_step_size

                if agent.direction_to_change == 7: # left
                    agent.x -= self.player_step_size

                if agent.direction_to_change == 8: # left up
                    agent.x -= self.player_step_size
                    agent.y += self.player_step_size

    def throw_frisbee(self):
        agent_num = 0
        for agent in self.agents:
            if agent.has_frisbee:
                # agent must throw frisbee
                if self.prev_agent_num == agent_num:
                    self.turnover = True
                    agent.has_frisbee = False
                else:
                    self.prev_agent_num = agent_num
                    
                # set distance from endzone
                self.offense_distance_from_endzone = (self.field_length - self.endzone_length) - agent.y

                # throw frisbee
                if random() < self.catch_frisbee_probability and agent.target_player_num is not agent_num:
                    agent.has_frisbee = False
                    self.agents[agent.target_player_num].has_frisbee = True
                elif agent.target_player_num != agent_num:
                    self.turnover = True
                    agent.has_frisbee = False
                return
            agent_num += 1
        for _ in range(100):
            print('frisbee not in state')

    def check_in_bounds(self):
        for agent in self.agents:
            if agent.has_frisbee and (agent.x < 0. or agent.x > self.field_width or agent.y < 0. or agent.y > self.field_length):
                self.turnover = True
                agent.has_frisbee = False
                return

