import numpy as np
from random import random

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        self.direction_to_change = None
        self.throw = None
        self.target_player_num = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
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
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3


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
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        self.move_agents()
        self.throw_frisbee()

    def move_agents(self):
        for agent in self.agents:
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
        for agent in self.agents:
            if agent.has_frisbee:
                # probability of catching. target agent must be in bounds
                if random() < self.catch_frisbee_probability and self.agents[agent.target_player_num].x > 0. and self.agents[agent.target_player_num].x < self.field_width and self.agents[agent.target_player_num].y > 0 and self.agents[agent.target_player_num].y < self.field_length:
                        self.agents[agent.target_player_num].has_frisbee = True
                else:
                    self.turnover = True
                agent.has_frisbee = False
                return
