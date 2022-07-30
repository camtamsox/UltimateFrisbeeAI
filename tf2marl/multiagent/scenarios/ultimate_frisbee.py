import numpy as np
from tf2marl.multiagent.core import World, Agent, Landmark
from tf2marl.multiagent.scenario import BaseScenario
from random import randint


class Scenario(BaseScenario):
    
    def make_world(self):
        world = World()
        # set any world properties first
        num_agents = 14
        world.num_agents = num_agents
        agents_per_team = 7
        world.agents_per_team = agents_per_team

        world.characteristics_per_agent = 4
        world.field_width = 37.
        world.endzone_length = 18.
        world.field_length = world.endzone_length * 2 + 64
        world.player_step_size = 1.
        world.done = False
        world.turnover = False
        world.catch_frisbee_probability = 0.9

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.is_offense = True if i < agents_per_team else False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # evenly separate along front of endzone
        world.done = False
        world.turnover = False
        for i in range(world.num_agents):
            if world.agents[i].is_offense:
                world.agents[i].x = world.field_width/(world.num_agents+1) * (i+1)
                world.agents[i].y = world.endzone_length
            else:
                world.agents[i].x = world.field_width/(world.num_agents+1) * (i+1)
                world.agents[i].y = world.field_length - world.endzone_length
        
        # randomly choose who gets frisbee
        for i in range(world.num_agents):
            world.agents[i].has_frisbee = False
        player_with_frisbee = randint(0, world.num_agents-1)
        world.agents[player_with_frisbee].has_frisbee = True

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if agent.is_offense]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if not agent.is_offense]

    def reward(self, agent, world):
        return self.offense_reward(agent, world) if agent.is_offense else self.defense_reward(agent, world)

    def defense_reward(self, agent, world):
        reward = 1
        # check if offense has frisbee in endzone
        offense_agents = self.good_agents(world)
        for offense_agent in offense_agents:
            if offense_agent.has_frisbee and offense_agent.y > world.field_length:
                reward -= 1000
                world.done = True

        # check if player is in bounds
        if agent.x < 0 or agent.x > world.field_width or agent.y < 0 or agent.y > world.field_length - world.endzone_length:
            reward -=5
        
        # check if turnover has occured
        if world.turnover:
            reward += 10000
            world.done = True
            #print('done: turnover')

        return reward

    def offense_reward(self, agent, world):
        reward = -1

        # check if offense has frisbee in endzone
        offense_agents = self.good_agents(world)
        for offense_agent in offense_agents:
            if offense_agent.has_frisbee and offense_agent.y > world.field_length - world.endzone_length:
                reward += 10000
                world.done = True
                #print('done: frisbee in endzone')

        # check if player is in bounds
        if agent.x < 0 or agent.x > world.field_width or agent.y < 0 or agent.y > world.field_length:
            reward -=5
        
        if world.turnover:
            reward -= 1000
            world.done = True

        return reward


    def observation(self, world):
        observation = []
        for agent in world.agents:
            observation.append(agent.x)
            observation.append(agent.y)
            observation.append(agent.has_frisbee)

        return np.array(observation)