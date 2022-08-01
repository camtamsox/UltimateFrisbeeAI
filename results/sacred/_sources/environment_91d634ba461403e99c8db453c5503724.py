import gym
from gym import spaces
import numpy as np

from tf2marl.multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.agents
        # set required vectorized gym env property
        self.n = len(world.agents)
        self.agents_per_team = world.agents_per_team
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:

            self.action_space.append(spaces.Discrete(18)) # direction to change (9) + throw frisbee (2) who to throw to + (7)

            obs_dim = len(observation_callback(self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))


        self.game_history = [] # for saving
        self.step_num = 0
        self.episode = 0
        self.save_interval = 50

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)
        # advance world state
        self.world.step()
        self.step_num += 1
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # each agent's observation represents entire world state
        if self.game_history[-1] is not obs_n[0]:
            self.game_history.append(obs_n[0]) 

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.episode += 1
        
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        # save every save_interval episodes
        if self.game_history != [] and (self.episode % self.save_interval == 0 or self.step_num > 9999):
            with open('test_game.npy', 'wb') as f:
                np.save(f, np.array(self.game_history))
            print('game has been saved from episode: %d' % self.episode)
        self.game_history = [obs_n[0]]
        self.step_num = 0

        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        return self.observation_callback(self.world).astype(np.float32)

    # get dones for a particular agent
    def _get_done(self, agent):
        return self.world.done

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent):
        # action array is softmax with each value representing one posibility in action space
        agent.direction_to_change = np.argmax(action[0:8])
        if action[9] > action[10]: # action[9] means throw
            agent.throw = True
        else:
            agent.throw = False
        agent.target_player_num = np.argmax(action[11:18])



# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
