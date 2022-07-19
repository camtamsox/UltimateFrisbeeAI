
import gym
from gym import spaces
import numpy as np
import random
from shapely.geometry.polygon import Polygon
from datetime import datetime
import os


from tf_agents.environments import gym_wrapper, tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
import tf_agents.trajectories.time_step as ts
from tf_agents.utils import common
from tf_agents.policies.policy_saver import PolicySaver

from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.agents.ppo import ppo_agent, ppo_policy

from tensorflow.python.client import device_lib


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


class PPOTrainer:
    
    def __init__(self, ppo_agents, train_env, eval_env, use_tensorboard=True, add_to_video=True, use_lstm=False, total_epochs=1000, collect_steps_per_episode=1000,
                eval_interval=100, num_eval_episodes=5, epsilon=0.0, save_interval=500, log_interval=1, experiment_name=""):

        self.train_env = train_env
        self.eval_env = eval_env

        self.use_lstm=use_lstm
        self.num_agents = 14

        self.max_buffer_size = collect_steps_per_episode
        self.collect_steps_per_episode = collect_steps_per_episode
        self.epochs = total_epochs
        self.global_step = 0 # global step count
        self.epsilon = epsilon # probability of using greedy policy

        self.agents = ppo_agents
        for agent in self.agents:
            agent.initialize()
        
        self.actor_nets = []
        self.value_nets = []
        self.eval_policies = []
        self.collect_policies = []
        self.replay_buffers = []
        for i in range(self.num_agents):
            self.actor_nets.append(self.agents[i]._actor_net)
            self.value_nets.append(self.agents[i]._value_net)
            self.eval_policies.append(self.agents[i].policy)
            self.collect_policies.append(self.agents[i].collect_policy)
            self.replay_buffers.append(
                tf_uniform_replay_buffer.TFUniformReplayBuffer(
                    self.agents[i].collect_data_spec,
                    batch_size=self.train_env.batch_size,
                    max_length=self.max_buffer_size))

        self.num_eval_episodes = num_eval_episodes
        self.eval_interval = eval_interval
        self.eval_returns = []
        for i in range(self.num_agents):
            self.eval_returns.append([])
        # logging
        self.time_string = datetime.now().strftime("%Y-%m-%d-%H%M")
        self.log_interval = log_interval

        # creating video
        self.video_train = []
        self.video_eval = []
        self.add_to_video = add_to_video
        self.FPS = 50

        # saving
        self.policy_save_dir = os.path.join(os.path.split(__file__)[0], "models", experiment_name.format(self.time_string))
        self.save_interval = save_interval
        if not os.path.exists(self.policy_save_dir):
            print("Directory {} does not exist. Creating it now".format(self.policy_save_dir))
            os.makedirs(self.policy_save_dir, exist_ok=True)
    
        self.train_savers = []
        self.eval_savers = []
        for i in range(self.num_agents):
            self.train_savers.append(PolicySaver(self.collect_policies[i], batch_size=None))
            self.eval_savers.append(PolicySaver(self.eval_policies[i], batch_size=None))

        # tensorboard


        # devices
        local_devices = device_lib.list_local_devices()
        num_gpus = len([x.name for x in local_devices if x.device_type == 'GPU'])
        self.use_gpu = num_gpus > 0

    def is_last(self, mode='train'): # not sure if I need this because env should say if the episode is done
        if mode == 'train':
            step_types = self.train_env.current_time_step().step_type.numpy()
        elif mode == 'eval':
            step_types = self.eval_env.current_time_step().step_type.numpy()
        
        is_last = bool(min(np.count_nonzero(step_types == 2), 1)) # not sure why min is used
        return is_last
    
    def get_agent_timesteps(self, time_step, step_type): # create timestep compatible w/ tf-agents
        discount = time_step.discount # will this be list/array?
        discount = tf.convert_to_tensor(discount, dtype=tf.float32, name='discount')

        rewards = []
        for i in range(self.num_agents):
            rewards.append(tf.convert_to_tensor(time_step.reward[i], dtype=tf.float32, name='reward'))

        processed_observations = self.process_observations(time_step)
        new_time_steps = []
        for i in range(self.num_agents):
            new_time_steps.append(ts.TimeStep(step_type, rewards[i], discount, processed_observations))
        return new_time_steps

    def process_observations(self, time_step):
        # placeholder for doing partial observations of environment

        processed_observations = time_step.observations
        return processed_observations

    # collect steps
    def collect_step(self, step=0, use_greedy=False, add_to_video=False):
        time_step = self.train_env.current_time_step()
        agent_timesteps = self.get_agent_timesteps(time_step, time_step.step_type) # step_type might be an array

        actions = []
        action_steps = []
        for i in range(self.num_agents):
            agent_ts = agent_timesteps[i]
            action_steps.append(self.collect_policies[i].action(agent_ts))
            actions.append(action_steps[i].action)

        # tensorboard



        action_tensor = tf.convert_to_tensor([tf.stack(tuple(actions), axis=1)])

        next_time_step = self.train_env.step(action_tensor)
        next_agent_timesteps = self.get_agent_timesteps(next_time_step, next_time_step.step_type)
        for i in range(self.num_agents):
            traj = trajectory.from_transition(agent_timesteps[i], action_steps[i], next_agent_timesteps[i])
            self.replay_buffers[i].add_batch(traj)

        # add observation to video




        rewards = []
        for i in range(self.num_agents):
            rewards.append(agent_timesteps[i].reward)
        return rewards


    # collect episode

    # compute average reward

    # collect step lstm

    # reset policy states

    # collect episode lstm

    # compute average reward lstm

    # train agents

    # create video

    # plot evaluation returns

    # save policies

    # load policies

def make_networks(env, in_fc_params, out_fc_params, lstm_size, use_lstm=False):
    
    if use_lstm:
        actor_net = ActorDistributionRnnNetwork(env.observation_spec(), env.action_spec(), conv_layer_params=[], input_fc_layer_params=in_fc_params, lstm_size=lstm_size, output_fc_layer_params=out_fc_params)

        value_net = ValueRnnNetwork(env.observation_spec(), conv_layer_params=[], input_fc_layer_params=in_fc_params, lstm_size=lstm_size, output_fc_layer_params=out_fc_params)

    else:
        actor_net = ActorDistributionNetwork(env.observation_spec(), env.action_spec(), conv_layer_params=[])

        value_net = ValueNetwork(env.observation_spec(), conv_layer_params=[])

    return actor_net, value_net

def make_agent(env, in_fc_params, out_fc_params, lstm_size, use_lstm=False, lr=8e-5):

    actor_net, value_net = make_networks(env, in_fc_params, out_fc_params, lstm_size, use_lstm)

    agent = ppo_agent.PPOAgent(env.time_step_spec(), env.action_spec(), actor_net=actor_net, value_net=value_net, optimizer=tf.compat.v1.train.AdamOptimizer(lr))

    return agent
