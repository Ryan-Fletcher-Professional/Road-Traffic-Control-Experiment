import math
import random
import tkinter as tk
from tkinter import filedialog
from collections import namedtuple, deque
from itertools import count
import numpy as np
from datetime import datetime
from os import mkdir, getcwd
import gymnasium as gym
import sumo_rl
import sys # getsysteminfo can be used to check memory usage

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from custom_parallel_envs.my_parallel_wrapper_fns import my_parallel_env
from custom_parallel_envs.my_sumo_env import MySumoEnvironment
import custom_parallel_envs.rewards as rewards
import custom_parallel_envs.observations as observations

"""This file runs multiple small neural networks that each take inputs from and control a single traffic signal in the traffic network."""

if(len(sys.argv) > 5):
    net_file = sys.argv[1]
    route_file = sys.argv[2]
    additional_sumo_cmd = sys.argv[3]
    begin_time_s = int(sys.argv[4])
    action_step_length = int(sys.argv[5])
else:
    exit(1)

network_name = net_file[net_file.rindex('\\') + 1:net_file.index('.')]

# Get string representation of current date and time
now = datetime.now()
output_dir = getcwd() + "\\output\\" + f"CDQN uncompressed Obs {network_name} " + now.strftime("%m-%d-%Y %H-%M-%S")
mkdir(output_dir)

env = my_parallel_env(
                           sumo_env=MySumoEnvironment,
                           net_file=net_file,
                           route_file=route_file,
                           additional_sumo_cmd=additional_sumo_cmd,
                           out_csv_name=output_dir + "\\" + network_name,
                           delta_time=action_step_length,
                           num_seconds=36000,  # Only 4000 seconds per episode for the ingolstadt21 network that has 21 traffic signals
                           begin_time=begin_time_s,
                           use_gui=False,
                           reward_fn=rewards.coordinated_mean_max_impedence_reward,
                           observation_class=observations.DefaultObservationFunction
                           )

# if gpu is to be used
# Compute Unified Device Architecture (CUDA) is an NVIDIA API for general purpose computing on GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# nn.Module is the base class for all NNs, has 3 linear layers
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 128)
        self.layer3 = nn.RNN(128, 32, 1)
        self.layer4 = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        h0 = F.relu(self.layer1(x))
        h1 = F.relu(self.layer2(h0))
        for i in range(1, 10):
            compressedh0 = torch.mean(h0, 0, True)
            h2 = self.layer3(h1, compressedh0)
        return self.layer4(h2[0])


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get the number of state observations
observations = env.reset()

policy_nets = {}
target_nets = {}
optimizers = {}
memories = {}

for agent in env.agents:
    # Get number of actions from gym action space
    n_actions = env.action_space(agent).n
    n_observations = len(observations[agent])
    policy_net = DQN(n_observations, n_actions).to(device) # Cast the DQN parameters and buffers and move them to the device
    target_net = DQN(n_observations, n_actions).to(device) # Cast the DQN parameters and buffers and move them to the device
    target_net.load_state_dict(policy_net.state_dict())    # Copy parameters and buffers from the state_dict of policy_net into the target_net
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    policy_nets[agent] = policy_net
    target_nets[agent] = target_net
    optimizers[agent] = optimizer
    memories[agent] = memory

steps_done = 0

obs = observations
observations = np.array([], dtype=np.float32)  # Flattened version of observations dictionary
for agent in env.agents:
    observations = np.concatenate((observations, obs[agent]), axis=None)
        
states = torch.tensor(observations, dtype=torch.float32, device=device).unsqueeze(0)

def select_actions(states):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
    actions = {}
    agent_observation_index = 0
    for agent in env.agents:
        observations_number = env.observation_space(agent).shape[0]
        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                net_out = policy_nets[agent](states[0][agent_observation_index:agent_observation_index+observations_number].unsqueeze(0))
                opt_action_index = net_out.numpy()[0].argmax()
                actions[agent] = opt_action_index
        else:
            actions[agent] = torch.tensor([[env.action_space(agent).sample()]], device=device, dtype=torch.long)[0].item()
        agent_observation_index += observations_number
    steps_done += 1
    return actions


episode_durations = []
steps = []

def optimize_model(agent):
    if len(memories[agent]) < BATCH_SIZE:
        return
    transitions = memories[agent].sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.Tensor([list(action_dict.values()) for action_dict in batch.action])
    reward_batch = torch.Tensor([list(rewards) for rewards in batch.reward])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_nets[agent](state_batch).gather(1, action_batch.type(torch.int64))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, 1, device=device)
                                               #^used to be env.num_agents, before witching optimize_model to be per-agent
    with torch.no_grad():
        opt_actions = {}
        net_out = target_nets[agent](non_final_next_states)
        for i in range(BATCH_SIZE):
            opt_actions[i] = {agent: net_out.numpy()[i].max()}
        next_state_values[non_final_mask] = torch.Tensor([list(dictionary.values()) for dictionary in [opt_actions[index] for index in opt_actions]])  # torch.Tensor(opt_actions.values()).unsqueeze(0)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizers[agent].zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_nets[agent].parameters(), 100)
    optimizers[agent].step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50


for i_episode in range(0, num_episodes):
    # Initialize the environment and get its state
    if i_episode > 0:
        obs = env.reset()
        observations = np.array([], dtype=np.float32)

        for agent in env.agents:
            observations = np.concatenate((observations, obs[agent]), axis=None)

        states = torch.tensor(observations, dtype=torch.float32, device=device).unsqueeze(0)
        
    for t in count():

        actions = select_actions(states)
        obs, rews, terminations, truncations, infos = env.step(actions)
        observations = np.array([], dtype=np.float32)
        for agent in env.agents:
            observations = np.concatenate((observations, obs[agent]), axis=None)
        rewards_new = []
        for agent in env.agents:
            rewards_new.append(rews[agent])
        terminated = True if True in list(terminations.values()) else False
        truncated = True if True in list(truncations.values()) else False
        reward = torch.tensor(rewards_new, device=device)
        done = terminated or truncated
        
        if done:
            episode_durations.append(t + 1)
            break

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observations, dtype=torch.float32, device=device).unsqueeze(0)

        agent_observation_index = 0
        for agent in env.agents:
            observations_number = env.observation_space(agent).shape[0]
            memories[agent].push(states[0][agent_observation_index:agent_observation_index+observations_number].unsqueeze(0),
                                {agent: actions[agent]},
                                next_state[0][agent_observation_index:agent_observation_index+observations_number].unsqueeze(0),
                                [rews[agent]])
                                # The action is stored in a dictionary for compatibility with current version of optimize_model()
                                # Same with the reward being stored in a list
            agent_observation_index += observations_number
        # Store the transition in memory

        # Move to the next state
        states = next_state

        for agent in env.agents:
            # Perform one step of the optimization (on the policy networks)
            optimize_model(agent)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_nets[agent].state_dict()
            policy_net_state_dict = policy_nets[agent].state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_nets[agent].load_state_dict(target_net_state_dict)

        steps.append(steps_done)


print('Complete')
