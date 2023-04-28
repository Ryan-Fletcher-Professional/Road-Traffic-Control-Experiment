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

import rewards
import observations

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


"""This file runs a single large neural network that takes inputs from and controls all the traffic signals in the traffic network."""

# Get string representation of current date and time
now = datetime.now()
output_dir = getcwd() + "\\output\\" + "adaptive " + now.strftime("%m-%d-%Y %H-%M-%S")
mkdir(output_dir)


# create random generator
rng = np.random.default_rng(seed=98765)
random.seed("98765")

DEFAULT_BEGIN_TIME = 1000

if(len(sys.argv) > 2):
    net_file = sys.argv[1]
    route_file = sys.argv[2]
    begin_time_s = int(sys.argv[3])
else:
    begin_time_s = DEFAULT_BEGIN_TIME
    root = tk.Tk()
    root.withdraw()
    directory = ""
    directory = filedialog.askdirectory(title="Select Traffic Data Directory")
    files = filedialog.askopenfiles(title="Select Network and Route Files",initialdir=directory)
    if len(files) < 2:
        print("Less than 2 files.")
        exit(0)
    else:
        if(files[0].name.endswith("net.xml")):
            net_file = files[0].name
            route_file = files[1].name
        else:
            net_file = files[1].name
            route_file=files[0].name

network_name = net_file[net_file.rindex('\\') + 1:net_file.index('.')]


env = sumo_rl.parallel_env(
                           net_file=net_file,
                           route_file=route_file,
                           out_csv_name=output_dir + "\\" + network_name,
                           num_seconds=61600,  # Only 4000 seconds per episode for the ingolstadt21 network that has 21 traffic signals
                           begin_time=begin_time_s,
                           use_gui=True,
                           reward_fn=rewards.coordinated_mean_max_impedence_reward,
                           observation_class=observations.CompressedObservationFunction
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
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 1024)
        self.layer3 = nn.RNN(1024, 256, 1)
        self.layer4 = nn.Linear(256, n_actions)

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

# Get number of actions from gym action space
n_actions = np.sum([env.action_space(a).n for a in env.agents])

n_observations = sum([len(observation) for observation in list(observations.values())])

policy_net = DQN(n_observations, n_actions).to(device) # Cast the DQN parameters and buffers and move them to the device
target_net = DQN(n_observations, n_actions).to(device) # Cast the DQN parameters and buffers and move them to the device
target_net.load_state_dict(policy_net.state_dict())    # Copy parameters and buffers from the state_dict of policy_net into the target_net

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

obs = observations
observations = np.array([], dtype=np.float32)  # Flattened version of observations dictionary
for agent in env.agents:
    observations = np.concatenate((observations, obs[agent]), axis=None)
        
states = torch.tensor(observations, dtype=torch.float32, device=device).unsqueeze(0)

def select_actions(states):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            net_out = policy_net(states)
            actions = {}
            
            agent_action_index = 0
            for agent in env.agents:
                actions_number = env.action_space(agent).n
                opt_action_index = net_out.numpy()[0][agent_action_index:agent_action_index+actions_number].argmax()
                agent_action_index += actions_number
                actions[agent] = opt_action_index            
            return actions
    else:
        actions = {agent: torch.tensor([[env.action_space(agent).sample()]], device=device, dtype=torch.long)[0].item() for agent in env.agents}
        return actions


episode_durations = []
steps = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    if env.num_agents != 21:
        return

    transitions = memory.sample(BATCH_SIZE)
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
    state_action_values = policy_net(state_batch).gather(1, action_batch.type(torch.int64))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, env.num_agents, device=device)
    with torch.no_grad():
        opt_actions = {}
        net_out = target_net(non_final_next_states)
        for i in range(BATCH_SIZE):
            actions = {}
            agent_action_index = 0
            for agent in env.agents:
                actions_number = env.action_space(agent).n
                opt_action_Q = net_out.numpy()[i][agent_action_index:agent_action_index+actions_number].max()
                agent_action_index += actions_number
                actions[agent] = opt_action_Q 
            opt_actions[i] = actions
        next_state_values[non_final_mask] = torch.Tensor([list(dictionary.values()) for dictionary in [opt_actions[index] for index in opt_actions]]) 
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(0, num_episodes):
    if i_episode > 0:
        # Initialize the environment and get it's state

        obs = env.reset()
        observations = np.array([], dtype=np.float32)

        for agent in env.agents:
            observations = np.concatenate((observations, obs[agent]), axis=None)

        states = torch.tensor(observations, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():

        actions = select_actions(states)
        obs, rews, terminations, truncations, infos = env.step(actions)
        rewards_new = []
        for agent in env.agents:
            rewards_new.append(rews[agent])
        reward = torch.tensor(rewards_new, device=device)

        dones = {}
        next_states = {}
        for agent in env.agents:
            dones[agent] = terminations[agent] or truncations[agent]

        for agent in env.agents:
            if dones[agent]:
                next_states[agent] = None
            else:
                next_states[agent] = obs[agent] 

        if all(done == True for done in dones.values()):
            episode_durations.append(t + 1)
            break

        observations = np.concatenate([next_states[agent] for agent in env.agents], axis=None)

        next_states_tensor = torch.tensor(observations, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(states, actions, next_states_tensor, reward)

        # Move to the next state
        states = next_states_tensor

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        steps.append(steps_done)

print('Complete')
