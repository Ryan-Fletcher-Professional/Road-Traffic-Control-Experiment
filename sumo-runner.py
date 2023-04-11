# use existing python traffic simulator with predefined traffic conditions to do it

import math
import random
import tkinter as tk
from tkinter import filedialog
from collections import namedtuple, deque
from itertools import count
import numpy as np
from datetime import datetime

import gymnasium as gym
import sumo_rl
import sys # getsysteminfo can be used to check memory usage

import rewards

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# create random generator
rng = np.random.default_rng(seed=98765)
random.seed("98765")

root = tk.Tk()
root.withdraw()
directory = ""
directory = filedialog.askdirectory(title="Select Traffic Data Directory")
files = filedialog.askopenfiles(title="Select Network and Route Files",initialdir=directory)
if len(files) < 2:
    print("Less than 2 files.")
    exit(0)

env = sumo_rl.parallel_env(
                           net_file=files[0].name,
                           route_file=files[1].name,
                           out_csv_name=directory+"\\output.csv",
                           num_seconds=100000,
                           use_gui=False,
                           reward_fn=rewards.impedance_reward,
                           min_green=30,
                           max_green=60
                           )

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    # https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
    from IPython import display 

# Turn on interactive mode to plot onto the screen
# https://matplotlib.org/3.1.1/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
plt.ion()

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
        if(h0.size()[0] != 1):
            randomNums = rng.integers(low=0, high=h0.size()[1]-1, size=h0.size()[1])
            indices = torch.tensor([randomNums])
            compressedh0 = torch.gather(h0, dim=1, index=indices)
            h2 = self.layer3(h1, compressedh0)
        else:
            h2 = self.layer3(h1, h0)
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

agents_list = [agent for agent in env.agents]
# Dictionaries containing agent data aren't necessarily consistently ordered.
# Using agents_list.index(agent) should allow an arbitrary but consistent ordering of the intersections in the output layer of the NN.

#print(observations)
# Get number of actions from gym action space
n_actions = np.sum([env.action_space(a).n for a in env.agents])
#print("\nActions: ", n_actions)
n_observations = np.sum([len(list(observations.values())[i]) for i in range(len(list(observations.values())))])
#print("\nObservations ", n_observations)

policy_net = DQN(n_observations, n_actions).to(device) # Cast the DQN parameters and buffers and move them to the device
target_net = DQN(n_observations, n_actions).to(device) # Cast the DQN parameters and buffers and move them to the device
target_net.load_state_dict(policy_net.state_dict())    # Copy parameters and buffers from the state_dict of policy_net into the target_net

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_actions(state):
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
            net = policy_net(state)
            #print("\nNet: ", net)
            actions = {}
            for agent in env.agents:
                actions[agent] = 0 if net[0][2*agents_list.index(agent)].item() >= net[0][(2*agents_list.index(agent))+1].item() else 1
                # CHANGE NN OUTPUT LAYER SIZE TO env.num_agents AND READ +/-???
            return actions
    else:
        actions = {agent: torch.tensor([[env.action_space(agent).sample()]], device=device, dtype=torch.long)[0].item() for agent in env.agents}
        #print("\nActions: ", actions)
        return actions


episode_durations = []
steps = []
rewards = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def plot_rewards(show_result=False):
    plt.figure(2)
    steps_t = torch.tensor(steps,dtype=torch.int64)
    rewards_t = torch.tensor([], dtype=torch.float)
    for reward in rewards:
        rewards_t = torch.cat((rewards_t, torch.unsqueeze(reward, 0)))
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.plot(steps_t.numpy(), rewards_t.numpy())
    #for i in range(len(rewards_t)):  # Use this to get number labels for the intersections
        #plt.plot(steps_t.numpy(), rewards_t[i].numpy(), str(i))

    # Take 100 episode averages and plot them too
    # if len(rewards_t) >= 100:
    #     means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
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
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch               # MAY NEED TO RESHAPE/COMPRESS

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

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


for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    obs = env.reset()
    # print("\nObs: ", obs)
    obs_lists = [[] for i in range(env.num_agents)]
    for agent in env.agents:  # Sort observations into proper agent ordering
        obs_lists[agents_list.index(agent)] = obs[agent]
    observations = np.array([], dtype=np.float32)
    for ob in obs_lists:  # Flatten observations into ordered inputs for NN
        observations = np.concatenate((observations, ob))
    # print("\nAgents: ", env.agents)
    # print("\nObservations flattened: ", observations)
    states = torch.tensor(observations, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        #while env.agents:
        actions = select_actions(states)
        # for agent in env.agents:
        #     actions[agent] = None#select_action(, agent)
            # actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
        obs, rews, terminations, truncations, infos = env.step(actions)
        #print("\nRewards: ", rews)
        obs_lists = [[] for i in range(env.num_agents)]
        for agent in env.agents:  # Sort observations into proper agent ordering
            obs_lists[agents_list.index(agent)] = obs[agent]
        observations = np.array([], dtype=np.float32)
        for ob in obs_lists:  # Flatten observations into ordered inputs for NN
            observations = np.concatenate((observations, ob))
        rew_lists = [[] for i in range(env.num_agents)]
        for agent in env.agents:
                rew_lists[agents_list.index(agent)] = rews[agent]
        rewards_new = []
        for r in rew_lists:
            rewards_new.append(r)
        #print("\nTerminations: ", terminations)
        #print("\nTruncations: ", truncations)
        terminated = True if True in list(terminations.values()) else False
        truncated = True if True in list(truncations.values()) else False
        # action = select_action(state)
        # observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor(rewards_new, device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observations, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(states, actions, next_state, reward)

        # Move to the next state
        states = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        rewards.append(reward)
        steps.append(steps_done)
        plot_rewards()

        if done:
            episode_durations.append(t + 1)
            #rewards.append(reward)
            #steps.append(steps_done)
            #plot_rewards()             # UNCOMMENT???
            break

print('Complete')
plot_rewards(show_result=True)
plt.ioff()
plt.show()