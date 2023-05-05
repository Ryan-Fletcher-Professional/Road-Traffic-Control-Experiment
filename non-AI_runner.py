
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


net_file = getcwd() + "\\" + net_file
route_file = getcwd() + "\\" + route_file

env = MySumoEnvironment(
                           net_file=net_file,
                           route_file=route_file,
                           out_csv_name=output_dir + "\\" + network_name,
                           num_seconds=61600,  # Only 4000 seconds per episode for the ingolstadt21 network that has 21 traffic signals
                           begin_time=begin_time_s,
                           fixed_ts=True,
                           single_agent=False,
                           use_gui=True,
                           reward_fn=rewards.coordinated_mean_max_impedence_reward,
                           observation_class=observations.CompressedObservationFunction
                           )

# if gpu is to be used
# Compute Unified Device Architecture (CUDA) is an NVIDIA API for general purpose computing on GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the number of state observations
observations = env.reset()

# Perform fixed time traffic signal control
for t in count():
    observations, rewards, dones, infos = env.step({})

    if dones['__all__']:
        break

# Save infos to a CSV
env.reset()

print('Complete')
