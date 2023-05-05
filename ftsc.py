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

"""This file simulates FTSC."""


if(len(sys.argv) > 4):
    net_file = sys.argv[1]
    route_file = sys.argv[2]
    additional_sumo_cmd = sys.argv[3]
    begin_time_s = int(sys.argv[4])
else:
    exit(1)

network_name = net_file[net_file.rindex('\\') + 1:net_file.index('.')]

# Get string representation of current date and time
now = datetime.now()
output_dir = getcwd() + "\\output\\" + "FTSC  " + network_name + " " + now.strftime("%m-%d-%Y %H-%M-%S")
mkdir(output_dir)

env = MySumoEnvironment(
                           net_file=net_file,
                           route_file=route_file,
                           additional_sumo_cmd=additional_sumo_cmd,
                           out_csv_name=output_dir + "\\",
                           num_seconds=36000,  # Only 4000 seconds per episode for the ingolstadt21 network that has 21 traffic signals
                           fixed_ts = True,
                           single_agent=False,
                           begin_time=begin_time_s,
                           use_gui=False
                           )

# Get the number of state observations
observations = env.reset()

# Perform fixed time traffic signal control
for t in count():
    observations, rewards, dones, infos = env.step({})

    if dones['__all__']:
        break

# Save infos to a CSV
env.reset()


