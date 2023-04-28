# Based on implementation for Cologne8 network https://github.com/LucasAlegre/sumo-rl/blob/9619f82c454020d03926835be31748db2b511274/experiments/sarsa_resco.py


import os
from symbol import term
import sys

import fire


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
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
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
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

# Get string representation of current date and time
now = datetime.now()
output_dir = getcwd() + "\\output\\" + "adaptive true online sarsa lambda " + now.strftime("%m-%d-%Y %H-%M-%S")
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

def run(use_gui=False, episodes=50):
    fixed_tl = False

    env = my_parallel_env(
                           sumo_env=MySumoEnvironment,
                           net_file=net_file,
                           route_file=route_file,
                           out_csv_name=output_dir + "\\" + network_name,
                           num_seconds=61600,  # Only 4000 seconds per episode for the ingolstadt21 network that has 21 traffic signals
                           begin_time=begin_time_s,
                           use_gui=True,
                           reward_fn=rewards.coordinated_mean_max_impedence_reward,
                           observation_class=observations.CompressedObservationFunction
                           )
    env.reset()

    agents = {
        ts_id: TrueOnlineSarsaLambda(
            env.observation_spaces[ts_id],
            env.action_spaces[ts_id],
            alpha=0.0001,
            gamma=0.95,
            epsilon=0.05,
            lamb=0.1,
            fourier_order=7,
        )
        for ts_id in env.agents
    }

    for ep in range(1, episodes + 1):
        obs = env.reset()
        dones = {agent: False for agent in env.agents}

        if fixed_tl:
            while not done["__all__"]:
                _, _, done, _ = env.step(None)
        else:
            while True:
                actions = {ts_id: agents[ts_id].act(obs[ts_id]) for ts_id in obs.keys()}
                next_obs, r, terminations, truncations, infos = env.step(actions=actions)
                if len(env.agents) != 0:
                    for agent in env.agents:
                        dones[agent] = terminations[agent] or truncations[agent]
                
                if all(truncations.values()) == True:
                    break

                for ts_id in next_obs.keys():
                    agents[ts_id].learn(
                        state=obs[ts_id], action=actions[ts_id], reward=r[ts_id], next_state=next_obs[ts_id], done=dones[ts_id]
                    )
                    obs[ts_id] = next_obs[ts_id]

    env.close()


if __name__ == "__main__":
    fire.Fire(run)
    #run() To debug