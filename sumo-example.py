# use existing python traffic simulator with predefined traffic conditions to do it
import gymnasium as gym
import sumo_rl
import tkinter as tk
from tkinter import filedialog
import rewards

root = tk.Tk()
root.withdraw()
directory = filedialog.askdirectory(title="Select Traffic Data Directory", initialdir="C:\\Users\\jancy\\Desktop\\mathew\\Amherst\\Spring2023\\Neural_Networks\\gym-examples\\sumo_rl_repo\\nets")
files = filedialog.askopenfiles(title="Select Network and Route Files",initialdir=directory)
if len(files) < 2:
    print("Less than 2 files.")
    exit(0)

env = gym.make('sumo-rl-v0',
                net_file=files[0].name,
                route_file=files[1].name,
                out_csv_name=directory+"\\output.csv",
                use_gui=True,
                num_seconds=100000,
                reward_fn=rewards.max_WAVE_reward_fn,
                min_green=30,
                max_green=60
                )

obs, info = env.reset()

done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
               