from itertools import count
from datetime import datetime
from os import mkdir, getcwd
from datetime import datetime
from custom_parallel_envs.my_sumo_env import MySumoEnvironment 
from threading import Thread, Lock, get_ident

"""This file simulates FTSC."""

# Print the starting time of this program
print(f"Starting at {datetime.now()}")

# Contains the path of the SUMO network file
net_file = r"sumo_networks\TUM-VT\ingolstadt_24h.net.xml.gz"

# Contains the path of the SUMO route files
route_file = r"sumo_networks\TUM-VT\bicycle_routes_24h.rou.xml.gz,sumo_networks\TUM-VT\motorized_routes_2020-09-16_24h.rou.xml.gz"

# Contains the network name
network_name = net_file[net_file.rindex('\\') + 1:net_file.index('.')]

# Get string representation of current date and time
now = datetime.now()
output_dir = getcwd() + "\\output\\" + "FTSC  " + network_name + " " + now.strftime("%m-%d-%Y %H-%M-%S")

# Create a directory with a unique name to store CSV output
mkdir(output_dir)

# A present issue with this approach is that it is not possible to accurately simulate a subportion of the 
# entire 86400s like 6:00AM to 10:00 AM using the following code because if we don't start from t=0s and we 
# start at t=t0s we'll be excluding vehicles that got into the intersection after t=0s and before t=t0s 
# but didn't get out. Therefore, even if we start the simulation at 6:00 AM there won't be any vehicles on 
# the road although there could be vehicles on the road that reached Ingolstadt before 6:00 AM. 
# All vehicles with a departure time (depart) lower than the begin time are discarded.

# The step length is going to be 0.25 because that is what is defined in the network configuration file 

# The following environment parameters must be configured for realistic simulation
# delta_time (int): Simulation seconds between actions in integers. Default: 5 seconds
# yellow_time (int): Duration of the yellow phase. Default: 2 seconds
# https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html
# delta_time > yellow_time, "Time between actions must be at least greater than yellow time."
# min_green (int): Minimum green time in a phase. Default: 5 seconds
# max_green (int): Max green time in a phase. Default: 60 seconds. Warning: This parameter is currently ignored!

DELTA_TIME = 3 # Set to smallest possible integer value for action step length

# URL: https://sumo.dlr.de/docs/Simulation/Output/index.html
# SUMO gui produces a step-log
# Device rerouting probability documentation: https://sumo.dlr.de/docs/Demand/Automatic_Routing.html
# Step length documentation: https://sumo.dlr.de/docs/Simulation/Basic_Definition.html#:~:text=sumo%20%2F%20sumo-gui%20use%20a%20time%20step%20of,a%20value%20in%20seconds%20between%20%5B0.001%20and%201.0%5D.
# For environment default values, https://github.com/LucasAlegre/sumo-rl/blob/master/sumo_rl/environment/env.py

# Initialize a MySumoEnvironment lasting for 720s starting at t=14400s and ending at t=15120s
env_early = MySumoEnvironment(
                           net_file=net_file,
                           route_file=route_file,
                           out_csv_name=output_dir + "\\" + network_name,
                           additional_sumo_cmd="--step-length 0.25 --ignore-junction-blocker 15 --device.rerouting.probability 0.2",                           
                           delta_time=DELTA_TIME,
                           time_to_teleport=300,
                           max_depart_delay=100,
                           num_seconds=15120,  # Only 720 seconds per episode for the ingolstadt21 network
                           fixed_ts = True,
                           single_agent=False,
                           begin_time=14400,
                           sumo_seed=123456,
                           use_gui=False
                           )

# Initialize a MySumoEnvironment lasting for 720s starting at t=21600s and ending at t=22320s
env = MySumoEnvironment(
                           net_file=net_file,
                           route_file=route_file,
                           out_csv_name=output_dir + "\\" + network_name,                         
                           additional_sumo_cmd="--step-length 0.25 --ignore-junction-blocker 15 --device.rerouting.probability 0.2",                           
                           delta_time=DELTA_TIME,
                           time_to_teleport=300,
                           max_depart_delay=100,
                           num_seconds=22320,  # Only 720 seconds per episode for the ingolstadt21 network
                           fixed_ts = True,
                           single_agent=False,
                           begin_time=21600,
                           sumo_seed=123456,
                           use_gui=False
                           )

# Initialize a MySumoEnvironment lasting for 720s starting at t=28800s and ending at t=29520s
env_late = MySumoEnvironment(
                           net_file=net_file,
                           route_file=route_file,
                           out_csv_name=output_dir + "\\" + network_name,                           
                           additional_sumo_cmd="--step-length 0.25  --ignore-junction-blocker 15 --device.rerouting.probability 0.2",                           
                           delta_time=DELTA_TIME,
                           time_to_teleport=300,
                           max_depart_delay=100,
                           num_seconds=29520,  # Only 720 seconds per episode for the ingolstadt21 network
                           fixed_ts = True,
                           single_agent=False,
                           sumo_seed=123456,
                           begin_time=28800,
                           use_gui=False
                           )

# Reset each of the environments
env_early.reset()
env.reset()
env_late.reset()

# Initialize a lock for multithreading to prevent data race conditions if possible. 
# Multithreading is preferred because the 3 MySumoEnvironments already 
# involve multiprocessing in the form of 3 different sumo binaries.
# A better application of multithreading could be to run the simulation with different seeds
lock = Lock()

# The Thread target function
def simulate_env(env):
    # Perform fixed time traffic signal control
    for t in count():
        # Output simulation state info
        print(f"(Thread #{get_ident()}) Step #{t} Start Time:{env.begin_time} End Time: {env.sim_max_time}")
        
        # Acquire the lock before making a step
        lock.acquire()

        # Make a step for the traffic signal agent that does fixed time signaling
        observations, rewards, dones, infos = env.step({})

        # Release the lock
        lock.release()

        if dones['__all__']:
            # If all agents are done, quit the loop
            break
    # Save infos to a CSV
    env.reset()

# Initialize the 3 threads to run each of the TSC experiments simultaneously
t1 = Thread(target = simulate_env, args = (env_early,), daemon=True)
t2 = Thread(target = simulate_env, args = (env,), daemon=True)
t3 = Thread(target = simulate_env, args = (env_late,), daemon=True)

# Start each of the threads
t1.start()
t2.start()
t3.start()

# Block the calling main thread until all of the threads terminate
t1.join()
# Check if thread was blocked or not
print("Thread t1 was blocked") if (t1.is_alive()) else print("Thread t1 terminated")
t2.join()
# Check if thread was blocked or not
print("Thread t2 was blocked") if (t2.is_alive()) else print("Thread t2 terminated")
t3.join()
# Check if thread was blocked or not
print("Thread t3 was blocked") if (t3.is_alive()) else print("Thread t3 terminated")

# Keyboard interrupts work properly only after all threads have completed their work

# Print the ending time of this program
print(f"Ending at {datetime.now()}")

