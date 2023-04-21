import traci
from traci import FatalTraCIError
import sumolib
import traci.constants as tc
import tkinter as tk
from tkinter import filedialog
import os
import io
from os import getcwd
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

print("Current Working Directory: " + getcwd())

if(len(sys.argv) > 1):
    # Requires full file path, not relative file paths
    sumo_cfg = sys.argv[1]
else:
    root = tk.Tk()
    root.withdraw()
    directory = ""
    directory = filedialog.askdirectory(title="Select Traffic Data Directory")
    files = filedialog.askopenfiles(title="Select Network and Route Files",initialdir=directory)
    if len(files) < 1:
        print("Less than 1 file.")
        exit(0)
    sumo_cfg = files[0].name.replace('/', '\\')

traci.start([sumolib.checkBinary("sumo"), "-c", sumo_cfg], label="init_connection")
conn = traci.getConnection("init_connection")

# Here, Linux type file paths work too
with open("powershell/junctions.txt", 'r') as junctions:       
    line = junctions.readline();
    junctionIDs = [line.strip() for line in junctions]

    for junctionID in junctionIDs:
        traci.junction.subscribeContext(junctionID, tc.CMD_GET_VEHICLE_VARIABLE, 42, [tc.VAR_SPEED, tc.VAR_WAITING_TIME])
        print("Subscription Results for Junction " + junctionID + ": " + str(traci.junction.getContextSubscriptionResults(junctionID)))

with io.open("output\\fixed_ts.txt", 'w+') as f:
    for step in range(100000):
        print("Step is: " + str(step))
        traci.simulationStep()
        vehiclelist = traci.vehicle.getIDList()
        vehicle_dict = {}
        for vehicle in vehiclelist:
            vehicle_dict[vehicle] = traci.vehicle.getAccumulatedWaitingTime(vehicle)
        str(vehicle_dict)
        waiting_times = [traci.vehicle.getAccumulatedWaitingTime(vehicle) for vehicle in vehiclelist]
        f.write(str(sum(waiting_times)))
        f.write('\n')
traci.close()