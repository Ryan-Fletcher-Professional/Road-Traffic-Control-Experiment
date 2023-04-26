import traci
from traci import FatalTraCIError
import sumolib
import traci.constants as tc
import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime
import io
from os import mkdir, getcwd
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
    output_type = sys.argv[2]
    begin_time = sys.argv[3]

traci.start([sumolib.checkBinary("sumo"), "-c", sumo_cfg, "--begin", begin_time], label="init_connection")
conn = traci.getConnection("init_connection")


network_name = sumo_cfg[sumo_cfg.rindex('\\') + 1:sumo_cfg.index('.')]
# Get string representation of current date and time
now = datetime.now()
output_dir = getcwd() + "\\output\\" + now.strftime("%m-%d-%Y %H-%M-%S") + " " + network_name + " fixed_time_using_TraCI"
mkdir(output_dir)

# Here, Linux type file paths work too
with open("powershell/junctions.txt", 'r') as junctions:       
    line = junctions.readline();
    junctionIDs = [line.strip() for line in junctions]

    for junctionID in junctionIDs:
        traci.junction.subscribeContext(junctionID, tc.CMD_GET_VEHICLE_VARIABLE, 42, [tc.VAR_SPEED, tc.VAR_WAITING_TIME])
        print("Subscription Results for Junction " + junctionID + ": " + str(traci.junction.getContextSubscriptionResults(junctionID)))

with io.open(output_dir + "\\fixed_ts.txt", 'w+') as f: 
    if output_type == "--stopped":
        f.write("Step Number of Vehicles Stopped")
        for step in range(4000):
            print("Step is: " + str(step))
            traci.simulationStep()
            vehiclelist = traci.vehicle.getIDList()
            vehicle_dict = {}
            for vehicle in vehiclelist:
                vehicle_dict[vehicle] = traci.vehicle.getAccumulatedWaitingTime(vehicle)
            str(vehicle_dict)
            waiting_times = [traci.vehicle.getAccumulatedWaitingTime(vehicle) for vehicle in vehiclelist]
            f.write(str(step + begin_time) + " ")
            f.write(str(sum(waiting_times)))
            f.write('\n')
    elif output_type == "--waitingTime":
        f.write("Step Accumulated Waiting Time")
        for step in range(4000):
            print("Step is: " + str(step))
            traci.simulationStep()
            vehiclelist = traci.vehicle.getIDList()
            vehicle_dict = {}
            for vehicle in vehiclelist:
                vehicle_dict[vehicle] = traci.vehicle.isStopped(vehicle)
            str(vehicle_dict)
            stopped_vehicles = [vehicle for vehicle in vehiclelist if traci.vehicle.isStopped(vehicle)]
            f.write(str(step + begin_time) + " ")
            f.write(len(stopped_vehicles))
            f.write('\n')
traci.close()