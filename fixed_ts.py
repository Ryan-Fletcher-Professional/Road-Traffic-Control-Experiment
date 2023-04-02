import traci
import sumolib
import traci.constants as tc
import tkinter as tk
from tkinter import filedialog
import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

root = tk.Tk()
root.withdraw()
directory = ""
directory = filedialog.askdirectory(title="Select Traffic Data Directory of the network")
files = filedialog.askopenfiles(title="Select Sumocfg",initialdir=directory)
sumocfg = files[0].name.replace('/', '\\')

traci.start([sumolib.checkBinary("sumo"), "-c", sumocfg], label="init_connection")
conn = traci.getConnection("init_connection")


junctionID = "t"
traci.junction.subscribeContext(junctionID, tc.CMD_GET_VEHICLE_VARIABLE, 42, [tc.VAR_SPEED, tc.VAR_WAITING_TIME])
print(traci.junction.getContextSubscriptionResults(junctionID))

with open(directory+"_fixed_ts.txt", 'w') as f:
    for step in range(10000):
        traci.simulationStep()
        vehiclelist = traci.vehicle.getIDList()
        waiting_times = [traci.vehicle.getAccumulatedWaitingTime(vehicle) for vehicle in vehiclelist]
        f.write(str(sum(waiting_times)))
        f.write('\n')
traci.close()