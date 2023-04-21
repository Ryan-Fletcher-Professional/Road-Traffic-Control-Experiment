import traci
from traci import FatalTraCIError
import numpy as np
import sumolib
import traci.constants as tc
import tkinter as tk
from tkinter import filedialog
import os
import io
from os import getcwd
import sys
from math import sqrt

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

junction_disp_dict = {}

# Here, Linux type file paths work too
with open("powershell/junctions.txt", 'r') as junctions:       
    line = junctions.readline();
    junctionIDs = [line.strip() for line in junctions]

    for junctionID1 in junctionIDs:
        pos1 = traci.junction.getPosition(junctionID1)
        junction_distance = {}
        for junctionID2 in junctionIDs:
            if junctionID1 != junctionID2:
                pos2 = traci.junction.getPosition(junctionID2)
                dist = (pos2[0] - pos1[0], pos2[1] - pos2[1])
                vec = sqrt(dist[0] * dist[0] + dist[1] * dist[1])
                junction_distance[junctionID2] = vec
        junction_disp_dict[junctionID1]  = junction_distance
    str(junction_disp_dict)
traci.close()