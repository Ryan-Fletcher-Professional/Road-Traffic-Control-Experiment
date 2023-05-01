#! /usr/bin/bash

python3 multi_network_runner_uncompressed_obs.py "sumo_networks/TUM-VT/ingolstadt_24h.net.xml/ingolstadt_24h.net.xml" "sumo_networks/TUM-VT/bicycle_routes_24h.rou.xml,sumo_networks/TUM-VT/motorized_routes_2020-09-16_24h.rou.xml" "--additional-files sumo_networks/TUM-VT/ingolstadt.poly.xml,sumo_networks/TUM-VT/tlLogics_WAUT_2020-09-16_24h.add.xml --step-length 5 --ignore-junction-blocker 15 --device.rerouting.probability 0.2" 21600 5 
