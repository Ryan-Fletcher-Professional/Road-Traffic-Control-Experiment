"""Custome SUMO Environment for Traffic Signal Control."""
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from .observations import DefaultObservationFunction
from sumo_rl import SumoEnvironment

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

class MySumoEnvironment(SumoEnvironment): 
    """Custom Sumo Environment implemented to gain access to functionality not accessible without subclassing"""
    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        CO2_emissions = [self.sumo.vehicle.getCO2Emission(vehicle) * self.delta_time for vehicle in vehicles]
        fuel_consumption = [self.sumo.vehicle.getFuelConsumption(vehicle) * self.delta_time for vehicle in vehicles]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
            "system_total_CO2_emissions_mg": sum(CO2_emissions),
            "fuel_consumption_mg": sum(fuel_consumption)
        }

