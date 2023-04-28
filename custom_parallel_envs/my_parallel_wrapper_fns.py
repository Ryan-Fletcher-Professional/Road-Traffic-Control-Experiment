"""SUMO Environment for Traffic Signal Control."""
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .my_sumo_environment_PZ import MySumoEnvironmentPZ


LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ


def env(sumo_env, **kwargs):
    """Instantiate a PettingoZoo environment."""
    env = MySumoEnvironmentPZ(sumo_env, **kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


my_parallel_env = parallel_wrapper_fn(env)