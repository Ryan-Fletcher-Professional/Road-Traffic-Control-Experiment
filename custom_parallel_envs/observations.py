"""Our Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from sumo_rl import ObservationFunction, TrafficSignal

class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )

class CompressedObservationFunction(ObservationFunction):
    """Compressed observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize compressed observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        density_mean = sum(density) / len(density)
        density_stdev = np.std(density)
        queue = self.ts.get_lanes_queue()
        queue_mean = sum(queue) / len(queue)
        queue_stdev = np.std(queue)
        agent_total_waiting_time = sum(self.ts.get_accumulated_waiting_time_per_lane())
        observation = np.array(phase_id + min_green + [density_mean, density_stdev] + [queue_mean, queue_stdev] + [agent_total_waiting_time], dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 + 2 + 1, dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 + 2 + 1, dtype=np.float32),
        )
