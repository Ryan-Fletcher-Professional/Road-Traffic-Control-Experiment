from math import e

# Calculates maximum number of waiting and approaching vehicles
def max_WAVE_reward_fn(traffic_signal):
    max_so_far = 0.0
    for lane in traffic_signal.lanes:
        num_wave = traffic_signal.sumo.lane.getLastStepHaltingNumber(lane) + traffic_signal.sumo.lane.getLastStepVehicleNumber(lane) 
        if(num_wave > max_so_far):
            max_so_far = num_wave
    return max_so_far

# Calculates average speed
def my_reward_fn(traffic_signal):
    return traffic_signal.get_average_speed

# Calculates an impedance-based reward
def impedance_reward(traffic_signal):
    beta = 0.5
    halted_vehicles=sum(traffic_signal.sumo.lane.getLastStepHaltingNumber(lane) for lane in traffic_signal.lanes)
    approaching_vehicles=sum(traffic_signal.sumo.lane.getLastStepVehicleNumber(lane) for lane in traffic_signal.lanes)
    outgoing_vehicles=sum(traffic_signal.sumo.lane.getLastStepVehicleNumber(lane) for lane in traffic_signal.out_lanes)
    vehicle_count=halted_vehicles+approaching_vehicles+outgoing_vehicles
    return vehicle_count / e ** (beta * traffic_signal.get_average_speed())
