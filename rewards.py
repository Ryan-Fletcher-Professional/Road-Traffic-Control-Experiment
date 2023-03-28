from math import e

# Calculates maximum number of waiting and approaching vehicles
def max_WAVE_reward_fn(traffic_signal):
    max_so_far = 0.0
    for lane in traffic_signal.lanes:
        num_wave = traffic_signal.sumo.lane.getLastStepHaltingNumber(lane) + traffic_signal.sumo.lane.getLastStepVehicleNumber(lane) 
        if(num_wave > max_so_far):
            max_so_far = num_wave
    if traffic_signal.last_measure < max_so_far:
        reward = traffic_signal.last_measure - max_so_far
    else:
        reward = traffic_signal.last_measure - max_so_far
    traffic_signal.last_measure = max_so_far
    return reward



# Calculates average speed
def my_reward_fn(traffic_signal):
    return traffic_signal.get_average_speed()

# Calculates an impedance-based reward
def impedance_reward(traffic_signal):
    beta = 5
    max_so_far = 0.0
   
    for lane in traffic_signal.lanes:
        impedance = (traffic_signal.sumo.lane.getLastStepHaltingNumber(lane) \
            + traffic_signal.sumo.lane.getLastStepVehicleNumber(lane)) / e ** (beta * traffic_signal.get_average_speed())
        if(impedance > max_so_far):
            max_so_far = impedance

    for lane in traffic_signal.out_lanes:
        impedance = (traffic_signal.sumo.lane.getLastStepVehicleNumber(lane) \
            + traffic_signal.sumo.lane.getLastStepHaltingNumber(lane))  / e ** (beta * traffic_signal.get_average_speed())
        if(impedance > max_so_far):
            max_so_far = impedance

    if traffic_signal.last_measure < max_so_far:
        reward = traffic_signal.last_measure - max_so_far
    else:
        reward = traffic_signal.last_measure - max_so_far
    traffic_signal.last_measure = max_so_far
    return reward
