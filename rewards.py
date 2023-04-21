from math import e
from asyncio import wait, get_event_loop, sleep

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

# no reward for fixed time signalling
def no_reward(traffic_signal):
    return 0


# Calculates average speed
def my_reward_fn(traffic_signal):
    return traffic_signal.get_average_speed()

async def calc_impedance_reward(traffic_signal):
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
    await sleep(0.01)
    return max_so_far

async def async_full_coordinated_impedance_reward(traffic_signal, loop):
    all_traffic_signals = traffic_signal.env.traffic_signals
    async_fns = {}
    for agent in all_traffic_signals:
        curr_ts = all_traffic_signals[agent]
        async_fns[agent] = loop.create_task(calc_impedance_reward(curr_ts))
    await wait(list(async_fns.values()))

# Calculates a multi agent impedance-based reward with full coordination among agents
def fully_coordinated_impedance_reward_async(traffic_signal):
    loop = get_event_loop()
    future = loop.run_until_complete(async_full_coordinated_impedance_reward(traffic_signal, loop))
    loop.close()
    return future
    #if traffic_signal.last_measure < f:
    #    reward = traffic_signal.last_measure - max_so_far
    #else:
    #    reward = traffic_signal.last_measure - max_so_far
    #traffic_signal.last_measure = max_so_far
    #return reward

# Calculates a multi agent impedance-based reward with full coordination among agents
def fully_coordinated_impedance_reward(traffic_signal):
    beta = 5
    max_so_far = 0.0
   
    all_traffic_signals = traffic_signal.env.traffic_signals
    for agent in all_traffic_signals:
        curr_ts = all_traffic_signals[agent]
        for lane in curr_ts.lanes:
            impedance = (curr_ts.sumo.lane.getLastStepHaltingNumber(lane) \
                + curr_ts.sumo.lane.getLastStepVehicleNumber(lane)) / e ** (beta * curr_ts.get_average_speed())
            if(impedance > max_so_far):
                max_so_far = impedance

        for lane in curr_ts.out_lanes:
            impedance = (curr_ts.sumo.lane.getLastStepVehicleNumber(lane) \
                + curr_ts.sumo.lane.getLastStepHaltingNumber(lane))  / e ** (beta * curr_ts.get_average_speed())
            if(impedance > max_so_far):
                max_so_far = impedance

    if traffic_signal.last_measure < max_so_far:
        reward = traffic_signal.last_measure - max_so_far
    else:
        reward = traffic_signal.last_measure - max_so_far
    traffic_signal.last_measure = max_so_far
    return reward

# Calculates a single agent impedance-based reward
def single_agent_impedance_reward(traffic_signal):
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
