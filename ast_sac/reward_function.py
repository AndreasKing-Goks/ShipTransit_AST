""" 
This module provides classes to construct reward function for the SAC.
"""

import numpy as np
from ast_sac.failure_modes import isMechanicalFailure, isNavigationFailure, isHitMapHorizon, isBeachingFailure, isBlackOutFailure
from simulator.obstacle import StaticObstacle

def non_terminal_state(pos):
    # reward = JONSWAP_reward_signal(pos) # JONSWAP func
    reward = 0
    return reward
    
def non_failure_terminal_state(pos, 
                               los_ct_error, 
                               heading_error, 
                               obstacles:StaticObstacle):
        
    # Reward 1 - Cross-track error
    reward_e_ct = -np.abs(los_ct_error)/250
        
    # Reward 2 - Miss heading alignment
    reward_e_hea = -np.abs(heading_error)
        
    # # Reward 3 - Miss distance from collision
    reward_col = obstacles.obstacles_distance(pos[0], pos[1])/1000
        
    return reward_e_ct + reward_e_hea + reward_col
    
def terminal_state(measured_shaft_rpm, 
                           los_ct_error, 
                           pos, 
                           engine_load, 
                           av_engine_load,
                           auto_pilot, 
                           obstacles):
    reward_terminal = 0
    status = " "
    done = False
    
    # Reaching the Endpoint
    # Get the route end_point
    n_route_end = auto_pilot.navigate.north[-1]
    e_route_end = auto_pilot.navigate.east[-1]
    arrival_radius = 10
    
    # Get the relative distance between the ship and endpoint
    relative_dist = (pos[0] - n_route_end)**2 + (pos[1] - e_route_end)**2
    
    if  relative_dist <= arrival_radius:
        reward_end =  1000
        reward_terminal += reward_end
        done = True
        status += " Reach endpoint "
        # print("Ship has reached destination point!")
    
    # Mechanical Failure
    if isMechanicalFailure(measured_shaft_rpm):
        reward_mf = -10
        reward_terminal += reward_mf
        done = True
        status += "Mechanical failure "
        # print('Ship experiencing Mechanical Failure!')
            
    # Navigation Failure
    if isNavigationFailure(los_ct_error):
        reward_nf = -10
        reward_terminal += reward_nf
        done = True
        status += "Navigation failure "
        # print('Ship experiencing Navigation Failure!')
        
    if isHitMapHorizon(pos):
        reward_hmhf = -100
        reward_terminal += reward_hmhf
        done = True
        status += "Hit map horizon"
            
    # Beaching Failure
    if isBeachingFailure(pos, obstacles):
        reward_bf = -100
        reward_terminal += reward_bf
        done = True
        status += "Beaching failure "
        # print('Ship is beached out!')
            
    # Black Out Failure
    if isBlackOutFailure(engine_load, av_engine_load):
        reward_bof = -10
        reward_terminal += reward_bof
        done = True
        status += "Blackout failure "
        # print('Ship experiencing Blackout Failure')
        
    if done == False:
        status = " Not in terminal state "
        
    return reward_terminal, done, status
        
def reward_function(pos, 
                    heading_error, 
                    measured_shaft_rpm, 
                    los_ct_error, 
                    engine_load, 
                    av_engine_load, auto_pilot, obstacles):
    
    # Compute non terminal state reward
    reward_non_terminal = non_terminal_state(pos)
    
    # Compute non failure terminal state reward
    reward_non_failure_terminal = non_failure_terminal_state(pos, los_ct_error, heading_error, obstacles)
    
    # Compute failure terminal state reward
    reward_terminal, done, status = terminal_state(measured_shaft_rpm, los_ct_error, pos, engine_load, av_engine_load, auto_pilot, obstacles)
    
    # Compute overall reward
    reward = reward_non_terminal + reward_non_failure_terminal + reward_terminal
    
    return reward, done, status