""" 
This module provides classes to construct reward function for the SAC.
"""

import numpy as np
from ast_sac.failure_modes import isMechanicalFailure, isNavigationFailure, isBeachingFailure, isBlackOutFailure
from simulator.obstacle import StaticObstacle

def non_terminal_state(pos):
    # reward = JONSWAP_reward_signal(pos) # JONSWAP func
    reward = 1
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
    reward_col = obstacles.obstacles_distance(pos[0], pos[1])
        
    return reward_e_ct + reward_e_hea + reward_col
    
def failure_terminal_state(measured_shaft_rpm, 
                           los_ct_error, 
                           pos, 
                           engine_load, 
                           av_engine_load, 
                           obstacles):
    reward_terminal = 0
    done = False
    
    # Reaching the Endpoint
    
    
    # Mechanical Failure
    if isMechanicalFailure(measured_shaft_rpm):
        reward_mf = 10
        reward_terminal += reward_mf
        done = True
        print('MF')
            
    # Navigation Failure
    if isNavigationFailure(los_ct_error):
        reward_nf = 10
        reward_terminal += reward_nf
        done= True
        print('NF')
            
    # Beaching Failure
    if isBeachingFailure(pos, obstacles):
        reward_bf = 10
        reward_terminal += reward_bf
        done = True
        print('BF')
            
    # Black Out Failure
    if isBlackOutFailure(engine_load, av_engine_load):
        reward_bof = 10
        reward_terminal += reward_bof
        done = True
        print('BOF')
        
    return reward_terminal, done
        
def reward_function(pos, heading_error, measured_shaft_rpm, los_ct_error, engine_load, av_engine_load, obstacles):
    
    # Compute non terminal state reward
    reward_non_terminal = non_terminal_state(pos)
    
    # Compute non failure terminal state reward
    reward_non_failure_terminal = non_failure_terminal_state(pos, los_ct_error, heading_error, obstacles)
    
    # Compute failure terminal state reward
    reward_failure_terminal, done = failure_terminal_state(measured_shaft_rpm, los_ct_error, pos, engine_load, av_engine_load, obstacles)
    
    # Compute overall reward
    reward = reward_non_terminal + reward_non_failure_terminal + reward_failure_terminal
    
    return reward, done