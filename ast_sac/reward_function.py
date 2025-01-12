""" 
This module provides classes to construct reward function for the SAC.
"""

import numpy as np
from failure_modes import isMechanicalFailure, isNavigationFailure, isBeachingFailure, isBlackOutFailure

from typing import NamedTuple, List


class RFInput(NamedTuple):
    position: float
    measured_shaft_rpm: float
    engine_load: float
    los_ct_error: float


class RewardFunction:
    def __init__(self, input: RFInput):
        super().__init__(RFInput)
    
    def non_terminal_state(self, pos):
        # reward = JONSWAP_reward_signal(pos) # JONSWAP func
        reward = 0
        return reward
    
    def non_failure_terminal_state(self, pos):
        # Unpack pos
        n_pos, e_pos, heading_pos = pos
        
        # Reward 1 - Cross-track error
        reward_1 = input.los_ct_error
        
        # Reward 2 - Miss heading alignment
        reward_2 = heading_pos - heading_pos_target
        
        # # Reward 3 - Miss distance from collision
        # reward_3 = np.sqrt((n_pos - n_pos_obs)**2 +(e_pos - e_pos_obs)**2)
        
        return reward_1 + reward_2 
    
    def failure_terminal_state(self):
        # Mechanical Failure
        if isMechanicalFailure(input.measured_shaft_rpm):
            reward_mf = 0
            
        # Navigation Failure
        if isNavigationFailure(input.los_ct_error):
            reward_nf = 0
            
        # Beaching Failure
        if isBeachingFailure(input.position):
            reward_bf = 0
            
        # Black Out Failure
        if isBlackOutFailure(input.engine_load):
            reward_bof = 0
        
        return reward_mf + reward_nf + reward_bf + reward_bof
        
        return
    