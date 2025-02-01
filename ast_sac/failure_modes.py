""" 
This module outlines the definition of posible failure modes during the simulation
using the Ship Transit Simulator
"""

import numpy as np
from simulator.obstacle import StaticObstacle

def isMechanicalFailure(measured_shaft_rpm):
    ## Check this website:
    ## https://spicerparts.com/calculators/critical-speed-rpm-calculator
    shaft_rpm_max = 2000 # rpm
    return np.abs(measured_shaft_rpm) > shaft_rpm_max

def isNavigationFailure(e_ct):
    ## Ship deviates off the course beyond tolerance
    e_tolerance = 250
    return np.abs(e_ct) > e_tolerance

def isBeachingFailure(pos,
                      obstacle:StaticObstacle): # Pos = [n_pos, e_pos]
    ## Ship stuck in the terminal position state
    n_ship, e_ship, _ = pos
    return obstacle.obstacles_inside(n_ship, e_ship)

def isBlackOutFailure(power_load, available_power):
    ## Diesel engine overloaded
    return power_load > available_power