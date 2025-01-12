""" 
This module outlines the definition of posible failure modes during the simulation
using the Ship Transit Simulator
"""

def isMechanicalFailure(measured_shaft_rpm):
    ## Check this website:
    ## https://spicerparts.com/calculators/critical-speed-rpm-calculator
    shaft_rpm_max = 2000 # rpm
    return measured_shaft_rpm > shaft_rpm_max

def isNavigationFailure(e_ct):
    ## Ship deviates off the course beyond tolerance
    e_tolerance = 20
    return e_ct > e_tolerance

def isBeachingFailure(pos): # Pos = [n_pos, e_pos]
    ## Ship stuck in the terminal position state
    terminal_position = []
    return False

def isBlackOutFailure(power_load, available_power):
    ## Diesel engine overloaded
    return power_load > available_power