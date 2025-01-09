""" 
This module outlines the definition of posible failure modes during the simulation
using the Ship Transit Simulator
"""

def isShaftBreak(shaft_rpm):
    ## Check this website:
    ## https://spicerparts.com/calculators/critical-speed-rpm-calculator
    shaft_rpm_max = 2000 # rpm
    return shaft_rpm >= shaft_rpm_max

def isLOSNavigationConvergenceFailure(x, x_trajectory):
    ## - Longer time convergence to the line segment
    return 1

def isLOSNavigationOscillationFailure(x, x_trajectory):
    ## - Oscillation or even divergence from the line segment
    return 1