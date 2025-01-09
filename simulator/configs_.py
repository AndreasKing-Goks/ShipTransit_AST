""" 
This module provides classes that that can be used to store
subsystem configuration into a NamedTuple object.
"""

from typing import NamedTuple, List
from simulator.ship_engine import MachineryMode

###################################################################################################################
####################################### CONFIGURATION FOR SHIP MODEL ##############################################
###################################################################################################################

class ShipConfiguration(NamedTuple):
    dead_weight_tonnage: float
    coefficient_of_deadweight_to_displacement: float
    bunkers: float
    ballast: float
    length_of_ship: float
    width_of_ship: float
    added_mass_coefficient_in_surge: float
    added_mass_coefficient_in_sway: float
    added_mass_coefficient_in_yaw: float
    mass_over_linear_friction_coefficient_in_surge: float
    mass_over_linear_friction_coefficient_in_sway: float
    mass_over_linear_friction_coefficient_in_yaw: float
    nonlinear_friction_coefficient__in_surge: float
    nonlinear_friction_coefficient__in_sway: float
    nonlinear_friction_coefficient__in_yaw: float


class EnvironmentConfiguration(NamedTuple):
    current_velocity_component_from_north: float
    current_velocity_component_from_east: float
    wind_speed: float
    wind_direction: float


class SimulationConfiguration(NamedTuple):
    initial_north_position_m: float
    initial_east_position_m: float
    initial_yaw_angle_rad: float
    initial_forward_speed_m_per_s: float
    initial_sideways_speed_m_per_s: float
    initial_yaw_rate_rad_per_s: float
    integration_step: float
    simulation_time: float
    
    
###################################################################################################################
###################################### CONFIGURATION FOR MACHINERY MODEL ##########################################
###################################################################################################################


class LoadOnPowerSources(NamedTuple):
    load_on_main_engine: float
    load_on_electrical: float
    load_percentage_on_main_engine: float
    load_percentage_on_electrical: float

    
class MachineryModeParams(NamedTuple):
    main_engine_capacity: float
    electrical_capacity: float
    shaft_generator_state: str


class MachineryModes:
    def __init__(self, list_of_modes: List[MachineryMode]):
        self.list_of_modes = list_of_modes


class FuelConsumptionCoefficients(NamedTuple):
    a: float
    b: float
    c: float


class MachinerySystemConfiguration(NamedTuple):
    hotel_load: float
    machinery_modes: MachineryModes
    machinery_operating_mode: int
    rated_speed_main_engine_rpm: float
    linear_friction_main_engine: float
    linear_friction_hybrid_shaft_generator: float
    gear_ratio_between_main_engine_and_propeller: float
    gear_ratio_between_hybrid_shaft_generator_and_propeller: float
    propeller_inertia: float
    propeller_speed_to_torque_coefficient: float
    propeller_diameter: float
    propeller_speed_to_thrust_force_coefficient: float
    rudder_angle_to_sway_force_coefficient: float
    rudder_angle_to_yaw_force_coefficient: float
    max_rudder_angle_degrees: float
    specific_fuel_consumption_coefficients_me: FuelConsumptionCoefficients
    specific_fuel_consumption_coefficients_dg: FuelConsumptionCoefficients


class WithoutMachineryModelConfiguration(NamedTuple):
    thrust_force_dynamic_time_constant: float
    rudder_angle_to_sway_force_coefficient: float
    rudder_angle_to_yaw_force_coefficient: float
    max_rudder_angle_degrees: float


class SimplifiedPropulsionMachinerySystemConfiguration(NamedTuple):
    hotel_load: float
    machinery_modes: MachineryModes
    machinery_operating_mode: int
    specific_fuel_consumption_coefficients_me: FuelConsumptionCoefficients
    specific_fuel_consumption_coefficients_dg: FuelConsumptionCoefficients
    thrust_force_dynamic_time_constant: float
    rudder_angle_to_sway_force_coefficient: float
    rudder_angle_to_yaw_force_coefficient: float
    max_rudder_angle_degrees: float
    
    
class RudderConfiguration(NamedTuple):
    rudder_angle_to_sway_force_coefficient: float
    rudder_angle_to_yaw_force_coefficient: float
    max_rudder_angle_degrees: float
    

###################################################################################################################
#################################### CONFIGURATION FOR PID CONTROLLER #############################################
###################################################################################################################


class ThrottleControllerGains(NamedTuple):
    kp_ship_speed: float
    ki_ship_speed: float
    kp_shaft_speed: float
    ki_shaft_speed: float
    
    
class HeadingControllerGains(NamedTuple):
    kp: float
    kd: float
    ki: float
    
    
class HeadingControllerGains(NamedTuple):
    kp: float
    kd: float
    ki: float

class LosParameters(NamedTuple):
    radius_of_acceptance: float
    lookahead_distance: float
    integral_gain: float
    integrator_windup_limit: float


###################################################################################################################
####################################### CONFIGURATION FOR OBSERVER ################################################
###################################################################################################################