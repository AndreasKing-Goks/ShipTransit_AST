import numpy as np
import math


from simulator.configs_ import ThrottleControllerGains, HeadingControllerGains, LosParameters
from simulator.LOS_guidance import NavigationSystem 

class PiController:
    def __init__(self, kp: float, ki: float, time_step: float, initial_integral_error=0):
        self.kp = kp
        self.ki = ki
        self.error_i = initial_integral_error
        self.time_step = time_step

    def pi_ctrl(self, setpoint, measurement, *args):
        ''' Uses a proportional-integral control law to calculate a control
            output. The optional argument is an 2x1 array and will specify lower
            and upper limit for error integration [lower, upper]
        '''
        error = setpoint - measurement
        error_i = self.error_i + error * self.time_step
        if args:
            error_i = self.sat(error_i, args[0], args[1])
        self.error_i = error_i
        return error * self.kp + error_i * self.ki

    @staticmethod
    def sat(val, low, hi):
        ''' Saturate the input val such that it remains
        between "low" and "hi"
        '''
        return max(low, min(val, hi))


class PidController:
    def __init__(self, kp: float, kd: float, ki: float, time_step: float):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.error_i = 0
        self.prev_error = 0
        self.time_step = time_step

    def pid_ctrl(self, setpoint, measurement, *args):
        ''' Uses a proportional-derivative-integral control law to calculate
            a control output. The optional argument is a 2x1 array and will
            specify lower and upper [lower, upper] limit for error integration
        '''
        error = setpoint - measurement
        d_error = (error - self.prev_error) / self.time_step
        error_i = self.error_i + error * self.time_step
        if args:
            error_i = self.sat(error_i, args[0], args[1])
        self.prev_error = error
        self.error_i = error_i
        return error * self.kp + d_error * self.kd + error_i * self.ki

    @staticmethod
    def sat(val, low, hi):
        ''' Saturate the input val such that it remains
        between "low" and "hi"
        '''
        return max(low, min(val, hi))
    
    
###################################################################################################################
################################## DESCENDANT CLASS FROM THE BASE CONTROLLER ######################################
###################################################################################################################


class EngineThrottleFromSpeedSetPoint:
    """
    Calculates throttle setpoint for power generation based on the ship´s speed, the propeller shaft speed
    and the desires ship speed.
    """

    def __init__(
            self,
            gains: ThrottleControllerGains,
            max_shaft_speed: float,
            time_step: float,
            initial_shaft_speed_integral_error: float
    ):
        self.ship_speed_controller = PiController(
            kp=gains.kp_ship_speed, ki=gains.ki_ship_speed, time_step=time_step
        )
        self.shaft_speed_controller = PiController(
            kp=gains.kp_shaft_speed,
            ki=gains.ki_shaft_speed,
            time_step=time_step,
            initial_integral_error=initial_shaft_speed_integral_error
        )
        self.max_shaft_speed = max_shaft_speed

    def throttle(self, speed_set_point, measured_speed, measured_shaft_speed):
        desired_shaft_speed = self.ship_speed_controller.pi_ctrl(setpoint=speed_set_point, measurement=measured_speed)
        desired_shaft_speed = self.ship_speed_controller.sat(val=desired_shaft_speed, low=0, hi=self.max_shaft_speed)
        throttle = self.shaft_speed_controller.pi_ctrl(setpoint=desired_shaft_speed, measurement=measured_shaft_speed)
        return self.shaft_speed_controller.sat(val=throttle, low=0, hi=1.1)


class ThrottleFromSpeedSetPointSimplifiedPropulsion:
    """
    Calculates throttle setpoint for power generation based on the ship´s speed, the propeller shaft speed
    and the desires ship speed.
    """

    def __init__(
            self,
            kp: float,
            ki: float,
            time_step: float,
    ):
        self.ship_speed_controller = PiController(
            kp=kp, ki=ki, time_step=time_step
        )

    def throttle(self, speed_set_point, measured_speed):
        throttle = self.ship_speed_controller.pi_ctrl(setpoint=speed_set_point, measurement=measured_speed)
        return self.ship_speed_controller.sat(val=throttle, low=0, hi=1.1)
    
    
class HeadingByReferenceController:
    def __init__(self, gains: HeadingControllerGains, time_step, max_rudder_angle):
        self.ship_heading_controller = PidController(kp=gains.kp, kd=gains.kd, ki=gains.ki, time_step=time_step)
        self.max_rudder_angle = max_rudder_angle

    def rudder_angle_from_heading_setpoint(self, heading_ref: float, measured_heading: float):
        ''' This method finds a suitable rudder angle for the ship to
            sail with the heading specified by "heading_ref" by using
            PID-controller. The rudder angle is saturated according to
            |self.rudder_ang_max|. The mathod should be called from within
            simulation loop if the user want the ship to follow a specified
            heading reference signal.
        '''
        rudder_angle = -self.ship_heading_controller.pid_ctrl(setpoint=heading_ref, measurement=measured_heading)
        return self.ship_heading_controller.sat(rudder_angle, -self.max_rudder_angle, self.max_rudder_angle)


class HeadingByRouteController:
    def __init__(
            self, route_name,
            heading_controller_gains: HeadingControllerGains,
            los_parameters: LosParameters,
            time_step: float,
            max_rudder_angle: float,
    ):
        self.heading_controller = HeadingByReferenceController(
            gains=heading_controller_gains, time_step=time_step, max_rudder_angle=max_rudder_angle
        )
        self.navigate = NavigationSystem(
            route=route_name,
            radius_of_acceptance=los_parameters.radius_of_acceptance,
            lookahead_distance=los_parameters.lookahead_distance,
            integral_gain=los_parameters.integral_gain,
            integrator_windup_limit=los_parameters.integrator_windup_limit
        )
        self.next_wpt = 1
        self.prev_wpt = 0

    def rudder_angle_from_route(self, north_position, east_position, heading):
        ''' This method finds a suitable rudder angle for the ship to follow
            a predefined route specified in the "navigate"-instantiation of the
            "NavigationSystem"-class.
        '''
        self.next_wpt, self.prev_wpt = self.navigate.next_wpt(self.next_wpt, north_position, east_position)
        psi_d = self.navigate.los_guidance(self.next_wpt, north_position, east_position)
        return self.heading_controller.rudder_angle_from_heading_setpoint(heading_ref=psi_d, measured_heading=heading)