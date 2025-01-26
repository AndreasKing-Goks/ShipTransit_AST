""" 
This modules provide classes for the reinforcement learning environment based on the Ship-Transit Simulator
"""

from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from collections import defaultdict
import numpy as np
import torch

from simulator.ship_model import ShipModel
from simulator.controllers import EngineThrottleFromSpeedSetPoint, HeadingBySampledRouteController
from simulator.LOS_guidance import NavigationSystem
from ast_sac.reward_function import reward_function


class ShipRLEnv(Env):
    """
    This class is the main class for the reinforcement learning environment based on the Ship-Transit Simulator
    """
    def __init__(self, 
                 ship_model: ShipModel,
                 auto_pilot: HeadingBySampledRouteController,
                 throttle_controller: EngineThrottleFromSpeedSetPoint,
                 integrator_term:list,
                 times:list,
                 ship_draw:bool,
                 time_since_last_ship_drawing:float):
    # def __init__(self):
        super().__init__()
        # Store the ship model, controllers, and reward function instances in self variables
        self.ship_model = ship_model
        self.auto_pilot = auto_pilot
        self.throttle_controller = throttle_controller
        
        # Set container for integration process
        self.integrator_term = integrator_term
        self.times = times
        
        # Ship drawing configuration
        self.ship_draw = ship_draw
        self.time_since_last_ship_drawing = time_since_last_ship_drawing
        
        # Define observation space [n_pos, e_pos, headings, forward speed, shaft_speed, los_e_ct, power_load] (7 states)
        self.observation_space = Box(
            low = np.array([-2000, -2000, -np.pi, -25, -3000, 0, 0], dtype=np.float32),
            high = np.array([2000, 2000, np.pi, 25, 3000, 100, 2000], dtype=np.float32),
        )
        
        # Define action space [route_point_n, route_point_e, desired_speed]
        self.action_space = Box(
            low = np.array([-100, -100, -5], dtype=np.float32),
            high = np.array([100, 100, 10], dtype=np.float32),
        )
        
        # Define initial state
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Desired speed
        self.desired_forward_speed = 10.0
        
        # # Initialize navigation system
        # self.nav_sys = NavigationSystem(
        #     ship_model = self.ship_model,
        #     route = self.auto_pilot.route,
        #     times = self.times
        # )
        
        # Initialize random seed
        self.seed()
    
    def reset(self):
        # Reset the simulation results dictionary
        self.ship_model.simulation_results = defaultdict(list)
        
        # Reset the route countainer
        self.auto_pilot.navigate.load_waypoints(self.auto_pilot.navigate.route)
        
        # Reset the changing states into its initial state
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self.state
    
    def step(self, action, sample_flag):       
        # Measure ship position and speed
        north_position = self.ship_model.north
        east_position = self.ship_model.east
        heading = self.ship_model.yaw_angle
        forward_speed = self.ship_model.forward_speed
        
        if sample_flag:
            route_shift_n, route_shift_e, desired_forward_speed = action
        
            # Update route_point based on the action
            route_shifts = route_shift_n, route_shift_e
            self.auto_pilot.update_route(route_shifts)
            
            # Update desired_forward_speed based on the action
            self.desired_forward_speed = desired_forward_speed
        
        # Find appropriate rudder angle and engine throttle
        rudder_angle = self.auto_pilot.rudder_angle_from_sampled_route(
            north_position=north_position,
            east_position=east_position,
            heading=heading
        )
        
        throttle = self.throttle_controller.throttle(
            speed_set_point = self.desired_forward_speed,
            measured_speed = forward_speed,
            measured_shaft_speed = forward_speed
        )
        
        # Update and integrate differential equations for current time step
        self.ship_model.store_simulation_data(throttle)
        self.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        self.ship_model.integrate_differentials()
        
        self.integrator_term.append(self.auto_pilot.navigate.e_ct_int)
        self.times.append(self.ship_model.int.time)        
        
        # Apply ship drawing (set as optional function)
        if self.ship_draw:
            if self.time_since_last_ship_drawing > 30:
                self.ship_model.ship_snap_shot()
                self.time_since_last_ship_drawing = 0
            self.time_since_last_ship_drawing += self.ship_model.int.dt
        
        # Compute reward
        pos = [self.ship_model.north, self.ship_model.east, self.ship_model.yaw_angle]
        heading_pos_target = 0
        measured_shaft_rpm = self.ship_model.simulation_results['propeller shaft speed [rpm]'][-1]
        los_ct_error = self.auto_pilot.navigate.e_ct
        engine_load = self.ship_model.simulation_results['power electrical [kw]'][-1]
        av_engine_load = self.ship_model.simulation_results['available power electrical [kw]'][-1]
        
        next_state = np.array([pos[0], pos[1], pos[2], self.ship_model.forward_speed, measured_shaft_rpm, los_ct_error, engine_load]) 
        
        reward, done = reward_function(pos, heading_pos_target, measured_shaft_rpm, los_ct_error, engine_load, av_engine_load)
        
        # Step up the simulator
        self.ship_model.int.next_time()
        
        return next_state, reward, done
    
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility"""
        self.np_random, seed = seeding.np_random(seed)