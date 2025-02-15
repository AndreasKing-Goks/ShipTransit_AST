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
from simulator.obstacle import StaticObstacle
from ast_sac.reward_function import reward_function


class ShipRLEnv(Env):
    """
    This class is the main class for the reinforcement learning environment based on the Ship-Transit Simulator
    """
    def __init__(self, 
                 ship_model: ShipModel,
                 auto_pilot: HeadingBySampledRouteController,
                 throttle_controller: EngineThrottleFromSpeedSetPoint,
                 obstacles: StaticObstacle,
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
        self.obstacles = obstacles
        
        # Set container for integration process
        self.init_intergrator_term = integrator_term
        self.integrator_term = self.init_intergrator_term
        
        self.init_times = times
        self.times = self.init_times
        
        # Ship drawing configuration
        self.ship_draw = ship_draw
        self.time_since_last_ship_drawing = time_since_last_ship_drawing
        
        # Define observation space [n_pos, e_pos, headings, forward speed, shaft_speed, los_e_ct, power_load] (7 states)
        self.observation_space = Box(
            low = np.array([-1000, -1000, -np.pi, -25, -3000, 0, 0], dtype=np.float32),
            high = np.array([1000, 1000, np.pi, 25, 3000, 100, 2000], dtype=np.float32),
        )
        
        # Define action space [route_point_n, route_point_e, desired_speed]
        self.action_space = Box(
            low = np.array([-200, -200, 5], dtype=np.float32),
            high = np.array([200, 200, 8], dtype=np.float32),
        )
        
        # Define initial state
        self.initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.state = self.initial_state
        
        # Desired speed (Not used)
        self.desired_forward_speed = 8.0
        
        # Initialize random seed
        self.seed()
    
    def reset(self):
        # Reset the simulator and the list
        self.ship_model.reset()
        self.integrator_term = self.init_intergrator_term
        self.times = self.init_times
        
        # Throttle controller reset
        self.throttle_controller.reset()
        
        # Autopilot controller reset
        self.auto_pilot.reset()
        
        # Reset the changing states into its initial state
        self.state = self.initial_state
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
        self.ship_model.store_simulation_data(throttle, 
                                              self.auto_pilot.get_cross_track_error(),
                                              self.auto_pilot.get_heading_error())
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
        heading_error = self.ship_model.simulation_results['heading error [deg]'][-1]
        measured_shaft_rpm = self.ship_model.simulation_results['propeller shaft speed [rpm]'][-1]
        los_ct_error = self.ship_model.simulation_results['cross track error [m]'][-1]
        engine_load = self.ship_model.simulation_results['power me [kw]'][-1]
        av_engine_load = self.ship_model.simulation_results['available power me [kw]'][-1]
        
        next_state = np.array([pos[0], pos[1], pos[2], self.ship_model.forward_speed, measured_shaft_rpm, los_ct_error, engine_load]) 
        
        reward, done, status = reward_function(pos, heading_error, measured_shaft_rpm, los_ct_error, engine_load, av_engine_load, self.auto_pilot, self.obstacles)
        
        # Step up the simulator
        self.ship_model.int.next_time()
        
        return next_state, reward, done, status
    
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility"""
        self.np_random, seed = seeding.np_random(seed)