""" 
This modules provide classes for the reinforcement learning environment based on the Ship-Transit Simulator
"""

from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from collections import defaultdict
import numpy as np
import torch

from simulator.ship_model import ShipModelAST
from simulator.controllers import EngineThrottleFromSpeedSetPoint, HeadingBySampledRouteController
from simulator.obstacle import StaticObstacle
# from ast_sac.reward_function import reward_function


class ShipRLEnv(Env):
    """
    This class is the main class for the reinforcement learning environment based on the Ship-Transit Simulator
    """
    def __init__(self, 
                 ship_model: ShipModelAST,
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
            low = np.array([0, 0, -np.pi, -25, -3000, 0, 0], dtype=np.float32),
            high = np.array([10000, 10000, np.pi, 25, 3000, 100, 2000], dtype=np.float32),
        )
        
        # # Define action space [route_point_n, route_point_e, desired_speed]
        # self.action_space = Box(
        #     low = np.array([-1000, -1000, 5], dtype=np.float32),
        #     high = np.array([1000, 1000, 8], dtype=np.float32),
        # )
        
        # Define action space [route_point_shift, desired_speed]
        self.action_space = Box(
            low = np.array([-10000/np.sqrt(2), 0], dtype=np.float32),
            high = np.array([10000/np.sqrt(2), 8], dtype=np.float32),
        )
        
        # Define initial state
        self.initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.state = self.initial_state
        
        # Expected forward speed (Normal cruising speed, predetermined)
        self.expected_forward_speed = 8.0
        
        # Initialize random seed
        self.seed()
        
        # Reward Function parameters
        self.e_tolerance = 250
        AB_distance_n = self.auto_pilot.navigate.north[-1] - self.auto_pilot.navigate.north[0]
        AB_distance_e = self.auto_pilot.navigate.east[-1] - self.auto_pilot.navigate.east[0]
        self.AB_distance = np.sqrt(AB_distance_n ** 2 + AB_distance_e ** 2)
        self.AB_alpha = np.arctan2(AB_distance_e, AB_distance_n)
        self.AB_beta = np.pi/2 - self.AB_alpha 
        self.prev_route_coordinate = None
    
    def reset(self):
        # Reset the simulator and the list
        self.ship_model.reset()
        self.integrator_term = self.init_intergrator_term
        self.times = self.init_times
        
        # Throttle controller reset
        self.throttle_controller.reset()
        
        # Autopilot controller reset
        self.auto_pilot.reset()
        
        # Route coordinate hold variable
        self.prev_route_coordinate = None
        
        # Reset the changing states into its initial state
        self.state = self.initial_state
        return self.state
    
    def step(self, 
             simu_input, 
             action_to_simu_input,
             sampling_time_record,
             debug=False):       
        # Measure ship position and speed
        north_position = self.ship_model.north
        east_position = self.ship_model.east
        heading = self.ship_model.yaw_angle
        forward_speed = self.ship_model.forward_speed
        
        if debug:
            print("---------------")
            print('Debug Mode 1')
            print("Previous route coordinate =",self.prev_route_coordinate)
            print("---------------")
        
        if action_to_simu_input:
            # Unpack simulation input
            # if len(simu_input) != 3:
            #     print(f" Debug: Invalid simu_input = {simu_input}, length = {len(simu_input)}")
            route_coord_n, route_coord_e, desired_forward_speed = simu_input
        
            # Update route_point based on the action
            route_coordinate = route_coord_n, route_coord_e
            self.auto_pilot.update_route(route_coordinate)
            
            # Update desired_forward_speed based on the action
            self.desired_forward_speed = desired_forward_speed
            
            # Store the sampled route coordinate to the holder variable
            self.prev_route_coordinate = route_coordinate
        
        # If it is not the time to use action as simulation input, use saved route coordinate
        else:
            if debug:
                print("---------------")
                print('Debug Mode 2')
                print("Previous route coordinate =",self.prev_route_coordinate)
                print("---------------")
            route_coordinate = self.prev_route_coordinate
        
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
                                              rudder_angle,
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
        power_load = self.ship_model.simulation_results['power me [kw]'][-1]
        available_power_load = self.ship_model.simulation_results['available power me [kw]'][-1]
        
        next_state = np.array([pos[0], pos[1], pos[2], self.ship_model.forward_speed, measured_shaft_rpm, los_ct_error, power_load]) 
        
        reward, done, status = self.reward_function(pos,
                                                    route_coordinate,
                                                    los_ct_error,
                                                    power_load,
                                                    available_power_load,
                                                    heading_error,
                                                    measured_shaft_rpm,
                                                    sampling_time_record)
        
        # Step up the simulator
        self.ship_model.int.next_time()
        
        return next_state, reward, done, status
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility"""
        self.np_random, seed = seeding.np_random(seed)
        
###################################################################################################################
############################################### FAILURE MODES #####################################################
###################################################################################################################
        
    def is_pos_outside_horizon(self, n_pos, e_pos, margin=100):
        ''' Checks if the ship positions are outside the map horizon. 
            Map horizons are determined by the edge point of the determined route point. 
            Allowed additional margin are default 100 m for North and East boundaries.
            Only works with start to end route points method (Two initial points).
        '''
        n_route_point = self.auto_pilot.navigate.north
        e_route_point = self.auto_pilot.navigate.east
        
        # min_bound and max_bound
        n_route_bound = [n_route_point[0]-margin, n_route_point[-1]+margin]
        e_route_bound = [e_route_point[0]-margin, e_route_point[-1]+margin]
        
        # Check if position is outside bound
        outside_n = n_pos <= n_route_bound[0] or n_pos >= n_route_bound[1]
        outside_e = e_pos <= e_route_bound[0] or e_pos >= e_route_bound[1]
        
        is_outside = outside_n or outside_e
        
        return is_outside
    
    def is_pos_inside_obstacles(self, n_pos, e_pos):
        ''' Checks if the sampled routes are inside any obstacle 
        '''
        
        distances_squared = (n_pos - np.array(self.obstacles.n_obs)) ** 2 + (e_pos - np.array(self.obstacles.e_obs)) ** 2
        radii_squared = np.array(self.obstacles.r_obs) ** 2

        if np.any(distances_squared <= radii_squared):  # True if inside any obstacle
          return True
      
        return False
    
    def is_too_slow(self, recorded_time):
        ''' Expected time = Scaling Factor * Start-to-end point distances / Expected forward speed
            
            Expected time is defined as the time needed to travel form start to end
            point with the expected forward speed.
            
            If the ship couldn't travelled enough distances for the sampling within the
            expected time, it is deemed as a termination and give huge negative rewards.
            
            The reason is this occurence most likely happened because the sampled speed
            is too slow 
        '''
        scale_factor = 1.0
        
        time_expected = scale_factor * self.AB_distance / self.expected_forward_speed
        
        return recorded_time > time_expected
    
    def is_mechanical_failure(self, measured_shaft_rpm):
        ## Check this website:
        ## https://spicerparts.com/calculators/critical-speed-rpm-calculator
        shaft_rpm_max = 2000 # rpm
        return np.abs(measured_shaft_rpm) > shaft_rpm_max

    def is_navigation_failure(self, e_ct):
        ## Ship deviates off the course beyond tolerance
        return np.abs(e_ct) > self.e_tolerance

    def is_blackout_failure(self,
                            power_load, 
                            available_power_load):
        ## Diesel engine overloaded
        return power_load > available_power_load
    
###################################################################################################################
############################################## REWARD FUNCTION ####################################################
###################################################################################################################

    def non_terminal_state_reward(self, 
                                  pos,
                                  los_ct_error,
                                  heading_error):
        ''' Reward per simulation time step should be in order 10**0
            As it will be accumulated over time. Negative reward proportional
            to positive reward shall be added for each simulation time step as
            well.
        '''
        # print(pos)
        n_pos, e_pos, _ = pos
        
        # Directional JONSWAP reward
        reward_jonswap = 1 # NEED ALGORITHM
        
        # Cross-track error reward        
        # Normalized cross_track error by the tolerance
        reward_e_ct = -np.abs(los_ct_error) / self.e_tolerance 
        
        # Miss heading alignment reward
        # In radian
        # Normalized by maximum rudder angle
        reward_e_hea = -np.abs(heading_error) / self.ship_model.ship_machinery_model.rudder_ang_max
        
        # Miss distance from collision reward
        # Normalized by start_to_end distance (NOT FINAL)
        reward_col = self.obstacles.obstacles_distance(n_pos, e_pos) / self.AB_distance
        
        return reward_jonswap + reward_e_ct + reward_e_hea + reward_col
    
    def terminal_state_reward(self,
                              pos,
                              route_coordinate,
                              los_ct_error,
                              power_load,
                              available_power_load,
                              measured_shaft_rpm,
                              recorded_time):
        ## Initial value
        reward_terminal = 0
        status = " "
        done = False
        
        n_pos, e_pos, _ = pos
        
        # print(route_coordinate)
        n_route, e_route = route_coordinate
        
        ## Reward for reaching the end point
        # Get the relative distance between the ship and the end point
        n_route_end = self.auto_pilot.navigate.north[-1]
        e_route_end = self.auto_pilot.navigate.east[-1]
        relative_dist = np.sqrt((n_pos - n_route_end)**2 + (e_pos - e_route_end)**2)
        
        # Check if the ship arrive at the end point
        arrival_radius = 10 # Arrival radius zone
        if relative_dist <= arrival_radius:
            reward_terminal += 1000
            done = True
            status += "|Reach endpoint|"
            
        ## Reward for ship hitting map horizon
        # print(pos)
        if self.is_pos_outside_horizon(n_pos, e_pos):
            reward_terminal += -1000
            done = True
            status += "|Map horizon hit failure|"
                    
        ## Reward for ship hitting obstacles
        if self.is_pos_inside_obstacles(n_pos, e_pos):
            reward_terminal += -1000
            done = True
            status += "|Collision failure|"
            
        ## Reward for route action sampled inside obstacles or outside map horizon
        if self.is_pos_outside_horizon(n_route, e_route) or\
            self.is_pos_inside_obstacles(n_route, e_route):
            reward_terminal += -1000
            done = True
            status += "|Route point is sampled in terminal state|"
        
        ## Reward for unnecessary slow ship movement
        if self.is_too_slow(recorded_time):
            reward_terminal += -1000
            done = True
            status += "|Slow progress failure|"
        
        ## Reward for Mechanical Failure 
        if self.is_mechanical_failure(measured_shaft_rpm):
            reward_terminal += -1000
            done = True
            status += "|Mechanical failure|"
        
        ## Reward for Navigation Failure    
        if self.is_navigation_failure(los_ct_error):
            reward_terminal += -1000
            done = True
            status += "|Navigation failure|"
        
        ## Reward for Blackout Failure    
        if self.is_blackout_failure(power_load, available_power_load):
            reward_terminal += -1000
            done = True
            status += "|Blackout failure|"
        
        if done == False:
            status = "|Not in terminal state|"
        
        return reward_terminal, done, status
    
    def reward_function(self,
                        pos,
                        route_coordinate,
                        los_ct_error,
                        power_load,
                        available_power_load,
                        heading_error,
                        measured_shaft_rpm,
                        recorded_time):
        
        # Compute non terminal state reward
        reward_non_terminal = self.non_terminal_state_reward(pos, 
                                                             los_ct_error, 
                                                             heading_error)
        
        # Compute termial state reward
        reward_terminal, done, status = self.terminal_state_reward(pos,
                                                                   route_coordinate,
                                                                   los_ct_error,
                                                                   power_load,
                                                                   available_power_load,
                                                                   measured_shaft_rpm,
                                                                   recorded_time)
        
        # Compute overal reward
        reward = reward_non_terminal + reward_terminal
        
        return reward, done, status