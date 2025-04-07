""" 
This modules provide classes for the reinforcement learning environment based on the Ship-Transit Simulator
"""

from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from collections import defaultdict
import numpy as np
import torch

from simulator.ship_model import ShipModel, ShipModelAST
from simulator.controllers import EngineThrottleFromSpeedSetPoint, HeadingByRouteController, HeadingBySampledRouteController
from simulator.obstacle import StaticObstacle, PolygonObstacle

from dataclasses import dataclass
from typing import Union, List
# from ast_sac.reward_function import reward_function

@dataclass
class ShipAssets:
    ship_model: Union[ShipModel, ShipModelAST]
    throttle_controller: EngineThrottleFromSpeedSetPoint
    auto_pilot: Union[HeadingBySampledRouteController, HeadingByRouteController]
    desired_forward_speed: float
    integrator_term: List[float]
    time_list: List[float]
    type_tag: str
    

class MultiShipRLEnv(Env):
    """
    This class is the main class for the reinforcement learning environment based on the Ship-Transit Simulator suited for 
    two ship actor
    """
    def __init__(self, 
                 assets:List[ShipAssets],
                 map: PolygonObstacle,
                 ship_draw:bool,
                 time_since_last_ship_drawing:float,
                 args):
        super().__init__()
        
        # # Convention keys
        # self.keys = [
        #     'ship_model',           # 0
        #     'throttle_controller', # 1
        #     'autopilot',           # 2
        #     'desired_speed',       # 3
        #     'integrator_term',     # 4
        #     'time_list',           # 5
        #     'type_tag'             # 6
        # ]
        
        #  For test ship
        self.test, self.obs = assets
        
        ## Unpack assets [test, obs]
        self.assets = [self.test, self.obs]
        
        # Ship drawing configuration
        self.ship_draw = ship_draw
        self.time_since_last_ship_drawing = time_since_last_ship_drawing
        
        # Define observation space 
        # [test_n_pos, test_e_pos, test_headings, test_forward speed, 
        #   obs_n_pos, obs_e_pos, obs_headings, obs_forward_speed, \
        #   test_shaft_speed, test_los_e_ct, test_power_load \] (11 states)
        self.observation_space = Box(
            low = np.array([0, 0, -np.pi, -25, 
                            0, 0, -np.pi, -25,
                            -3000, 0, 0], dtype=np.float32),
            high = np.array([10000, 20000, np.pi, 25, 
                             10000, 20000, np.pi, 25,
                             3000, 100, 2000], dtype=np.float32),
        )
        
        # Define action space [route_point_shift, desired_forward_speed] # FOR LATER
        # Define action space [route_point_shift] 
        self.action_space = Box(
            low = np.array([-np.pi/4], dtype=np.float32),
            high = np.array([np.pi/4], dtype=np.float32),
        )
        
        # Define initial desired speed
        self.init_test_desired_forward_speed = self.test.desired_forward_speed
        self.test_desired_forward_speed = self.init_test_desired_forward_speed
        self.init_obs_desired_forward_speed = self.obs.desired_forward_speed
        self.obs_desired_forward_speed = self.init_obs_desired_forward_speed
        
        # Define initial state
        self.initial_state = np.array([self.test.ship_model.north, self.test.ship_model.east, self.test.ship_model.yaw_angle, self.test.ship_model.forward_speed,
                                       self.obs.ship_model.north, self.obs.ship_model.east, self.obs.ship_model.yaw_angle, self.obs.ship_model.forward_speed,
                                       0.0, 0.0, 0.0], dtype=np.float32)
        self.state = self.initial_state
        
        # Important Lists
        self.init_test_integrator_term = self.test.integrator_term
        self.init_test_times = self.test.time_list
        self.init_obs_integrator_term = self.obs.integrator_term
        self.init_obs_times = self.obs.time_list
        
        # Store the map class as attribute
        self.map = map 
        
        # Store args as attribute
        self.args = args
        
        # Simulation time and travel distance counter
        self.eps_simu_time = 0
        self.eps_distance_travelled = 0
        
        # Initialize Reward Function Parameters
        # self.reward_function_params()
        

    def reward_function_params(self):
        # Reward Function parameters
        self.e_tolerance = 1000
        AB_distance_n = self.auto_pilot.navigate.north[-1] - self.auto_pilot.navigate.north[0]
        AB_distance_e = self.auto_pilot.navigate.east[-1] - self.auto_pilot.navigate.east[0]
        self.AB_distance = np.sqrt(AB_distance_n ** 2 + AB_distance_e ** 2)
        self.AB_alpha = np.arctan2(AB_distance_e, AB_distance_n)
        self.AB_beta = np.pi/2 - self.AB_alpha 
        self.prev_route_coordinate = None
        
        self.total_distance_travelled = 0
        self.distance_travelled = 0
        self.theta = self.args.theta
        
    
    def reset(self):
        ''' Reset the ship environment for each model (or both)
            Args:
            'test_ship', 'obs_ship', 'all'
        '''
        # Reset the assets
        for ship in self.assets:
            #  Reset the ship simulator
            ship.ship_model.reset()
            
            # Reset the ship throttle controller
            ship.throttle_controller.reset() 
            
            # Reset the autopilot controlller
            ship.auto_pilot.reset()
            
            # Reset the important lists an desired speed
            if ship.type_tag == 'test_ship':
                ship.integrator_term = self.init_test_integrator_term
                ship.time_list = self.init_test_times
                ship.desired_forward_speed = self.init_test_desired_forward_speed
            elif ship.type_tag == 'obs_ship':
                ship.integrator_term = self.init_obs_integrator_term
                ship.time_list = self.init_obs_times
                ship.desired_forward_speed = self.init_obs_desired_forward_speed
        
        # Reset the simulation time and travel distance counter
        self.simu_time = 0
        self.eps_distance_travelled = 0
        
        # Reset the changing states into its initial state
        self.state = self.initial_state
        
        return self.state
    
    def init_step(self):
        ''' The initial step to place the ship and the controller 
            to work inside the digital simulation (WITHOUT STORING)
        '''
        # For all assets
        for ship in self.assets:
            # Measure ship position and speed
            north_position = ship.ship_model.north
            east_position = ship.ship_model.east
            heading = ship.ship_model.yaw_angle
            forward_speed = ship.ship_model.forward_speed
        
            # Find appropriate rudder angle and engine throttle
            rudder_angle = ship.auto_pilot.rudder_angle_from_sampled_route(
                north_position=north_position,
                east_position=east_position,
                heading=heading
            )
        
            throttle = ship.throttle_controller.throttle(
                speed_set_point = ship.desired_forward_speed,
                measured_speed = forward_speed,
                measured_shaft_speed = forward_speed
            )

            # Step
            ship.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
            ship.ship_model.integrate_differentials()
        return 
    
    def step(self, 
             sampled_route, 
             SAC_update,
             sampling_time_record,
             init):
        ''' The method is used for stepping up the simulator for all the reinforcement
            learning assets
        '''
        # For all assets
        for ship in self.assets:
            # Measure ship position and speed
            north_position = ship.ship_model.north
            east_position = ship.ship_model.east
            heading = ship.ship_model.yaw_angle
            forward_speed = ship.ship_model.forward_speed

            # If the it is time for the SAC update, alter the obstacle ship autopilot
            # Test ship autopilot shall not be altered y the reinforcement learning agent
            if SAC_update and ship.type_tag == 'obs_ship':
                route_coord_n, route_coord_e = sampled_route
            
                # Update route_point based on the action
                route_coordinate = route_coord_n, route_coord_e
                ship.auto_pilot.update_route(route_coordinate)
            
                # Update desired_forward_speed based on the action
                # self.desired_forward_speed = desired_forward_speed
            
                # Store the sampled route coordinate to the holder variable
                self.prev_route_coordinate = route_coordinate
        
            # If it is not the time to use action as simulation input, use saved route coordinate
            else:
                route_coordinate = self.prev_route_coordinate
        
            # Find appropriate rudder angle and engine throttle
            rudder_angle = ship.auto_pilot.rudder_angle_from_sampled_route(
                north_position=north_position,
                east_position=east_position,
                heading=heading
            )
        
            throttle = ship.throttle_controller.throttle(
                measured_speed = ship.desired_forward_speed,
                measured_shaft_speed = ship.desired_forward_speed,
            )
        
            # Update and integrate differential equations for current time step
            ship.ship_model.store_simulation_data(throttle, 
                                                  rudder_angle,
                                                  self.auto_pilot.get_cross_track_error(),
                                                  self.auto_pilot.get_heading_error())
            ship.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
            ship.ship_model.integrate_differentials()
        
            ship.integrator_term.append(ship.auto_pilot.navigate.e_ct_int)
            ship.time_list.append(ship.ship_model.int.time)
        
            # Apply ship drawing (set as optional function)
            if self.ship_draw:
                if self.time_since_last_ship_drawing > 30:
                    ship.ship_model.ship_snap_shot()
                    self.time_since_last_ship_drawing = 0 # The ship draw timer is reset here
                self.time_since_last_ship_drawing += ship.ship_model.int.dt
        
            ####### START AGAIN HERE #######
            
            # Compute reward
            pos = [self.ship_model.north, self.ship_model.east, self.ship_model.yaw_angle]
            heading_error = self.ship_model.simulation_results['heading error [deg]'][-1]
            measured_shaft_rpm = self.ship_model.simulation_results['propeller shaft speed [rpm]'][-1]
            los_ct_error = self.ship_model.simulation_results['cross track error [m]'][-1]
            power_load = self.ship_model.simulation_results['power me [kw]'][-1]
            available_power_load = self.ship_model.simulation_results['available power me [kw]'][-1]
        
            next_state = np.array([self.ensure_scalar(pos[0]), 
                                    self.ensure_scalar(pos[1]), 
                                    self.ensure_scalar(pos[2]), 
                                    self.ensure_scalar(self.ship_model.forward_speed), 
                                    self.ensure_scalar(measured_shaft_rpm), 
                                    self.ensure_scalar(los_ct_error),
                                    self.ensure_scalar(power_load)], dtype=np.float32) 
        
            reward, done, status = self.reward_function(pos,
                                                        route_coordinate,
                                                        los_ct_error,
                                                        power_load,
                                                        available_power_load,
                                                        heading_error,
                                                        measured_shaft_rpm,
                                                        sampling_time_record)
        
            # Compute travelled distance
            if init == False:
                dist_trav_north = self.ship_model.simulation_results['north position [m]'][-1] - self.ship_model.simulation_results['north position [m]'][-2]
                dist_trav_east = self.ship_model.simulation_results['east position [m]'][-1] - self.ship_model.simulation_results['east position [m]'][-2]
                self.distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)
                self.total_distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)
        
            # Step up the simulator
            self.ship_model.int.next_time()
        
        return next_state, reward, done, status
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility"""
        self.np_random, seed = seeding.np_random(seed)
        
    # Make sure all values are scalars
    def ensure_scalar(self, x):
        return float(x[0]) if isinstance(x, (np.ndarray, list)) else float(x)
        
###################################################################################################################
############################################### FAILURE MODES #####################################################
###################################################################################################################
        
    def is_pos_outside_horizon(self, n_pos, e_pos, margin=0):
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
        outside_n = n_pos < n_route_bound[0] or n_pos > n_route_bound[1]
        outside_e = e_pos < e_route_bound[0] or e_pos > e_route_bound[1]
        
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
        scale_factor = 2.5
        
        time_expected = scale_factor * self.AB_distance / self.expected_forward_speed
        
        return recorded_time > time_expected
    
    def is_mechanical_failure(self, measured_shaft_rpm):
        ## Check this website:
        ## https://spicerparts.com/calculators/critical-speed-rpm-calculator
        shaft_rpm_max = 2000 # rpm
        return np.abs(measured_shaft_rpm) > shaft_rpm_max

    def is_navigation_failure(self, e_ct):
        ## Ship deviates off the course beyond tolerance defined by these two conditions
        condition_1 = np.abs(e_ct) > self.e_tolerance
        condition_2 = self.distance_travelled > self.AB_distance * self.theta
        
        return condition_1 or condition_2

    def is_blackout_failure(self,
                            power_load, 
                            available_power_load):
        ## Diesel engine overloaded
        # print(power_load)
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
        
        # Base stepping reward
        reward_base = 0.1
        
        # Directional JONSWAP reward
        reward_jonswap = 0 # NEED ALGORITHM
        
        # Cross-track error reward        
        # Normalized cross_track error by the tolerance
        # Huge cross track error is not preferable. But when navigation loss happened give more reward
        reward_e_ct = -np.abs(los_ct_error) / self.e_tolerance 
        
        # Miss heading alignment reward
        # In radian
        # Normalized by maximum rudder angle
        # reward_e_hea = -np.abs(heading_error) / self.ship_model.ship_machinery_model.rudder_ang_max
        
        # Distance to end point
        # Get the relative distance between the ship and the end point
        n_route_end = self.auto_pilot.navigate.north[-1]
        e_route_end = self.auto_pilot.navigate.east[-1]
        distance_to_reward = np.sqrt((n_pos - n_route_end)**2 + (e_pos - e_route_end)**2)
        
        # Closer to end point is more rewarding
        reward_to_distance = 1 - distance_to_reward/self.AB_distance
        
        # d_zones = [6000, 5000, 4000, 3000]
        # f_zones = [2.0, 3.0, 4.0, 5.0]
        
        # reward_to_distance = 1
        
        # if distance_to_reward < d_zones[0] and distance_to_reward > d_zones[1]:
        #     reward_to_distance += self.AB_distance / (distance_to_reward)
        # elif distance_to_reward < d_zones[1] and distance_to_reward > d_zones[2]:
        #     reward_to_distance += self.AB_distance / (distance_to_reward)
        # elif distance_to_reward < d_zones[2] and distance_to_reward > d_zones[1]:
        #     reward_to_distance += self.AB_distance / (distance_to_reward)
        # elif distance_to_reward < d_zones[3]:
        #     reward_to_distance += self.AB_distance/ (distance_to_reward)
        
        # Miss distance from collision reward
        # Normalized by start_to_end distance (NOT FINAL)
        # Farther from collision the better. Although unintentionally close to collision is rewarded
        reward_col = self.obstacles.obstacles_distance(n_pos, e_pos) / self.AB_distance
        
        # return reward_jonswap + reward_e_ct + reward_e_hea + reward_col + reward_to_distance
        return float(reward_base + reward_jonswap + reward_e_ct + reward_col + reward_to_distance)
    
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
        arrival_radius = 200 # Arrival radius zone
        if relative_dist <= arrival_radius:
            reward_terminal += 1000
            done = True
            status = "|Reach endpoint|"
            
        ## Reward for ship hitting map horizon
        # print(pos)
        if self.is_pos_outside_horizon(n_pos, e_pos):
            reward_terminal += 700
            done = True
            status = "|Map horizon hit failure|"
                    
        ## Reward for ship hitting obstacles
        # if self.is_pos_inside_obstacles(n_pos, e_pos): # When using circular obstacle
        if self.obstacles.if_pos_inside_obstacles(n_pos, e_pos): # When using polygon obstacle
            reward_terminal += 700
            done = True
            status = "|Collision failure|"
            
        ## Reward for route action sampled inside obstacles or outside map horizon
        # if self.is_pos_outside_horizon(n_route, e_route) or\
        #     self.is_pos_inside_obstacles(n_route, e_route): # When using circular obstacle
        if self.is_pos_outside_horizon(n_route, e_route) or\
            self.obstacles.if_pos_inside_obstacles(n_route, e_route): # When using polygon obstacle
            reward_terminal += -2500
            done = True
            status = "|Route point is sampled in terminal state|"
        
        # ## Reward for unnecessary slow ship movement
        # if self.is_too_slow(recorded_time):
        #     reward_terminal += -1000
        #     done = True
        #     status += "|Slow progress failure|"
        
        ## Reward for Mechanical Failure 
        if self.is_mechanical_failure(measured_shaft_rpm):
            reward_terminal += 1200
            done = True
            status = "|Mechanical failure|"
        
        ## Reward for Navigation Failure    
        if self.is_navigation_failure(los_ct_error):
            reward_terminal += 1000
            done = True
            status = "|Navigation failure|"
        
        ## Reward for Blackout Failure    
        if self.is_blackout_failure(power_load, available_power_load):
            reward_terminal += 1500
            done = True
            status = "|Blackout failure|"
        
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