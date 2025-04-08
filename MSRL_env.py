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

from dataclasses import dataclass, field
from typing import Union, List

import copy
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
    init_copy: 'ShipAssets' = field(default=None, repr=False, compare=False)

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
        
        # Store init values
        for i, asset in enumerate(self.assets):
            asset.init_copy=copy.deepcopy(asset)
        
        # Ship drawing configuration
        self.ship_draw = ship_draw
        self.time_since_last_ship_drawing = time_since_last_ship_drawing
        
        # Define observation space 
        # [test_n_pos, test_e_pos, test_headings, test_forward speed, 
        #   test_shaft_speed, test_los_e_ct, test_power_load, \
        #   obs_n_pos, obs_e_pos, obs_headings, obs_forward_speed] (11 states)
        self.observation_space = Box(
            low = np.array([0, 0, -np.pi, -25, 
                            0, 0, -np.pi, -25,
                            -3000, 0, 0], dtype=np.float32),
            high = np.array([10000, 20000, np.pi, 25, 
                             3000, 100, 2000,
                             10000, 20000, np.pi, 25,], dtype=np.float32),
        )
        
        # Define action space [route_point_shift, desired_forward_speed] # FOR LATER
        # Define action space [route_point_shift] 
        self.action_space = Box(
            low = np.array([-np.pi/4], dtype=np.float32),
            high = np.array([np.pi/4], dtype=np.float32),
        )
        
        # Define initial state
        self.initial_state = np.array([self.test.ship_model.north, self.test.ship_model.east, self.test.ship_model.yaw_angle, self.test.ship_model.forward_speed,
                                       self.obs.ship_model.north, self.obs.ship_model.east, self.obs.ship_model.yaw_angle, self.obs.ship_model.forward_speed,
                                       0.0, 0.0, 0.0], dtype=np.float32)
        self.state = self.initial_state
        
        # Container for the next state
        # [test_n_pos, test_e_pos, test_headings, test_forward speed, 
        #   test_shaft_speed, test_los_e_ct, test_power_load, \
        #   obs_n_pos, obs_e_pos, obs_headings, obs_forward_speed] (11 states)
        self.initial_next_state = np.zeros((11,), dtype=np.float32)
        self.next_state = self.initial_next_state
        
        # Store the map class as attribute
        self.map = map 
        
        # Store args as attribute
        self.args = args
        
        # Simulation time and travel distance counter
        self.eps_simu_time = 0
        self.eps_distance_travelled = 0
        
        # Previously sampled route coordinate
        self.prev_route_coordinate = None
        
        
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
        for i, ship in enumerate(self.assets):
            # Call upon the copied initial values
            init = ship.init_copy
            
            #  Reset the ship simulator
            ship.ship_model.reset()
            
            # Reset the ship throttle controller
            ship.throttle_controller.reset() 
            
            # Reset the autopilot controlller
            ship.auto_pilot.reset()
            
            # Reset parameters and lists
            ship.desired_forward_speed = init.desired_forward_speed
            ship.integrator_term = copy.deepcopy(init.integrator_term)
            ship.time_list = copy.deepcopy(init.time_list)
        
        # Reset the simulation time and travel distance counter
        self.simu_time = 0
        self.eps_distance_travelled = 0
        
        # Reset the changing states into its initial state
        self.state = self.initial_state
        
        # Reset the previous sampled route coordinate
        self.prev_route_coordinate = None
        
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
                                                  ship.auto_pilot.get_cross_track_error(),
                                                  ship.auto_pilot.get_heading_error())
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
        
            # Compute reward
            pos = [ship.ship_model.north, ship.ship_model.east, ship.ship_model.yaw_angle]
            heading_error = ship.ship_model.simulation_results['heading error [deg]'][-1]
            measured_shaft_rpm = ship.ship_model.simulation_results['propeller shaft speed [rpm]'][-1]
            los_ct_error = ship.ship_model.simulation_results['cross track error [m]'][-1]
            power_load = ship.ship_model.simulation_results['power me [kw]'][-1]
            available_power_load = ship.ship_model.simulation_results['available power me [kw]'][-1]
            
            # Store the states required for RL method
            self.get_next_state(self, 
                                pos, 
                                forward_speed, 
                                measured_shaft_rpm, 
                                los_ct_error, 
                                power_load, 
                                ship.type_tag)
        
            # Compute travelled distance for the obstacle ship
            if init == False and ship.type_tag == "obs_ship":
                dist_trav_north = ship.ship_model.simulation_results['north position [m]'][-1] - ship.ship_model.simulation_results['north position [m]'][-2]
                dist_trav_east = ship.ship_model.simulation_results['east position [m]'][-1] - ship.ship_model.simulation_results['east position [m]'][-2]
                self.distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)
                self.total_distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)
        
            # Step up the simulator
            ship.ship_model.int.next_time()
        
        # Set the next state, then reset the next_state container to zero
        next_state = self.next_state
        self.next_state = self.initial_next_state
        
        ## MORE WORK ON THE REWARD FUNCTION NOW
        reward, done, status = self.reward_function(pos,
                                                    route_coordinate,
                                                    los_ct_error,
                                                    power_load,
                                                    available_power_load,
                                                    heading_error,
                                                    measured_shaft_rpm,
                                                    sampling_time_record)
        
        return next_state, reward, done, status
    
    def get_next_state(self, pos, forward_speed, measured_shaft_rpm, los_ct_error, power_load, type_tag):
        ''' This method is used to get the next RL steps required to compute the reward function and to update the policy.
            It is based on the simulator step from each asset. The RL states take various of asset simulator step, hence 
            this method is fully depends on the context of the agent's learning purposes.
        '''
        
        if type_tag == 'test_ship':
            self.next_state[0] = self.ensure_scalar(pos[0])
            self.next_state[1] = self.ensure_scalar(pos[1]) 
            self.next_state[2] = self.ensure_scalar(pos[2])
            self.next_state[3] = self.ensure_scalar(forward_speed)
            self.next_state[4] = self.ensure_scalar(measured_shaft_rpm)
            self.next_state[5] = self.ensure_scalar(los_ct_error)
            self.next_state[6] = self.ensure_scalar(power_load)
        elif type_tag == 'obs_ship':
            self.next_state[7] = self.ensure_scalar(pos[0]) 
            self.next_state[8] = self.ensure_scalar(pos[1]) 
            self.next_state[9] = self.ensure_scalar(pos[2])
            self.next_state[10] = self.ensure_scalar(forward_speed)
        
        next_state = self.next_state
        
        return next_state
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility"""
        self.np_random, seed = seeding.np_random(seed)
        
    # Make sure all values are scalars
    def ensure_scalar(self, x):
        return float(x[0]) if isinstance(x, (np.ndarray, list)) else float(x)
        
###################################################################################################################
############################################### FAILURE MODES #####################################################
###################################################################################################################
        
    def is_pos_outside_horizon(self, pos, ship_length, margin=0):
        ''' Checks if the ship positions are outside the map horizon. 
            Map horizons are determined by the edge point of the determined route point. 
            Allowed additional margin are default 100 m for North and East boundaries.
            Only works with start to end route points method (Two initial points).
        '''
        # Unpack ship position
        n_pos, e_pos, _ = pos
        
        # Get the map boundaries
        min_north = self.map.min_north
        min_east = self.map.min_east
        max_north = self.map.max_north
        max_east = self.map.max_east
        
        # Get the obstacle margin due to all assets ship length
        margin = ship_length/2
            
        # min_bound and max_bound
        n_route_bound = [min_north + margin , max_north - margin]
        e_route_bound = [min_east + margin, max_east - margin]
        
        # Check if position is outside bound
        outside_n = n_pos < n_route_bound[0] or n_pos > n_route_bound[1]
        outside_e = e_pos < e_route_bound[0] or e_pos > e_route_bound[1]
        
        is_outside = outside_n or outside_e
        
        return is_outside
    
    def is_pos_inside_obstacles(self, n_pos, e_pos, ship_length):
        ''' Checks if the tagged position is inside any obstacle
        '''
        # Get the obstacle margin due to all assets ship length
        # Margin is defined as a square patch enveloping the ships
        margin = ship_length/2
        
        # Get the max reach and min reach of the ship
        min_north = n_pos - margin
        min_east = e_pos - margin
        max_north = n_pos + margin
        max_east = e_pos + margin
        
        # All patch's hard point
        hard_points = [(min_north, min_east), (min_north, max_east), (max_north, min_east), (max_north, max_east)]
        
        is_inside = False
        
        for hard_point in hard_points:
            if self.map.if_pos_inside_obstacles(hard_point):
                is_inside =  True
      
        return is_inside
    
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
    
    def is_ship_collision(self, test_pos, obs_pos):
        ''' If test ship and obstacle ship distance is below some threshold, categorized it as collision
        '''
        # Unpack ship position
        n_test, e_test, _ = test_pos
        n_obs, e_obs , _ = obs_pos
        
        # Set minimum ship distacne
        minimum_ship_distance = 50 # arbitrary number
        
        # Compute the ship distance
        ship_distance =  (n_test - n_obs)**2 + (e_test - e_obs)**2
        
        # Collision logic
        is_collide = False
        
        if ship_distance < minimum_ship_distance ** 2:
            is_collide = True
        
        return is_collide
    
    #### EXPERIMENTAL #### NOT USED FOR NOW ####
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
    
###################################################################################################################
############################################## REWARD FUNCTION ####################################################
###################################################################################################################

    def test_ship_non_terminal_state_reward(self, 
                                  test_pos,
                                  test_los_ct_error):
        ''' Reward per simulation time step should be in order 10**0
            As it will be accumulated over time. Negative reward proportional
            to positive reward shall be added for each simulation time step as
            well.
            
            For Test Ship
        '''
        ## Unpack test_ship
        n_test, e_test, _ = test_pos
        
        ## Base stepping reward
        reward_base = 0.0
        
        ## Cross-track error reward        
        # Normalized cross_track error by the tolerance
        # Big cross track error is preferable. But when navigation loss happened give more reward
        reward_e_ct = np.abs(test_los_ct_error) / self.e_tolerance  
        
        ## Distance to end point
        # Get the relative distance between the ship and the end point
        n_route_end = self.test.auto_pilot.navigate.north[-1]
        e_route_end = self.test.auto_pilot.navigate.east[-1]
        distance_to_reward = np.sqrt((n_test - n_route_end)**2 + (e_test - e_route_end)**2)
        # Closer to end point is more rewarding, normalized by the maximum east position
        reward_to_distance = 1 - distance_to_reward/self.map.max_east
        
        # Miss distance from collision reward
        # Normalized by the maximum north position
        # Closer to obstacle is better
        reward_near_col = 1 - self.map.obstacles_distance(n_test, e_test) / self.map.max_north
        
        return float(reward_base  + reward_e_ct + reward_near_col + reward_to_distance)
    
    def obs_ship_non_terminal_state_reward(self, 
                                  obs_pos,
                                  obs_los_ct_error):
        ''' Reward per simulation time step should be in order 10**0
            As it will be accumulated over time. Negative reward proportional
            to positive reward shall be added for each simulation time step as
            well.
            
            For Obstacle Ship
        '''
        ## Unpack test_ship
        n_obs, e_obs, _ = obs_pos
        
        ## Base stepping reward
        reward_base = 0.0
        
        ## Cross-track error reward        
        # Normalized cross_track error by the tolerance
        # Huge cross track error is not preferable. But when navigation loss happened give more reward
        reward_e_ct = -np.abs(obs_los_ct_error) / self.e_tolerance 
        
        ## Distance to end point
        # Get the relative distance between the ship and the end point
        n_route_end = self.obs.auto_pilot.navigate.north[-1]
        e_route_end = self.obs.auto_pilot.navigate.east[-1]
        distance_to_reward = np.sqrt((n_obs - n_route_end)**2 + (e_obs - e_route_end)**2)
        # Closer to end point is more rewarding, normalized by the maximum east position
        reward_to_distance = 1 - distance_to_reward/self.map.max_east
        
        # Miss distance from collision reward
        # Normalized by the maximum north position
        # Closer to obstacle is worse
        reward_near_col = -(1 - self.map.obstacles_distance(n_obs, e_obs) / self.map.max_north)
        
        return float(reward_base  + reward_e_ct + reward_near_col + reward_to_distance)
    
    def shared_non_terminal_state_reward(self,
                                         test_pos,
                                         obs_pos):
        ''' For computing reward function based on the test and obstacle ship distance
        '''
        # Unpack ship position
        n_test, e_test, _ = test_pos
        n_obs, e_obs , _ = obs_pos
        
        # Compute the ship distance
        ship_distance =  np.sqrt((n_test - n_obs)**2 + (e_test - e_obs)**2)
        
        reward =  1 - ship_distance/self.map.max_north
        
        return reward
    
    def test_ship_terminal_state_reward(self,
                                        pos,
                                        route_coordinate,
                                        los_ct_error,
                                        power_load,
                                        available_power_load,
                                        measured_shaft_rpm):
        ## Initial value
        reward_terminal = 0
        status = " "
        done = False
        
        n_test, e_test, _ = pos
        
        # print(route_coordinate)
        n_route, e_route = route_coordinate
        
        ## Reward for reaching the end point
        # Get the relative distance between the ship and the end point
        n_route_end = self.auto_pilot.navigate.north[-1]
        e_route_end = self.auto_pilot.navigate.east[-1]
        relative_dist = np.sqrt((n_test - n_route_end)**2 + (e_test - e_route_end)**2)
        
        # Check if the ship arrive at the end point
        arrival_radius = 200 # Arrival radius zone
        if relative_dist <= arrival_radius:
            reward_terminal += 1000
            done = True
            status = "|Test ship reaches endpoint|"
            
        ## Reward for ship hitting map horizon
        if self.is_pos_outside_horizon(n_test, e_test):
            reward_terminal += 0
            done = True
            status = "|Test ship hits map horizon|"
                    
        ## Reward for ship hitting obstacles
        # We want the test ship to hit the obstacle
        if self.map.if_pos_inside_obstacles(n_test, e_test): # When using polygon obstacle
            reward_terminal += 700
            done = True
            status = "|Test ship collide with the terrain|"

        ## Reward for Mechanical Failure 
        if self.is_mechanical_failure(measured_shaft_rpm):
            reward_terminal += 1200
            done = True
            status = "|Test ship mechanical failure|"
        
        ## Reward for Navigation Failure    
        if self.is_navigation_failure(los_ct_error):
            reward_terminal += 1000
            done = True
            status = "|Test ship navigation failure|"
        
        ## Reward for Blackout Failure    
        if self.is_blackout_failure(power_load, available_power_load):
            reward_terminal += 1500
            done = True
            status = "|Test ship blackout failure|"
        
        if done == False:
            status = "|Test ship not in terminal state|"
        
        return reward_terminal, done, status
    
    def obs_ship_terminal_state_reward(self,
                                        pos,
                                        route_coordinate,
                                        los_ct_error,
                                        power_load,
                                        available_power_load,
                                        measured_shaft_rpm):
        ## Initial value
        reward_terminal = 0
        status = " "
        done = False
        
        n_obs, e_obs, _ = pos
        
        # print(route_coordinate)
        n_route, e_route = route_coordinate
        
        ## Reward for reaching the end point
        # Get the relative distance between the ship and the end point
        n_route_end = self.auto_pilot.navigate.north[-1]
        e_route_end = self.auto_pilot.navigate.east[-1]
        relative_dist = np.sqrt((n_obs - n_route_end)**2 + (e_obs - e_route_end)**2)
        
        # Check if the ship arrive at the end point
        arrival_radius = 200 # Arrival radius zone
        if relative_dist <= arrival_radius:
            reward_terminal += 1000
            done = True
            status = "|Obsatcle ship reaches endpoint|"
            
        ## Reward for ship hitting map horizon
        if self.is_pos_outside_horizon(n_obs, e_obs):
            reward_terminal += 0
            done = True
            status = "|Test ship hits map horizon|"
                    
        ## Reward for ship hitting obstacles
        ## We don't want the obstacle ship to hit the obstacle
        if self.map.if_pos_inside_obstacles(n_obs, e_obs): # When using polygon obstacle
            reward_terminal -= 1000
            done = True
            status = "|Obs ship collide with the terrain|"
            
        ## Reward for route action sampled inside obstacles or outside map horizon
        # Exclusive for obstacle ship
        if self.is_pos_outside_horizon(n_route, e_route) or\
            self.map.if_pos_inside_obstacles(n_route, e_route): # When using polygon obstacle
            reward_terminal += -2500
            done = True
            status = "|Obstacle ship intermediate route point is sampled in terminal state|"
        
        # ## Reward for unnecessary slow ship movement
        # if self.is_too_slow(recorded_time):
        #     reward_terminal += -1000
        #     done = True
        #     status += "|Slow progress failure|"
        
        ## Reward for Navigation Failure    
        if self.is_navigation_failure(los_ct_error):
            reward_terminal += 1000
            done = True
            status = "|Test ship navigation failure|"
        
        ## Reward for Blackout Failure    
        if self.is_blackout_failure(power_load, available_power_load):
            reward_terminal += 1500
            done = True
            status = "|Test ship blackout failure|"
        
        if done == False:
            status = "|Test ship not in terminal state|"
        
        return reward_terminal, done, status
    
    # TO WORK LATER
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