from simulator.ship_model import  ShipConfiguration, EnvironmentConfiguration, SimulationConfiguration, ShipModelAST
from simulator.ship_engine import MachinerySystemConfiguration, MachineryMode, MachineryModeParams, MachineryModes, SpecificFuelConsumptionBaudouin6M26Dot3, SpecificFuelConsumptionWartila6L26
from simulator.LOS_guidance import LosParameters
from simulator.obstacle import PolygonObstacle
from simulator.controllers import ThrottleControllerGains, HeadingControllerGains, EngineThrottleFromSpeedSetPoint, HeadingBySampledRouteController

from MSRL_env import MultiShipRLEnv, ShipAssets
from ast_sac.nn_models import *
from ast_sac.sac import SAC
from ast_sac.replay_memory import ReplayMemory
from utils.utils import action_record_to_df

from log_function.log_function import LogMessage

import json
import os
import time
import numpy as np
import pandas as pd
import itertools
import datetime
import argparse

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch
from torch.utils.tensorboard import SummaryWriter

from typing import Union, List

np.set_printoptions(precision=2, suppress=True)

# Argument Parser
parser = argparse.ArgumentParser(description='Ship Transit Soft Actor-Critic Args')

# Coefficient and boolean parameters
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every scoring episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--theta', type=float, default=2, metavar='G',
                    help='action sampling frequency coefficient(θ) (default: 1.5)')
parser.add_argument('--sampling_frequency', type=int, default=4, metavar='G',
                    help='maximum amount of action sampling per episode (default: 9)')
parser.add_argument('--max_route_resampling', type=int, default=1000, metavar='G',
                    help='maximum amount of route resampling if route is sampled inside\
                        obstacle (default: 1000)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')

# Neural networks parameters
parser.add_argument('--seed', type=int, default=25350, metavar='Q',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=64, metavar='Q',
                    help='batch size (default: 256)')
parser.add_argument('--replay_size', type=int, default=1000, metavar='Q',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='Q',
                    help='hidden size (default: 256)')
parser.add_argument('--cuda', action="store_true", default=True,
                    help='run on CUDA (default: False)')

# Timesteps and episode parameters
parser.add_argument('--time_step', type=int, default=0.5, metavar='N',
                    help='time step size in second for ship transit simulator (default: 0.5)')
parser.add_argument('--num_steps', type=int, default=100000, metavar='N',
                    help='maximum number of steps across all episodes (default: 100000)')
parser.add_argument('--num_steps_episode', type=int, default=600, metavar='N',
                    help='Maximum number of steps per episode to avoid infinite recursion (default: 600)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--update_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--sampling_step', type=int, default=1000, metavar='N',
                    help='Step for doing full action sampling (default:1000')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--scoring_episode_every', type=int, default=20, metavar='N',
                    help='Number of every episode to evaluate learning performance(default: 40)')
parser.add_argument('--num_scoring_episodes', type=int, default=20, metavar='N',
                    help='Number of episode for learning performance assesment(default: 20)')

# Others
parser.add_argument('--radius_of_acceptance', type=int, default=300, metavar='O',
                    help='Radius of acceptance for LOS algorithm(default: 600)')
parser.add_argument('--lookahead_distance', type=int, default=1000, metavar='O',
                    help='Lookahead distance for LOS algorithm(default: 450)')

args = parser.parse_args()

# Engine configuration
main_engine_capacity = 2160e3
diesel_gen_capacity = 510e3
hybrid_shaft_gen_as_generator = 'GEN'
hybrid_shaft_gen_as_motor = 'MOTOR'
hybrid_shaft_gen_as_offline = 'OFF'

# Configure the simulation
ship_config = ShipConfiguration(
    coefficient_of_deadweight_to_displacement=0.7,
    bunkers=200000,
    ballast=200000,
    length_of_ship=80,
    width_of_ship=16,
    added_mass_coefficient_in_surge=0.4,
    added_mass_coefficient_in_sway=0.4,
    added_mass_coefficient_in_yaw=0.4,
    dead_weight_tonnage=3850000,
    mass_over_linear_friction_coefficient_in_surge=130,
    mass_over_linear_friction_coefficient_in_sway=18,
    mass_over_linear_friction_coefficient_in_yaw=90,
    nonlinear_friction_coefficient__in_surge=2400,
    nonlinear_friction_coefficient__in_sway=4000,
    nonlinear_friction_coefficient__in_yaw=400
)
env_config = EnvironmentConfiguration(
    current_velocity_component_from_north=-1,
    current_velocity_component_from_east=-1,
    wind_speed=2,
    wind_direction=-np.pi/4
)
pto_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=0,
    shaft_generator_state=hybrid_shaft_gen_as_generator
)
pto_mode = MachineryMode(params=pto_mode_params)

pti_mode_params = MachineryModeParams(
    main_engine_capacity=0,
    electrical_capacity=2*diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_motor
)
pti_mode = MachineryMode(params=pti_mode_params)

mec_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_offline
)
mec_mode = MachineryMode(params=mec_mode_params)
mso_modes = MachineryModes(
    [pti_mode]
)
fuel_spec_me = SpecificFuelConsumptionWartila6L26()
fuel_spec_dg = SpecificFuelConsumptionBaudouin6M26Dot3()
machinery_config = MachinerySystemConfiguration(
    machinery_modes=mso_modes,
    machinery_operating_mode=0,
    linear_friction_main_engine=68,
    linear_friction_hybrid_shaft_generator=57,
    gear_ratio_between_main_engine_and_propeller=0.6,
    gear_ratio_between_hybrid_shaft_generator_and_propeller=0.6,
    propeller_inertia=6000,
    propeller_diameter=3.1,
    propeller_speed_to_torque_coefficient=7.5,
    propeller_speed_to_thrust_force_coefficient=1.7,
    hotel_load=200000,
    rated_speed_main_engine_rpm=1000,
    rudder_angle_to_sway_force_coefficient=50e3,
    rudder_angle_to_yaw_force_coefficient=500e3,
    max_rudder_angle_degrees=30,
    specific_fuel_consumption_coefficients_me=fuel_spec_me.fuel_consumption_coefficients(),
    specific_fuel_consumption_coefficients_dg=fuel_spec_dg.fuel_consumption_coefficients()
)

## CONFIGURE THE SHIP SIMULATION MODELS
# Ship in Test
ship_in_test_simu_setup = SimulationConfiguration(
    initial_north_position_m=100,
    initial_east_position_m=100,
    initial_yaw_angle_rad=45 * np.pi / 180,
    initial_forward_speed_m_per_s=0,
    initial_sideways_speed_m_per_s=0,
    initial_yaw_rate_rad_per_s=0,
    integration_step=args.time_step,
    simulation_time=7200,
)
test_ship = ShipModelAST(ship_config=ship_config,
                       machinery_config=machinery_config,
                       environment_config=env_config,
                       simulation_config=ship_in_test_simu_setup,
                       initial_propeller_shaft_speed_rad_per_s=400 * np.pi / 30)

# Obstacle Ship
ship_as_obstacle_simu_setup = SimulationConfiguration(
    initial_north_position_m=9900,
    initial_east_position_m=14900,
    initial_yaw_angle_rad=-90 * np.pi / 180,
    initial_forward_speed_m_per_s=0,
    initial_sideways_speed_m_per_s=0,
    initial_yaw_rate_rad_per_s=0,
    integration_step=args.time_step,
    simulation_time=7200,
)
obs_ship = ShipModelAST(ship_config=ship_config,
                       machinery_config=machinery_config,
                       environment_config=env_config,
                       simulation_config=ship_as_obstacle_simu_setup,
                       initial_propeller_shaft_speed_rad_per_s=400 * np.pi / 30)

## Configure the map data
map_data = [
    [(0,10000), (10000,10000), (9200,9000) , (7600,8500), (6700,7300), (4900,6500), (4300, 5400), (4700, 4500), (6000,4000), (5800,3600), (4200, 3200), (3200,4100), (2000,4500), (1000,4000), (900,3500), (500,2600), (0,2350)],   # Island 1 
    [(10000, 0), (11500,750), (12000, 2000), (11700, 3000), (11000, 3600), (11250, 4250), (12300, 4000), (13000, 3800), (14000, 3000), (14500, 2300), (15000, 1700), (16000, 800), (17500,0)], # Island 2
    [(15500, 10000), (16000, 9000), (18000, 8000), (19000, 7500), (20000, 6000), (20000, 10000)]
    ]

map = PolygonObstacle(map_data)

## Set the throttle and autopilot controllers for the test ship
test_ship_throttle_controller_gains = ThrottleControllerGains(
    kp_ship_speed=7, ki_ship_speed=0.13, kp_shaft_speed=0.05, ki_shaft_speed=0.005
)
test_ship_throttle_controller = EngineThrottleFromSpeedSetPoint(
    gains=test_ship_throttle_controller_gains,
    max_shaft_speed=test_ship.ship_machinery_model.shaft_speed_max,
    time_step=args.time_step,
    initial_shaft_speed_integral_error=114
)
test_route_name = r'D:\OneDrive - NTNU\PhD\PhD_Projects\ShipTransit_OptiStress\ShipTransit_AST\data\test_ship_route.txt'
test_heading_controller_gains = HeadingControllerGains(kp=1, kd=90, ki=0.01)
test_los_guidance_parameters = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)
test_auto_pilot = HeadingBySampledRouteController(
    test_route_name,
    heading_controller_gains=test_heading_controller_gains,
    los_parameters=test_los_guidance_parameters,
    time_step=args.time_step,
    max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180,
    num_of_samplings=2
)
test_desired_forward_speed = 6.0

test_integrator_term = []
test_times = []

## Set the throttle and autopilot controllers for the obstacle ship
obs_ship_throttle_controller_gains = ThrottleControllerGains(
    kp_ship_speed=7, ki_ship_speed=0.13, kp_shaft_speed=0.05, ki_shaft_speed=0.005
)
obs_ship_throttle_controller = EngineThrottleFromSpeedSetPoint(
    gains=test_ship_throttle_controller_gains,
    max_shaft_speed=obs_ship.ship_machinery_model.shaft_speed_max,
    time_step=args.time_step,
    initial_shaft_speed_integral_error=114
)
obs_route_name = r'D:\OneDrive - NTNU\PhD\PhD_Projects\ShipTransit_OptiStress\ShipTransit_AST\data\obs_ship_route.txt'
obs_heading_controller_gains = HeadingControllerGains(kp=1, kd=90, ki=0.01)
obs_los_guidance_parameters = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)
obs_auto_pilot = HeadingBySampledRouteController(
    obs_route_name,
    heading_controller_gains=obs_heading_controller_gains,
    los_parameters=test_los_guidance_parameters,
    time_step=args.time_step,
    max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180,
    num_of_samplings=2
)
obs_desired_forward_speed = 8.0

obs_integrator_term = []
obs_times = []

# Wraps simulation objects based on the ship type using a dictionary
test = ShipAssets(
    ship_model=test_ship,
    throttle_controller=test_ship_throttle_controller,
    auto_pilot=test_auto_pilot,
    desired_forward_speed=test_desired_forward_speed,
    integrator_term=test_integrator_term,
    time_list=test_times,
    type_tag='test_ship'
)

obs = ShipAssets(
    ship_model=obs_ship,
    throttle_controller=obs_ship_throttle_controller,
    auto_pilot=obs_auto_pilot,
    desired_forward_speed=obs_desired_forward_speed,
    integrator_term=obs_integrator_term,
    time_list=obs_times,
    type_tag='obs_ship'
)

# Package the assets for reinforcement learning agent
assets: List[ShipAssets] = [test, obs]

# Timer for drawing the ship
ship_draw = True
time_since_last_ship_drawing = 30

################################### RL SPACE ###################################
# Initiate Multi-Ship Reinforcement Learning Environment Class Wrapper
env = MultiShipRLEnv(assets=assets,
                     map=map,
                     ship_draw=ship_draw,
                     time_since_last_ship_drawing=time_since_last_ship_drawing,
                     args=args)

# Pseudorandom seeding for Reinforcement Learning algorithm
random_seed = False
random_seed = True
if random_seed:
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
# Setup reinforcement learnig agent
agent = SAC(env, args)

# Memory buffer for SAC learnig
memory = ReplayMemory(args.replay_size, args.seed)

# Log message
log_ID = 3
log_dir = f"D:\OneDrive - NTNU\PhD\PhD_Projects\ShipTransit_OptiStress\ShipTransit_AST\MSlogs/MS_run_{log_ID}"
logging = LogMessage(log_dir, log_ID, args)
save_record = True

## STORE RESULTS
# STEPWISE - EPISODICAL LOGGING RECORD
sampled_action = None
episode_record = defaultdict(lambda: {"sampled_action": [], "termination": [], "rewards": [], "states": []})
action_record = defaultdict(lambda: {"scoping_angle [deg]": [], "route_north [m]": [], "route_east [m]": [], "cumulative_rewards":[]})

# REWARD STORE
best_reward = float('-inf')
best_episode = 0
best_policy_wegihts = None

### START TRAINING LOOP
# Initial log message
logging.initial_log()

## Training loop
total_numsteps = 0
updates = 0
testing_count = 0

################################################################################################################################
################################################################################################################################
# Count the episode
for i_episode in itertools.count(1):
    # Set up the initial episodical metrics
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    
    # Set up the sac update cumulative rewards
    # because reward update to the agent update
    # only during sac reward
    sac_update_reward = 0
    
    # Reset the select_action method for each new episode
    agent.select_action_reset()
    
    # Start the timer for each episode
    start_time = time.time()
    
################################################################################################################################
    # Run the simulator unitl it terminates, iteratively
    while not done:
        
        ## INITIALIZE SIMULATOR MODE BASED ON TIMESTEP, SIGNIFIED BY INIT FLAG
        # At episode steps 0, kickstart the simulator by placing the init_step.
        if episode_steps == 0:
            env.init_step()
            episode_steps += 1
            continue
        # Then, we sampled the next intermediate route point immediately. 
        elif episode_steps == 1:
            init = True
        # For the rest of the episode we run 
        else:
            init = False

        ## SELECT ACTION BASED ON MODE
        if args.start_steps > total_numsteps:
            # action_to_simu_input is a boolean to determine wheter the action is implemented as simulation input
            action, route_scope_angle, SAC_update, sampling_time_record = agent.select_action(state,
                                                                                              done,
                                                                                              init=init,
                                                                                              mode='random') # Random sampling 
            
        else:
            action, route_scope_angle, SAC_update, sampling_time_record = agent.select_action(state, 
                                                                                              done,
                                                                                              init=init,
                                                                                              mode='train') # Policy based sampling - train
        
        ## STEP UP THE SIMULATION
        next_state, reward, done, obs_stop, status = env.step(action, 
                                                              SAC_update, 
                                                              init) 
        episode_steps += 1
        total_numsteps += 1
        sac_update_reward += reward
        episode_reward += reward
    
        # Ignore the "done" signal if it comes from hitting the time horizon
        mask = 1 if episode_steps == args.num_steps_episode else float(not done)
        
        ## DURING SAC UPDATE
        if SAC_update:
            # Push the current visited state, 
            # the action taken,
            # the next visited state due to the action taken,
            # the reward associated to state-action transition
            # and the mask to memory
            
            memory.push(state, action, sac_update_reward, next_state, mask)
            
            # Update the policy networks  if we have enough visited state-action transition inside the memory buffer
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range (args.update_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
            
                updates += 1
                
            # Store action record
            action_record[i_episode]["scoping_angle [deg]"].append(route_scope_angle * 180/np.pi)
            action_record[i_episode]["route_north [m]"].append(action[0]) 
            action_record[i_episode]["route_east [m]"].append(action[1])
            action_record[i_episode]["cumulative_rewards"].append(sac_update_reward) 
            
            # Store episodic results
            episode_record[i_episode]["sampled_action"].append(action)
            episode_record[i_episode]["termination"].append(done)
            episode_record[i_episode]["rewards"].append(sac_update_reward)
            episode_record[i_episode]["states"].append(state.tolist())
            
            # Then reset the cumulated reward
            sac_update_reward = 0
        else:
            episode_record[i_episode]["sampled_action"].append(None)
            episode_record[i_episode]["termination"].append(done)
            episode_record[i_episode]["rewards"].append(None)
            episode_record[i_episode]["states"].append(state.tolist())
            
        # Set the next state as current state for the next simulator step
        state = next_state
################################################################################################################################
     
    # Stop the episode timer
    elapsed_time = time.time() - start_time
    
    # Do the training log
    logging.training_log(i_episode, elapsed_time, total_numsteps, episode_steps, episode_reward, status)
    
    # CHECK THE BEST REWARD AND SAVE THE POLICY WEIGHTS
    if episode_reward > best_reward:
        best_reward = episode_reward # Update the best reward
        best_episode = i_episode     # Track which episode achieved this reward
        
        # Define the path to save the best model
        best_model_path = log_dir
        
         # Save the best model
        agent.save_checkpoint(log_dir, best_reward, best_episode, total_numsteps)

        logging.input_log(f"New best policy saved at Episode {best_episode} with Reward: {best_reward:.2f}")
     
    # Log the simulation steps on this episode
    logging.simulation_step_log(episode_record, i_episode, log=False)
    
    # SAVE THE EPISODE RECORD INTO A FILE
    logging.save_episode_record(episode_record, save_record)
    
################################################################################################################################
    ## DO POLICY EVALUATION
    if i_episode % args.scoring_episode_every == 0 and args.eval is True:
        # Set up initials value for evaluation logging
        status_record = [0, 0, 0, 0, 0, 0, 0] # BF, MF, NF, CF, arrival_F, terminal_route_F, not_in_terminal_f
        avg_reward = 0.
        
        for i in range (args.num_scoring_episodes):
            # Set up the initial episodical metrics
            episode_reward = 0
            episode_steps = 0
            done = False
            state = env.reset()
    
            # Reset the select_action method for each new episode
            agent.select_action_reset()
    
            # Start the timer for each episode
            start_time = time.time()
            
            while not done:
                ## INITIALIZE SIMULATOR MODE BASED ON TIMESTEP, SIGNIFIED BY INIT FLAG
                # At episode steps 0, kickstart the simulator by placing the init_step.
                if episode_steps == 0:
                    env.init_step()
                    episode_steps += 1
                    continue
                # Then, we sampled the next intermediate route point immediately. 
                elif episode_steps == 1:
                    init = True
                # For the rest of the episode we run 
                else:
                    init = False

                ## SELECT ACTION BASED ON EVALUATION MODE
                action, _, SAC_update, sampling_time_record = agent.select_action(state, 
                                                                                  done,
                                                                                  init=init,
                                                                                  mode='eval') # Policy based sampling - train
        
                ## STEP UP THE SIMULATION
                next_state, reward, done, obs_stop, status = env.step(action, 
                                                                      SAC_update, 
                                                                      init)
                episode_reward += reward
                episode_steps += 1
                                
                # Count the termination occurence
                if done:
                    if "Blackout failure" in status:
                        status_record[0] += 1
                    if "Mechanical failure" in status:
                        status_record[1] += 1
                    if "Navigation failure" in status:
                        status_record[2] += 1
                    if "Collision failure" in status:
                        status_record[3] += 1
                    if "Reach endpoint" in status:
                        status_record[4] += 1
                    if "Route point is sampled in terminal state" in status or "Map horizon hit failure" in status:
                        status_record[5] += 1
                    if "Not in terminal state" in status:
                        status_record[6] += 1
            
            # Add up each episode reward to the container variable
            avg_reward += episode_reward
        
        # Average the reward on the evaluation episodes
        avg_reward /= args.num_scoring_episodes
        testing_count += 1
        
        # Log the evaluation results
        logging.evaluation_log(testing_count, avg_reward, status_record)
        
    # FOR DEBUG
    if i_episode == 1:
        break
################################################################################################################################
################################################################################################################################          

## OBS
## HANDLE WHEN THE OBSTACLE SHIP FINISHED RUNNING

## Convert action record to dataframe
action_record_df = action_record_to_df(action_record)

# Get the last epsiode action_record
last_action_record = action_record_df[action_record_df["episode"] == best_episode]
print(last_action_record)

## Store the simulation results in a pandas dataframe
test_ship_results_df = pd.DataFrame().from_dict(test.ship_model.simulation_results)
obs_ship_results_df = pd.DataFrame().from_dict(obs.ship_model.simulation_results)

# Create a single figure and axis instead of a grid
plt.figure(figsize=(15, 10))

# Plot 1.1: Ship trajectory with sampled route
# Test ship
plt.plot(test_ship_results_df['east position [m]'].to_numpy(), test_ship_results_df['north position [m]'].to_numpy())
plt.scatter(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north, marker='x', color='blue')  # Waypoints
plt.plot(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north, linestyle='--', color='blue')  # Line
for x, y in zip(test.ship_model.ship_drawings[1], test.ship_model.ship_drawings[0]):
    plt.plot(x, y, color='blue')
for i, (east, north) in enumerate(zip(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north)):
    test_radius_circle = Circle((east, north), args.radius_of_acceptance, color='blue', alpha=0.3, fill=True)
    plt.gca().add_patch(test_radius_circle)
# Obs ship    
plt.plot(obs_ship_results_df['east position [m]'].to_numpy(), obs_ship_results_df['north position [m]'].to_numpy())
plt.scatter(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north, marker='x', color='red')  # Waypoints
plt.plot(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north, linestyle='--', color='red')  # Line
for x, y in zip(obs.ship_model.ship_drawings[1], obs.ship_model.ship_drawings[0]):
    plt.plot(x, y, color='red')
for i, (east, north) in enumerate(zip(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north)):
    obs_radius_circle = Circle((east, north), args.radius_of_acceptance, color='red', alpha=0.3, fill=True)
    plt.gca().add_patch(obs_radius_circle)
map.plot_obstacle(plt.gca())  # get current Axes to pass into map function

plt.xlim(0, 20000)
plt.ylim(0, 10000)
plt.title('Ship Trajectory with the Sampled Route')
plt.xlabel('East position (m)')
plt.ylabel('North position (m)')
plt.gca().set_aspect('equal')
plt.grid(color='0.8', linestyle='-', linewidth=0.5)



# # Create a No.1 2x2 grid for subplots
# fig_1, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
# axes = axes.flatten()  # Flatten the 2D array for easier indexing

# # Plot 1.1: Ship trajectory with sampled route
# axes[0].plot(test_ship_results_df['east position [m]'].to_numpy(), test_ship_results_df['north position [m]'].to_numpy())
# axes[0].scatter(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north, marker='x', color='black')  # Waypoints
# axes[0].plot(test.auto_pilot.navigate.east, test.auto_pilot.navigate.north, linestyle='--', color='black')  # Line
# for x, y in zip(test.ship_model.ship_drawings[1], test.ship_model.ship_drawings[0]):
#     axes[0].plot(x, y, color='black')
# axes[0].plot(obs_ship_results_df['east position [m]'].to_numpy(), obs_ship_results_df['north position [m]'].to_numpy())
# axes[0].scatter(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north, marker='x', color='red')  # Waypoints
# axes[0].plot(obs.auto_pilot.navigate.east, obs.auto_pilot.navigate.north, linestyle='--', color='red')  # Line
# for x, y in zip(obs.ship_model.ship_drawings[1], obs.ship_model.ship_drawings[0]):
#     axes[0].plot(x, y, color='red')
# map.plot_obstacle(axes[0])
# axes[0].set_xlim(0,20000)
# axes[0].set_ylim(0,10000)
# axes[0].set_title('Ship Trajectory with the Sampled Route')
# axes[0].set_xlabel('East position (m)')
# axes[0].set_ylabel('North position (m)')
# axes[0].set_aspect('equal')
# axes[0].grid(color='0.8', linestyle='-', linewidth=0.5)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()