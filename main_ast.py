from simulator.ship_model import  ShipConfiguration, EnvironmentConfiguration, SimulationConfiguration, ShipModelAST
from simulator.ship_engine import MachinerySystemConfiguration, MachineryMode, MachineryModeParams, MachineryModes, SpecificFuelConsumptionBaudouin6M26Dot3, SpecificFuelConsumptionWartila6L26
from simulator.LOS_guidance import LosParameters
from simulator.obstacle import StaticObstacle
from simulator.controllers import ThrottleControllerGains, HeadingControllerGains, EngineThrottleFromSpeedSetPoint, HeadingBySampledRouteController

from RL_env import ShipRLEnv
from ast_sac.nn_models import *
from ast_sac.sac import SAC
from ast_sac.replay_memory import ReplayMemory

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
parser.add_argument('--theta', type=float, default=1.5, metavar='G',
                    help='action sampling frequency coefficient(θ) (default: 1.5)')
parser.add_argument('--sampling_frequency', type=int, default=8, metavar='G',
                    help='maximum amount of action sampling per episode (default: 9)')
parser.add_argument('--max_route_resampling', type=int, default=1000, metavar='G',
                    help='maximum amount of route resampling if route is sampled inside\
                        obstacle (default: 1000)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')

# Neural networks parameters
parser.add_argument('--seed', type=int, default=25450, metavar='Q',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=64, metavar='Q',
                    help='batch size (default: 256)')
parser.add_argument('--replay_size', type=int, default=1000, metavar='Q',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='Q',
                    help='hidden size (default: 256)')
parser.add_argument('--cuda', action="store_true",
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
parser.add_argument('--radius_of_acceptance', type=int, default=600, metavar='O',
                    help='Radius of acceptance for LOS algorithm(default: 600)')
parser.add_argument('--lookahead_distance', type=int, default=450, metavar='O',
                    help='Lookahead distance for LOS algorithm(default: 450)')

args = parser.parse_args()

# Time settings
# time_step = 0.5
# desired_forward_speed_meters_per_second = 8.5
# time_since_last_ship_drawing = 30

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
    current_velocity_component_from_north=-2,
    current_velocity_component_from_east=-2,
    wind_speed=0,
    wind_direction=0
)
mec_mode_params = MachineryModeParams(
    main_engine_capacity=main_engine_capacity,
    electrical_capacity=diesel_gen_capacity,
    shaft_generator_state=hybrid_shaft_gen_as_offline
)
mec_mode = MachineryMode(params=mec_mode_params)
mso_modes = MachineryModes(
    [mec_mode]
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
simulation_setup = SimulationConfiguration(
    initial_north_position_m=0,
    initial_east_position_m=0,
    initial_yaw_angle_rad=45 * np.pi / 180,
    initial_forward_speed_m_per_s=7,
    initial_sideways_speed_m_per_s=0,
    initial_yaw_rate_rad_per_s=0,
    integration_step=args.time_step,
    simulation_time=3600,
)
ship_model = ShipModelAST(ship_config=ship_config,
                       machinery_config=machinery_config,
                       environment_config=env_config,
                       simulation_config=simulation_setup,
                       initial_propeller_shaft_speed_rad_per_s=400 * np.pi / 30)

# Place obstacles
obstacle_data = r'D:\OneDrive - NTNU\PhD\PhD_Projects\ShipTransit_OptiStress\ShipTransit_AST\data\obstacles.txt'
obstacles = StaticObstacle(obstacle_data)

# Set up throttle controller
throttle_controller_gains = ThrottleControllerGains(
    kp_ship_speed=7, ki_ship_speed=0.13, kp_shaft_speed=0.05, ki_shaft_speed=0.005
)
throttle_controller = EngineThrottleFromSpeedSetPoint(
    gains=throttle_controller_gains,
    max_shaft_speed=ship_model.ship_machinery_model.shaft_speed_max,
    time_step=args.time_step,
    initial_shaft_speed_integral_error=114
)

# Set up autopilot controller
route_name = r'D:\OneDrive - NTNU\PhD\PhD_Projects\ShipTransit_OptiStress\ShipTransit_AST\data\start_to_end.txt'
heading_controller_gains = HeadingControllerGains(kp=4, kd=90, ki=0.01)
los_guidance_parameters = LosParameters(
    radius_of_acceptance=args.radius_of_acceptance,
    lookahead_distance=args.lookahead_distance,
    integral_gain=0.002,
    integrator_windup_limit=4000
)

auto_pilot = HeadingBySampledRouteController(
    route_name,
    heading_controller_gains=heading_controller_gains,
    los_parameters=los_guidance_parameters,
    time_step=args.time_step,
    max_rudder_angle=machinery_config.max_rudder_angle_degrees * np.pi/180,
    num_of_samplings=2
)

# Instantiate RL Environment
integrator_term = []
times = []

RL_env = ShipRLEnv(
    ship_model=ship_model,
    auto_pilot=auto_pilot,
    throttle_controller=throttle_controller,
    obstacles = obstacles,
    integrator_term=integrator_term,
    times=times,
    ship_draw=True,
    time_since_last_ship_drawing=30,
)

# # Pseudorandom seeding
random_seed = False
# random_seed = True
if random_seed:
    RL_env.seed(args.seed)
    RL_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

# Agent
agent = SAC(RL_env, args)

# Tensorboard 
writer = SummaryWriter('runs/{}_AST_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'Ship Transit AST_SAC',
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

## Training loop
total_numsteps = 0
updates = 0
testing_count = 0

## STORE SAMPLED ACTION
action_record = defaultdict(list)
# After this develop a function that can read the sampled action record based on episode

## NOTE:
# We have SAC iterations and simulation time steps
# SAC iterations consists of several simulation time steps
# Action consists of: route_point_north, route_point_east, desired_speed
# desired_speed is sampled for each time step
# route points are sampled for each SAC iteration
# For each time step, route points are hold until new route points are sampled

# Count the episode
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = RL_env.reset()
    
    while not done:
        if episode_steps == 0:
            init = True
        else:
            init = False
        
        if args.start_steps > total_numsteps:
            action, action_to_simu_input, sampling_time_record = agent.select_action(state,
                                                                                    done,
                                                                                    init=init,
                                                                                    mode=0) # Random sampling 
            
        else:
            action, action_to_simu_input, sampling_time_record = agent.select_action(state, 
                                                                                    done,
                                                                                    init=init,
                                                                                    mode=1) # Policy based sampling
            
        ## STORE SAMPLED ACTION
        if action_to_simu_input:
            sampled_action_info = np.insert(action, 0, RL_env.ship_model.int.time) # time is not reset here
            action_record[i_episode].append(sampled_action_info)

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range (args.update_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                
                # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                # writer.add_scalar('loss/policy', policy_loss, updates)
                # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1   
                        
        # Convert action to simulation input
        simu_input = agent.convert_action_to_simu_input(action)
                    
        # Step up the simulation
        next_state, reward, done, status = RL_env.step(simu_input, 
                                                       action_to_simu_input, 
                                                       sampling_time_record)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
    
        # Ignore the "done" signal if it comes from hitting the time horizon
        mask = 1 if episode_steps == args.num_steps_episode else float(not done)
        
        # Push the transtition to memory
        ###OBS####
        # Input action inside memory
        memory.push(state, action, reward, next_state, mask)
    
        # Set the next state as current state for the next step
        state = next_state
        
    # Reset the action sampling internal state at the end of episode
    agent.select_action_reset()
    
    # print(episode_steps)

    if total_numsteps > args.num_steps:
        break

    # writer.add_scalar('reward/train', reward, i_episode)
    print("Episode: {}, Total numsteps: {}, Episode steps: {}, Reward: {}, Status:{}".\
        format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2), status))
    
    ## Asses learning performance
    if i_episode % args.scoring_episode_every == 0 and args.eval is True:
        avg_reward = 0.
        for _ in range (args.num_scoring_episodes):
            state = RL_env.reset()
            episode_reward = 0
            episode_steps_eval = 0
            done = False
            while not done:
                if episode_steps_eval < 1:
                    init_eval = True
                else:
                    init_eval = False
                    
                action, action_to_simu_input, sampling_time_record = agent.select_action(state, 
                                                                                        done,
                                                                                        init=init_eval,
                                                                                        mode=2) # Policy based sampling
                
                next_state, reward, done, _ = RL_env.step(action, 
                                                          action_to_simu_input,
                                                          sampling_time_record)
                episode_reward += reward
                
                state = next_state
                
                episode_steps_eval += 1
                
                # Limit the simulator stepping to avoid infinite recursion for debugging
                if episode_steps_eval > args.num_steps_episode:
                    break
                
            avg_reward += episode_reward
            
        avg_reward /= args.num_scoring_episodes
        testing_count += 1
        
        # writer.add_scalar('avg_reward/test', avg_reward, i_episode)
        
        print("----------------------------------------")
        print("Test Number: {}, Avg. Reward: {}".format(testing_count, round(avg_reward, 2)))
        print("----------------------------------------")
        
    # print(np.array(auto_pilot.navigate.north))
    # print(np.array(auto_pilot.navigate.east))

    
    # if i_episode == 1:
    #     # print(ship_model.simulation_results['power me [kw]'])
    #     # print(ship_model.simulation_results['propeller shaft speed [rpm]'])
    #     break

# ## Convert action_record to data frame
# all_action_record = []

# print(len(memory))

# for episode, data in action_record.items():
#     ep_action_record_df = pd.DataFrame(data, columns=["sample time [s]", "route_north [m]", "route_east [m]", "velocity [m/s]"])
#     ep_action_record_df["episode"] = episode
#     all_action_record.append(ep_action_record_df)

# # Concatenate all episodes into one large DataFrame
# action_record_df = pd.concat(all_action_record, ignore_index=True)   

# # Convert episode to a categorical type for efficient memory usage
# action_record_df["episode"] = action_record_df["episode"].astype("category")
# # print(action_record_df.head())

# ## HOW TO RETRIEVE DATA
# # episode_5_action_record = action_record_df[action_record_df["episode"] == 5]
# # sample_time_list = episode_5_action_record["sample_time"].tolist()
# # route_north_list = episode_5_action_record["route_north"].tolist()
# # route_east_list = episode_5_action_record["route_east"].tolist()
# # velocity_list = episode_5_action_record["velocity"].tolist()

# last_action_record = action_record_df[action_record_df["episode"] == i_episode]
# sample_time_list = last_action_record["sample time [s]"].to_list()
# route_north_list = last_action_record["route_north [m]"].to_list()
# route_east_list = last_action_record["route_east [m]"].to_list()
# velocity_list = last_action_record["velocity [m/s]"].to_list()

# ## Store the simulation results in a pandas dataframe
# results_df = pd.DataFrame().from_dict(ship_model.simulation_results)
# # print(results_df.head())

# # # Create a No.2 2x2 grid for subplots
# # fig_2, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
# # axes = axes.flatten()  # Flatten the 2D array for easier indexing

# # # Plot 2.1: Propeller Shaft Speed
# # axes[0].plot(results_df['time [s]'], results_df['propeller shaft speed [rpm]'])
# # axes[0].set_title('Propeller Shaft Speed [rpm]')
# # axes[0].set_xlabel('Time (s)')
# # axes[0].set_ylabel('Propeller Shaft Speed (rpm)')

# # # Plot 2.2: Power vs Available Power
# # axes[2].plot(results_df['time [s]'], results_df['power me [kw]'], label="Power")
# # axes[2].plot(results_df['time [s]'], results_df['available power me [kw]'], label="Available Power")
# # axes[2].set_title('Power vs Available Power [kw]')
# # axes[2].set_xlabel('Time (s)')
# # axes[2].set_ylabel('Power (kw)')
# # axes[2].legend()

# # # Plot 2.3: Cross Track error
# # axes[1].plot(results_df['time [s]'], results_df['cross track error [m]'])
# # axes[1].set_title('Cross Track Error [m]')
# # axes[1].set_xlabel('Time (s)')
# # axes[1].set_ylabel('Cross track error (m)')

# # # Plot 2.4: Fuel Consumption
# # axes[3].plot(results_df['time [s]'], results_df['fuel consumption [kg]'])
# # axes[3].set_title('Fuel Consumption [kg]')
# # axes[3].set_xlabel('Time (s)')
# # axes[3].set_ylabel('Fuel Consumption (kg)')

# # Create a No.1 2x2 grid for subplots
# fig_1, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
# axes = axes.flatten()  # Flatten the 2D array for easier indexing

# # Plot 1.1: Ship trajectory with sampled route
# axes[0].plot(results_df['east position [m]'].to_numpy(), results_df['north position [m]'].to_numpy())
# axes[0].scatter(auto_pilot.navigate.east, auto_pilot.navigate.east, marker='x', color='green')  # Waypoints
# for x, y in zip(ship_model.ship_drawings[1], ship_model.ship_drawings[0]):
#     axes[0].plot(x, y, color='black')
# obstacles.plot_obstacle(axes[0])
# axes[0].set_title('Ship Trajectory with the Sampled Route')
# axes[0].set_xlabel('East position (m)')
# axes[0].set_ylabel('North position (m)')
# axes[0].set_aspect('equal')

# # Plot 1.2: Sampled Route with the Order
# axes[2].scatter(auto_pilot.navigate.east, auto_pilot.navigate.east, marker='x', color='green')
# for i, (east, north) in enumerate(zip(auto_pilot.navigate.east, auto_pilot.navigate.east)):
#     if i == 0 or i == len(auto_pilot.navigate.east)-1: 
#         string = str(i)
#     # elif i == 0:
#     #     string = f"{i}, vel={velocity_list[i-1]:.2f} m/s"
#     else:
#         string = f"{i}, vel={velocity_list[i-1]:.2f} m/s, time={sample_time_list[i-1]:.1f} s"
    
#     axes[2].text(east, north, string, fontsize=8, ha='left', color='blue')  # Label with index
#     radius_circle = Circle((east, north), args.radius_of_acceptance, color='red', alpha=0.3, fill=True)
#     axes[2].add_patch(radius_circle)
# axes[2].set_title('Sampled Route with the Order')
# axes[2].set_xlabel('East position (m)')
# axes[2].set_ylabel('North position (m)')

# # Plot 1.3: Forward Speed
# axes[1].plot(results_df['time [s]'], results_df['forward speed [m/s]'])
# axes[1].set_title('Forward Speed [m/s]')
# axes[1].set_xlabel('Time (s)')
# axes[1].set_ylabel('Forward Speed (m/s)')

# # Plot 1.4: Heading error
# axes[3].plot(results_df['time [s]'], results_df['rudder angle [deg]'])
# axes[3].set_title('Rudder angle [deg]')
# axes[3].set_xlabel('Time (s)')
# axes[3].set_ylabel('Rudder angle [deg]')


# # print(action_record[1])
# # print(times)

# # Adjust layout for better spacing
# plt.tight_layout()
# plt.show()

