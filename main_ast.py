from simulator.ship_model import  ShipConfiguration, EnvironmentConfiguration, SimulationConfiguration, ShipModelAST
from simulator.ship_engine import MachinerySystemConfiguration, MachineryMode, MachineryModeParams, MachineryModes, SpecificFuelConsumptionBaudouin6M26Dot3, SpecificFuelConsumptionWartila6L26
from simulator.LOS_guidance import LosParameters, StaticObstacle
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
parser.add_argument('--sampling_frequency', type=int, default=5, metavar='G',
                    help='maximum amount of action sampling per episode (default: 9)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')

# Neural networks parameters
parser.add_argument('--seed', type=int, default=123456, metavar='Q',
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
parser.add_argument('--num_steps', type=int, default=3000, metavar='N',
                    help='maximum number of steps across all episodes (default: 100000)')
parser.add_argument('--num_steps_episode', type=int, default=2000, metavar='N',
                    help='Maximum number of steps per episode to avoid infinite recursion')
parser.add_argument('--start_steps', type=int, default=100, metavar='N',
                    help='Steps sampling random actions (default: 1000)')
parser.add_argument('--update_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--sampling_step', type=int, default=10, metavar='N',
                    help='Step for doing full action sampling (default:1000')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--scoring_episode_every', type=int, default=40, metavar='N',
                    help='Number of every episode to evaluate learning performance(default: 40)')
parser.add_argument('--num_scoring_episodes', type=int, default=20, metavar='N',
                    help='Number of episode for learning performance assesment(default: 20)')

# Others
parser.add_argument('--radius_of_acceptance', type=int, default=100, metavar='O',
                    help='Radius of acceptance for LOS algorithm(default: 600)')
parser.add_argument('--lookahead_distance', type=int, default=150, metavar='O',
                    help='Lookahead distance for LOS algorithm(default: 600)')

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
obstacle_data = np.loadtxt('D:\OneDrive - NTNU\PhD\PhD_Projects\ShipTransit_OptiStress\ShipTransit_Simu\obstacles.txt')
list_of_obstacles = []
for obstacle in obstacle_data:
    list_of_obstacles.append(StaticObstacle(obstacle[0], obstacle[1], obstacle[2]))

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
route_name = r'D:\OneDrive - NTNU\PhD\PhD_Projects\ShipTransit_OptiStress\ShipTransit_AST\start_to_end'
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
    integrator_term=integrator_term,
    times=times,
    ship_draw=True,
    time_since_last_ship_drawing=30,
)

# Pseudorandom seeding
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
            action, sample_flag = agent.select_action(state,
                                         done,
                                         init=init,
                                         mode=0) # Random sampling
        else:
            action, sample_flag = agent.select_action(state, 
                                         done,
                                         init=init,
                                         mode=1) # Policy based sampling

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range (args.update_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1   
                        
        # print('here2')            
        next_state, reward, done = RL_env.step(action, sample_flag)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
    
        # Ignore the "done" signal if it comes from hitting the time horizon
        mask = 1 if episode_steps == args.num_steps_episode else float(not done)
        
        # ONLY FOR TRAINING, WHEN EPISODE STEPS IS LIMITED
        # Limit the simulator stepping to avoid infinite recursion for debugging
        if episode_steps > args.num_steps_episode:
            break
    
        # Push the transtition to memory
        memory.push(state, action, reward, next_state, mask)
    
        # Set the next state as current state for the next step
        state = next_state
        
    # print(episode_steps)

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    
    ## Asses learning performance
    if i_episode % args.scoring_episode_every == 0 and args.eval is True:
        avg_reward = 0.
        testing_count = 0
        for _ in range (args.num_scoring_episodes):
            state = RL_env.reset()
            episode_reward = 0
            episode_steps_eval = 0
            done = False
            while not done:
                if episode_steps_eval < 2:
                    init_eval = True
                else:
                    init_eval = False
                    
                action, sample_flag = agent.select_action(state, 
                                         done,
                                         init=init_eval,
                                         mode=2) # Policy based sampling
                
                next_state, reward, done = RL_env.step(action, sample_flag)
                episode_reward += reward
                
                state = next_state
                
                episode_steps_eval += 1
                
                # Limit the simulator stepping to avoid infinite recursion for debugging
                if episode_steps_eval > args.num_steps_episode:
                    break
                
            avg_reward += episode_reward
            
        avg_reward /= args.num_scoring_episodes
        testing_count += 1
        
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)
        
        print("----------------------------------------")
        print("Test Number: {}, Avg. Reward: {}".format(testing_count, round(avg_reward, 2)))
        print("----------------------------------------")
        
# Store the simulation results in a pandas dataframe
results = pd.DataFrame().from_dict(ship_model.simulation_results)

# Example on how a map-view can be generated
map_fig, map_ax = plt.subplots()
map_ax.plot(results['east position [m]'].to_numpy(), results['north position [m]'].to_numpy())
map_ax.scatter(auto_pilot.navigate.east, auto_pilot.navigate.north, marker='x', color='green')  # Plot the waypoints
for x, y in zip(ship_model.ship_drawings[1], ship_model.ship_drawings[0]):
    map_ax.plot(x, y, color='black')
# for obstacle in list_of_obstacles:
#     obstacle.plot_obst(ax=map_ax)

map_ax.set_aspect('equal')

route_fig, route_ax = plt.subplots()
route_ax.scatter(auto_pilot.navigate.east, auto_pilot.navigate.north, marker='x', color='green')
for i, (east, north) in enumerate(zip(auto_pilot.navigate.east, auto_pilot.navigate.north)):
    route_ax.text(east, north, str(i), fontsize=8, ha='right', color='blue')  # Label with the index
    # `ha='right'` aligns the text to the right of the point; adjust as needed
    radius_circle = Circle((east, north), args.radius_of_acceptance, color='red', alpha=0.3, fill=True)
    route_ax.add_patch(radius_circle)

# # Example on plotting time series
# fuel_fig, fuel_ax = plt.subplots()
# results.plot(x='time [s]', y='power [kw]', ax=fuel_ax)

# int_fig, int_ax = plt.subplots()
# int_ax.plot(times, integrator_term)

forward_speed_fig, forward_speed_ax = plt.subplots()
forward_speed_ax.set_title('Forward Speed [m/s]')
forward_speed_ax.plot(ship_model.simulation_results['forward speed [m/s]'])

shaft_speed_fig, shaft_speed_ax = plt.subplots()
shaft_speed_ax.set_title('Propeller shaft speed [rpm]')
shaft_speed_ax.plot(ship_model.simulation_results['propeller shaft speed [rpm]'])

power_fig, power_ax = plt.subplots()
power_ax.set_title('Power vs Available Power [kw] ')
power_ax.plot(ship_model.simulation_results['power me [kw]'])
power_ax.plot(ship_model.simulation_results['available power me [kw]'])

# print(np.array(auto_pilot.navigate.north))
# print(np.array(auto_pilot.navigate.east))
                      
plt.show()
