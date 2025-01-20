from simulator.ship_model import  ShipConfiguration, EnvironmentConfiguration, SimulationConfiguration, ShipModel
from simulator.ship_engine import MachinerySystemConfiguration, MachineryMode, MachineryModeParams, MachineryModes, SpecificFuelConsumptionBaudouin6M26Dot3, SpecificFuelConsumptionWartila6L26
from simulator.LOS_guidance import LosParameters, StaticObstacle
from simulator.controllers import ThrottleControllerGains, HeadingControllerGains, EngineThrottleFromSpeedSetPoint, HeadingBySampledRouteController

from RL_env import ShipRLEnv
from ast_sac.nn_models import *
from ast_sac.sac import SAC
from ast_sac.replay_memory import ReplayMemory

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import datetime
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

# Argument Parser
parser = argparse.ArgumentParser(description='Ship Transit Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every scoring episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--scoring_episode', type=int, default=20, metavar='N',
                    help='Number of every episode to evaluate learning performance(default: 20)')
parser.add_argument('--sampling_step', type=int, default=1000, metavar='N',
                    help='Step for doing full action sampling (default:1000')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=100001, metavar='N',
                    help='maximum number of steps across all episodes (default: 100000)')
parser.add_argument('--time_step', type=int, default=0.5, metavar='N',
                    help='time step size in second for ship transit simulator (default: 0.5)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--update_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 1000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
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
ship_model = ShipModel(ship_config=ship_config,
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
    radius_of_acceptance=600,
    lookahead_distance=500,
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
agent = SAC(RL_env.observation_space.shape[0], RL_env.action_space, args)

# Tensorboard 
writer = SummaryWriter('runs/{}_AST_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'Ship Transit AST_SAC',
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

def partial_sample_action(episode_steps, 
                          state, 
                          hold, 
                          total_numsteps, 
                          evaluate=False):
    """
    Perform partial sampling: full sampling every args.sampling_step, otherwise partial sampling.
    """
    # Full sampling at sampling step or episode start
    # During evaluation, no need for random sampling
    if evaluate:
        if episode_steps % args.sampling_step == 0 or episode_steps == 0:
            action = agent.select_action(state, evaluate=evaluate) # Policy based sampling
            hold[0], hold[1] = action[0], action[1]
        else:
            action = agent.select_action(state, evaluate=evaluate) # Policy based sampling
            action[0], action[1] = hold[0], hold[1]
            
    # For training
    else:
        if episode_steps % args.sampling_step == 0 or episode_steps == 0:
            if args.start_steps > total_numsteps:
                action = RL_env.action_space.sample() # Random sampling
            else:
                action = agent.select_action(state, evaluate=evaluate) # Policy based sampling
            hold[0], hold[1] = action[0], action[1]
        else:
            if args.start_steps > total_numsteps:
                action = RL_env.action_space.sample()  # Random sampling
                action[0], action[1] = hold[0], hold[1]
            else:
                action = agent.select_action(state, evaluate=evaluate) # Policy based sampling
                action[0], action[1] = hold[0], hold[1]
    
    return action

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
        ## Do partial sampling for every time step
        # And full sampling for every 'sampling step'
        route_point_north_hold = 0
        route_point_east_hold = 0
        
        ## Sample an action
        # action = partial_sample_action(
        #     episode_steps=episode_steps,
        #     state=state,
        #     hold=[route_point_north_hold, route_point_east_hold],
        #     total_numsteps=total_numsteps,
        #     evaluate=False
        #     )
        
        if args.start_steps > total_numsteps:
            action = RL_env.action_space.sample() # Random sampling
        else:
            action = agent.select_action(state, evaluate=False) # Policy based sampling

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
            
        next_state, reward, done = RL_env.step(action)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
    
        # Ignore the "done" signal if it comes from hitting the time horizon
        mask = 1 if episode_steps == RL_env._max_episode_steps else float(not done)
    
        # Push the transtition to memory
        memory.push(state, action, reward, next_state, mask)
    
        # Set the next state as current state for the next step
        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    
    ## Asses learning performance
    if i_episode % args.scoring_episode == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _ in range (episodes):
            state = RL_env.reset()
            episode_reward = 0
            episode_steps_eval = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                
                # action = partial_sample_action(
                #     episode_steps=episode_steps_eval,
                #     state=state,
                #     hold=[route_point_north_hold, route_point_east_hold],
                #     total_numsteps=None,
                #     evaluate=False
                #     )
                
                next_state, reward, done = RL_env.step(action)
                episode_reward += reward
                
                state = next_state
                
                episode_steps_eval += 1
            avg_reward += episode_reward
        avg_reward /= episodes
        
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)
        
        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")