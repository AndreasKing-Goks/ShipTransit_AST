import os
import numpy as np
from numpy import ndarray
import torch
import torch.nn.functional as F
from torch.optim import Adam
from RL_env import ShipRLEnv
from simulator.obstacle import StaticObstacle
from ast_sac.utils import soft_update, hard_update
from ast_sac.nn_models import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, 
                 RL_env: ShipRLEnv, 
                 args):

        self.env = RL_env
        
        self.num_inputs = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space
        
        self.args = args
        
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.max_route_resampling = args.max_route_resampling

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(self.num_inputs, self.action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(self.num_inputs, self.action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(self.num_inputs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(self.num_inputs, self.action_space.shape[0], args.hidden_size, self.action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        # Initialize variables for route point sampling
        self.sampling_frequency = args.sampling_frequency
        self.theta = args.theta
        
        self.AB_north = RL_env.auto_pilot.navigate.north[-1] - RL_env.auto_pilot.navigate.north[0]
        self.AB_east = RL_env.auto_pilot.navigate.east[-1] - RL_env.auto_pilot.navigate.east[0]
        self.AB_alpha = np.arctan2(self.AB_east, self.AB_north)
        self.AB_beta = np.pi/2 - self.AB_alpha
        self.segment_AB_north = self.AB_north / (self.sampling_frequency + 1)
        self.segment_AB_east = self.AB_east / (self.sampling_frequency + 1)
        self.segment_AB = np.sqrt(self.segment_AB_north**2 + self.segment_AB_east**2)
        
        self.total_distance_travelled = 0
        self.distance_travelled = 0
        self.sampling_count = 0
        
        self.last_action = 0
        self.last_route_point_north = 0 
        self.last_route_point_east = 0
        self.last_desired_forward_speed = self.env.expected_forward_speed
        
        self.omega = np.pi/2 - self.AB_beta 
        
        self.x_s = 0
        self.y_s = 0
        self.x_base = self.segment_AB_north
        self.y_base = self.segment_AB_north
        
        self.time_record = 0
        
        self.stop_sampling = False
        
        self.i = 0

    def select_action(self, state, done: bool, init: bool, mode: int):
        
        # Compute action based on mode
        # Transform the state array to tensor
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        action_to_simu_input = False
    
        # print(init, 'eps ', i)
        
        if not self.stop_sampling:

            # Compute traveled distance
            # Only compute travelled distance on the second action sampling
            # First action sampling is directly done at the first time step (flagged as init)
            if not init and len(self.env.ship_model.simulation_results['north position [m]']) > 1:
            # if not init:
                # print(init, 'eps ', self.i)
                # print(self.env.ship_model.north, self.env.ship_model.east)
                # print(self.env.ship_model.simulation_results['north position [m]'])
                # print(self.env.ship_model.simulation_results['east position [m]'])
                dist_trav_north = self.env.ship_model.simulation_results['north position [m]'][-1] - self.env.ship_model.simulation_results['north position [m]'][-2]
                dist_trav_east = self.env.ship_model.simulation_results['east position [m]'][-1] - self.env.ship_model.simulation_results['east position [m]'][-2]
                self.distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)
                self.total_distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)
            
            # Handle sampling condition
            # Do sample action at "init condition" or after the ship has travelled a certain distances
            # OBS
            # self.distance_travelled > self.segment_AB * self.theta should have been the navigatinal error trigger
            # if init or if_reach_radius_of_acceptance
            
            # Input for checking radius of acceptance
            n_pos = self.env.ship_model.north
            e_pos = self.env.ship_model.east
            r_o_a = self.args.radius_of_acceptance
            
            reach_radius_of_acceptance = self.env.auto_pilot.if_reach_radius_of_acceptance(n_pos, e_pos, r_o_a)
            
            if init or self.distance_travelled > self.segment_AB * self.theta:
                # print(init, 'epis ', self.i)
                # print(self.env.ship_model.north, self.env.ship_model.east)
                # print(self.env.ship_model.simulation_results['north position [m]'])
                # print(self.env.ship_model.simulation_results['east position [m]'])
            # if init or reach_radius_of_acceptance:
                ## Sample new action
                # First check if the we still allowed to sample an action
                # according to the allowed sampling frequency
                if self.sampling_count < self.sampling_frequency:
                    ## ACTION SAMPLING MODE EXPLANATION
                    # Mode 0 = Random action sampling directly from action space
                    # Mode 1 = Policy-based action sampling with noise (For training only)
                    # Mode 2 = Policy-based mmean-action sampling (For evaluation only)
                    if mode == 0:
                        action = self.env.action_space.sample().item() if isinstance(self.env.action_space.sample(), ndarray) else self.env.action_space.sample()
                        # action = self.env.action_space.sample()
                    elif mode == 1:
                        action, _, _ = self.policy.sample(state)
                        action = action.detach().cpu().numpy()[0]
                    elif mode == 2:
                        _, _, action = self.policy.sample(state)
                        action = action.detach().cpu().numpy()[0]
                    
                    # Store the sampled action until the next action sampling
                    # The stored action will be used for parameter update
                    # until the new action sampling
                    self.last_action = action
                
                    # Reset distance and increment sampling count
                    self.distance_travelled = 0
                    self.sampling_count += 1
                
                    # Set a flag to implement action to the simulator run
                    # and record time
                    action_to_simu_input = True
                    self.time_record += self.env.ship_model.int.dt
                    
                    sampling_time_record = self.time_record
                    
                    # Reset time record  when we truly sample new action
                    self.time_record = 0
                    
                    self.i += 1
                
                    return action, action_to_simu_input, sampling_time_record
            
            # Reset if sampling limit is reached
            if self.sampling_count == self.sampling_frequency:
                self.distance_travelled = 0
                self.sampling_count = 0  # Reset to start a new cycle
                self.stop_sampling = True # Set as True because we don't want to 
                                          # sample anymore until we reach terminal state
                
            # Or if terminal state is reached
            if done:
                self.distance_travelled = 0
                self.sampling_count = 0  # Reset to start a new cycle
                self.stop_sampling = False # Set as False because we have reached terminal state 
                                           # thus we want to start fresh for the nex action sampling
        
        # INACTION IS ALSO AN ACTION
        # action = np.array([0, self.last_action[1]])    
        
        ## Return last known action because no sampling occurred
        # Keep recording the time
        action = self.last_action
        self.time_record += self.env.ship_model.int.dt
        
        sampling_time_record = self.time_record
        
        self.i += 1
        
        return action, action_to_simu_input, sampling_time_record
    
    
    def convert_action_to_simu_input(self, 
                                     action):
        # Unpack action
        # chi, desired_forward_speed = action
        chi = action
        
        # Compute x_s and _y_s
        l_s = np.abs(self.segment_AB * np.tan(chi))
        self.x_s = l_s * np.cos(self.omega)
        self.y_s = l_s * np.sin(self.omega)
        
        # Compute x_base and y_base
        # Positive sampling angle turn left
        if chi > 0:
            self.x_s *= -1
        else:
            self.y_s *= -1
        
        # print(self.x_base, self.y_base)
        # print(self.x_s, self.y_s)
            
        # Compute next route coordinate
        route_coord_n = self.x_base + self.x_s
        route_coord_e = self.y_base + self.y_s
        
        
        # Update new base for the next route coordinate
        next_segment_factor = self.sampling_count + 1
        self.x_base = (self.segment_AB_north * next_segment_factor) + self.x_s
        self.y_base = (self.segment_AB_east * next_segment_factor) + self.y_s
        
        # Repack into simulation input
        # simu_input = [route_coord_n, route_coord_e, desired_forward_speed]
        simu_input = [route_coord_n, route_coord_e]
        
        return simu_input  
    
    # def convert_action_to_simu_input(self, 
    #                                  action):
    #     # Unpack action
    #     route_shift, desired_forward_speed = action
        
    #     # Check sign and magnitude
    #     route_shift_mg = np.abs(route_shift)
        
    #     ## Add segment
    #     # If route shift negative (shifting to the right)
    #     if route_shift < 0:
    #         route_coord_n = (self.segment_AB_north * self.sampling_count) + (route_shift_mg * np.cos(self.AB_beta))
    #         route_coord_e = (self.segment_AB_east * self.sampling_count) - (route_shift_mg * np.sin(self.AB_beta))
    #     # If route shift positive (shifting to the left)
    #     else:
    #         route_coord_n = (self.segment_AB_north * self.sampling_count) - (route_shift_mg * np.cos(self.AB_beta))
    #         route_coord_e = (self.segment_AB_east * self.sampling_count) + (route_shift_mg * np.sin(self.AB_beta))
        
    #     # Repack into simulation input
    #     simu_input = [route_coord_n, route_coord_e, desired_forward_speed]
        
    #     return simu_input  
    
    def convert_action_reset(self):
        self.distance_travelled = 0
        self.time_record = 0
        self.sampling_count = 0
        self.stop_sampling = False
        self.last_route_point_north = 0 
        self.last_route_point_east = 0
        self.x_s = 0
        self.y_s = 0
        self.x_base = self.segment_AB_north
        self.y_base = self.segment_AB_north
        self.last_desired_forward_speed = self.env.expected_forward_speed          

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, log_dir, best_reward, best_episode, total_numsteps, suffix="best"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Define checkpoint path inside log_dir
        ckpt_path = os.path.join(log_dir, f"sac_checkpoint_{suffix}.pth")
    
        print(f"Saving best model to {ckpt_path} (Episode {best_episode}, Reward {float(best_reward):.2f})")
    
        # Save model, optimizers, and training metadata for resuming training
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'critic_optimizer_state_dict': self.critic_optim.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'best_reward': float(best_reward),  # Track best reward
            'best_episode': int(best_episode),  # Track best episode
            'total_steps': int(total_numsteps)  # Track training progress
        }, ckpt_path)
        
    # Load model parameters
    def load_checkpoint(self, log_dir, suffix="best", evaluate=False, weights_only=True):
        ckpt_path = os.path.join(log_dir, f"sac_checkpoint_{suffix}.pth")

        if not os.path.exists(ckpt_path):
            print(f"No checkpoint found at {ckpt_path}")
            return None, None, None  # No checkpoint found

        print(f"Loading model from {ckpt_path}")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, weights_only=weights_only)

        # Load model weights
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])

        # Load optimizer states
        self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

        # Load training progress
        best_reward = checkpoint.get("best_reward", float('-inf'))  # Default -inf if not found
        best_episode = checkpoint.get("best_episode", 0)  # Default episode 0
        total_steps = checkpoint.get("total_steps", 0)  # Default 0

        # Set model to eval mode if testing
        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()

        print(f"Restored best reward: {best_reward:.2f}, Best episode: {best_episode}, Total steps: {total_steps}")
    
        return best_reward, best_episode, total_steps
    
    
    # def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
    #     if not os.path.exists('checkpoints/'):
    #         os.makedirs('checkpoints/')
    #     if ckpt_path is None:
    #         ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
    #     print('Saving models to {}'.format(ckpt_path))
    #     torch.save({'policy_state_dict': self.policy.state_dict(),  
    #                 'critic_state_dict': self.critic.state_dict(),
    #                 'critic_target_state_dict': self.critic_target.state_dict(),
    #                 'critic_optimizer_state_dict': self.critic_optim.state_dict(),
    #                 'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # # Load model parameters
    # def load_checkpoint(self, ckpt_path, evaluate=False):
    #     print('Loading models from {}'.format(ckpt_path))
    #     if ckpt_path is not None:
    #         checkpoint = torch.load(ckpt_path)
    #         self.policy.load_state_dict(checkpoint['policy_state_dict'])
    #         self.critic.load_state_dict(checkpoint['critic_state_dict'])
    #         self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
    #         self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    #         self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

    #         if evaluate:
    #             self.policy.eval()
    #             self.critic.eval()
    #             self.critic_target.eval()
    #         else:
    #             self.policy.train()
    #             self.critic.train()
    #             self.critic_target.train()

# def sample_action_on_mode(self, state, mode):
#     if mode == 0:
#         act = self.env.action_space.sample()
#         # print(act)
#     elif mode == 1:
#         act, _, _ = self.policy.sample(state)
#         act = act.detach().cpu().numpy()[0]
#     elif mode == 2:
#         _, _, act = self.policy.sample(state)
#         act = act.detach().cpu().numpy()[0]
#     return act
    
# def select_action_lastroute(self, state, done: bool, init: bool, mode: int):
        
#         # Compute action based on mode
#         # Transform the state array to tensor
#         state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

#         # print(self.env.ship_model.simulation_results['north position [m]'])

#         if not self.stop_sampling:

#             # Compute traveled distance
#             if not init and len(self.env.ship_model.simulation_results['north position [m]']) > 1:
#                 dist_trav_north = self.env.ship_model.simulation_results['north position [m]'][-1] - self.env.ship_model.simulation_results['north position [m]'][-2]
#                 dist_trav_east = self.env.ship_model.simulation_results['east position [m]'][-1] - self.env.ship_model.simulation_results['east position [m]'][-2]
#                 self.distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)

#             # Handle sampling condition
#             if init or self.distance_travelled > self.segment_AB * self.theta:
#                 # if init:
#                 #     print(f"Initial route sampling")
#                 # else:
#                 #     print(f"Distance travelled: {self.distance_travelled:.2f}, Threshold: {self.segment_AB * self.theta:.2f}")
            
#                 # Sample action based on mode
#                 if mode == 0:
#                     act = self.env.action_space.sample()
#                 elif mode == 1:
#                     act, _, _ = self.policy.sample(state)
#                     act = act.detach().cpu().numpy()[0]
#                 elif mode == 2:
#                     _, _, act = self.policy.sample(state)
#                     act = act.detach().cpu().numpy()[0]

#                 # Unpack action
#                 north_deviation, east_deviation, desired_forward_speed = act

#                 # Compute new route point
#                 if self.sampling_count < self.sampling_frequency:
#                     route_point_north = north_deviation + self.last_route_point_north
#                     route_point_east = east_deviation + self.last_route_point_east
                    
#                     i = 0
                    
#                     route_is_inside = self.env.obstacles.if_route_inside_obstacles(route_point_north, route_point_east)
                    
#                     while route_is_inside:
                        
#                         # Sample action based on mode
#                         if mode == 0:
#                             act = self.env.action_space.sample()
#                         elif mode == 1:
#                             act, _, _ = self.policy.sample(state)
#                             act = act.detach().cpu().numpy()[0]
#                         elif mode == 2:
#                             _, _, act = self.policy.sample(state)
#                             act = act.detach().cpu().numpy()[0]
                        
#                         # Unpack action
#                         north_deviation, east_deviation, desired_forward_speed = act
                        
#                         # Sample new route until the new route point is not inside the obstacles
#                         route_point_north = north_deviation + self.last_route_point_north
#                         route_point_east = east_deviation + self.last_route_point_east
                        
#                         # Set up counter to limit the the auto-sampling
#                         i += 1
                        
#                         if i == self.max_route_resampling:
#                             print('Achieved maximum route resampling')
#                             break
                        
#                     action = np.array([route_point_north, route_point_east, desired_forward_speed])

#                     # Store the sampled action until the next sampling
#                     self.last_route_point_north = route_point_north 
#                     self.last_route_point_east = route_point_east
#                     self.last_desired_forward_speed = desired_forward_speed
                
#                     # Reset distance and increment sampling count
#                     self.distance_travelled = 0
#                     self.sampling_count += 1
                
#                     sample_flag = True

#                     # np.set_printoptions(precision=2, suppress=True)
#                     # print(f"Sampled action with policy: {action:}, Distance reset.")
                
#                     return action, sample_flag

#             # print(self.sampling_count)
            
#             # Reset if sampling limit or terminal state is reached
#             if self.sampling_count == self.sampling_frequency:
#                 self.distance_travelled = 0
#                 self.sampling_count = 0  # Reset to start a new cycle
#                 self.stop_sampling = True
        
#             if done:
#                 self.distance_travelled = 0
#                 self.sampling_count = 0  # Reset to start a new cycle
#                 self.stop_sampling = False
            

#         # Return last known action if no sampling occurred
#         action = np.array([self.last_route_point_north, self.last_route_point_east, self.last_desired_forward_speed])
#         sample_flag = False
        
#         return action, sample_flag
    
    # def select_action_lastpos(self, state, done: bool, init: bool, mode: int):
        
    #     # Compute action based on mode
    #     # Transform the state array to tensor
    #     state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

    #     # print(self.env.ship_model.simulation_results['north position [m]'])

    #     if not self.stop_sampling:

    #         # Compute traveled distance
    #         if not init and len(self.env.ship_model.simulation_results['north position [m]']) > 1:
    #             dist_trav_north = self.env.ship_model.simulation_results['north position [m]'][-1] - self.env.ship_model.simulation_results['north position [m]'][-2]
    #             dist_trav_east = self.env.ship_model.simulation_results['east position [m]'][-1] - self.env.ship_model.simulation_results['east position [m]'][-2]
    #             self.distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)

    #         # Handle sampling condition
    #         if init or self.distance_travelled > self.segment_AB * self.theta:
    #             # if init:
    #             #     print(f"Initial route sampling")
    #             # else:
    #             #     print(f"Distance travelled: {self.distance_travelled:.2f}, Threshold: {self.segment_AB * self.theta:.2f}")
            
    #             # Sample action based on mode
    #             if mode == 0:
    #                 act = self.env.action_space.sample()
    #             elif mode == 1:
    #                 act, _, _ = self.policy.sample(state)
    #                 act = act.detach().cpu().numpy()[0]
    #             elif mode == 2:
    #                 _, _, act = self.policy.sample(state)
    #                 act = act.detach().cpu().numpy()[0]

    #             # Unpack action
    #             north_deviation, east_deviation, desired_forward_speed = act

    #             # Compute new route point
    #             if self.sampling_count < self.sampling_frequency:
    #                 route_point_north = north_deviation + self.env.ship_model.north
    #                 route_point_east = east_deviation + self.env.ship_model.east
                    
    #                 i = 0
                    
    #                 route_is_inside = self.env.obstacles.if_route_inside_obstacles(route_point_north, route_point_east)
                    
    #                 while route_is_inside:
                        
    #                     # Sample action based on mode
    #                     if mode == 0:
    #                         act = self.env.action_space.sample()
    #                     elif mode == 1:
    #                         act, _, _ = self.policy.sample(state)
    #                         act = act.detach().cpu().numpy()[0]
    #                     elif mode == 2:
    #                         _, _, act = self.policy.sample(state)
    #                         act = act.detach().cpu().numpy()[0]
                        
    #                     # Unpack action
    #                     north_deviation, east_deviation, desired_forward_speed = act
                        
    #                     # Sample new route until the new route point is not inside the obstacles
    #                     route_point_north = north_deviation + self.env.ship_model.north
    #                     route_point_east = east_deviation + self.env.ship_model.east
                        
    #                     # Set up counter to limit the the auto-sampling
    #                     i += 1
                        
    #                     if i == self.max_route_resampling:
    #                         print('Achieved maximum route resampling')
    #                         break
                        
    #                 action = np.array([route_point_north, route_point_east, desired_forward_speed])

    #                 # Store the sampled action until the next sampling
    #                 self.last_route_point_north = route_point_north 
    #                 self.last_route_point_east = route_point_east
    #                 self.last_desired_forward_speed = desired_forward_speed
                
    #                 # Reset distance and increment sampling count
    #                 self.distance_travelled = 0
    #                 self.sampling_count += 1
                
    #                 sample_flag = True

    #                 # np.set_printoptions(precision=2, suppress=True)
    #                 # print(f"Sampled action with policy: {action:}, Distance reset.")
                
    #                 return action, sample_flag

    #         # print(self.sampling_count)
            
    #         # Reset if sampling limit or terminal state is reached
    #         if self.sampling_count == self.sampling_frequency:
    #             self.distance_travelled = 0
    #             self.sampling_count = 0  # Reset to start a new cycle
    #             self.stop_sampling = True
        
    #         if done:
    #             self.distance_travelled = 0
    #             self.sampling_count = 0  # Reset to start a new cycle
    #             self.stop_sampling = False
            

    #     # Return last known action if no sampling occurred
    #     action = np.array([self.last_route_point_north, self.last_route_point_east, self.last_desired_forward_speed])
    #     sample_flag = False
        
    #     return action, sample_flag
    
    # def select_action_segment(self, state, done: bool, init: bool, mode: int):
        
    #     # Compute action based on mode
    #     # Transform the state array to tensor
    #     state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

    #     # print(self.env.ship_model.simulation_results['north position [m]'])

    #     if not self.stop_sampling:

    #         # Compute traveled distance
    #         if not init and len(self.env.ship_model.simulation_results['north position [m]']) > 1:
    #             dist_trav_north = self.env.ship_model.simulation_results['north position [m]'][-1] - self.env.ship_model.simulation_results['north position [m]'][-2]
    #             dist_trav_east = self.env.ship_model.simulation_results['east position [m]'][-1] - self.env.ship_model.simulation_results['east position [m]'][-2]
    #             self.distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)

    #         # Handle sampling condition
    #         if init or self.distance_travelled > self.segment_AB * self.theta:
    #             # if init:
    #             #     print(f"Initial route sampling")
    #             # else:
    #             #     print(f"Distance travelled: {self.distance_travelled:.2f}, Threshold: {self.segment_AB * self.theta:.2f}")
            
    #             # Sample action based on mode
    #             if mode == 0:
    #                 act = self.env.action_space.sample()
    #             elif mode == 1:
    #                 act, _, _ = self.policy.sample(state)
    #                 act = act.detach().cpu().numpy()[0]
    #             elif mode == 2:
    #                 _, _, act = self.policy.sample(state)
    #                 act = act.detach().cpu().numpy()[0]

    #             # Unpack action
    #             north_deviation, east_deviation, desired_forward_speed = act

    #             # Compute new route point
    #             if self.sampling_count < self.sampling_frequency:
    #                 route_point_north = north_deviation + self.segment_AB_north * self.sampling_count
    #                 route_point_east = east_deviation + self.segment_AB_east * self.sampling_count
                    
    #                 i = 0
                    
    #                 route_is_inside = self.env.obstacles.if_route_inside_obstacles(route_point_north, route_point_east)
                    
    #                 while route_is_inside:
                        
    #                     # Sample action based on mode
    #                     if mode == 0:
    #                         act = self.env.action_space.sample()
    #                     elif mode == 1:
    #                         act, _, _ = self.policy.sample(state)
    #                         act = act.detach().cpu().numpy()[0]
    #                     elif mode == 2:
    #                         _, _, act = self.policy.sample(state)
    #                         act = act.detach().cpu().numpy()[0]
                        
    #                     # Unpack action
    #                     north_deviation, east_deviation, desired_forward_speed = act
                        
    #                     # Sample new route until the new route point is not inside the obstacles
    #                     route_point_north = north_deviation + self.segment_AB_north * self.sampling_count
    #                     route_point_east = east_deviation + self.segment_AB_east * self.sampling_count
                        
    #                     # Set up counter to limit the the auto-sampling
    #                     i += 1
                        
    #                     if i == self.max_route_resampling:
    #                         print('Achieved maximum route resampling')
    #                         break
                        
    #                 action = np.array([route_point_north, route_point_east, desired_forward_speed])

    #                 # Store the sampled action until the next sampling
    #                 self.last_route_point_north = route_point_north 
    #                 self.last_route_point_east = route_point_east
    #                 self.last_desired_forward_speed = desired_forward_speed
                
    #                 # Reset distance and increment sampling count
    #                 self.distance_travelled = 0
    #                 self.sampling_count += 1
                
    #                 sample_flag = True

    #                 # np.set_printoptions(precision=2, suppress=True)
    #                 # print(f"Sampled action with policy: {action:}, Distance reset.")
                
    #                 return action, sample_flag

    #         # print(self.sampling_count)
            
    #         # Reset if sampling limit or terminal state is reached
    #         if self.sampling_count == self.sampling_frequency:
    #             self.distance_travelled = 0
    #             self.sampling_count = 0  # Reset to start a new cycle
    #             self.stop_sampling = True
        
    #         if done:
    #             self.distance_travelled = 0
    #             self.sampling_count = 0  # Reset to start a new cycle
    #             self.stop_sampling = False
            

    #     # Return last known action if no sampling occurred
    #     action = np.array([self.last_route_point_north, self.last_route_point_east, self.last_desired_forward_speed])
    #     sample_flag = False
        
    #     return action, sample_flag