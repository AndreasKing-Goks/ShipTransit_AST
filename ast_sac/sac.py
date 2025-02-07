import os
import numpy as np
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
        
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

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
        self.segment_AB_north = self.AB_north / (self.sampling_frequency + 1)
        self.segment_AB_east = self.AB_east / (self.sampling_frequency + 1)
        self.segment_AB = np.sqrt(self.segment_AB_north**2 + self.segment_AB_east**2)
        
        self.distance_travelled = 0
        self.sampling_count = 0
        
        self.last_route_point_north = 0 
        self.last_route_point_east = 0
        self.last_desired_forward_speed = self.env.desired_forward_speed
        
        self.stop_sampling = False
        
    
    def select_action(self, state, done: bool, init: bool, mode: int):
        
        # Compute action based on mode
        # Transform the state array to tensor
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        # print(self.env.ship_model.simulation_results['north position [m]'])

        if not self.stop_sampling:

            # Compute traveled distance
            if not init and len(self.env.ship_model.simulation_results['north position [m]']) > 1:
                dist_trav_north = self.env.ship_model.simulation_results['north position [m]'][-1] - self.env.ship_model.simulation_results['north position [m]'][-2]
                dist_trav_east = self.env.ship_model.simulation_results['east position [m]'][-1] - self.env.ship_model.simulation_results['east position [m]'][-2]
                self.distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)

            # Handle sampling condition
            if init or self.distance_travelled > self.segment_AB * self.theta:
                # if init:
                #     print(f"Initial route sampling")
                # else:
                #     print(f"Distance travelled: {self.distance_travelled:.2f}, Threshold: {self.segment_AB * self.theta:.2f}")
            
                # Sample action based on mode
                if mode == 0:
                    act = self.env.action_space.sample()
                elif mode == 1:
                    act, _, _ = self.policy.sample(state)
                    act = act.detach().cpu().numpy()[0]
                elif mode == 2:
                    _, _, act = self.policy.sample(state)
                    act = act.detach().cpu().numpy()[0]

                # Unpack action
                north_deviation, east_deviation, desired_forward_speed = act

                # Compute new route point
                if self.sampling_count < self.sampling_frequency:
                    route_point_north = north_deviation + self.segment_AB_north * self.sampling_count
                    route_point_east = east_deviation + self.segment_AB_east * self.sampling_count
                    
                    i = 0
                    
                    route_is_inside = self.env.obstacles.if_route_inside_obstacles(route_point_north, route_point_east)
                    
                    # print(self.sampling_count)
                    
                    while route_is_inside  or i == 10:
                        
                        # Sample action based on mode
                        if mode == 0:
                            act = self.env.action_space.sample()
                        elif mode == 1:
                            act, _, _ = self.policy.sample(state)
                            act = act.detach().cpu().numpy()[0]
                        elif mode == 2:
                            _, _, act = self.policy.sample(state)
                            act = act.detach().cpu().numpy()[0]
                        
                        # Unpack action
                        north_deviation, east_deviation, desired_forward_speed = act
                        
                        # Sample new route until the new route point is not inside the obstacles
                        route_point_north = north_deviation + self.segment_AB_north * self.sampling_count
                        route_point_east = east_deviation + self.segment_AB_east * self.sampling_count
                        
                        # Set up counter to limit the the auto-sampling
                        i += 1
                        
                        if i == 100:
                            print('Achieved max re-sampling')
                            break
                        
                    action = np.array([route_point_north, route_point_east, desired_forward_speed])

                    # Store the sampled action until the next sampling
                    self.last_route_point_north = route_point_north 
                    self.last_route_point_east = route_point_east
                    self.last_desired_forward_speed = desired_forward_speed
                
                    # Reset distance and increment sampling count
                    self.distance_travelled = 0
                    self.sampling_count += 1
                
                    sample_flag = True

                    # np.set_printoptions(precision=2, suppress=True)
                    # print(f"Sampled action with policy: {action:}, Distance reset.")
                
                    return action, sample_flag

            # print(self.sampling_count)
            
            # Reset if sampling limit or terminal state is reached
            if self.sampling_count == self.sampling_frequency:
                self.distance_travelled = 0
                self.sampling_count = 0  # Reset to start a new cycle
                self.stop_sampling = True
        
            if done:
                self.distance_travelled = 0
                self.sampling_count = 0  # Reset to start a new cycle
                self.stop_sampling = False
            

        # Return last known action if no sampling occurred
        action = np.array([self.last_route_point_north, self.last_route_point_east, self.last_desired_forward_speed])
        sample_flag = False
        
        return action, sample_flag
      
    def select_action_reset(self):
        self.sampling_count = 0
        self.stop_sampling = False

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
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
