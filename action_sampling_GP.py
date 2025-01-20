import numpy as np
from ast_sac.nn_models import *
from RL_env import *
from torch.distributions import Normal

# Instantiate RL Environment
env = ShipRLEnv()

observation_space = env.observation_space
action_space = env.action_space

# Instantiate Policy

num_inputs = 7 # [n_pos, e_pos, heading, forward speed, shaft_speed, e_ct, power_load]
num_actions = 3 # [route_point_n, route_point_e, desired_speed]
hidden_dim =  36

gaussian_policy = GaussianPolicy(num_inputs, num_actions, hidden_dim, action_space)

state = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32)

# mean, log_std = gaussian_policy.forward(state)

# std = log_std.exp()

# # print(mean, std)

# normal = Normal(mean, std)

# x_t = normal.rsample()

# y_t = torch.tanh(x_t)

# action = y_t * gaussian_policy.action_scale + gaussian_policy.action_bias

# log_prob = normal.log_prob(action)

# print(action)
# print(log_prob)


action, log_prob, mean = gaussian_policy.sample(state)

print(action, log_prob, mean)
