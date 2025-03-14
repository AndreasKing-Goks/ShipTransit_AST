import numpy as np
import os

def LogMessage(message,
               logID,
               log_dir,
               mode,
               initLog=False):
    ## Helper function to log or print the message
    # Log file path
    file_name = f"log_result_{logID}_{mode}.txt"
    log_file_path = os.path.join(log_dir, file_name)

    # Check directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Log the message
    if initLog:
        if log_file_path:
            with open(log_file_path, "w") as log_file: # Overwrite mode
                log_file.write(message + "\n")
    else:
        if log_file_path:
            with open(log_file_path, "a") as log_file:
                log_file.write(message + "\n")

def SampleEpisode(env,            # Environment Class
                  policy,         # Policy class 
                  max_n_steps: int=10000,
                  reset:      bool=True):
    
    # Intitialize transition info container
    disc_observations = []
    observations      = []
    actions           = []
    rewards           = []
    dones             = []

    # If reset, initial state is now is the current state
    if reset:
        env.reset()                                         # Reset the environment
        init_states = env.states                            # Get the initial continuous states (from the resetted environment)
        disc_init_states = env.DiscretizeState(init_states) # Discretizes the initial states

    else:
        init_states = env.states                            # Directly get the initial continouos states
        disc_init_states = env.DiscretizeState(init_states) # Discretizes the initial states

    # Add the initial states to the observation container
    disc_observations.append(disc_init_states)
    observations.append(init_states)

    # Go through the timestep
    for step in range(max_n_steps):
        # Sample an action based on the policy
        action = policy.sample(disc_observations[-1])

        # Obtain the next observation by stepping with action
        next_states, reward, done = env.step(action)

        # Discretizes the next_states
        disc_next_states = env.DiscretizeState(next_states)

        # Store the transition info into the containers
        disc_observations.append(disc_next_states)
        observations.append(next_states)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        # Terminate the episode if termination status is active
        if done:
            break
    
    return disc_observations, observations, actions, rewards, dones

def ScorePolicy(env, policy, max_n_steps, n_episodes: int = 10000):
    # Initialize containers
    total_reward_per_episode = []
    total_step_per_episode = []
    # total_score_per_episode = []

    # Go through the policy for each episode
    for episode in range(n_episodes):
        # Sample the episode
        _, _, _, rewards, _ = SampleEpisode(env, policy, max_n_steps, reset=True)

        # Record the goal occurrences and the steps to achieve it
        episode_reward = sum(rewards)
        episode_steps = len(rewards)

        # Record total reward and step count per episode
        total_reward_per_episode.append(episode_reward)
        total_step_per_episode.append(episode_steps)

        # Calculate the normalized score for the episode
        # score_ratio = episode_reward / episode_steps
        # if episode_steps > 0:
        #     score_ratio = episode_reward / episode_steps
        # else:
        #     score_ratio = 0  # Handle case where no steps were taken
        # total_score_per_episode.append(score_ratio)

    # Compute the scores: normalized score across episodes
    score = np.mean(total_reward_per_episode)

    return score, total_reward_per_episode, total_step_per_episode