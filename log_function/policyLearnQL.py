from utils.utilCosim import *
from utils.utilPolicy import *
from utils.utilRL import *

import numpy as np
import time
import os
import pickle    

def PolicyLearnQL(env,
                  policy,
                  n_episode_value,
                  n_episode_score,
                  logID,
                  logMode,
                  alpha:                float=0.1,
                  gamma:                float=0.9,
                  max_n_steps:            int=10000,
                  print_every:            int=500,
                  save_result:           bool=True,
                  print_last_eps_res:    bool=False,
                  save_dir:               str="./02_Saved/Scores",
                  test_log_dir:           str="./02_Saved/Log/Test",
                  learn_log_dir:          str="./02_Saved/Log/Learn"):
    # Start timer
    start_time = time.time()

    # Learning type
    mode = "QL"

    # Logging mode
    if logMode == "test":
        log_dir = test_log_dir
    elif logMode == "learn":
        log_dir = learn_log_dir
    elif logMode == "none":
        return
    else:
        raise ValueError(f"Log mode '{logMode}' is not available. Please choose from 'test', 'learn', or 'none'.")
    
    # Intial log to rewrite the log text file
    initLog = True
    
    LogMessage("############################### LEARNING PARAMETERS ################################", logID, log_dir, mode, initLog)
    LogMessage("", logID, log_dir, mode)
    LogMessage(f"Policy improvement maximum episodes         : {n_episode_value}", logID, log_dir, mode)
    LogMessage(f"Policy scoring maximum episodes             : {n_episode_score}", logID, log_dir, mode)
    LogMessage(f"Log ID                                      : {logID}", logID, log_dir, mode)
    LogMessage(f"Log mode                                    : {logMode}", logID, log_dir, mode)
    LogMessage(f"Learning rate                               : {alpha}", logID, log_dir, mode)
    LogMessage(f"Maximum number of simulation step           : {max_n_steps}", logID, log_dir, mode)
    LogMessage(f"Observation space (Position [m])            : {env.observation_space.MSDPosition}", logID, log_dir, mode)
    LogMessage(f"Observation space (Velocity [m/s])          : {env.observation_space.MSDVelocity}", logID, log_dir, mode)
    LogMessage(f"Terminal state bound (Position [m])         : {env.terminal_state.MSDPositionTerminal}", logID, log_dir, mode)
    LogMessage(f"Desired height (Position [m])               : {env.y_desired}", logID, log_dir, mode)
    # LogMessage(f"Allowed desired height bound (Position [m]) : {env.y_desiredBound}", logID, log_dir, mode)
    LogMessage(f"Printing for every                          : {print_every} episodes", logID, log_dir, mode)
    
    LogMessage("", logID, log_dir, mode)
    
    LogMessage("############################# Q-LEARNING RESULT LOG ################################", logID, log_dir, mode)
    
    LogMessage("", logID, log_dir, mode)
    
    LogMessage("Note:", logID, log_dir, mode)
    LogMessage("Score is defined as the ratio between the obtained rewards and the total steps", logID, log_dir, mode)
    LogMessage("taken, averaged across all the scoring episodes.", logID, log_dir, mode)

    LogMessage("", logID, log_dir, mode)

    # Create directory if it doesn't exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set q tables file name
    qt_filename = f"best_q_table_{logID}_{logMode}_QL.pkl"

    # Intialize the best score
    best_score = -np.inf

    # Results container
    discrete_states_list_all_episode = []
    states_list_all_episode = []
    rewards_list_all_episode = []
    actions_list_all_episode = []
    terminates_list_all_episode = []

    # Initiate training mode
    policy.train()
    
    ############## EVALUATION-EXPLOITATION CYCLE ##############
    # Through each episode
    for episode in range(n_episode_value):

        # Episodic container
        discrete_states_per_episode = []
        states_per_episode = []
        rewards_per_episode = []
        actions_per_episode = []
        terminates_per_episode = []

        # Set the policy to handle the epsilon decay
        policy.begin_episode(episode)

        ############## SCORING START ##############
        # Print for every couple episode
        if not ((episode + 1) % print_every):
            # Print current episode and epsilon
            LogMessage(f"Q-Learning episode: {episode + 1}, epsilon: {policy.epsilon: .2f}", logID, log_dir, mode)

            # Switch to evaluation mode
            policy.eval()

            # Do policy scoring
            score, _ , total_step = ScorePolicy(env, 
                                                policy, 
                                                max_n_steps, 
                                                n_episode_score)

            ############## SAVING SCORE STARTS ##############
            if save_result:
                # Save the best QL scores
                if score > best_score:
                    best_score = score

                    # Save the best scores and the Q-table
                    save_path = os.path.join(save_dir, qt_filename)
                    with open(save_path, "wb") as f:
                        pickle.dump({
                            "Episode" : episode + 1,
                            "Score"   : score,
                            "Q-table" : policy.q,
                        }, f)
                    
                    # LogMessage(f'New best score of {score:.2%} achieved at episode {episode+1}', logID, log_dir, mode)
                    LogMessage(f'New best score of {score} achieved at episode {episode+1}', logID, log_dir, mode)
                    LogMessage(f'Best Q-table saved to {save_path}', logID, log_dir, mode)
            ############## SAVING SCORES ENDS ##############

            # Print the score in terminal
            LogMessage("Over all scoring episodes:", logID, log_dir, mode)
            # LogMessage(f"Score: {score:.2%} ; Average step taken {np.mean(total_step)}: , Average rewards: {np.mean(total_reward)}",
            #             logID, log_dir, mode)
            LogMessage(f"Score: {score} ; Average step taken {np.mean(total_step)}:",
                        logID, log_dir, mode)

            # Switch to training mode
            policy.train()

            # Print whitespace for readability
            LogMessage("", logID, log_dir, mode)
        ############## SCORING  ENDS ##############

        # THE SNIPPET BELOW IS ONLY FOR INITIALIZING BEFORE THE LOOP PROCESS
        # Reset the environment for RL and Co-Simulation
        env.reset()

        # Get the initial state and discretize it
        states = env.states
        discrete_states = env.DiscretizeState(states)

        discrete_states_per_episode.append(discrete_states)
        
        states_per_episode.append(states)
        discrete_states_per_episode.append(discrete_states)

        # Sample an action based on the policy using the discrete state
        action = policy.sample(discrete_states)

        # Do action value estimation and improvement method using Q-Learning
        for step in range(max_n_steps):
            # Perform forward step for the agent
            next_states, reward, done = env.step(action)
            states_per_episode.append(next_states)
            rewards_per_episode.append(reward)
            actions_per_episode.append(action)
            terminates_per_episode.append(done)

            # Discretize the next state
            next_discrete_states = env.DiscretizeState(next_states)
            discrete_states_per_episode.append(next_discrete_states)

            # Sample next action using next states
            next_action = policy.sample(next_discrete_states)

            ## Compute the state-action value
            # Obtained reward
            term_1 = reward

            # Maximum expected state-action pair value
            term_2 = gamma * np.max(policy.q[next_discrete_states])

            # Current state-action pair value
            term_3 = policy.q[discrete_states, action]

            # Update the Q-tables
            policy.q[discrete_states, action] += alpha * (term_1 + term_2 -term_3)

            # If the episode has finished, compute the action value and then break the loop
            if done:
                break

            # Set the next state-action pair as the current state-action pair for the next action-value update
            discrete_states = next_discrete_states
            action = next_action

            # Add st
            discrete_states_list_all_episode.append(discrete_states_per_episode)
            states_list_all_episode.append(states_per_episode)
            rewards_list_all_episode.append(rewards_per_episode)
            actions_list_all_episode.append(actions_per_episode)
            terminates_list_all_episode.append(terminates_per_episode)

    # Stop timer
    end_time = time.time()
    e_time = end_time - start_time

    # Print result in logfiles
    if print_last_eps_res:
        LogMessage("############################# LAST EPISODE RESULTS #################################", logID, log_dir, mode)
        pos = states_list_all_episode[-1][0][0]
        vel = states_list_all_episode[-1][0][1]
        len_state = len(states_list_all_episode[-1])
        len_step = len(actions_list_all_episode[-1])
        tot_rew = sum(rewards_list_all_episode[-1])
        LogMessage("------------------------------------------------------------------------------------", logID, log_dir, mode)
        LogMessage(f"States visited: {len_state} | Step taken: {len_step} | Total rewards: {tot_rew} | Elapsed Time: {e_time:.1f}", logID, log_dir, mode)
        LogMessage("------------------------------------------------------------------------------------", logID, log_dir, mode)
        LogMessage(f"act:  -    | ter:   -   | rew: -  |  pos: {pos:.3f} | vel: {vel:.3f}", logID, log_dir, mode)
        for i in range(len(rewards_list_all_episode[-1])):
            if actions_list_all_episode[-1][i]==0:
                msg = 'Idle'
            elif actions_list_all_episode[-1][i]==1:
                msg = 'Up'
            else:
                msg = 'Down'
            rew = rewards_list_all_episode[-1][i]
            act = msg
            ter = "False" if terminates_list_all_episode[-1][i]==0 else "True"
            pos = states_list_all_episode[-1][i+1][0]
            vel = states_list_all_episode[-1][i+1][1]
            LogMessage(f"act: {act:<5} | ter: {ter:<5} | rew: {rew:<2} | pos: {pos:>6.3f} | vel: {vel:>6.3f}", logID, log_dir, mode)
    # return discrete_states_list_all_episode, actions_list_all_episode, rewards_list_all_episode
    return