import os
import json
import numpy as np

class LogMessage:
    ''' Class for log functionality
    '''
    def __init__(self,
                 log_dir,
                 log_ID,
                 args):
        
        # Arguments
        self.args = args
        
        # Log file path
        log_file_name = f"log_result_{log_ID}.log"
        self.log_file_path = os.path.join(log_dir, log_file_name) 
        
        # Episode record path
        record_file_name = f"episodes_record_{log_ID}.json"
        self.all_episodes_log_path = os.path.join(log_dir, record_file_name)
        
        # Check directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    def input_log(self, 
                  message,
                  overwrite=False):
        if overwrite:
            if self.log_file_path:
                with open(self.log_file_path, "w") as log_file:
                    log_file.write(message + "\n")
        else:
            if self.log_file_path:
                with open(self.log_file_path, "a") as log_file:
                    log_file.write(message + "\n")
            
    def initial_log(self):
        self.input_log("-------------------------------------------------------- STRESS TEST PARAMETERS ---------------------------------------------------------", overwrite=True)
        self.input_log("* Coefficient and boolean parameters")
        self.input_log(f" Policy type                                   : {self.args.policy}")
        self.input_log(f" Do evaluation                                 : {self.args.eval}")
        self.input_log(f" Reward discount factor (gamma)                : {self.args.gamma}")
        self.input_log(f" Target smoothing coefficient (tau)            : {self.args.tau}")
        self.input_log(f" Action sampling frequency coefficient (theta) : {self.args.theta}")
        self.input_log(f" Maximum action sampling count per episode     : {self.args.sampling_frequency}")
        self.input_log(f" RL learning rate                              : {self.args.lr}")
        self.input_log(f" SAC's entropy temperature parameters (alpha)  : {self.args.alpha}")
        self.input_log(f" Do automatic entroypy tuning                  : {self.args.automatic_entropy_tuning}")
        self.input_log("")
        self.input_log("* Neural networks parameters")
        self.input_log(f" Set random seed                               : {self.args.seed}")
        self.input_log(f" SAC batch size                                : {self.args.batch_size}")
        self.input_log(f" SAC replay size                               : {self.args.replay_size}")
        self.input_log(f" SAC hidden network size                       : {self.args.hidden_size}")
        self.input_log(f" Run training on CUDA                          : {self.args.cuda}")
        self.input_log("")
        self.input_log("* Timesteps and episode parameters")
        self.input_log(f" Simulator time step size                      : {self.args.time_step}")
        self.input_log(f" Maximum steps count over all episodes         : {self.args.num_steps}")
        self.input_log(f" Steps count for initial exploration           : {self.args.start_steps}")
        self.input_log(f" Steps count for model update                  : {self.args.update_per_step}")
        self.input_log(f" Steps count for target update                 : {self.args.target_update_interval}")
        self.input_log(f" Episodes count for evaluation                 : {self.args.scoring_episode_every}")
        self.input_log(f" Episodes count during evaluation              : {self.args.num_scoring_episodes}")
        self.input_log("")
        self.input_log("* Others")
        self.input_log(f" Autopilot LOS's radius of acceptance          : {self.args.radius_of_acceptance}")
        self.input_log(f" Autopilot LOS's lookahead distance            : {self.args.lookahead_distance}")
        self.input_log("") 
        self.input_log("------------------------------------------------------------ TRAINING PHASE -------------------------------------------------------------") 
        self.input_log("") 
        # self.input_log(f"  : {self.args.}")
        
    def training_log(self, 
                     i_episode, 
                     total_numsteps, 
                     episode_steps, 
                     episode_reward, 
                     travel_distance,
                     travel_time,
                     status):
        reward = round(episode_reward, 2)
        self.input_log(f"Episode: {i_episode:<4}, Total numsteps: {total_numsteps:<6}, Episode steps: {episode_steps:<5}, Travel distanced: {travel_distance:<6.2f}, Travel time: {travel_time:<5.2f} Reward: {reward:<8.2f}")
        self.input_log(f"Status:{status}")
    
    def evaluation_log(self,
                       testing_count,
                       avg_reward):
        self.input_log("") 
        self.input_log("----------------------------------------------------------- EVALUATION PHASE ------------------------------------------------------------")
        self.input_log(f"Test Number: {testing_count}, Avg. Reward: {round(avg_reward, 2):.2f}")
        self.input_log("-----------------------------------------------------------------------------------------------------------------------------------------")
        self.input_log("") 
    
    def simulation_step_log(self,
                            episode_record,
                            i_episode,
                            log=False):
        if log:
            # Initial log formatting
            self.input_log(" || ")
            self.input_log("_||_")
            self.input_log("\  /")
            self.input_log(" \/ ")
            self.input_log("*********************************************************** Simulation Step *************************************************************")
            self.input_log("# Step 0, act: -, - , ter: - , rew: - ")
            self.input_log("  n_pos: -        | e_pos: -        | head: -      | f_spd: -    | s_spd: -       | e_ct: -       | p_load: -")

            # Loop through each step in the episode record
            for step in range(len(episode_record[i_episode]["sampled_action"])):
                # Extract stepwise values
                act = episode_record[i_episode]["sampled_action"][step]
                done = episode_record[i_episode]["termination"][step]
                reward = episode_record[i_episode]["rewards"][step]
                state = episode_record[i_episode]["states"][step]
                
                # Unpack action values
                    # Convert NumPy array to list if necessary
                if isinstance(act, np.ndarray):
                    act = act.tolist()
                
                if isinstance(act, (list, tuple)) and len(act) == 2:
                    scoping_angle = act[0] 
                    desired_forward_speed = act [1]
                else:
                    scoping_angle, desired_forward_speed = "Unknown", "Unknown"  # Fallback in case of unexpected format

                # Unpack state values
                if isinstance(state, (list, tuple)) and len(state) >= 7:
                    n_pos, e_pos, head, f_spd, s_spd, e_ct, p_load = state[:7]  # Ensure only first 7 elements are used
                else:
                    n_pos, e_pos, head, f_spd, s_spd, e_ct, p_load = ["Unknown"] * 7  # Fallback for incorrect state size

                # Log the extracted values
                self.input_log(f"# Step {step+1}, act: {scoping_angle:.2f}, {desired_forward_speed:.2f}, ter: {done}, rew: {reward:.2f} ")
                self.input_log(f"  n_pos: {n_pos:>8.2f} | e_pos: {e_pos:>8.2f} | head: {head:>6.2f} | f_spd: {f_spd:>4.2f} | s_spd: {s_spd::>7.2f} | e_ct: {e_ct:>7.2f} | p_load: {p_load:>7.2f}")
        
    def save_episode_record(self, 
                            episode_record, 
                            save_record=True):
        if save_record:
            # Convert NumPy arrays to lists for JSON compatibility
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj

            # Convert before saving
            episode_record_json_compatible = convert_numpy(episode_record)

            # Save the converted dictionary
            with open(self.all_episodes_log_path, "w") as f:
                json.dump(episode_record_json_compatible, f, indent=4)