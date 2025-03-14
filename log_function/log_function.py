import os

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
        file_name = f"log_result_{log_ID}.log"
        self.log_file_path = os.path.join(log_dir, file_name) 
        
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
        self.input_log("------------------------------- STRESS TEST PARAMETERS --------------------------------", overwrite=True)
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
        self.input_log("----------------------------------- TRAINING PHASE ------------------------------------") 
        self.input_log("") 
        # self.input_log(f"  : {self.args.}")
        
    def training_log(self, 
                     i_episode, 
                     total_numsteps, 
                     episode_steps, 
                     episode_reward, 
                     status):
        reward = round(episode_reward, 2)
        self.input_log(f"Episode: {i_episode:<3}, Total numsteps: {total_numsteps:<6}, Episode steps: {episode_steps:<5}, Reward: {reward:<8}, Status:{status}")
    
    def evaluation_log(self,
                       testing_count,
                       avg_reward):
        self.input_log("") 
        self.input_log("---------------------------------- EVALUATION PHASE -----------------------------------")
        self.input_log(f"Test Number: {testing_count}, Avg. Reward: {round(avg_reward, 2)}")
        self.input_log("---------------------------------------------------------------------------------------")
        self.input_log("") 
    
    def stepwise_log(self):
        return
    
    def final_log(self):
        return