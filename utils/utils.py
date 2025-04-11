import pandas as pd

def action_record_to_df(action_record):
    all_action_record = []
    
    for episode, data in action_record.items():
        ep_action_record_df = pd.DataFrame(data, columns=["scoping_angle [deg]", "route_north [m]", "route_east [m]", "cumulative_rewards"])
        ep_action_record_df["episode"] = episode
        all_action_record.append(ep_action_record_df)
    
    action_record_df = pd.concat(all_action_record, ignore_index=True) 
    
    # Convert episode to a categorical type for efficient memory usage
    action_record_df["episode"] = action_record_df["episode"].astype("category")
    
    return action_record_df