import os
import pickle
import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # print(f"Debug (Before Storing): action shape = {action.shape}, action = {action}")
        
        # Reshape to a consistent shape
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        action = np.array([action]) if np.isscalar(action) else np.array(action, dtype=np.float32)
        reward = float(reward)
        done = float(done) 
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # print(self.buffer)
        # print(batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        # print(f"Debug (Sampling): action batch shape = {action.shape}")
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, env_name, suffix="", save_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        if save_path is None:
            save_path = "checkpoints/sac_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
