from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity, seed=None):
        self.memory = deque([], maxlen=capacity)
        
        if seed is not None:
            random.seed(seed)

    """
    Each transition is a tuple of (state, action, reward, next_state, done)
    """
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)