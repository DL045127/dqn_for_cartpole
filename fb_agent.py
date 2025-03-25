import gymnasium as gym
import flappy_bird_gymnasium
import torch

import matplotlib
import matplotlib.pyplot as plt

from torch import nn
from dqn import DQN
from experience_replay import ReplayBuffer

from datetime import datetime, timedelta
import argparse
import itertools
import yaml
import random
import os
import numpy as np

DATE_FORMAT = "%n-%d %H:%H:%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Use to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')


# Chec if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force CPU

# Agent class for DQN
class Agent:
    """
    A Deep Q-Network (DQN) agent for training and running 
    on a gym environment


    Parameters:
        hp_set (str): hyperparameter set to use
        hidden_dim (int, optional): Number of hidden units in the network
        device (str, optional): Device to run the agent on ('cpu' or 'cuda') 
    """
    def __init__(self, hp_set, hidden_dim=256, device='cpu'):
        self.hidden_dim = hidden_dim
        # self.dqn = DQN(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.device = device
        
        # load in hyperparameters
        with open('config.yaml', 'r') as file:
            all_hyperparameters = yaml.safe_load(file)
            hyperparameters = all_hyperparameters[hp_set]\
            
        self.hp_set = hp_set

        # initialize hyperparameters
        self.capacity = hyperparameters['capacity']
        self.batch_size = hyperparameters['batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_final = hyperparameters['epsilon_final']
        self.df = hyperparameters['discount_factor']
        self.lr = hyperparameters['learning_rate']
        self.sync_rate = hyperparameters['sync_rate']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.env_id = hyperparameters['env_id']

        # initialize loss function adn optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hp_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hp_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hp_set}.png')

    
    # function to run the agent
    def run(self, training=True, render=False):
        """
        Runs the agent in the environment

        Parameters:
            training (bool): Flag to indicate training or testing
        """

        if training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f'{start_time.strftime(DATE_FORMAT)}: Training starting...'
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')


        env = gym.make(self.env_id, render_mode='human' if render else None)
        action_dim = env.action_space.n
        state_dim = env.observation_space.shape[0]
        rewards_per_episode = []

        policy_dqn = DQN(state_dim, action_dim, self.hidden_dim).to(self.device)

        # initialize replay buffer, epsilon, target network, and optimizer if we are in training mode
        if training:
            memory = ReplayBuffer(capacity=self.capacity)
            epsilon = self.epsilon_init
            target_dqn = DQN(state_dim, action_dim, self.hidden_dim).to(self.device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.lr)

            epsilon_history = []
            best_reward = -9999999

            step_count = 0
        else:
            # Load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_dqn.eval()


        # loop for episodes
        for episode in itertools.count():
            # reset state
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=self.device)
            terminated = False
            episode_reward = 0.0

            while (not terminated) and (episode_reward < self.stop_on_reward):
                # epsilon-greedy policy
                if training and random.random() < epsilon:
                    # take random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=self.device)
                else:
                    # estimate q-values and retrieve highest q-value action
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                    
                # take action
                new_state, reward, terminated, _, info = env.step(action.item())
                episode_reward += reward

                # convert new_state and reward to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float, device=self.device)

                if training:
                    # append transition to memory
                    memory.append((state, action, reward, new_state, terminated))

                    step_count += 1

                # update state
                state = new_state

            # update rewards and epsilon
            rewards_per_episode.append(episode_reward)

            if training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # train the agent
                if len(memory) > self.batch_size:
                    batch = memory.sample(self.batch_size)
                    self.optimize(batch, policy_dqn, target_dqn)

                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_final)
                    epsilon_history.append(epsilon)

                    # copy policy network to target network after number of steps
                    if step_count > self.sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
    
    def optimize(self, batch, policy_dqn, target_dqn):
        """
        Optimizes the policy network using a batch of transitions

        Parameters:
            batch (list): A batch of experiences sampled from the replay buffer
            policy_dqn (torch.nn.Module): The policy network (DQN)
            target_dqn (torch.nn.Module): The target network (DQN)
        """
        # retrieve all experiences at once
        states, actions, rewards, new_states, terminations = zip(*batch)

        # convert to tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(self.device)

        # calculate target q-values
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.df * target_dqn(new_states).max(dim=1)[0]

        # calculate current q-values
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # calculate loss and optimize
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range((len(mean_rewards))):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='training mode', action='store_true')
    args = parser.parse_args()

    agent = Agent(args.hyperparameters, device=device)

    if args.train:
        agent.run(training=True)
    else:
        agent.run(training=False, render=True)