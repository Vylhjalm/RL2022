import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
import random
from collections import deque

#Hyperparameters
gamma        = 0.98
seed = 1
random.seed(seed)
torch.manual_seed(seed)

tau          = 0.05
alpha        = 0.2
lr = 0.0005
hidden_size = 256
epsilon = 1e-8
replay_size = 10000
start_steps = 10000
max_steps = 1000000
expl_noise = 0.1
batch_size = 32
update_iteration = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_std = nn.Linear(256, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

    def get_action(self, x):
        mu, std = self.forward(x)
        normal = Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        return action.item(), log_prob.sum().item()

    def train_net(self, state, target_action, log_prob):
        mu, std = self.forward(state)
        normal = Normal(mu, std)
        new_log_prob = normal.log_prob(target_action) - torch.log(1 - target_action.pow(2) + epsilon)
        ratio = torch.exp(new_log_prob - log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        return actor_loss

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_net(self, state, target):
        value = self.forward(state)
        critic_loss = F.mse_loss(value, target)
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()
        return critic_loss

class PPO():
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.memory = []

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor.get_action(state)

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def update(self):
        state, action, reward, next_state, done, log_prob = zip(*self.memory)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)
        log_prob = torch.FloatTensor(log_prob).to(device)

        with torch.no_grad():
            target_value = reward + (1 - done) * gamma * self.critic(next_state)

        advantage = target_value - self.critic(state)
        advantage = (advantage - advantage.mean()) / (advantage.std() + epsilon)

        for i in range(update_iteration):
            for j in range(0, len(self.memory), batch_size):
                mini_batch = zip(state[j:j+batch_size], action[j:j+batch_size], advantage[j:j+batch_size], log_prob[j:j+batch_size])
                for state_b, action_b, advantage_b, log_prob_b in mini_batch:
                    self.actor.train_net(state_b, action_b, log_prob_b)
                    self.critic.train_net(state_b, target_value[j:j+batch_size])

        self.memory = []

def main():
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(state_dim, action_dim)
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(10000):
                a, log_prob = agent.select_action(s)
                s_prime, r, done, info = env.step([a])
                agent.store_transition(s, a, r/100.0, s_prime, done, log_prob)
                s = s_prime

                score += r
                if done:
                    break

            agent.update()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()
