#this is a main folder known as agents.py
#inside agents.py there are 4 files that are
#1.discountagent
#2.fixedpriceagent
#3.Qlearningagent
#4.DQNagent


#discount_agent.py

class DiscountAgent:
    def act(self, state):
        seats_remaining, days_remaining = state

        if days_remaining > 20:
            return 8
        elif days_remaining > 10:
            return 5
        else:
            return 2
#fixed agent

class FixedPriceAgent:
    def __init__(self):
        self.action = 4

    def act(self, state):
        return self.action


# Project 2: # Reinforcement Learning for Dynamic Pricing 
# Role : Q-Learning Developer &  Baseline Strategies and Q-Learning 
# WHAT THEY BUILD Q-LearningAgent.py

import numpy as np 
import random

class QLearningAgent:
    def __init__(self):
        pass
    # Hyperparameters
        self.learning_rate = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        #(x_bin,y_bin,action)
        self.q_table = np.zeros((10,10,5))

    def get_bins(self,state):
        x = min(9,  max(0,  int(state[0])))
        y = min(9,  max(0,  int(state[1])))
        return x,y

    def act(self,state):
        x, y = self.get_bins(state)

        #Epsilon-greedy starategy
        if np.random.rand() < self.epsilon:
            return np.random.randint(5)  # explore
        else:
            return np.argmax(self.q_table[x,y])  # exploit

    def learn(self, state, action, reward, next_state, done):
          x, y = self.get_bins(state)
          nx, ny =self.get_bins(next_state)

          current_q = self.q_table[x,y, action]

          if done:
              target_q = reward
          else:
             target_q = reward + self.gamma * np.max(
                 self.q_table[nx,ny]
             )
          self.q_table[x, y, action] += self.learning_rate  * (
             target_q - current_q
         )
          if self.epsilon > self.epsilon_min:
             self.epsilon *=  self.epsilon_decay 
             
            
            #DQN network agent

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

class DQNAgent:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=5000)

        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        # Models
        self.model = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()
    

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # Each e[0] and e[3] will now be [state_size] numpy arrays because of the change in remember
        # Concatenate them to form [batch_size, state_size] tensors
        states = torch.FloatTensor(np.array([e[0] for e in minibatch]))
        actions = torch.LongTensor([e[1] for e in minibatch])
        rewards = torch.FloatTensor([e[2] for e in minibatch])
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch]))
        dones = torch.FloatTensor([e[4] for e in minibatch])

        # Get current Q values from model. self.model(states) will now output [batch_size, action_size]
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get next Q values from model (used as target model in vanilla DQN for stability)
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]

        # Compute the expected Q values
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = self.criterion(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    
        
        
        
    
    
    
    
    

    