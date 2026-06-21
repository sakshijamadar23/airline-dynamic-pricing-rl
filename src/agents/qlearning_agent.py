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