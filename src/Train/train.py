# Project 2: # Reinforcement Learning for Dynamic Pricing 
# Role : Q-Learning Developer &  Baseline Strategies and Q-Learning 
# WHAT THEY BUILD Q-LearningAgent.py

import numpy as np 
import matplotlib.pyplot as plt 
from qlearning_agent import QLearningAgent
import random

#Environment settings  
GRID_SIZE = 10
ACTIONS = 5 # UP, DOWN, LEFT ,RIGHT , STAY 
EPISODES = 100
MAX_STEPS = 200


def step(state, action):
    x,y = state

    if action == 0:  # up 
        x = max(0, x-1)

    elif action == 1: # down 
        x =min(GRID_SIZE - 1, x + 1)

    elif action == 2: # left 
        y = max(0, y - 1)

    elif action == 3: # right 
        y = min(GRID_SIZE - 1, y + 1)

    elif action == 4: # stay 
        pass
    next_state = ( x , y)

    # Goal at bottom-right corner 
    done = (x == GRID_SIZE-1 and y == GRID_SIZE-1)

    reward = 10 if done else -0.1

    return next_state,reward,done, {}
def reset():
    return(
        random.randint(0,GRID_SIZE-1),
        random.randint(0,GRID_SIZE-1)
    )

agent = QLearningAgent()

rewards = []

print("Training for 100 episodes....\n")

for episode in range(1,EPISODES + 1):

    state = reset()
    total_reward = 0 
    done = False
    steps = 0 

    while not done and steps < MAX_STEPS:

        action = agent.act(state)

        next_state, reward, done, _ = step(state, action)

        agent.learn(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state
        total_reward += reward
        steps += 1

    rewards.append(total_reward)

    print(

         f"Episode {episode:3d}: Reward = {total_reward:3f}"


      )

    print ("\nTraining completed.")   

    plt.plot(rewards)
    plt.title("Reward over 100 Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)


    plt.savefig("results.png")

    print("Results saved to results.png")
  




    

