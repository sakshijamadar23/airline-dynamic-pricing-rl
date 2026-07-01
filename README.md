✈️ Airline Dynamic Pricing Using Reinforcement Learning 

📖  About the Project :

Airline Dynamic Pricing Using Reinforcement Learning is an AI-based project that demonstrates how machine learning can optimize airline ticket pricing. Instead of using fixed prices, the system learns pricing strategies through interaction with a simulated environment to maximize revenue. 

📌 Project Overview

Dynamic pricing helps airlines adjust ticket prices based on market conditions and demand. This project evaluates four pricing strategies:

🟥 Fixed Price
🟨 Discount Pricing
🟩 Q-Learning
🟦 Deep Q-Network (DQN)

The objective is to compare these approaches and determine which strategy achieves the highest revenue over multiple training episodes.  

🚀 Features
✈️ Airline Dynamic Pricing Simulation
🤖 Reinforcement Learning (Q-Learning & DQN)
📈 AI Learning Curve Visualization
📊 Revenue Comparison Dashboard
📉 Agent Improvement Analysis
🌙 Modern Dark-Themed Dashboard
💾 Automatic Result Saving (results.json) 


🛠️ Technologies Used
Python 3.x
NumPy
Matplotlib 

1)Python: Core language
2)OpenAI Gymnasium: RL Environment
3)NumPy: Math operations
4)PyTorch: Neural networks
5)Matplotlib: Visualizations
6)Seaborn: Chart styling
7)tqdm : Progress bars
8)Pandas: data handling 

📂 Project Structure
Airline-Dynamic-Pricing/
│
├── train.py
├── environment.py
├── agents.py
├── dashboard.py
├── results.json
├── dashboard.png
├── README.md
└── requirements.txt  

# The 4 AI Agents

| Agent | Strategy | Intelligence |
|-------|----------|-------------|
| 🔴 Fixed Price | Always charges ₹700 | Zero — Baseline |
| 🟡 Discount | Human logic rules | Rule-based |
| 🟢 Q-Learning | Learns Q-table | Machine Learning |
| 🔵 DQN | Neural Network brain | Deep Learning |

### Why 4 Agents?
Like a science experiment needs a control group —
we need comparison to **PROVE** our AI works better!

 🧠 Technical Implementation
 Environment Design
```python
State Space:  [seats_remaining, days_remaining]
Action Space: Discrete(5) → ₹300/₹500/₹700/₹900/₹1100
Reward:       seats_sold × price_charged
Episodes:     2000 per agent
Total Seats:  100
Total Days:   30

📊 Dashboard

The dashboard includes:
📈 How AI Gets Smarter Over Time
📊 Revenue Comparison
📉 Agent Improvement
🏆 Best Revenue
🤖 Best Performing Agent
💰 Average Revenue

📈 Dashboard Output
Our professional dashboard shows 3 charts:
Chart 1 — Learning Curves
Shows how all 4 agents perform across 2000
episodes. Proves learning agents improve over time!
Chart 2 — Revenue Comparison
Bar chart comparing average revenue of all agents
with exact values labeled on each bar.
Chart 3 — Improvement Analysis
Shows improvement percentage from first 100
to last 100 episodes for each agent.

🤖 AI Agents Compared
   Agent                              Description
Fixed Price                   Uses a constant ticket price
Discount Pricing              Applies discount-based pricing
Q-Learning                    Reinforcement Learning algorithm
Deep Q-Network (DQN)          Deep Reinforcement Learning algorithm

📈 Results
The project compares all pricing strategies using:
Average Revenue 
Revenue Trends
Learning Curves
Improvement Percentage
AI Dashboard Visualizations










