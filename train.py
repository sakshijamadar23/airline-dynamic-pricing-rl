## AIRLINE DYNAMIC PRICING PROJECT
# ══════════════════════════════════
# Author: Sakshi Jamadar
# Role: Project Lead
# Date: June 2026
# ══════════════════════════════════

import numpy as np
import json
from tqdm import tqdm

# ══════════════════════════════════
# TEAM FILES IMPORTS
# Remove # when files are ready!
# ══════════════════════════════════

from environment import AirlinePricingEnv
from agents import FixedPriceAgent
from agents import DiscountAgent
from agents import QLearningAgent
from agents import DQNAgent

print("train.py created successfully!")
print("Waiting for team files...")


# train.py
# ══════════════════════════════════
# AIRLINE DYNAMIC PRICING PROJECT
# ══════════════════════════════════
# Author: Sakshi Jamadar
# Role: Project Lead
# Date: June 2026
# ══════════════════════════════════

import numpy as np
import json
from tqdm import tqdm

# All team files connected here!
from environment import AirlinePricingEnv
from agents import FixedPriceAgent
from agents import DiscountAgent
from agents import QLearningAgent
from agents import DQNAgent

# ══════════════════════════════════
# SETTINGS
# ══════════════════════════════════
TOTAL_EPISODES = 2000
PRINT_EVERY    = 100

# ══════════════════════════════════
# FUNCTION 1: Run One Episode
# ══════════════════════════════════
def run_episode(env, agent,
                train=False,
                agent_type='baseline'):
    state, _ = env.reset()
    total_revenue = 0

    while True:
        action = agent.act(state)
        next_state, reward, done, _, info = env.step(action)
        total_revenue += reward

        if train:
            if agent_type == 'qlearning':
                agent.learn(
                    state, action,
                    reward, next_state, done
                )
            elif agent_type == 'dqn':
                agent.remember(
                    state, action,
                    reward, next_state, done
                )
                agent.learn()

        state = next_state
        if done:
            break

    return total_revenue

# ══════════════════════════════════
# FUNCTION 2: Train All Agents
# ══════════════════════════════════
def train_all_agents():

    print("\n" + "="*50)
    print("   AIRLINE DYNAMIC PRICING")
    print("   TRAINING BEGINS!")
    print("="*50 + "\n")

    env = AirlinePricingEnv()
    results = {}

    # ── Agent 1: Fixed Price ───────
    print("Running Fixed Price Agent...")
    fixed_agent = FixedPriceAgent()
    fixed_revenues = []

    for episode in tqdm(range(TOTAL_EPISODES),
                        desc="Fixed Price"):
        revenue = run_episode(
            env, fixed_agent,
            train=False,
            agent_type='baseline'
        )
        fixed_revenues.append(revenue)

    results['Fixed Price'] = fixed_revenues
    avg = np.mean(fixed_revenues[-100:])
    print(f"Fixed Price Done!")
    print(f"Avg Revenue: Rs {avg:,.0f}\n")

    # ── Agent 2: Discount ──────────
    print("Running Discount Agent...")
    discount_agent = DiscountAgent()
    discount_revenues = []

    for episode in tqdm(range(TOTAL_EPISODES),
                        desc="Discount"):
        revenue = run_episode(
            env, discount_agent,
            train=False,
            agent_type='baseline'
        )
        discount_revenues.append(revenue)

    results['Discount'] = discount_revenues
    avg = np.mean(discount_revenues[-100:])
    print(f"Discount Agent Done!")
    print(f"Avg Revenue: Rs {avg:,.0f}\n")

    # ── Agent 3: Q-Learning ────────
    print("Training Q-Learning Agent...")
    q_agent = QLearningAgent()
    q_revenues = []

    for episode in tqdm(range(TOTAL_EPISODES),
                        desc="Q-Learning"):
        revenue = run_episode(
            env, q_agent,
            train=True,
            agent_type='qlearning'
        )
        q_revenues.append(revenue)

    results['Q-Learning'] = q_revenues
    avg = np.mean(q_revenues[-100:])
    print(f"Q-Learning Done!")
    print(f"Avg Revenue: Rs {avg:,.0f}\n")

    # ── Agent 4: DQN ───────────────
    print("Training DQN Agent...")
    dqn_agent = DQNAgent()
    dqn_revenues = []

    for episode in tqdm(range(TOTAL_EPISODES),
                        desc="DQN"):
        revenue = run_episode(
            env, dqn_agent,
            train=True,
            agent_type='dqn'
        )
        dqn_revenues.append(revenue)

    results['DQN'] = dqn_revenues
    avg = np.mean(dqn_revenues[-100:])
    print(f"DQN Done!")
    print(f"Avg Revenue: Rs {avg:,.0f}\n")

    return results

# ══════════════════════════════════
# FUNCTION 3: Print Summary
# ══════════════════════════════════
def print_summary(results):

    print("\n" + "="*50)
    print("   FINAL RESULTS")
    print("="*50)

    for name, revenues in results.items():
        avg   = np.mean(revenues[-100:])
        best  = np.max(revenues)
        print(f"\n{name}:")
        print(f"  Avg Revenue: Rs {avg:,.0f}")
        print(f"  Best:        Rs {best:,.0f}")

    # Find winner
    averages = {
        name: np.mean(rev[-100:])
        for name, rev in results.items()
    }
    winner = max(averages, key=averages.get)
    best_avg = averages[winner]

    print("\n" + "="*50)
    print(f"WINNER: {winner}!")
    print(f"Best Avg: Rs {best_avg:,.0f}")
    print("="*50 + "\n")

# ══════════════════════════════════
# FUNCTION 4: Save Results
# ══════════════════════════════════
def save_results(results):

    json_results = {}
    for name, revenues in results.items():
        json_results[name] = [
            float(r) for r in revenues
        ]

    with open('results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    print("Results saved to results.json ✅")
    print("Member 6 can now build dashboard!")

# ══════════════════════════════════
# MAIN — Run Everything
# ══════════════════════════════════
if __name__ == "__main__":

    print("\n" + "="*50)
    print("   PROJECT: AIRLINE DYNAMIC PRICING")
    print("   Team: 6 Members")
    print("="*50)

    # Step 1: Train all agents
    results = train_all_agents()

    # Step 2: Print summary
    print_summary(results)

    # Step 3: Save results
    save_results(results)

    print("\n" + "="*50)
    print("   TRAINING COMPLETE! 🎉")
    print("   Run dashboard.py for charts!")
    print("="*50 + "\n")