# =============================================================
#  PROJECT 2 — Dynamic Pricing with Reinforcement Learning
#  FILE: environment.py
#  WHAT: The Market Simulator — the "Game World" our AI lives in
# =============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AirlinePricingEnv(gym.Env):
    """
    A simulated airline ticket market.

    The AI (agent) lives inside this environment.
    Every day it picks a price, the environment responds
    with how many tickets sold and the revenue earned.

    STATE  : [seats_remaining, days_remaining]
    ACTION : index (0-9) → maps to a price in price_levels
    REWARD : revenue earned that day (price × tickets_sold)
    """

    def __init__(self):
        super().__init__()

        # ── Problem Settings ──────────────────────────────────────
        self.total_seats  = 50      # total tickets available
        self.total_days   = 30      # days in the selling window

        # Price options the AI can choose from (its "actions")
        self.price_levels = [500, 1500, 3000, 5000, 7000]
        self.n_actions = 5

        # ── Gymnasium: define action & observation spaces ─────────
        # Action space: AI picks one index (0 to 9) → maps to a price
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space: [seats_remaining, days_remaining]
        self.observation_space = spaces.Box(
            low   = np.array([0, 0]),
            high = np.array([self.total_seats, self.total_days]),
            dtype = np.float32
        )

        # ── Internal State ────────────────────────────────────────
        self.seats_remaining = None
        self.days_remaining  = None


    # ─────────────────────────────────────────────────────────────
    # RESET — called at the start of every new "season"
    # ─────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seats_remaining = self.total_seats
        self.days_remaining  = self.total_days
        state = np.array([self.seats_remaining,
                          self.days_remaining], dtype=np.float32)
        return state, {}


    # ─────────────────────────────────────────────────────────────
    # DEMAND FUNCTION — how many customers buy at this price?
    # ─────────────────────────────────────────────────────────────
    def _get_demand(self, price):
        """
        Higher price  → fewer customers buy.
        Fewer days left → customers slightly less likely to buy.
        Randomness is added to mimic real-world unpredictability.
        """
        # Base purchase probability drops as price rises
        base_prob = 0.9 * np.exp(-0.00025 * (price - 500))

        # Urgency factor: as days run out, demand slightly drops
        urgency_factor = 0.5 + 0.5 * (self.days_remaining / self.total_days)

        # Final probability of a single customer buying
        buy_prob = np.clip(base_prob * urgency_factor, 0, 1)

        # Simulate how many customers arrive today (random, 0 to 5)
        potential_customers = np.random.randint(0, 6)

        # Each customer independently decides to buy or not
        tickets_sold = np.random.binomial(potential_customers, buy_prob)

        # Can't sell more than what's available!
        return min(tickets_sold, self.seats_remaining)


    # ─────────────────────────────────────────────────────────────
    # STEP — main function: AI picks action, world responds
    # ─────────────────────────────────────────────────────────────
    def step(self, action):
        # Convert action index → actual price
        price = self.price_levels[action]

        # Simulate today's sales
        tickets_sold = self._get_demand(price)

        # Calculate reward (revenue earned today)
        reward = price * tickets_sold

        # Update internal state
        self.seats_remaining -= tickets_sold
        self.days_remaining  -= 1

        # Is the season over? (no days left OR no seats left)
        terminated = (self.days_remaining <= 0) or (self.seats_remaining <= 0)

        # Build new state to return to the AI
        state = np.array([self.seats_remaining,
                          self.days_remaining], dtype=np.float32)

        info = {"tickets_sold": tickets_sold, "price": price}
        return state, reward, terminated, False, info


# =============================================================
#  QUICK TEST — runs only when you execute this file directly
#  In your terminal: python environment.py
# =============================================================
if __name__ == "__main__":

    env = AirlinePricingEnv()
    state, _ = env.reset()

    print("=" * 55)
    print("   SIMULATOR TEST RUN — 5 Days")
    print("=" * 55)
    print(f"Start → Seats: {int(state[0])}, Days Left: {int(state[1])}")
    print("-" * 55)

    total_revenue = 0

    for day in range(1, 6):
        action = env.action_space.sample()          # random price for now
        price  = env.price_levels[action]
        state, reward, terminated, _, info = env.step(action)
        total_revenue += reward

        print(f"Day {day:2d} | Price: Rs.{price:,} | "
              f"Sold: {info['tickets_sold']} ticket(s) | "
              f"Revenue: Rs.{reward:,} | "
              f"Seats Left: {int(state[0])} | "
              f"Days Left: {int(state[1])}")

        if terminated:
            print("Season ended early — all seats sold!")
            break

    print("-" * 55)
    print(f"Total Revenue after 5 days: Rs.{total_revenue:,}")
    print("=" * 55)
    print("Simulator is working correctly!")
