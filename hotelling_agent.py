"""
Agent definitions for Hotelling Location Model
Includes loyalty strategies and Q-Learning agent
"""

from abc import ABC, abstractmethod
import numpy as np


# ============================================================================
# LOYALTY STRATEGIES
# ============================================================================


class LoyaltyStrategy(ABC):
    """Abstract base class for customer loyalty strategies"""

    @abstractmethod
    def get_retention_rate(
        self, time_at_position: int, total_customers: float
    ) -> float:
        """Return loyalty rate based on seller's tenure and customer base"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name for visualization"""
        pass


class HighBrandLoyalty(LoyaltyStrategy):
    """Strong brand loyalty that increases with tenure"""

    def __init__(
        self, base_rate: float = 0.5, tenure_bonus: float = 0.05, max_rate: float = 0.8
    ):
        self.base_rate = base_rate
        self.tenure_bonus = tenure_bonus
        self.max_rate = max_rate

    def get_retention_rate(
        self, time_at_position: int, total_customers: float
    ) -> float:
        rate = min(self.base_rate + self.tenure_bonus * time_at_position, self.max_rate)
        return rate

    def get_name(self) -> str:
        return f"HighBrand (base={self.base_rate:.0%})"


class ModerateBrandLoyalty(LoyaltyStrategy):
    """Moderate brand loyalty"""

    def __init__(
        self, base_rate: float = 0.3, tenure_bonus: float = 0.02, max_rate: float = 0.5
    ):
        self.base_rate = base_rate
        self.tenure_bonus = tenure_bonus
        self.max_rate = max_rate

    def get_retention_rate(
        self, time_at_position: int, total_customers: float
    ) -> float:
        rate = min(self.base_rate + self.tenure_bonus * time_at_position, self.max_rate)
        return rate

    def get_name(self) -> str:
        return f"ModBrand (base={self.base_rate:.0%})"


class WeakBrandLoyalty(LoyaltyStrategy):
    """Weak brand loyalty"""

    def __init__(self, base_rate: float = 0.15):
        self.base_rate = base_rate

    def get_retention_rate(
        self, time_at_position: int, total_customers: float
    ) -> float:
        return self.base_rate

    def get_name(self) -> str:
        return f"WeakBrand ({self.base_rate:.0%})"


# ============================================================================
# Q-LEARNING AGENT
# ============================================================================


class QLearningAgent:
    """Q-Learning agent with loyalty strategy"""
    
    def __init__(self, name: str, n_positions: int, loyalty_strategy: LoyaltyStrategy):
        self.name = name
        self.n_positions = n_positions
        self.n_actions = n_positions
        self.loyalty_strategy = loyalty_strategy
        self.q_table = np.zeros((n_positions, n_positions, n_positions))
        self.alpha = 0.2
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95

    def get_loyalty_rate(self, time_at_position: int, customer_base: float) -> float:
        """Get loyalty rate from strategy"""
        return self.loyalty_strategy.get_retention_rate(time_at_position, customer_base)

    def act(self, my_pos: int, opponent_pos: int) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[my_pos, opponent_pos])

    def learn(
        self,
        my_pos: int,
        opp_pos: int,
        action: int,
        reward: float,
        next_my: int,
        next_opp: int,
    ):
        """Update Q-table using Q-learning update rule"""
        current_q = self.q_table[my_pos, opp_pos, action]
        max_next_q = np.max(self.q_table[next_my, next_opp])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[my_pos, opp_pos, action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: str):
        """Save agent's Q-table to file"""
        np.save(filepath, self.q_table)
        print(f"Agent {self.name} saved to {filepath}")

    def load(self, filepath: str):
        """Load agent's Q-table from file"""
        self.q_table = np.load(filepath)
        print(f"Agent {self.name} loaded from {filepath}")
