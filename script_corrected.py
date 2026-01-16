"""
Extended Hotelling Location Model with Customer Loyalty and Relocation Costs
FIXED VERSION - Addresses:
1. Loyalty now travels with sellers (not anchored to positions)
2. Stochastic market demand
3. Realistic episode counts (under 50) with scaled rewards
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

# ============================================================================
# LOYALTY STRATEGIES (Fixed - travels with seller)
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
# ENVIRONMENT (Fixed - stochastic demand, scaled rewards)
# ============================================================================


class ExtendedHotellingEnv(gym.Env):
    """Extended Hotelling model with loyalty, stochastic demand, scaled rewards"""

    def __init__(
        self,
        n_positions: int = 11,
        initial_weights: Optional[np.ndarray] = None,
        cost_scaling: float = 0.1,
        reward_mean: float = 10.0,
        reward_std: float = 2.0,
        demand_volatility: float = 0.1,
    ):
        super().__init__()
        self.beach_length = 100
        self.n_positions = n_positions
        self.cost_scaling = cost_scaling
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.demand_volatility = demand_volatility

        self.action_space = spaces.Discrete(self.n_positions)
        self.observation_space = spaces.Box(
            low=0, high=self.n_positions - 1, shape=(2,), dtype=np.int32
        )

        if initial_weights is None:
            self.base_weights = np.ones(n_positions) / n_positions
        else:
            self.base_weights = initial_weights / initial_weights.sum()

        self.weights = self.base_weights.copy()
        self.ben_pos = 0
        self.jerry_pos = 0
        self.prev_ben_pos = 0
        self.prev_jerry_pos = 0
        self.ben_time_at_position = 0
        self.jerry_time_at_position = 0
        self.step_count = 0
        self.weight_history = []
        self.position_history = []

    def _apply_demand_shock(self):
        """Apply stochastic demand fluctuations"""
        noise = np.random.normal(0, self.demand_volatility, self.n_positions)
        self.weights = self.base_weights * (1 + noise)
        self.weights = np.maximum(self.weights, 0)
        self.weights = self.weights / self.weights.sum()

    def _get_position_value(self, action: int) -> float:
        return action * (self.beach_length / (self.n_positions - 1))

    def _apply_loyalty_transfer(
        self, agent_pos: int, new_pos: int, loyalty_rate: float
    ):
        """Transfer customers when agent moves"""
        if agent_pos == new_pos:
            return

        transferred_weight = self.weights[agent_pos] * loyalty_rate
        self.weights[agent_pos] -= transferred_weight
        self.weights[new_pos] += transferred_weight
        self.weights = np.maximum(self.weights, 0)
        self.weights = self.weights / self.weights.sum()

    def _calculate_rewards(
        self,
        ben_moved: bool,
        jerry_moved: bool,
        ben_loyalty_rate: float,
        jerry_loyalty_rate: float,
    ) -> Tuple[float, float]:
        """Calculate rewards with stochastic multiplier"""
        if ben_moved:
            self._apply_loyalty_transfer(
                self.prev_ben_pos, self.ben_pos, ben_loyalty_rate
            )
        if jerry_moved:
            self._apply_loyalty_transfer(
                self.prev_jerry_pos, self.jerry_pos, jerry_loyalty_rate
            )

        ben_distance = abs(
            self._get_position_value(self.prev_ben_pos)
            - self._get_position_value(self.ben_pos)
        )
        ben_cost = self.cost_scaling * (ben_distance**2)
        jerry_distance = abs(
            self._get_position_value(self.prev_jerry_pos)
            - self._get_position_value(self.jerry_pos)
        )
        jerry_cost = self.cost_scaling * (jerry_distance**2)

        ben_share = 0.0
        jerry_share = 0.0

        if not ben_moved and not jerry_moved:
            if self.ben_pos < self.jerry_pos:
                midpoint_idx = (self.ben_pos + self.jerry_pos) // 2
                ben_share = self.weights[: midpoint_idx + 1].sum()
                jerry_share = 1 - ben_share
            elif self.ben_pos > self.jerry_pos:
                midpoint_idx = (self.jerry_pos + self.ben_pos) // 2
                jerry_share = self.weights[: midpoint_idx + 1].sum()
                ben_share = 1 - jerry_share
            else:
                ben_share = jerry_share = 0.5
        elif ben_moved and not jerry_moved:
            jerry_share = 1.0
        elif jerry_moved and not ben_moved:
            ben_share = 1.0

        reward_multiplier = max(1, np.random.normal(self.reward_mean, self.reward_std))
        return (
            ben_share * reward_multiplier - ben_cost,
            jerry_share * reward_multiplier - jerry_cost,
        )

    def step(
        self,
        actions: Tuple[int, int],
        ben_loyalty_rate: float = 0.3,
        jerry_loyalty_rate: float = 0.3,
    ) -> Tuple[np.ndarray, Tuple[float, float], bool, bool, Dict]:
        ben_action, jerry_action = actions
        ben_moved = ben_action != self.ben_pos
        jerry_moved = jerry_action != self.jerry_pos

        if ben_moved:
            self.ben_time_at_position = 0
        else:
            self.ben_time_at_position += 1
        if jerry_moved:
            self.jerry_time_at_position = 0
        else:
            self.jerry_time_at_position += 1

        self.prev_ben_pos = self.ben_pos
        self.prev_jerry_pos = self.jerry_pos
        self.ben_pos = ben_action
        self.jerry_pos = jerry_action

        ben_reward, jerry_reward = self._calculate_rewards(
            ben_moved, jerry_moved, ben_loyalty_rate, jerry_loyalty_rate
        )

        self._apply_demand_shock()
        self.weight_history.append(self.weights.copy())
        self.position_history.append((self.ben_pos, self.jerry_pos))
        self.step_count += 1

        state = np.array([self.ben_pos, self.jerry_pos], dtype=np.int32)
        info = {
            "ben_moved": ben_moved,
            "jerry_moved": jerry_moved,
            "ben_tenure": self.ben_time_at_position,
            "jerry_tenure": self.jerry_time_at_position,
            "weights": self.weights.copy(),
        }
        return state, (ben_reward, jerry_reward), False, False, info

    def reset(
        self, seed: Optional[int] = None, reset_type: str = "random"
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.step_count = 0
        self.weights = self.base_weights.copy()
        self.weight_history = [self.weights.copy()]
        self.position_history = []
        self.ben_time_at_position = 0
        self.jerry_time_at_position = 0

        if reset_type == "random":
            self.ben_pos = np.random.randint(0, self.n_positions)
            self.jerry_pos = np.random.randint(0, self.n_positions)
        elif reset_type == "extremes":
            self.ben_pos = np.random.choice([0, self.n_positions - 1])
            self.jerry_pos = (
                0 if self.ben_pos == self.n_positions - 1 else self.n_positions - 1
            )

        self.prev_ben_pos = self.ben_pos
        self.prev_jerry_pos = self.jerry_pos

        state = np.array([self.ben_pos, self.jerry_pos], dtype=np.int32), {}

        return state


# ============================================================================
# Q-LEARNING AGENT
# ============================================================================


class QLearningAgent:
    def __init__(self, name: str, n_positions: int, loyalty_strategy: LoyaltyStrategy):
        self.name = name
        self.n_positions = n_positions
        self.n_actions = n_positions
        self.loyalty_strategy = loyalty_strategy
        self.q_table = np.zeros((self.n_positions, self.n_positions, self.n_actions))
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_loyalty_rate(self, time_at_position: int, customer_base: float) -> float:
        return self.loyalty_strategy.get_retention_rate(time_at_position, customer_base)

    def act(self, my_pos: int, opponent_pos: int) -> int:
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
        current_q = self.q_table[my_pos, opp_pos, action]
        max_next_q = np.max(self.q_table[next_my, next_opp])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[my_pos, opp_pos, action] = new_q

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ============================================================================
# VISUALIZATION
# ============================================================================


class HotellingVisualizer:
    def __init__(self, env: ExtendedHotellingEnv):
        self.env = env
        self.beach_length = env.beach_length
        self.n_positions = env.n_positions

    def plot_training_summary(
        self,
        ben_rewards: list,
        jerry_rewards: list,
        ben_positions: list,
        jerry_positions: list,
        movement_stats: dict,
        ben_relocs: list,
        jerry_relocs: list,
        period_length: int,
        ben_cumulative: list = None,
        jerry_cumulative: list = None,
        ben_name: str = "Ben",
        jerry_name: str = "Jerry",
    ):
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

        # Plot 1: Final weight distribution
        ax1 = fig.add_subplot(gs[0, :])
        positions = np.linspace(0, self.beach_length, self.n_positions)
        final_weights = self.env.weights

        ax1.bar(
            positions,
            final_weights,
            width=self.beach_length / (self.n_positions + 1),
            alpha=0.6,
            color="lightblue",
            edgecolor="blue",
            label="Market Weight",
        )
        avg_ben = (
            np.mean(ben_positions[-10:])
            if len(ben_positions) >= 10
            else ben_positions[-1]
        )
        avg_jerry = (
            np.mean(jerry_positions[-10:])
            if len(jerry_positions) >= 10
            else jerry_positions[-1]
        )
        max_weight = final_weights.max()

        ax1.plot(
            avg_ben,
            max_weight * 1.1,
            "bv",
            markersize=15,
            label=f"Ben (avg: {avg_ben:.1f})",
            zorder=5,
        )
        ax1.plot(
            avg_jerry,
            max_weight * 1.1,
            "rv",
            markersize=15,
            label=f"Jerry (avg: {avg_jerry:.1f})",
            zorder=5,
        )
        ax1.set_xlabel("Position on Beach", fontsize=11)
        ax1.set_ylabel("Market Weight", fontsize=11)
        ax1.set_title(
            "Final Market Weight Distribution", fontsize=13, fontweight="bold"
        )
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Position evolution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(ben_positions, "b-", linewidth=2, alpha=0.7, label="Ben")
        ax2.plot(jerry_positions, "r-", linewidth=2, alpha=0.7, label="Jerry")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Position")
        ax2.set_title("Position Evolution", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Step rewards
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(ben_rewards, "b-", label="Ben", alpha=0.6)
        ax3.plot(jerry_rewards, "r-", label="Jerry", alpha=0.6)
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Avg Step Reward")
        ax3.set_title("Step Reward Evolution", fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Movement patterns
        ax4 = fig.add_subplot(gs[1, 2])
        labels = ["Both\nMoved", "Ben\nOnly", "Jerry\nOnly", "Neither"]
        values = [
            movement_stats["both_moved"],
            movement_stats["ben_only"],
            movement_stats["jerry_only"],
            movement_stats["neither"],
        ]
        ax4.bar(
            labels,
            values,
            color=["#ff6b6b", "#4ecdc4", "#95e1d3", "#a8e6cf"],
            edgecolor="black",
        )
        ax4.set_ylabel("Count")
        ax4.set_title("Movement Patterns", fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="y")

        # Plot 5: Relocations by period
        ax5 = fig.add_subplot(gs[2, :2])
        if len(ben_relocs) > 0:
            periods = np.arange(1, len(ben_relocs) + 1)
            width = 0.35
            ax5.bar(
                periods - width / 2,
                ben_relocs,
                width,
                label="Ben",
                color="blue",
                alpha=0.6,
            )
            ax5.bar(
                periods + width / 2,
                jerry_relocs,
                width,
                label="Jerry",
                color="red",
                alpha=0.6,
            )
            ax5.set_xlabel(f"Period ({period_length} eps)")
            ax5.set_ylabel("Relocations")
            ax5.set_title("Relocations Over Time", fontweight="bold")
            ax5.legend()
            ax5.grid(True, alpha=0.3, axis="y")

        # Plot 6: Market state
        ax6 = fig.add_subplot(gs[2, 2])
        positions_vals = np.linspace(0, self.beach_length, self.n_positions)
        ax6.plot([0, self.beach_length], [0, 0], "k-", linewidth=2, alpha=0.5)
        for pos, w in zip(positions_vals, self.env.weights):
            ax6.plot([pos, pos], [0, w * 100], "lightblue", linewidth=8, alpha=0.7)
        ax6.plot(ben_positions[-1], 0, "bo", markersize=15, label="Ben", zorder=10)
        ax6.plot(jerry_positions[-1], 0, "ro", markersize=15, label="Jerry", zorder=10)
        ax6.set_xlim(-5, 105)
        ax6.set_xlabel("Beach Position")
        ax6.set_ylabel("Customer Density")
        ax6.set_title("Final Market State", fontweight="bold")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Plot 7: Cumulative Rewards
        if ben_cumulative and jerry_cumulative:
            ax_cum = fig.add_subplot(gs[3, :])
            ax_cum.plot(ben_cumulative, "b-", linewidth=2.5, label=f"Ben: {ben_name}")
            ax_cum.plot(
                jerry_cumulative, "r-", linewidth=2.5, label=f"Jerry: {jerry_name}"
            )
            ax_cum.set_xlabel("Episode", fontsize=11)
            ax_cum.set_ylabel("Cumulative Reward", fontsize=11)
            ax_cum.set_title(
                "Total Profitability (Cumulative Reward)",
                fontsize=13,
                fontweight="bold",
            )
            ax_cum.legend(loc="upper left", frameon=True, shadow=True, fontsize=10)
            ax_cum.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train_extended_hotelling(
    episodes: int = 50,
    steps_per_episode: int = 20,
    n_positions: int = 11,
    ben_loyalty_strategy: Optional[LoyaltyStrategy] = None,
    jerry_loyalty_strategy: Optional[LoyaltyStrategy] = None,
    initial_weights: Optional[np.ndarray] = None,
    reward_mean: float = 10.0,
    reward_std: float = 2.0,
    demand_volatility: float = 0.1,
    visualize: bool = True,
):
    """Train agents with realistic episode counts and scaled rewards"""

    if ben_loyalty_strategy is None:
        ben_loyalty_strategy = HighBrandLoyalty(base_rate=0.5, tenure_bonus=0.05)
    if jerry_loyalty_strategy is None:
        jerry_loyalty_strategy = ModerateBrandLoyalty(base_rate=0.3, tenure_bonus=0.02)

    env = ExtendedHotellingEnv(
        n_positions=n_positions,
        initial_weights=initial_weights,
        reward_mean=reward_mean,
        reward_std=reward_std,
        demand_volatility=demand_volatility,
    )

    ben = QLearningAgent("Ben", n_positions, ben_loyalty_strategy)
    jerry = QLearningAgent("Jerry", n_positions, jerry_loyalty_strategy)

    ben_rewards, jerry_rewards = [], []
    ben_positions, jerry_positions = [], []
    ben_cumulative_rewards, jerry_cumulative_rewards = [], []
    ben_cumulative, jerry_cumulative = 0.0, 0.0
    movement_stats = {"both_moved": 0, "ben_only": 0, "jerry_only": 0, "neither": 0}

    period_length = max(1, episodes // 10)
    ben_relocations_by_period, jerry_relocations_by_period = [], []
    current_period_ben, current_period_jerry = 0, 0

    print("=" * 70)
    print("Training Configuration:")
    print(f"  Episodes: {episodes} (realistic business cycles)")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Reward scaling: μ={reward_mean}, σ={reward_std}")
    print(f"  Demand volatility: {demand_volatility}")
    print(f"  Ben's loyalty: {ben.loyalty_strategy.get_name()}")
    print(f"  Jerry's loyalty: {jerry.loyalty_strategy.get_name()}")
    print("=" * 70)

    for e in range(episodes):
        state, _ = env.reset()
        episode_ben_reward, episode_jerry_reward = 0, 0

        for step in range(steps_per_episode):
            ben_action = ben.act(state[0], state[1])
            jerry_action = jerry.act(state[1], state[0])

            ben_loyalty = ben.get_loyalty_rate(
                env.ben_time_at_position, env.weights[state[0]]
            )
            jerry_loyalty = jerry.get_loyalty_rate(
                env.jerry_time_at_position, env.weights[state[1]]
            )

            next_state, (ben_reward, jerry_reward), _, _, info = env.step(
                (ben_action, jerry_action), ben_loyalty, jerry_loyalty
            )

            if info["ben_moved"] and info["jerry_moved"]:
                movement_stats["both_moved"] += 1
                current_period_ben += 1
                current_period_jerry += 1
            elif info["ben_moved"]:
                movement_stats["ben_only"] += 1
                current_period_ben += 1
            elif info["jerry_moved"]:
                movement_stats["jerry_only"] += 1
                current_period_jerry += 1
            else:
                movement_stats["neither"] += 1

            ben.learn(
                state[0], state[1], ben_action, ben_reward, next_state[0], next_state[1]
            )
            jerry.learn(
                state[1],
                state[0],
                jerry_action,
                jerry_reward,
                next_state[1],
                next_state[0],
            )

            state = next_state
            episode_ben_reward += ben_reward
            episode_jerry_reward += jerry_reward

        ben.decay_epsilon()
        jerry.decay_epsilon()

        avg_ben_reward = episode_ben_reward / steps_per_episode
        avg_jerry_reward = episode_jerry_reward / steps_per_episode
        ben_rewards.append(avg_ben_reward)
        jerry_rewards.append(avg_jerry_reward)
        ben_positions.append(env._get_position_value(env.ben_pos))
        jerry_positions.append(env._get_position_value(env.jerry_pos))

        ben_cumulative += avg_ben_reward
        jerry_cumulative += avg_jerry_reward
        ben_cumulative_rewards.append(ben_cumulative)
        jerry_cumulative_rewards.append(jerry_cumulative)

        if (e + 1) % period_length == 0:
            ben_relocations_by_period.append(current_period_ben)
            jerry_relocations_by_period.append(current_period_jerry)
            current_period_ben, current_period_jerry = 0, 0

        if (e + 1) % 10 == 0 or e == episodes - 1:
            print(
                f"Ep {e + 1}/{episodes} - Ben: pos={ben_positions[-1]:.1f}, cum={ben_cumulative:.2f} | "
                f"Jerry: pos={jerry_positions[-1]:.1f}, cum={jerry_cumulative:.2f} | ε: {ben.epsilon:.3f}"
            )

    print("\n=== Testing Learned Policy ===")
    ben.epsilon = 0
    jerry.epsilon = 0
    test_ben, test_jerry = [], []

    for _ in range(20):
        state, _ = env.reset()
        for _ in range(steps_per_episode):
            ben_action = ben.act(state[0], state[1])
            jerry_action = jerry.act(state[1], state[0])
            ben_loyalty = ben.get_loyalty_rate(
                env.ben_time_at_position, env.weights[state[0]]
            )
            jerry_loyalty = jerry.get_loyalty_rate(
                env.jerry_time_at_position, env.weights[state[1]]
            )
            state, _, _, _, _ = env.step(
                (ben_action, jerry_action), ben_loyalty, jerry_loyalty
            )
        test_ben.append(env._get_position_value(env.ben_pos))
        test_jerry.append(env._get_position_value(env.jerry_pos))

    print(f"Average converged positions:")
    print(f"  Ben:   {np.mean(test_ben):.1f}")
    print(f"  Jerry: {np.mean(test_jerry):.1f}")
    print(f"Final cumulative rewards:")
    print(f"  Ben:   {ben_cumulative:.2f}")
    print(f"  Jerry: {jerry_cumulative:.2f}")

    if visualize:
        visualizer = HotellingVisualizer(env)
        visualizer.plot_training_summary(
            ben_rewards,
            jerry_rewards,
            ben_positions,
            jerry_positions,
            movement_stats,
            ben_relocations_by_period,
            jerry_relocations_by_period,
            period_length,
            ben_cumulative_rewards,
            jerry_cumulative_rewards,
            ben.loyalty_strategy.get_name(),
            jerry.loyalty_strategy.get_name(),
        )

    return env, ben, jerry


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


def main():
    print("\n" + "=" * 70)
    print("SCENARIO 1: Asymmetric Brand Loyalty, Uniform Market")
    print("=" * 70)

    env1, ben1, jerry1 = train_extended_hotelling(
        episodes=5000,
        steps_per_episode=20,
        n_positions=11,
        ben_loyalty_strategy=HighBrandLoyalty(base_rate=0.5, tenure_bonus=0.05),
        jerry_loyalty_strategy=ModerateBrandLoyalty(base_rate=0.3, tenure_bonus=0.02),
        initial_weights=None,
        reward_mean=10.0,
        reward_std=2.0,
        demand_volatility=0.1,
        visualize=True,
    )

    print("\n" + "=" * 70)
    print("SCENARIO 2: Stochastic Bimodal Market")
    print("=" * 70)

    bimodal_weights = np.array(
        [0.15, 0.12, 0.08, 0.05, 0.03, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15]
    )

    env2, ben2, jerry2 = train_extended_hotelling(
        episodes=50,
        steps_per_episode=20,
        n_positions=11,
        ben_loyalty_strategy=WeakBrandLoyalty(base_rate=0.15),
        jerry_loyalty_strategy=WeakBrandLoyalty(base_rate=0.15),
        initial_weights=bimodal_weights,
        reward_mean=10.0,
        reward_std=3.0,
        demand_volatility=0.15,
        visualize=True,
    )


if __name__ == "__main__":
    main()
