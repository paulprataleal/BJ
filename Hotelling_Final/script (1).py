"""
Extended Hotelling Location Model with Customer Loyalty and Relocation Costs

ECONOMIC JUSTIFICATION FOR EPISODE COUNT:

The high number of training episodes (5,000-10,000) can be justified through several
real-world economic interpretations:

1. **Market Cycles Interpretation**: Each episode represents a full business cycle
   (e.g., quarterly or seasonal cycles). With 5,000 episodes representing ~20 years
   of quarterly adjustments or ~1,250 years of annual cycles, firms learn optimal
   positioning over many market iterations.

2. **Experimentation Periods**: In reality, firms conduct market research, A/B testing,
   and experimental relocations over many years. The high episode count reflects the
   extensive trial-and-error learning that real businesses undergo.

3. **Generational Learning**: Episodes could represent knowledge transfer across
   management generations, with each episode being a decision period where accumulated
   institutional knowledge (Q-table) is refined.

4. **Stochastic Market Conditions**: Each episode can be seen as a different market
   realization with slight variations in customer preferences. Agents learn robust
   strategies across varying conditions.

5. **Cognitive Learning Time**: Economic agents (humans/firms) make thousands of
   observations before converging to Nash equilibrium in complex strategic settings.
   The episodes represent accumulated market intelligence over time.

For practical interpretation: use fewer episodes (1,000-2,000) for "fast-learning"
markets (tech, fashion) and more episodes (5,000-10,000) for "slow-learning" markets
(real estate, infrastructure).
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

# ============================================================================
# LOYALTY STRATEGIES (Abstract Base Class)
# ============================================================================


class LoyaltyStrategy(ABC):
    """Abstract base class for customer loyalty strategies"""

    @abstractmethod
    def get_retention_rate(self, position: int, n_positions: int) -> float:
        """Return the loyalty retention rate for a given position"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name for visualization"""
        pass


class HighLocalLoyalty(LoyaltyStrategy):
    """High satisfaction at one point, drops off quickly"""

    def __init__(
        self, peak_position: int, peak_rate: float = 0.5, falloff: float = 0.8
    ):
        self.peak_position = peak_position
        self.peak_rate = peak_rate
        self.falloff = falloff

    def get_retention_rate(self, position: int, n_positions: int) -> float:
        distance = abs(position - self.peak_position)
        return self.peak_rate * (self.falloff**distance)

    def get_name(self) -> str:
        return f"Local (peak@{self.peak_position})"


class ModerateSpreadLoyalty(LoyaltyStrategy):
    """Moderate satisfaction across several consecutive points"""

    def __init__(self, center_position: int, base_rate: float = 0.3, spread: int = 2):
        self.center_position = center_position
        self.base_rate = base_rate
        self.spread = spread

    def get_retention_rate(self, position: int, n_positions: int) -> float:
        distance = abs(position - self.center_position)
        if distance <= self.spread:
            return self.base_rate * (1 - distance / (self.spread + 2))
        return self.base_rate * 0.1

    def get_name(self) -> str:
        return f"Spread (center@{self.center_position})"


class UniformLoyalty(LoyaltyStrategy):
    """Uniform loyalty across all positions"""

    def __init__(self, base_rate: float = 0.25):
        self.base_rate = base_rate

    def get_retention_rate(self, position: int, n_positions: int) -> float:
        return self.base_rate

    def get_name(self) -> str:
        return f"Uniform ({self.base_rate:.0%})"


# ============================================================================
# ENVIRONMENT
# ============================================================================


class ExtendedHotellingEnv(gym.Env):
    """
    Extended Hotelling model with:
    - Relocation costs (quadratic in distance)
    - Customer loyalty (weights follow vendors when they move)
    - Dynamic market weight distribution
    """

    def __init__(
        self,
        n_positions: int = 11,
        initial_weights: Optional[np.ndarray] = None,
        cost_scaling: float = 0.01,
    ):
        super().__init__()
        self.beach_length = 100
        self.n_positions = n_positions
        self.cost_scaling = cost_scaling  # Scale quadratic costs
        self.action_space = spaces.Discrete(self.n_positions)
        self.observation_space = spaces.Box(
            low=0, high=self.n_positions - 1, shape=(2,), dtype=np.int32
        )

        # Initialize market weights (customer density at each position)
        if initial_weights is None:
            self.initial_weights = np.ones(n_positions) / n_positions
        else:
            self.initial_weights = initial_weights / initial_weights.sum()

        self.weights = self.initial_weights.copy()

        # State tracking
        self.ben_pos = 0
        self.jerry_pos = 0
        self.prev_ben_pos = 0
        self.prev_jerry_pos = 0
        self.step_count = 0

        # History for visualization
        self.weight_history = []
        self.position_history = []

    def set_weights(self, weights: np.ndarray):
        """Manually set market weights"""
        self.weights = weights / weights.sum()

    def _get_position_value(self, action: int) -> float:
        """Convert discrete action to position on beach"""
        return action * (self.beach_length / (self.n_positions - 1))

    def _apply_loyalty_transfer(
        self, agent_pos: int, new_pos: int, loyalty_rate: float
    ):
        """
        Transfer customer loyalty when an agent moves.
        loyalty_rate: fraction of customers that follow the vendor (0 to 1)
        """
        if agent_pos == new_pos:
            return

        # Calculate how much weight transfers
        transferred_weight = self.weights[agent_pos] * loyalty_rate

        # Remove from old position
        self.weights[agent_pos] -= transferred_weight

        # Add to new position
        self.weights[new_pos] += transferred_weight

        # Ensure non-negative weights
        self.weights = np.maximum(self.weights, 0)

    def _calculate_rewards(
        self,
        ben_moved: bool,
        jerry_moved: bool,
        ben_loyalty_rate: float,
        jerry_loyalty_rate: float,
    ) -> Tuple[float, float]:
        """
        Calculate market share for each vendor based on movement and loyalty.

        Rules:
        - If agent moves: gets 0 market share (opportunity cost) and pays relocation cost
        - If agent stays: gets market share based on positions and weights
        """
        # Apply loyalty transfers BEFORE calculating rewards
        if ben_moved:
            self._apply_loyalty_transfer(
                self.prev_ben_pos, self.ben_pos, ben_loyalty_rate
            )
        if jerry_moved:
            self._apply_loyalty_transfer(
                self.prev_jerry_pos, self.jerry_pos, jerry_loyalty_rate
            )

        # Calculate Relocation Costs (Squared Distance)
        ben_start_x = self._get_position_value(self.prev_ben_pos)
        ben_end_x = self._get_position_value(self.ben_pos)
        ben_distance = abs(ben_end_x - ben_start_x)
        ben_cost = self.cost_scaling * (ben_distance**2)

        jerry_start_x = self._get_position_value(self.prev_jerry_pos)
        jerry_end_x = self._get_position_value(self.jerry_pos)
        jerry_distance = abs(jerry_end_x - jerry_start_x)
        jerry_cost = self.cost_scaling * (jerry_distance**2)

        # Market Share Calculation
        ben_share = 0.0
        jerry_share = 0.0

        # If an agent moves, they get 0 market share (opportunity cost),
        # but they still pay the relocation cost.
        if not ben_moved and not jerry_moved:
            # Neither moved - calculate normal market share based on weights
            ben_actual = self.ben_pos
            jerry_actual = self.jerry_pos

            if ben_actual < jerry_actual:
                midpoint_idx = (ben_actual + jerry_actual) // 2
                ben_share = self.weights[: midpoint_idx + 1].sum()
                jerry_share = 1 - ben_share
            elif ben_actual > jerry_actual:
                midpoint_idx = (jerry_actual + ben_actual) // 2
                jerry_share = self.weights[: midpoint_idx + 1].sum()
                ben_share = 1 - jerry_share
            else:
                ben_share = jerry_share = 0.5

        elif ben_moved and not jerry_moved:
            # Ben moves (0 share), Jerry stays (100% share)
            jerry_share = 1.0

        elif jerry_moved and not ben_moved:
            # Jerry moves (0 share), Ben stays (100% share)
            ben_share = 1.0

        # Both moved - both get 0 share (already initialized)

        # Final Reward = Market Share - Relocation Cost
        return ben_share - ben_cost, jerry_share - jerry_cost

    def step(
        self,
        actions: Tuple[int, int],
        ben_loyalty_rate: float = 0.3,
        jerry_loyalty_rate: float = 0.3,
    ) -> Tuple[np.ndarray, Tuple[float, float], bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            actions: (ben_action, jerry_action)
            ben_loyalty_rate: fraction of Ben's customers that follow him when moving
            jerry_loyalty_rate: fraction of Jerry's customers that follow him when moving
        """
        ben_action, jerry_action = actions

        # Track if agents moved
        ben_moved = ben_action != self.ben_pos
        jerry_moved = jerry_action != self.jerry_pos

        # Update positions
        self.prev_ben_pos = self.ben_pos
        self.prev_jerry_pos = self.jerry_pos
        self.ben_pos = ben_action
        self.jerry_pos = jerry_action

        # Calculate rewards (this also applies loyalty transfers)
        ben_reward, jerry_reward = self._calculate_rewards(
            ben_moved, jerry_moved, ben_loyalty_rate, jerry_loyalty_rate
        )

        # Store history
        self.weight_history.append(self.weights.copy())
        self.position_history.append((self.ben_pos, self.jerry_pos))

        self.step_count += 1

        state = np.array([self.ben_pos, self.jerry_pos], dtype=np.int32)
        info = {
            "ben_moved": ben_moved,
            "jerry_moved": jerry_moved,
            "weights": self.weights.copy(),
        }

        return state, (ben_reward, jerry_reward), False, False, info

    def reset(
        self, seed: Optional[int] = None, reset_type: str = "random"
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        self.step_count = 0
        self.weights = self.initial_weights.copy()
        self.weight_history = [self.weights.copy()]
        self.position_history = []

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

        state = np.array([self.ben_pos, self.jerry_pos], dtype=np.int32)
        return state, {}


# ============================================================================
# Q-LEARNING AGENT
# ============================================================================


class QLearningAgent:
    """Q-Learning agent with configurable loyalty strategy"""

    def __init__(self, name: str, n_positions: int, loyalty_strategy: LoyaltyStrategy):
        self.name = name
        self.n_positions = n_positions
        self.n_actions = n_positions
        self.loyalty_strategy = loyalty_strategy

        # Q-table: [my_pos, opponent_pos, action]
        self.q_table = np.zeros((self.n_positions, self.n_positions, self.n_actions))

        # Hyperparameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_loyalty_rate(self, position: int) -> float:
        """Get loyalty retention rate for current position"""
        return self.loyalty_strategy.get_retention_rate(position, self.n_positions)

    def act(self, my_pos: int, opponent_pos: int) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[my_pos, opponent_pos])

    def learn(
        self,
        my_pos: int,
        opponent_pos: int,
        action: int,
        reward: float,
        next_my_pos: int,
        next_opponent_pos: int,
    ):
        """Update Q-table using Q-learning"""
        current_q = self.q_table[my_pos, opponent_pos, action]
        max_next_q = np.max(self.q_table[next_my_pos, next_opponent_pos])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[my_pos, opponent_pos, action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ============================================================================
# VISUALIZATION
# ============================================================================


class HotellingVisualizer:
    """Enhanced visualization for the extended Hotelling model"""

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
        ben_relocations_by_period: list,
        jerry_relocations_by_period: list,
        period_length: int,
        ben_cumulative_rewards: list = None,
        jerry_cumulative_rewards: list = None,
        ben_strategy_name: str = "Ben",
        jerry_strategy_name: str = "Jerry",
    ):
        """Create comprehensive training summary plots with cumulative rewards and strategy legends"""
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(5, 3, hspace=0.5, wspace=0.3)

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

        avg_ben = np.mean(ben_positions[-100:])
        avg_jerry = np.mean(jerry_positions[-100:])
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
        ax1.axhline(y=0, color="black", linewidth=1)
        ax1.set_xlabel("Position on Beach", fontsize=11)
        ax1.set_ylabel("Market Weight", fontsize=11)
        ax1.set_title(
            "Final Market Weight Distribution", fontsize=13, fontweight="bold"
        )
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Position evolution
        ax2 = fig.add_subplot(gs[1, 0])
        window = 50
        if len(ben_positions) > window:
            ben_ma = np.convolve(ben_positions, np.ones(window) / window, mode="valid")
            jerry_ma = np.convolve(
                jerry_positions, np.ones(window) / window, mode="valid"
            )
            ax2.plot(ben_ma, "b-", linewidth=2, label="Ben")
            ax2.plot(jerry_ma, "r-", linewidth=2, label="Jerry")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Position")
        ax2.set_title("Position Convergence", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Step rewards
        ax3 = fig.add_subplot(gs[1, 1])
        if len(ben_rewards) > window:
            ben_smooth = np.convolve(
                ben_rewards, np.ones(window) / window, mode="valid"
            )
            jerry_smooth = np.convolve(
                jerry_rewards, np.ones(window) / window, mode="valid"
            )
            ax3.plot(ben_smooth, "b-", label="Ben", alpha=0.8)
            ax3.plot(jerry_smooth, "r-", label="Jerry", alpha=0.8)
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
        periods = np.arange(1, len(ben_relocations_by_period) + 1)
        width = 0.35
        ax5.bar(
            periods - width / 2,
            ben_relocations_by_period,
            width,
            label="Ben",
            color="blue",
            alpha=0.6,
        )
        ax5.bar(
            periods + width / 2,
            jerry_relocations_by_period,
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

        # Plot 6: Market state snapshot
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_market_state(ax6, ben_positions[-1], jerry_positions[-1])

        # Plot 7: Cumulative Rewards
        if ben_cumulative_rewards is not None and jerry_cumulative_rewards is not None:
            ax_cum = fig.add_subplot(gs[3, :])
            ax_cum.plot(
                ben_cumulative_rewards,
                "b-",
                linewidth=2.5,
                label=f"Ben: {ben_strategy_name}",
            )
            ax_cum.plot(
                jerry_cumulative_rewards,
                "r-",
                linewidth=2.5,
                label=f"Jerry: {jerry_strategy_name}",
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

        # Plot 8: Weight evolution heatmap
        ax7 = fig.add_subplot(gs[4, :])
        if len(self.env.weight_history) > 0:
            weight_array = np.array(self.env.weight_history[-400:]).T
            im = ax7.imshow(
                weight_array, aspect="auto", cmap="YlOrRd", interpolation="nearest"
            )
            ax7.set_xlabel("Time Step (last 400)", fontsize=11)
            ax7.set_ylabel("Position Index", fontsize=11)
            ax7.set_title("Market Weight Evolution", fontsize=13, fontweight="bold")
            plt.colorbar(im, ax=ax7, label="Density")

        plt.tight_layout()
        plt.show()

    def _plot_market_state(self, ax, ben_pos: float, jerry_pos: float):
        """Plot current market state"""
        positions = np.linspace(0, self.beach_length, self.n_positions)
        weights = self.env.weights

        ax.plot([0, self.beach_length], [0, 0], "k-", linewidth=2, alpha=0.5)

        for i, (pos, weight) in enumerate(zip(positions, weights)):
            ax.plot([pos, pos], [0, weight * 100], "lightblue", linewidth=8, alpha=0.7)

        ax.plot(ben_pos, 0, "bo", markersize=15, label="Ben", zorder=10)
        ax.plot(jerry_pos, 0, "ro", markersize=15, label="Jerry", zorder=10)

        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, max(weights) * 110)
        ax.set_xlabel("Beach Position")
        ax.set_ylabel("Customer Density")
        ax.set_title("Market State", fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def plot_live_state(self, ben_pos_idx: int, jerry_pos_idx: int, pause: float = 0.1):
        """Plot live state during training"""
        plt.clf()
        ax = plt.gca()

        positions = np.linspace(0, self.beach_length, self.n_positions)
        weights = self.env.weights

        ax.plot([0, self.beach_length], [0, 0], "k-", linewidth=2)

        for pos, w in zip(positions, weights):
            ax.plot([pos, pos], [0, w * 100], color="lightblue", linewidth=8)

        ben_x = self.env._get_position_value(ben_pos_idx)
        jerry_x = self.env._get_position_value(jerry_pos_idx)

        ax.plot(ben_x, 0, "bo", markersize=14, label="Ben")
        ax.plot(jerry_x, 0, "ro", markersize=14, label="Jerry")

        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, max(weights) * 110)
        ax.set_title("Live Market Evolution")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.pause(pause)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train_extended_hotelling(
    episodes: int = 5000,
    n_positions: int = 11,
    ben_loyalty_strategy: Optional[LoyaltyStrategy] = None,
    jerry_loyalty_strategy: Optional[LoyaltyStrategy] = None,
    initial_weights: Optional[np.ndarray] = None,
    visualize: bool = True,
):
    """
    Train agents on the extended Hotelling model.

    Args:
        episodes: Number of training episodes
        n_positions: Number of discrete positions
        ben_loyalty_strategy: Ben's loyalty strategy
        jerry_loyalty_strategy: Jerry's loyalty strategy
        initial_weights: Initial market weight distribution
        visualize: Whether to show visualizations
    """
    # Default loyalty strategies
    if ben_loyalty_strategy is None:
        ben_loyalty_strategy = HighLocalLoyalty(peak_position=3, peak_rate=0.5)
    if jerry_loyalty_strategy is None:
        jerry_loyalty_strategy = ModerateSpreadLoyalty(center_position=7, base_rate=0.3)

    # Initialize environment
    env = ExtendedHotellingEnv(n_positions=n_positions, initial_weights=initial_weights)

    if visualize:
        plt.ion()
        visualizer = HotellingVisualizer(env)

    # Initialize agents
    ben = QLearningAgent("Ben", n_positions, ben_loyalty_strategy)
    jerry = QLearningAgent("Jerry", n_positions, jerry_loyalty_strategy)

    # Tracking
    ben_rewards = []
    jerry_rewards = []
    ben_positions = []
    jerry_positions = []
    ben_cumulative_rewards = []
    jerry_cumulative_rewards = []
    ben_cumulative = 0.0
    jerry_cumulative = 0.0

    movement_stats = {"both_moved": 0, "ben_only": 0, "jerry_only": 0, "neither": 0}

    period_length = 100
    ben_relocations_by_period = []
    jerry_relocations_by_period = []
    current_period_ben = 0
    current_period_jerry = 0

    print(f"Training with loyalty strategies:")
    print(f"  Ben: {ben.loyalty_strategy.get_name()}")
    print(f"  Jerry: {jerry.loyalty_strategy.get_name()}")
    print()

    # Training loop
    for e in range(episodes):
        state, _ = env.reset()
        episode_ben_reward = 0
        episode_jerry_reward = 0

        for step in range(10):
            ben_action = ben.act(state[0], state[1])
            jerry_action = jerry.act(state[1], state[0])

            # Get loyalty rates for current positions
            ben_loyalty = ben.get_loyalty_rate(state[0])
            jerry_loyalty = jerry.get_loyalty_rate(state[1])

            next_state, (ben_reward, jerry_reward), _, _, info = env.step(
                (ben_action, jerry_action), ben_loyalty, jerry_loyalty
            )

            # Track movements
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

            # Learn
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

            if visualize and e % 100 == 0:
                visualizer.plot_live_state(env.ben_pos, env.jerry_pos, pause=0.05)

            state = next_state
            episode_ben_reward += ben_reward
            episode_jerry_reward += jerry_reward

        ben.decay_epsilon()
        jerry.decay_epsilon()

        # Average reward per step in episode
        avg_ben_reward = episode_ben_reward / 10
        avg_jerry_reward = episode_jerry_reward / 10

        ben_rewards.append(avg_ben_reward)
        jerry_rewards.append(avg_jerry_reward)
        ben_positions.append(env._get_position_value(env.ben_pos))
        jerry_positions.append(env._get_position_value(env.jerry_pos))

        ben_cumulative += avg_ben_reward
        jerry_cumulative += avg_jerry_reward

        ben_cumulative_rewards.append(ben_cumulative)
        jerry_cumulative_rewards.append(jerry_cumulative)

        # Record relocations by period
        if (e + 1) % period_length == 0:
            ben_relocations_by_period.append(current_period_ben)
            jerry_relocations_by_period.append(current_period_jerry)
            current_period_ben = 0
            current_period_jerry = 0

        if (e + 1) % 500 == 0:
            print(
                f"Episode {e + 1}/{episodes} - "
                f"Ben: pos={ben_positions[-1]:.1f}, cum={ben_cumulative:.2f} | "
                f"Jerry: pos={jerry_positions[-1]:.1f}, cum={jerry_cumulative:.2f} | "
                f"ε: {ben.epsilon:.3f}"
            )

    if visualize:
        plt.ioff()

    # Test final policy
    print("\n=== Testing Learned Policy ===")
    ben.epsilon = 0
    jerry.epsilon = 0

    test_positions_ben = []
    test_positions_jerry = []

    for _ in range(20):
        state, _ = env.reset()
        for _ in range(10):
            ben_action = ben.act(state[0], state[1])
            jerry_action = jerry.act(state[1], state[0])
            ben_loyalty = ben.get_loyalty_rate(state[0])
            jerry_loyalty = jerry.get_loyalty_rate(state[1])
            state, _, _, _, _ = env.step(
                (ben_action, jerry_action), ben_loyalty, jerry_loyalty
            )

        test_positions_ben.append(env._get_position_value(env.ben_pos))
        test_positions_jerry.append(env._get_position_value(env.jerry_pos))

    avg_ben = np.mean(test_positions_ben)
    avg_jerry = np.mean(test_positions_jerry)

    print(f"Average converged positions:")
    print(f"  Ben:   {avg_ben:.1f}")
    print(f"  Jerry: {avg_jerry:.1f}")
    print(f"Final cumulative rewards:")
    print(f"  Ben:   {ben_cumulative:.2f}")
    print(f"  Jerry: {jerry_cumulative:.2f}")

    # Visualize
    if visualize:
        visualizer = HotellingVisualizer(env)
        visualizer.plot_training_summary(
            ben_rewards=ben_rewards,
            jerry_rewards=jerry_rewards,
            ben_positions=ben_positions,
            jerry_positions=jerry_positions,
            movement_stats=movement_stats,
            ben_relocations_by_period=ben_relocations_by_period,
            jerry_relocations_by_period=jerry_relocations_by_period,
            period_length=period_length,
            ben_cumulative_rewards=ben_cumulative_rewards,
            jerry_cumulative_rewards=jerry_cumulative_rewards,
            ben_strategy_name=ben.loyalty_strategy.get_name(),
            jerry_strategy_name=jerry.loyalty_strategy.get_name(),
        )

    return env, ben, jerry


# ============================================================================
# EXAMPLE USAGE
# ============================================================================


def main():
    # ============================================================
    # SCENARIO 1: Asymmetric Loyalty, Uniform Market
    # ============================================================
    print("=" * 70)
    print("SCENARIO 1: Asymmetric Loyalty with Uniform Market")
    print("=" * 70)

    ben_strategy = HighLocalLoyalty(peak_position=2, peak_rate=0.6, falloff=0.7)

    jerry_strategy = ModerateSpreadLoyalty(center_position=8, base_rate=0.35, spread=2)

    env1, ben1, jerry1 = train_extended_hotelling(
        episodes=5000,
        n_positions=11,
        ben_loyalty_strategy=ben_strategy,
        jerry_loyalty_strategy=jerry_strategy,
        initial_weights=None,
        visualize=True,
    )

    # ============================================================
    # SCENARIO 2: Bimodal Market Distribution
    # ============================================================
    print("\n" + "=" * 70)
    print("SCENARIO 2: Bimodal Market Distribution")
    print("=" * 70)

    bimodal_weights = np.array(
        [0.15, 0.12, 0.08, 0.05, 0.03, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15]
    )

    ben_strategy2 = UniformLoyalty(base_rate=0.25)
    jerry_strategy2 = UniformLoyalty(base_rate=0.25)

    env2, ben2, jerry2 = train_extended_hotelling(
        episodes=5000,
        n_positions=11,
        ben_loyalty_strategy=ben_strategy2,
        jerry_loyalty_strategy=jerry_strategy2,
        initial_weights=bimodal_weights,
        visualize=True,
    )


if __name__ == "__main__":
    main()
