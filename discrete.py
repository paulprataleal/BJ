import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces


# Hotelling Environment
class HotellingEnv(gym.Env):
    def __init__(self, n_positions=11):
        super(HotellingEnv, self).__init__()
        self.beach_length = 100
        self.n_positions = n_positions  # Fewer positions for simpler learning

        self.action_space = spaces.Discrete(self.n_positions)
        self.observation_space = spaces.Box(
            low=0, high=self.n_positions - 1, shape=(2,), dtype=np.int32
        )

        self.ben_pos = 0
        self.jerry_pos = 0

    def _get_position_value(self, action):
        """Convert discrete action to position on beach"""
        return action * (self.beach_length / (self.n_positions - 1))

    def _calculate_rewards(self):
        """Calculate market share for each vendor"""
        ben_actual = self._get_position_value(self.ben_pos)
        jerry_actual = self._get_position_value(self.jerry_pos)
        midpoint = (ben_actual + jerry_actual) / 2

        if ben_actual < jerry_actual:
            # Ben gets all customers from 0 to midpoint
            ben_share = midpoint / self.beach_length
            jerry_share = 1 - ben_share
        elif ben_actual > jerry_actual:
            # Jerry gets all customers from 0 to midpoint
            jerry_share = midpoint / self.beach_length
            ben_share = 1 - jerry_share
        else:  # Same position - split market
            ben_share = jerry_share = 0.5

        return ben_share, jerry_share

    def step(self, actions):
        """actions: tuple of (ben_action, jerry_action)"""
        ben_action, jerry_action = actions

        self.ben_pos = ben_action
        self.jerry_pos = jerry_action

        ben_reward, jerry_reward = self._calculate_rewards()

        state = np.array([self.ben_pos, self.jerry_pos], dtype=np.int32)

        return state, (ben_reward, jerry_reward), False, False, {}

    def reset(self, seed=None, type : str = "random"):
        super().reset(seed=seed)
        if type == "random":
            # Random initial positions
            self.ben_pos = np.random.randint(0, self.n_positions)
            self.jerry_pos = np.random.randint(0, self.n_positions)
            state = np.array([self.ben_pos, self.jerry_pos], dtype=np.int32)
        elif type == "extreme":
            self.
        return state, {}


# Simple Q-Learning Agent
class QLearningAgent:
    def __init__(self, name, n_positions):
        self.name = name
        self.n_positions = n_positions
        self.n_actions = n_positions

        # Q-table: rows = opponent's position, cols = my action
        self.q_table = np.zeros((self.n_positions, self.n_actions))

        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, opponent_pos):
        """Choose action based on epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[opponent_pos])

    def learn(self, opponent_pos, action, reward, next_opponent_pos):
        """Update Q-table"""
        current_q = self.q_table[opponent_pos, action]
        max_next_q = np.max(self.q_table[next_opponent_pos])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[opponent_pos, action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Training and Visualization
def train_and_visualize(episodes=2000, n_positions=11):
    env = HotellingEnv(n_positions=n_positions)
    ben = QLearningAgent("Ben", n_positions)
    jerry = QLearningAgent("Jerry", n_positions)

    ben_rewards = []
    jerry_rewards = []
    ben_positions = []
    jerry_positions = []

    for e in range(episodes):
        state, _ = env.reset()
        episode_ben_reward = 0
        episode_jerry_reward = 0

        # Multiple steps per episode to allow learning
        for step in range(10):
            ben_action = ben.act(state[1])  # Ben observes Jerry's position
            jerry_action = jerry.act(state[0])  # Jerry observes Ben's position

            next_state, (ben_reward, jerry_reward), _, _, _ = env.step(
                (ben_action, jerry_action)
            )

            # Both agents learn
            ben.learn(state[1], ben_action, ben_reward, next_state[1])
            jerry.learn(state[0], jerry_action, jerry_reward, next_state[0])

            state = next_state
            episode_ben_reward += ben_reward
            episode_jerry_reward += jerry_reward

        ben.decay_epsilon()
        jerry.decay_epsilon()

        # Record final positions and average rewards
        ben_rewards.append(episode_ben_reward / 10)
        jerry_rewards.append(episode_jerry_reward / 10)
        ben_positions.append(env._get_position_value(env.ben_pos))
        jerry_positions.append(env._get_position_value(env.jerry_pos))

        if (e + 1) % 200 == 0:
            print(
                f"Episode {e + 1}/{episodes} - Ben: {ben_positions[-1]:.1f}, Jerry: {jerry_positions[-1]:.1f}, "
                f"Epsilon: {ben.epsilon:.3f}, Rewards: Ben={ben_rewards[-1]:.3f}, Jerry={jerry_rewards[-1]:.3f}"
            )

    # Test final policy (no exploration)
    print("\n=== Testing Learned Policy (No Exploration) ===")
    ben.epsilon = 0
    jerry.epsilon = 0
    test_positions_ben = []
    test_positions_jerry = []

    for _ in range(20):
        state, _ = env.reset()
        for _ in range(10):
            ben_action = ben.act(state[1])
            jerry_action = jerry.act(state[0])
            state, _, _, _, _ = env.step((ben_action, jerry_action))
        test_positions_ben.append(env._get_position_value(env.ben_pos))
        test_positions_jerry.append(env._get_position_value(env.jerry_pos))

    avg_ben = np.mean(test_positions_ben)
    avg_jerry = np.mean(test_positions_jerry)

    print(f"Average Ben position: {avg_ben:.1f}")
    print(f"Average Jerry position: {avg_jerry:.1f}")
    print(
        f"Average distance from center (50): Ben={abs(50 - avg_ben):.1f}, Jerry={abs(50 - avg_jerry):.1f}"
    )

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Final positions on beach
    ax1 = axes[0, 0]
    ax1.plot([0, 100], [0, 0], "k-", linewidth=3, label="Beach")
    ax1.plot(avg_ben, 0, "bo", markersize=20, label="Ben (avg)", zorder=5)
    ax1.plot(avg_jerry, 0, "ro", markersize=20, label="Jerry (avg)", zorder=5)
    ax1.axvline(x=50, color="green", linestyle="--", alpha=0.5, label="Center (Nash)")
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel("Position on Beach")
    ax1.set_title(f"Converged Positions (Avg over 20 tests)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Position evolution during training
    ax2 = axes[0, 1]
    ax2.plot(ben_positions, "b-", label="Ben", alpha=0.5, linewidth=0.5)
    ax2.plot(jerry_positions, "r-", label="Jerry", alpha=0.5, linewidth=0.5)
    # Moving average
    window = 50
    if len(ben_positions) > window:
        ben_ma = np.convolve(ben_positions, np.ones(window) / window, mode="valid")
        jerry_ma = np.convolve(jerry_positions, np.ones(window) / window, mode="valid")
        ax2.plot(ben_ma, "b-", linewidth=2, label="Ben (MA)")
        ax2.plot(jerry_ma, "r-", linewidth=2, label="Jerry (MA)")
    ax2.axhline(y=50, color="green", linestyle="--", alpha=0.5, label="Center")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Position")
    ax2.set_title("Position Evolution During Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Rewards evolution
    ax3 = axes[1, 0]
    window = 50
    if len(ben_rewards) > window:
        ben_smooth = np.convolve(ben_rewards, np.ones(window) / window, mode="valid")
        jerry_smooth = np.convolve(
            jerry_rewards, np.ones(window) / window, mode="valid"
        )
        ax3.plot(ben_smooth, "b-", label="Ben", linewidth=2)
        ax3.plot(jerry_smooth, "r-", label="Jerry", linewidth=2)
    ax3.axhline(y=0.5, color="gray", linestyle="--", label="Equal share")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Average Market Share")
    ax3.set_title("Market Share Evolution (smoothed)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Q-table heatmaps
    ax4 = axes[1, 1]
    # Show Ben's Q-table
    im = ax4.imshow(ben.q_table, cmap="RdYlGn", aspect="auto")
    ax4.set_xlabel("Ben's Action (Position)")
    ax4.set_ylabel("Jerry's Position")
    ax4.set_title("Ben's Q-Table (Greener = Better)")
    plt.colorbar(im, ax=ax4)

    # Set tick labels
    positions = [f"{int(i * 100 / (n_positions - 1))}" for i in range(n_positions)]
    ax4.set_xticks(range(n_positions))
    ax4.set_xticklabels(positions, rotation=45)
    ax4.set_yticks(range(n_positions))
    ax4.set_yticklabels(positions)

    plt.tight_layout()
    plt.show()

    print(f"\n=== Theory Check ===")
    print(f"Hotelling's model predicts both vendors converge to center (50)")
    print(f"Your agents converged to: Ben={avg_ben:.1f}, Jerry={avg_jerry:.1f}")
    print(
        f"Success! ✓"
        if abs(avg_ben - 50) < 15 and abs(avg_jerry - 50) < 15
        else "Needs more training..."
    )


if __name__ == "__main__":
    train_and_visualize(episodes=10000, n_positions=11)
