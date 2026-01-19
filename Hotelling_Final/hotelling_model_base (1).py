import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

# --- 1. Environnement Hotelling ---
class HotellingEnv(gym.Env):
    def __init__(self, n_positions=11):
        super(HotellingEnv, self).__init__()
        self.beach_length = 100
        self.n_positions = n_positions  
        self.action_space = spaces.Discrete(self.n_positions)
        self.observation_space = spaces.Box(
            low=0, high=self.n_positions - 1, shape=(2,), dtype=np.int32
        )
        self.ben_pos = 0
        self.jerry_pos = 0

    def _get_position_value(self, action):
        return action * (self.beach_length / (self.n_positions - 1))

    def _calculate_rewards(self):
        ben_actual = self._get_position_value(self.ben_pos)
        jerry_actual = self._get_position_value(self.jerry_pos)
        
        if ben_actual == jerry_actual:
            return 0.5, 0.5
            
        midpoint = (ben_actual + jerry_actual) / 2
        if ben_actual < jerry_actual:
            ben_share = midpoint / self.beach_length
            jerry_share = 1 - ben_share
        else:
            jerry_share = midpoint / self.beach_length
            ben_share = 1 - jerry_share
        return ben_share, jerry_share

    def step(self, actions):
        ben_action, jerry_action = actions
        self.ben_pos = ben_action
        self.jerry_pos = jerry_action
        ben_reward, jerry_reward = self._calculate_rewards()
        state = np.array([self.ben_pos, self.jerry_pos], dtype=np.int32)
        return state, (ben_reward, jerry_reward), False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ben_pos = np.random.randint(0, self.n_positions)
        self.jerry_pos = np.random.randint(0, self.n_positions)
        state = np.array([self.ben_pos, self.jerry_pos], dtype=np.int32)
        return state, {}

# --- 2. Agent Q-Learning (Intelligent) ---
class QLearningAgent:
    def __init__(self, name, n_positions):
        self.name = name
        self.n_positions = n_positions
        self.n_actions = n_positions
        self.q_table = np.zeros((self.n_positions, self.n_actions))

        self.alpha = 0.1      
        
        # --- RETOUR À LA NORMALE ---
        # On remet une vision long terme.
        # Les valeurs dans la Q-table vont monter jusqu'à ~10 (car 0.5 / (1-0.95))
        self.gamma = 0.95     
        
        self.epsilon = 1.0    
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999 

    def act(self, opponent_pos):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        values = self.q_table[opponent_pos]
        best_actions = np.flatnonzero(values == values.max())
        return np.random.choice(best_actions)

    def learn(self, opponent_pos, action, reward, next_opponent_pos):
        current_q = self.q_table[opponent_pos, action]
        max_next_q = np.max(self.q_table[next_opponent_pos])
        
        # Formule standard du Q-Learning
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[opponent_pos, action] = new_q

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- 3. Entraînement et Visualisation ---
def train_and_visualize(episodes=5000, n_positions=11):
    env = HotellingEnv(n_positions=n_positions)
    ben = QLearningAgent("Ben", n_positions)
    jerry = QLearningAgent("Jerry", n_positions)

    ben_rewards = []
    jerry_rewards = []
    ben_positions = []
    jerry_positions = []

    print(f"Lancement avec Gamma={ben.gamma}...")

    for e in range(episodes):
        state, _ = env.reset()
        episode_ben_reward = 0
        episode_jerry_reward = 0

        for step in range(10):
            ben_action = ben.act(state[1]) 
            jerry_action = jerry.act(state[0])
            next_state, (ben_reward, jerry_reward), _, _, _ = env.step(
                (ben_action, jerry_action)
            )
            ben.learn(state[1], ben_action, ben_reward, next_state[1])
            jerry.learn(state[0], jerry_action, jerry_reward, next_state[0])
            state = next_state
            episode_ben_reward += ben_reward
            episode_jerry_reward += jerry_reward

        ben.decay_epsilon()
        jerry.decay_epsilon()
        ben_rewards.append(episode_ben_reward / 10)
        jerry_rewards.append(episode_jerry_reward / 10)
        ben_positions.append(env._get_position_value(env.ben_pos))
        jerry_positions.append(env._get_position_value(env.jerry_pos))

    # --- Visualisation ---
    ben.epsilon = 0
    jerry.epsilon = 0
    
    # Validation finale
    final_ben_pos = []
    final_jerry_pos = []
    for _ in range(20):
        s, _ = env.reset()
        for _ in range(5):
            ba = ben.act(s[1])
            ja = jerry.act(s[0])
            s, _, _, _, _ = env.step((ba, ja))
        final_ben_pos.append(env._get_position_value(s[0]))
        final_jerry_pos.append(env._get_position_value(s[1]))
    avg_ben = np.mean(final_ben_pos)
    avg_jerry = np.mean(final_jerry_pos)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1
    ax1 = axes[0, 0]
    ax1.plot([0, 100], [0, 0], "k-", linewidth=3, label="Plage")
    ax1.plot(avg_ben, 0, "bo", markersize=20, label="Ben (Final)", zorder=5)
    ax1.plot(avg_jerry, 0, "ro", markersize=20, label="Jerry (Final)", zorder=5)
    ax1.axvline(x=50, color="green", linestyle="--", alpha=0.5)
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-1, 1)
    ax1.set_title(f"Position Finale (Moyenne)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2
    ax2 = axes[0, 1]
    ax2.plot(ben_positions, "b-", alpha=0.15, linewidth=0.5)
    ax2.plot(jerry_positions, "r-", alpha=0.15, linewidth=0.5)
    window = 100
    if len(ben_positions) > window:
        ben_ma = np.convolve(ben_positions, np.ones(window)/window, mode='valid')
        jerry_ma = np.convolve(jerry_positions, np.ones(window)/window, mode='valid')
        ax2.plot(ben_ma, "b-", linewidth=2, label="Ben (Moy)")
        ax2.plot(jerry_ma, "r-", linewidth=2, label="Jerry (Moy)")
    ax2.axhline(y=50, color="green", linestyle="--")
    ax2.set_title("Évolution")
    ax2.legend()

    # Plot 3
    ax3 = axes[1, 0]
    if len(ben_rewards) > window:
        ben_smooth = np.convolve(ben_rewards, np.ones(window)/window, mode='valid')
        jerry_smooth = np.convolve(jerry_rewards, np.ones(window)/window, mode='valid')
        ax3.plot(ben_smooth, "b-", label="Ben")
        ax3.plot(jerry_smooth, "r-", label="Jerry")
    ax3.axhline(y=0.5, color="gray", linestyle="--")
    ax3.set_title("Parts de Marché")
    ax3.legend()

    # --- CORRECTION DE L'AFFICHAGE ---
    ax4 = axes[1, 1]
    
    # On récupère la Q-Table brute
    q_data = ben.q_table
    
    # NORMALISATION MIN-MAX LOCALE
    # On transforme les valeurs (ex: 9.5 à 10.0) en échelle (0.0 à 1.0)
    # Ainsi, le pire coup de tout le tableau sera 0 (Rouge) et le meilleur 1 (Vert)
    q_norm = (q_data - np.min(q_data)) / (np.max(q_data) - np.min(q_data))
    
    im = ax4.imshow(q_norm, cmap="RdYlGn", aspect="auto", origin='lower')
    
    ax4.set_xlabel("Action de Ben (Position 0-100)")
    ax4.set_ylabel("État (Position de Jerry 0-100)")
    ax4.set_title("Q-Table: Préférence Relative (Rouge=Pire, Vert=Meilleur)")
    
    # Ticks propres
    ticks = np.arange(n_positions)
    labels = [f"{int(i * 100 / (n_positions - 1))}" for i in ticks]
    ax4.set_xticks(ticks)
    ax4.set_xticklabels(labels, rotation=45)
    ax4.set_yticks(ticks)
    ax4.set_yticklabels(labels)

    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_visualize(episodes=5000, n_positions=11)