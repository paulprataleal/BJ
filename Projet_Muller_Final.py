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
    """
    Modèle Hotelling Étendu :
    1. Physique de foule : Les agents transportent 20% de la densité locale quand ils bougent.
    2. Exclusion : On ne peut pas Vendre ET Bouger au même tour.
    """

    def __init__(
        self,
        n_positions: int = 11,
        initial_weights: Optional[np.ndarray] = None,
        cost_scaling: float = 1.0,
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
        
        # Initialisation standard
        self.ben_pos = 0
        self.jerry_pos = 0
        self.prev_ben_pos = 0
        self.prev_jerry_pos = 0
        self.ben_time_at_position = 0
        self.jerry_time_at_position = 0
        self.step_count = 0
        self.weight_history = []
        self.position_history = []

    def _get_position_value(self, action: int) -> float:
        return action * (self.beach_length / (self.n_positions - 1))

    # --- NOUVELLE MÉTHODE : La foule grossit si on reste ---
    def _apply_tenure_growth(self, pos: int, growth_rate: float = 0.05):
        self.weights[pos] *= (1.0 + growth_rate)
        self.weights /= self.weights.sum()

    # --- NOUVELLE MÉTHODE : La foule se déplace avec l'agent ---
    def _apply_loyalty_transport(self, old_pos: int, new_pos: int, transport_rate: float = 0.20):
        if old_pos == new_pos:
            return
        
        # On prend X% de la densité de l'ancienne case
        transported_mass = self.weights[old_pos] * transport_rate
        
        # On déplace la masse
        self.weights[old_pos] -= transported_mass
        self.weights[new_pos] += transported_mass
        
        # On s'assure que rien n'est négatif (arrondi float) et on normalise
        self.weights = np.maximum(self.weights, 0)
        self.weights /= self.weights.sum()

    def _apply_demand_shock(self):
        """Fluctuations aléatoires mineures du marché"""
        noise = np.random.normal(0, self.demand_volatility, self.n_positions)
        self.weights = self.weights * (1 + noise)
        self.weights = np.maximum(self.weights, 0)
        self.weights = self.weights / self.weights.sum()

    # --- NOUVELLE LOGIQUE DE RÉCOMPENSE : Exclusion Mutuelle ---
    def _calculate_rewards_strict(self, ben_moved: bool, jerry_moved: bool) -> Tuple[float, float]:
        
        # 1. Coûts de déplacement
        ben_dist = abs(self._get_position_value(self.prev_ben_pos) - self._get_position_value(self.ben_pos))
        jerry_dist = abs(self._get_position_value(self.prev_jerry_pos) - self._get_position_value(self.jerry_pos))
        
        ben_cost = self.cost_scaling * (ben_dist ** 2)
        jerry_cost = self.cost_scaling * (jerry_dist ** 2)

        # 2. Calcul des parts de marché (SEULEMENT si on reste)
        ben_share = 0.0
        jerry_share = 0.0
        
        # Si au moins un joueur vend, on doit calculer la répartition géographique
        if not ben_moved or not jerry_moved:
            positions = np.linspace(0, self.beach_length, self.n_positions)
            ben_loc = self._get_position_value(self.ben_pos)
            jerry_loc = self._get_position_value(self.jerry_pos)
            
            # Distance de chaque segment de plage vers les vendeurs
            dist_ben = np.abs(positions - ben_loc)
            dist_jerry = np.abs(positions - jerry_loc)
            
            # Qui gagne quel segment ?
            ben_wins = dist_ben < dist_jerry
            jerry_wins = dist_jerry < dist_ben
            ties = dist_ben == dist_jerry
            
            # Somme des poids gagnés
            ben_share = np.sum(self.weights[ben_wins]) + 0.5 * np.sum(self.weights[ties])
            jerry_share = np.sum(self.weights[jerry_wins]) + 0.5 * np.sum(self.weights[ties])

        reward_multiplier = max(0.1, np.random.normal(self.reward_mean, self.reward_std))
        
        # 3. Attribution (L'exclusion stricte)
        # BEN
        if ben_moved:
            # Punitif : Coût pur, aucune vente. Gain stratégique futur uniquement.
            r_ben = -ben_cost 
        else:
            # Lucratif : On encaisse
            r_ben = (ben_share * reward_multiplier)

        # JERRY
        if jerry_moved:
            r_jerry = -jerry_cost
        else:
            r_jerry = (jerry_share * reward_multiplier)
            
        return r_ben, r_jerry

    def step(
        self,
        actions: Tuple[int, int],
        ben_loyalty_rate: float, 
        jerry_loyalty_rate: float,
    ) -> Tuple[np.ndarray, Tuple[float, float], bool, bool, Dict]:
        
        ben_action, jerry_action = actions
        ben_moved = ben_action != self.ben_pos
        jerry_moved = jerry_action != self.jerry_pos
        
        self.prev_ben_pos = self.ben_pos
        self.prev_jerry_pos = self.jerry_pos
        self.ben_pos = ben_action
        self.jerry_pos = jerry_action
        
        # --- PHYSIQUE ---
        
        # BEN
        if ben_moved:
            # Transport de foule (si je bouge, j'emmène ma fidélité acquise)
            self._apply_loyalty_transport(self.prev_ben_pos, self.ben_pos, transport_rate=ben_loyalty_rate)
            self.ben_time_at_position = 0
        else:
            # CROISSANCE DÉSACTIVÉE (Pour isoler l'effet du coût de déplacement)
            # self._apply_tenure_growth(self.ben_pos, growth_rate=0.05) 
            
            # On garde juste le compteur de temps pour le futur taux de fidélité
            self.ben_time_at_position += 1
            
        # JERRY
        if jerry_moved:
            self._apply_loyalty_transport(self.prev_jerry_pos, self.jerry_pos, transport_rate=jerry_loyalty_rate)
            self.jerry_time_at_position = 0
        else:
            # CROISSANCE DÉSACTIVÉE ICI AUSSI
            # self._apply_tenure_growth(self.jerry_pos, growth_rate=0.05)
            self.jerry_time_at_position += 1

        # Volatilité du marché
        self._apply_demand_shock()

        # Calcul des récompenses (Vente OU Mouvement)
        ben_reward, jerry_reward = self._calculate_rewards_strict(ben_moved, jerry_moved)
        
        # Historique
        self.weight_history.append(self.weights.copy())
        self.position_history.append((self.ben_pos, self.jerry_pos))
        self.step_count += 1

        state = np.array([self.ben_pos, self.jerry_pos], dtype=np.int32)
        info = {"ben_moved": ben_moved, "jerry_moved": jerry_moved}
        
        return state, (ben_reward, jerry_reward), False, False, info

    def reset(self, seed: Optional[int] = None, reset_type: str = "random") -> Tuple[np.ndarray, Dict]:
        """Reset stochastique : Le marché change à chaque épisode !"""
        super().reset(seed=seed)
        self.step_count = 0
        
        # Poids Stochastiques ---
        # On génère une distribution de clients aléatoire à chaque reset
        raw_weights = np.random.rand(self.n_positions)
        self.weights = raw_weights / raw_weights.sum()
        # ------------------------------------------

        self.weight_history = [self.weights.copy()]
        self.position_history = []
        self.ben_time_at_position = 0
        self.jerry_time_at_position = 0

        if reset_type == "random":
            self.ben_pos = np.random.randint(0, self.n_positions)
            self.jerry_pos = np.random.randint(0, self.n_positions)
        elif reset_type == "extremes":
            self.ben_pos = np.random.choice([0, self.n_positions - 1])
            self.jerry_pos = 0 if self.ben_pos == self.n_positions - 1 else self.n_positions - 1

        self.prev_ben_pos = self.ben_pos
        self.prev_jerry_pos = self.jerry_pos
        
        return np.array([self.ben_pos, self.jerry_pos], dtype=np.int32), {}
# ============================================================================
# Q-LEARNING AGENT
# ============================================================================


class QLearningAgent:
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
    """
    Visualisation complète incluant :
    - Distribution finale
    - Heatmap (Évolution des densités de foule)
    - Cumulative Rewards (Rentabilité)
    """

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
        ben_relocations: list,
        jerry_relocations: list,
        period_length: int,
        ben_cumulative: list,
        jerry_cumulative: list,
        ben_name: str,
        jerry_name: str,
        game_weight_history: list,  # <--- NOUVEAU : Pour la Heatmap
    ):
        fig = plt.figure(figsize=(20, 18))
        # Grille 5 lignes x 3 colonnes
        gs = fig.add_gridspec(5, 3, hspace=0.5, wspace=0.3)

        # --- PLOT 1: Distribution Finale des Poids ---
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
        
        # Moyenne des positions sur la fin
        avg_ben = np.mean(ben_positions[-50:]) if len(ben_positions) > 50 else np.mean(ben_positions)
        avg_jerry = np.mean(jerry_positions[-50:]) if len(jerry_positions) > 50 else np.mean(jerry_positions)
        max_weight = final_weights.max() if len(final_weights) > 0 else 0.1

        ax1.plot(avg_ben, max_weight * 1.1, "bv", markersize=15, label=f"Ben (avg: {avg_ben:.1f})", zorder=5)
        ax1.plot(avg_jerry, max_weight * 1.1, "rv", markersize=15, label=f"Jerry (avg: {avg_jerry:.1f})", zorder=5)
        ax1.set_title("Final Market Weight Distribution", fontsize=13, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # --- PLOT 2: Évolution des Positions ---
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(ben_positions, "b-", linewidth=1.5, alpha=0.8, label="Ben")
        ax2.plot(jerry_positions, "r-", linewidth=1.5, alpha=0.8, label="Jerry")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Position")
        ax2.set_title("Position Evolution (Game Phase)", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- PLOT 3: Rewards par Step ---
        ax3 = fig.add_subplot(gs[1, 1])
        # Lissage pour lisibilité
        window = max(1, len(ben_rewards) // 20)
        if len(ben_rewards) > window:
            b_smooth = np.convolve(ben_rewards, np.ones(window)/window, mode='valid')
            j_smooth = np.convolve(jerry_rewards, np.ones(window)/window, mode='valid')
            ax3.plot(b_smooth, "b-", label="Ben", alpha=0.8)
            ax3.plot(j_smooth, "r-", label="Jerry", alpha=0.8)
        else:
            ax3.plot(ben_rewards, "b-", label="Ben")
            ax3.plot(jerry_rewards, "r-", label="Jerry")
            
        ax3.set_title("Smoothed Reward Evolution", fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # --- PLOT 4: Stats Mouvements ---
        ax4 = fig.add_subplot(gs[1, 2])
        labels = ["Both\nMoved", "Ben\nOnly", "Jerry\nOnly", "Neither"]
        values = [
            movement_stats["both_moved"],
            movement_stats["ben_only"],
            movement_stats["jerry_only"],
            movement_stats["neither"],
        ]
        ax4.bar(labels, values, color=["#ff6b6b", "#4ecdc4", "#95e1d3", "#a8e6cf"], edgecolor="black")
        ax4.set_title("Movement Patterns", fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="y")

        # --- PLOT 5: Relocations (Pas très pertinent sur un run court mais on garde) ---
        ax5 = fig.add_subplot(gs[2, :2])
        # On affiche juste un histogramme simple des positions visitées si pas de périodes
        ax5.hist(ben_positions, bins=self.n_positions, alpha=0.5, color='blue', label='Ben Locs')
        ax5.hist(jerry_positions, bins=self.n_positions, alpha=0.5, color='red', label='Jerry Locs')
        ax5.set_title("Position Occupancy Histogram", fontweight="bold")
        ax5.legend()

        # --- PLOT 6: Snapshot État Marché ---
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_market_state(ax6, ben_positions[-1], jerry_positions[-1])

        # --- PLOT 7: Cumulative Rewards (Rentabilité) ---
        ax_cum = fig.add_subplot(gs[3, :])
        ax_cum.plot(ben_cumulative, "b-", linewidth=2.5, label=f"Ben: {ben_name}")
        ax_cum.plot(jerry_cumulative, "r-", linewidth=2.5, label=f"Jerry: {jerry_name}")
        ax_cum.set_xlabel("Time Step")
        ax_cum.set_ylabel("Total Profit")
        ax_cum.set_title("Total Profitability (Cumulative)", fontsize=13, fontweight="bold")
        ax_cum.legend(loc="upper left")
        ax_cum.grid(True, alpha=0.3)

        # --- PLOT 8: HEATMAP (Évolution des poids) ---
        ax7 = fig.add_subplot(gs[4, :])
        if len(game_weight_history) > 0:
            # On transpose pour avoir le temps en X et les positions en Y
            weight_array = np.array(game_weight_history).T 
            im = ax7.imshow(
                weight_array, 
                aspect="auto", 
                cmap="YlOrRd", 
                interpolation="nearest",
                origin="lower" # Pour que la position 0 soit en bas
            )
            ax7.set_xlabel("Time Step (Game Phase)", fontsize=11)
            ax7.set_ylabel("Position Index", fontsize=11)
            ax7.set_title("Market Crowd Density Evolution (Heatmap)", fontsize=13, fontweight="bold")
            plt.colorbar(im, ax=ax7, label="Customer Density")
        
        plt.tight_layout()
        plt.show()

    def _plot_market_state(self, ax, ben_pos: float, jerry_pos: float):
        positions = np.linspace(0, self.beach_length, self.n_positions)
        weights = self.env.weights
        ax.plot([0, self.beach_length], [0, 0], "k-", linewidth=2, alpha=0.5)
        for i, (pos, weight) in enumerate(zip(positions, weights)):
            ax.plot([pos, pos], [0, weight * 100], "lightblue", linewidth=8, alpha=0.7)
        ax.plot(ben_pos, 0, "bo", markersize=15, label="Ben", zorder=10)
        ax.plot(jerry_pos, 0, "ro", markersize=15, label="Jerry", zorder=10)
        ax.set_title("Final Market Snapshot", fontweight="bold")


# ============================================================================
# TRAINING FUNCTION
# ============================================================================


def train_extended_hotelling(
    train_episodes: int = 5000,
    game_episodes: int = 30, #Phase de test
    steps_per_episode: int = 20,
    n_positions: int = 11,
    ben_loyalty_strategy: Optional[LoyaltyStrategy] = None,
    jerry_loyalty_strategy: Optional[LoyaltyStrategy] = None,
    reward_mean: float = 10.0,
    reward_std: float = 2.0,
    cost_scaling: float = 0.01,
    demand_volatility: float = 0.1,
    visualize: bool = True,
):
    # 1. SETUP
    if ben_loyalty_strategy is None:
        ben_loyalty_strategy = HighBrandLoyalty(base_rate=0.5, tenure_bonus=0.05)
    if jerry_loyalty_strategy is None:
        jerry_loyalty_strategy = ModerateBrandLoyalty(base_rate=0.3, tenure_bonus=0.02)

    env = ExtendedHotellingEnv(
        n_positions=n_positions,
        initial_weights=None, # Stochastique géré par le reset
        cost_scaling=cost_scaling,
        reward_mean=reward_mean,
        reward_std=reward_std,
        demand_volatility=demand_volatility,
    )

    ben = QLearningAgent("Ben", n_positions, ben_loyalty_strategy)
    jerry = QLearningAgent("Jerry", n_positions, jerry_loyalty_strategy)

    # 2. PHASE D'ENTRAÎNEMENT (5000 Épisodes)
    print(f"\n>>> DÉBUT DE L'ENTRAÎNEMENT ({train_episodes} épisodes)...")
    
    for e in range(train_episodes):
        state, _ = env.reset()
        for step in range(steps_per_episode):
            ben_action = ben.act(state[0], state[1])
            jerry_action = jerry.act(state[1], state[0])
            
            # Loyalty = Transport rate
            ben_loyalty = ben.get_loyalty_rate(env.ben_time_at_position, env.weights[state[0]])
            jerry_loyalty = jerry.get_loyalty_rate(env.jerry_time_at_position, env.weights[state[1]])

            next_state, (ben_reward, jerry_reward), _, _, _ = env.step(
                (ben_action, jerry_action), ben_loyalty, jerry_loyalty
            )

            ben.learn(state[0], state[1], ben_action, ben_reward, next_state[0], next_state[1])
            jerry.learn(state[1], state[0], jerry_action, jerry_reward, next_state[1], next_state[0])
            state = next_state

        ben.decay_epsilon()
        jerry.decay_epsilon()
        if (e + 1) % 1000 == 0:
            print(f"  Training Progress: {e + 1}/{train_episodes} (Epsilon: {ben.epsilon:.3f})")

    # 3. PHASE DE JEU RÉEL (30 Épisodes)
    print(f"\n>>> DÉBUT DU 'ACTUAL GAME' ({game_episodes} épisodes de validation)...")
    ben.epsilon = 0
    jerry.epsilon = 0
    
    # Données pour la visualisation
    history_ben_pos_plot = []
    history_jerry_pos_plot = []
    history_rewards_ben_plot = []
    history_rewards_jerry_plot = []
    all_game_weights = [] # <--- Pour la Heatmap
    
    movement_stats = {"both_moved": 0, "ben_only": 0, "jerry_only": 0, "neither": 0}
    ben_wins = 0
    jerry_wins = 0
    game_ben_rewards = []
    game_jerry_rewards = []

    for e in range(game_episodes):
        state, _ = env.reset()
        # On enregistre le poids initial
        all_game_weights.append(env.weights.copy())
        
        ep_ben_rew, ep_jerry_rew = 0, 0
        ep_ben_pos = [env._get_position_value(env.ben_pos)]
        ep_jerry_pos = [env._get_position_value(env.jerry_pos)]

        for step in range(steps_per_episode):
            ben_action = ben.act(state[0], state[1])
            jerry_action = jerry.act(state[1], state[0])
            
            ben_loyalty = ben.get_loyalty_rate(env.ben_time_at_position, env.weights[state[0]])
            jerry_loyalty = jerry.get_loyalty_rate(env.jerry_time_at_position, env.weights[state[1]])

            next_state, (ben_reward, jerry_reward), _, _, info = env.step(
                (ben_action, jerry_action), ben_loyalty, jerry_loyalty
            )
            
            # Stats
            if info["ben_moved"] and info["jerry_moved"]: movement_stats["both_moved"] += 1
            elif info["ben_moved"]: movement_stats["ben_only"] += 1
            elif info["jerry_moved"]: movement_stats["jerry_only"] += 1
            else: movement_stats["neither"] += 1
            
            ep_ben_rew += ben_reward
            ep_jerry_rew += jerry_reward
            state = next_state
            
            ep_ben_pos.append(env._get_position_value(env.ben_pos))
            ep_jerry_pos.append(env._get_position_value(env.jerry_pos))
            
            # Pour la Heatmap (On enregistre les poids à chaque pas de temps)
            all_game_weights.append(env.weights.copy())

        game_ben_rewards.append(ep_ben_rew)
        game_jerry_rewards.append(ep_jerry_rew)
        
        history_ben_pos_plot.extend(ep_ben_pos)
        history_jerry_pos_plot.extend(ep_jerry_pos)
        history_rewards_ben_plot.append(ep_ben_rew / steps_per_episode)
        history_rewards_jerry_plot.append(ep_jerry_rew / steps_per_episode)

        if ep_ben_rew > ep_jerry_rew: ben_wins += 1
        elif ep_jerry_rew > ep_ben_rew: jerry_wins += 1

    # 4. RÉSULTATS & VISUALISATION
    print("-" * 50)
    print(f"RÉSULTATS ({game_episodes} parties) :")
    print(f"Ben Wins: {ben_wins} | Jerry Wins: {jerry_wins}")
    print("-" * 50)

    if visualize:
        visualizer = HotellingVisualizer(env)
        # Calcul des cumulés pour le graphique
        ben_cum = np.cumsum(history_rewards_ben_plot)
        jerry_cum = np.cumsum(history_rewards_jerry_plot)
        
        visualizer.plot_training_summary(
            ben_rewards=history_rewards_ben_plot,
            jerry_rewards=history_rewards_jerry_plot,
            ben_positions=history_ben_pos_plot,
            jerry_positions=history_jerry_pos_plot,
            movement_stats=movement_stats,
            ben_relocations=[], # Non utilisé ici
            jerry_relocations=[], 
            period_length=1,
            ben_cumulative=ben_cum,
            jerry_cumulative=jerry_cum,
            ben_name=ben.loyalty_strategy.get_name(),
            jerry_name=jerry.loyalty_strategy.get_name(),
            game_weight_history=all_game_weights # Passage des données Heatmap
        )

    return env, ben, jerry


# ============================================================================
# EXAMPLE USAGE (MAIN)
# ============================================================================


def main():
    # Stratégies de loyauté (identiques pour la comparaison)
    common_loyalty = HighBrandLoyalty(base_rate=0.5, tenure_bonus=0.05) 
    
    print("\n" + "=" * 70)
    print("COMPARATIVE STUDY : FLUID vs RIGID MARKET")
    print("Training: 5000 episodes | Actual Game: 30 episodes")
    print("=" * 70)

    # --- SCENARIO 1 : MARCHÉ FLUIDE ---
    print("\n>>> SCENARIO 1: MARCHÉ FLUIDE (Low Cost)")
    env1, ben1, jerry1 = train_extended_hotelling(
        train_episodes=5000,     # <--- C'est ICI que ça compte : 5000
        game_episodes=30,        # <--- Et là : 30
        steps_per_episode=20,
        n_positions=11,
        ben_loyalty_strategy=common_loyalty,
        jerry_loyalty_strategy=common_loyalty,
        reward_mean=10.0,
        reward_std=2.0,
        cost_scaling=0.001,      # Coût faible
        demand_volatility=0.1,
        visualize=True,
    )

    # --- SCENARIO 2 : MARCHÉ RIGIDE ---
    print("\n>>> SCENARIO 2: MARCHÉ RIGIDE (High Cost)")
    env2, ben2, jerry2 = train_extended_hotelling(
        train_episodes=5000,     # <--- 5000 ici aussi
        game_episodes=30,        # <--- 30 ici aussi
        steps_per_episode=20,
        n_positions=11,
        ben_loyalty_strategy=common_loyalty,
        jerry_loyalty_strategy=common_loyalty,
        reward_mean=10.0,
        reward_std=2.0,
        cost_scaling=0.20,       # Coût élevé
        demand_volatility=0.1,
        visualize=True,
    )

if __name__ == "__main__":
    main()