"""
Extended Hotelling Location Model Environment
Implements a gym environment with customer loyalty and relocation costs
"""

from typing import Dict, Optional, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ExtendedHotellingEnv(gym.Env):
    """
    Extended Hotelling Model with:
    1. Crowd physics: Agents transport 20% of local density when moving
    2. Mutual exclusion: Cannot sell AND move in the same turn
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
        
        # Standard initialization
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
        """Convert discrete position to continuous beach location"""
        return action * (self.beach_length / (self.n_positions - 1))

    def _apply_tenure_growth(self, pos: int, growth_rate: float = 0.05):
        """Crowd grows if agent stays at position"""
        self.weights[pos] *= (1.0 + growth_rate)
        self.weights /= self.weights.sum()

    def _apply_loyalty_transport(self, old_pos: int, new_pos: int, transport_rate: float = 0.20):
        """Crowd moves with the agent (loyalty transport)"""
        if old_pos == new_pos:
            return
        
        # Take X% of density from old position
        transported_mass = self.weights[old_pos] * transport_rate
        
        # Move the mass
        self.weights[old_pos] -= transported_mass
        self.weights[new_pos] += transported_mass
        
        # Ensure non-negative and normalize
        self.weights = np.maximum(self.weights, 0)
        self.weights /= self.weights.sum()

    def _apply_demand_shock(self):
        """
        Applique une fluctuation autour de la distribution DE BASE.
        Cela garantit que le marché a une structure (ex: les gens préfèrent le centre)
        mais qu'il y a du bruit à chaque tour.
        """
        # On génère un bruit normal centré sur 0
        noise = np.random.normal(0, self.demand_volatility, self.n_positions)
        
        # On applique ce bruit aux poids DE BASE (et non aux poids précédents)
        # Cela ancre le marché à sa réalité structurelle
        self.weights = self.base_weights * (1 + noise)
        
        # Nettoyage (pas de poids négatifs + normalisation à 1)
        self.weights = np.maximum(self.weights, 0)
        self.weights /= self.weights.sum()

    # ---------------------------------------------------------
    # REMPLACE L'ANCIENNE MÉTHODE _calculate_rewards_strict PAR CELLE-CI
    # ---------------------------------------------------------
    def _calculate_rewards_strict(
        self, 
        ben_moved: bool, 
        jerry_moved: bool,
        ben_loyalty_rate: float,  # Nouveaux arguments
        jerry_loyalty_rate: float # Nouveaux arguments
    ) -> Tuple[float, float]:
        
        # --- 1. MISE À JOUR PHYSIQUE SIMULTANÉE (Correction Architecture) ---
        # On calcule les transferts sur une copie pour ne pas biaiser le second joueur
        weights_update = np.zeros_like(self.weights)
        current_weights = self.weights.copy() # Snapshot avant mouvement

        # Ben Transfert
        if ben_moved and self.prev_ben_pos != self.ben_pos:
            mass = current_weights[self.prev_ben_pos] * ben_loyalty_rate
            weights_update[self.prev_ben_pos] -= mass
            weights_update[self.ben_pos] += mass

        # Jerry Transfert
        if jerry_moved and self.prev_jerry_pos != self.jerry_pos:
            mass = current_weights[self.prev_jerry_pos] * jerry_loyalty_rate
            weights_update[self.prev_jerry_pos] -= mass
            weights_update[self.jerry_pos] += mass

        # Application globale
        self.weights += weights_update
        self.weights = np.maximum(self.weights, 0)
        self.weights /= self.weights.sum() # Normalisation propre

        # --- 2. CALCUL DES COÛTS ---
        ben_dist = abs(self._get_position_value(self.prev_ben_pos) - self._get_position_value(self.ben_pos))
        jerry_dist = abs(self._get_position_value(self.prev_jerry_pos) - self._get_position_value(self.jerry_pos))
        
        ben_cost = self.cost_scaling * (ben_dist ** 2)
        jerry_cost = self.cost_scaling * (jerry_dist ** 2)

        # --- 3. CALCUL DES PARTS DE MARCHÉ ---
        ben_share = 0.0
        jerry_share = 0.0
        
        if not ben_moved or not jerry_moved:
            positions = np.linspace(0, self.beach_length, self.n_positions)
            ben_loc = self._get_position_value(self.ben_pos)
            jerry_loc = self._get_position_value(self.jerry_pos)
            
            dist_ben = np.abs(positions - ben_loc)
            dist_jerry = np.abs(positions - jerry_loc)
            
            ben_wins = dist_ben < dist_jerry
            jerry_wins = dist_jerry < dist_ben
            ties = dist_ben == dist_jerry
            
            ben_share = np.sum(self.weights[ben_wins]) + 0.5 * np.sum(self.weights[ties])
            jerry_share = np.sum(self.weights[jerry_wins]) + 0.5 * np.sum(self.weights[ties])

        # --- 4. REWARD SCALING ---
        reward_multiplier = max(1.0, np.random.normal(self.reward_mean, self.reward_std))
        
        if ben_moved:
            r_ben = -ben_cost 
        else:
            r_ben = (ben_share * reward_multiplier)

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
        
        # --- NOTE IMPORTANTE : ---
        # On a retiré les appels à "self._apply_loyalty_transport" ici
        # car c'est maintenant géré à l'intérieur de _calculate_rewards_strict
        
        # Gestion du temps passé (Tenure)
        if ben_moved:
            self.ben_time_at_position = 0
        else:
            self.ben_time_at_position += 1
            
        if jerry_moved:
            self.jerry_time_at_position = 0
        else:
            self.jerry_time_at_position += 1

        # Volatilité du marché
        self._apply_demand_shock()

        # Calcul des récompenses ET application de la physique
        # C'est ici qu'on passe les loyalty_rates !
        ben_reward, jerry_reward = self._calculate_rewards_strict(
            ben_moved, 
            jerry_moved, 
            ben_loyalty_rate, 
            jerry_loyalty_rate
        )
        
        # Historique
        self.weight_history.append(self.weights.copy())
        self.position_history.append((self.ben_pos, self.jerry_pos))
        self.step_count += 1

        state = np.array([self.ben_pos, self.jerry_pos], dtype=np.int32)
        info = {"ben_moved": ben_moved, "jerry_moved": jerry_moved}
        
        return state, (ben_reward, jerry_reward), False, False, info

    def reset(self, seed: Optional[int] = None, reset_type: str = "random") -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.step_count = 0
        
        # --- MODIFICATION : Retour à la structure de base ---
        # On repart de la distribution structurelle (ex: uniforme ou définie dans __init__)
        # Le bruit sera ajouté dès le premier step() via _apply_demand_shock
        self.weights = self.base_weights.copy()
        
        # On applique un premier choc pour ne pas commencer exactement pareil
        self._apply_demand_shock()
        
        self.weight_history = [self.weights.copy()]
        self.position_history = []
        self.ben_time_at_position = 0
        self.jerry_time_at_position = 0

        # Positionnement des agents (inchangé)
        if reset_type == "random":
            self.ben_pos = np.random.randint(0, self.n_positions)
            self.jerry_pos = np.random.randint(0, self.n_positions)
        elif reset_type == "extremes":
            self.ben_pos = np.random.choice([0, self.n_positions - 1])
            self.jerry_pos = 0 if self.ben_pos == self.n_positions - 1 else self.n_positions - 1

        self.prev_ben_pos = self.ben_pos
        self.prev_jerry_pos = self.jerry_pos
        
        return np.array([self.ben_pos, self.jerry_pos], dtype=np.int32), {}
