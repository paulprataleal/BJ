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
        """Apply random market fluctuations"""
        noise = np.random.normal(0, self.demand_volatility, self.n_positions)
        self.weights = self.weights * (1 + noise)
        self.weights = np.maximum(self.weights, 0)
        self.weights = self.weights / self.weights.sum()

    def _calculate_rewards_strict(self, ben_moved: bool, jerry_moved: bool) -> Tuple[float, float]:
        """Calculate rewards with mutual exclusion: sell OR move, not both"""
        
        # 1. Movement costs
        ben_dist = abs(self._get_position_value(self.prev_ben_pos) - self._get_position_value(self.ben_pos))
        jerry_dist = abs(self._get_position_value(self.prev_jerry_pos) - self._get_position_value(self.jerry_pos))
        
        ben_cost = self.cost_scaling * (ben_dist ** 2)
        jerry_cost = self.cost_scaling * (jerry_dist ** 2)

        # 2. Calculate market shares (ONLY if staying)
        ben_share = 0.0
        jerry_share = 0.0
        
        if not ben_moved or not jerry_moved:
            positions = np.linspace(0, self.beach_length, self.n_positions)
            ben_loc = self._get_position_value(self.ben_pos)
            jerry_loc = self._get_position_value(self.jerry_pos)
            
            # Distance from each beach segment to sellers
            dist_ben = np.abs(positions - ben_loc)
            dist_jerry = np.abs(positions - jerry_loc)
            
            # Who wins which segment?
            ben_wins = dist_ben < dist_jerry
            jerry_wins = dist_jerry < dist_ben
            ties = dist_ben == dist_jerry
            
            # Sum of weights won
            ben_share = np.sum(self.weights[ben_wins]) + 0.5 * np.sum(self.weights[ties])
            jerry_share = np.sum(self.weights[jerry_wins]) + 0.5 * np.sum(self.weights[ties])

        reward_multiplier = max(0.1, np.random.normal(self.reward_mean, self.reward_std))
        
        # 3. Attribution (Strict exclusion)
        # BEN
        if ben_moved:
            r_ben = -ben_cost  # Punitive: pure cost, no sales
        else:
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
        """Execute one environment step"""
        
        ben_action, jerry_action = actions
        ben_moved = ben_action != self.ben_pos
        jerry_moved = jerry_action != self.jerry_pos
        
        self.prev_ben_pos = self.ben_pos
        self.prev_jerry_pos = self.jerry_pos
        self.ben_pos = ben_action
        self.jerry_pos = jerry_action
        
        # --- PHYSICS ---
        
        # BEN
        if ben_moved:
            self._apply_loyalty_transport(self.prev_ben_pos, self.ben_pos, transport_rate=ben_loyalty_rate)
            self.ben_time_at_position = 0
        else:
            self.ben_time_at_position += 1
            
        # JERRY
        if jerry_moved:
            self._apply_loyalty_transport(self.prev_jerry_pos, self.jerry_pos, transport_rate=jerry_loyalty_rate)
            self.jerry_time_at_position = 0
        else:
            self.jerry_time_at_position += 1

        # Market volatility
        self._apply_demand_shock()

        # Calculate rewards (Sell OR Move)
        ben_reward, jerry_reward = self._calculate_rewards_strict(ben_moved, jerry_moved)
        
        # History tracking
        self.weight_history.append(self.weights.copy())
        self.position_history.append((self.ben_pos, self.jerry_pos))
        self.step_count += 1

        state = np.array([self.ben_pos, self.jerry_pos], dtype=np.int32)
        info = {"ben_moved": ben_moved, "jerry_moved": jerry_moved}
        
        return state, (ben_reward, jerry_reward), False, False, info

    def reset(self, seed: Optional[int] = None, reset_type: str = "random") -> Tuple[np.ndarray, Dict]:
        """Stochastic reset: Market changes each episode!"""
        super().reset(seed=seed)
        self.step_count = 0
        
        # Stochastic weights - generate random customer distribution each reset
        raw_weights = np.random.rand(self.n_positions)
        self.weights = raw_weights / raw_weights.sum()

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
