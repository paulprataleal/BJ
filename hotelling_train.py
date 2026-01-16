"""
Training script for Hotelling Location Model
Trains Q-Learning agents and saves them for later testing
"""

from typing import Optional

import numpy as np

from hotelling_agent import (
    HighBrandLoyalty,
    LoyaltyStrategy,
    ModerateBrandLoyalty,
    QLearningAgent,
)
from hotelling_env import ExtendedHotellingEnv
from hotelling_viz import HotellingVisualizer


def train_agents(
    train_episodes: int = 5000,
    game_episodes: int = 30,
    steps_per_episode: int = 20,
    n_positions: int = 11,
    ben_loyalty_strategy: Optional[LoyaltyStrategy] = None,
    jerry_loyalty_strategy: Optional[LoyaltyStrategy] = None,
    reward_mean: float = 10.0,
    reward_std: float = 2.0,
    cost_scaling: float = 0.01,
    demand_volatility: float = 0.1,
    visualize: bool = True,
    save_agents: bool = True,
    save_prefix: str = "agent",
):
    """
    Train two Q-Learning agents in Hotelling competition

    Args:
        train_episodes: Number of training episodes
        game_episodes: Number of validation episodes after training
        steps_per_episode: Steps per episode
        n_positions: Number of discrete positions on beach
        ben_loyalty_strategy: Loyalty strategy for Ben
        jerry_loyalty_strategy: Loyalty strategy for Jerry
        reward_mean: Mean of reward multiplier
        reward_std: Std dev of reward multiplier
        cost_scaling: Scaling factor for movement costs
        demand_volatility: Market volatility parameter
        visualize: Whether to show plots after training
        save_agents: Whether to save trained agents
        save_prefix: Prefix for saved agent files

    Returns:
        Tuple of (environment, ben_agent, jerry_agent)
    """

    # 1. SETUP
    if ben_loyalty_strategy is None:
        ben_loyalty_strategy = HighBrandLoyalty(base_rate=0.5, tenure_bonus=0.05)
    if jerry_loyalty_strategy is None:
        jerry_loyalty_strategy = ModerateBrandLoyalty(base_rate=0.3, tenure_bonus=0.02)

    env = ExtendedHotellingEnv(
        n_positions=n_positions,
        initial_weights=None,  # Stochastic managed by reset
        cost_scaling=cost_scaling,
        reward_mean=reward_mean,
        reward_std=reward_std,
        demand_volatility=demand_volatility,
    )

    ben = QLearningAgent("Ben", n_positions, ben_loyalty_strategy)
    jerry = QLearningAgent("Jerry", n_positions, jerry_loyalty_strategy)

    # 2. TRAINING PHASE
    print(f"\n>>> STARTING TRAINING ({train_episodes} episodes)...")

    for e in range(train_episodes):
        state, _ = env.reset()
        for step in range(steps_per_episode):
            ben_action = ben.act(state[0], state[1])
            jerry_action = jerry.act(state[1], state[0])

            # Loyalty = Transport rate
            ben_loyalty = ben.get_loyalty_rate(
                env.ben_time_at_position, env.weights[state[0]]
            )
            jerry_loyalty = jerry.get_loyalty_rate(
                env.jerry_time_at_position, env.weights[state[1]]
            )

            next_state, (ben_reward, jerry_reward), _, _, _ = env.step(
                (ben_action, jerry_action), ben_loyalty, jerry_loyalty
            )

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

        ben.decay_epsilon()
        jerry.decay_epsilon()
        if (e + 1) % 1000 == 0:
            print(
                f"  Training Progress: {e + 1}/{train_episodes} (Epsilon: {ben.epsilon:.3f})"
            )

    # 3. VALIDATION PHASE
    print(f"\n>>> STARTING VALIDATION ({game_episodes} episodes)...")
    ben.epsilon = 0
    jerry.epsilon = 0

    # Data for visualization
    history_ben_pos_plot = []
    history_jerry_pos_plot = []
    history_rewards_ben_plot = []
    history_rewards_jerry_plot = []
    all_game_weights = []

    movement_stats = {"both_moved": 0, "ben_only": 0, "jerry_only": 0, "neither": 0}
    ben_wins = 0
    jerry_wins = 0
    game_ben_rewards = []
    game_jerry_rewards = []

    for e in range(game_episodes):
        state, _ = env.reset()
        all_game_weights.append(env.weights.copy())

        ep_ben_rew, ep_jerry_rew = 0, 0
        ep_ben_pos = [env._get_position_value(env.ben_pos)]
        ep_jerry_pos = [env._get_position_value(env.jerry_pos)]

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

            # Stats
            if info["ben_moved"] and info["jerry_moved"]:
                movement_stats["both_moved"] += 1
            elif info["ben_moved"]:
                movement_stats["ben_only"] += 1
            elif info["jerry_moved"]:
                movement_stats["jerry_only"] += 1
            else:
                movement_stats["neither"] += 1

            ep_ben_rew += ben_reward
            ep_jerry_rew += jerry_reward
            state = next_state

            ep_ben_pos.append(env._get_position_value(env.ben_pos))
            ep_jerry_pos.append(env._get_position_value(env.jerry_pos))

            all_game_weights.append(env.weights.copy())

        game_ben_rewards.append(ep_ben_rew)
        game_jerry_rewards.append(ep_jerry_rew)

        history_ben_pos_plot.extend(ep_ben_pos)
        history_jerry_pos_plot.extend(ep_jerry_pos)
        history_rewards_ben_plot.append(ep_ben_rew / steps_per_episode)
        history_rewards_jerry_plot.append(ep_jerry_rew / steps_per_episode)

        if ep_ben_rew > ep_jerry_rew:
            ben_wins += 1
        elif ep_jerry_rew > ep_ben_rew:
            jerry_wins += 1

    # 4. RESULTS
    print("-" * 50)
    print(f"RESULTS ({game_episodes} games):")
    print(f"Ben Wins: {ben_wins} | Jerry Wins: {jerry_wins}")
    print(f"Total Ben Reward: {sum(game_ben_rewards):.2f}")
    print(f"Total Jerry Reward: {sum(game_jerry_rewards):.2f}")
    print("-" * 50)

    # 5. SAVE AGENTS
    if save_agents:
        ben.save(f"{save_prefix}_ben.npy")
        jerry.save(f"{save_prefix}_jerry.npy")

    # 6. VISUALIZATION
    if visualize:
        visualizer = HotellingVisualizer(env)
        ben_cum = np.cumsum(history_rewards_ben_plot)
        jerry_cum = np.cumsum(history_rewards_jerry_plot)

        visualizer.plot_training_summary(
            ben_rewards=history_rewards_ben_plot,
            jerry_rewards=history_rewards_jerry_plot,
            ben_positions=history_ben_pos_plot,
            jerry_positions=history_jerry_pos_plot,
            movement_stats=movement_stats,
            ben_relocations=[],
            jerry_relocations=[],
            period_length=1,
            ben_cumulative=ben_cum,
            jerry_cumulative=jerry_cum,
            ben_name=ben.loyalty_strategy.get_name(),
            jerry_name=jerry.loyalty_strategy.get_name(),
            game_weight_history=all_game_weights,
        )

    return env, ben, jerry


if __name__ == "__main__":
    # Example: Train agents with different cost scalings

    print("\n" + "=" * 70)
    print("SCENARIO 1: LOW MOVEMENT COST (Fluid Market)")
    print("=" * 70)

    env1, ben1, jerry1 = train_agents(
        train_episodes=5000,
        game_episodes=30,
        steps_per_episode=20,
        cost_scaling=0.001,
        save_prefix="fluid_market",
        visualize=True,
    )

    print("\n" + "=" * 70)
    print("SCENARIO 2: HIGH MOVEMENT COST (Rigid Market)")
    print("=" * 70)

    env2, ben2, jerry2 = train_agents(
        train_episodes=5000,
        game_episodes=30,
        steps_per_episode=20,
        cost_scaling=0.20,
        save_prefix="rigid_market",
        visualize=True,
    )
