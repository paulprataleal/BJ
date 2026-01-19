"""
Testing script for Hotelling Location Model
Loads trained agents and evaluates their performance
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


def test_agents(
    ben_filepath: str,
    jerry_filepath: str,
    test_episodes: int = 50,
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
    """
    Test trained Q-Learning agents

    Args:
        ben_filepath: Path to Ben's saved Q-table
        jerry_filepath: Path to Jerry's saved Q-table
        test_episodes: Number of test episodes
        steps_per_episode: Steps per episode
        n_positions: Number of discrete positions
        ben_loyalty_strategy: Loyalty strategy for Ben
        jerry_loyalty_strategy: Loyalty strategy for Jerry
        reward_mean: Mean of reward multiplier
        reward_std: Std dev of reward multiplier
        cost_scaling: Movement cost scaling
        demand_volatility: Market volatility
        visualize: Whether to show plots

    Returns:
        Dictionary with test results
    """

    # 1. SETUP
    if ben_loyalty_strategy is None:
        ben_loyalty_strategy = HighBrandLoyalty(base_rate=0.5, tenure_bonus=0.05)
    if jerry_loyalty_strategy is None:
        jerry_loyalty_strategy = ModerateBrandLoyalty(base_rate=0.3, tenure_bonus=0.02)

    env = ExtendedHotellingEnv(
        n_positions=n_positions,
        cost_scaling=cost_scaling,
        reward_mean=reward_mean,
        reward_std=reward_std,
        demand_volatility=demand_volatility,
    )

    # 2. LOAD AGENTS
    ben = QLearningAgent("Ben", n_positions, ben_loyalty_strategy)
    jerry = QLearningAgent("Jerry", n_positions, jerry_loyalty_strategy)

    ben.load(ben_filepath)
    jerry.load(jerry_filepath)

    # Set to greedy mode (no exploration)
    ben.epsilon = 0
    jerry.epsilon = 0

    # 3. TESTING PHASE
    print(f"\n>>> TESTING AGENTS ({test_episodes} episodes)...")

    history_ben_pos_plot = []
    history_jerry_pos_plot = []
    history_rewards_ben_plot = []
    history_rewards_jerry_plot = []
    all_game_weights = []

    movement_stats = {"both_moved": 0, "ben_only": 0, "jerry_only": 0, "neither": 0}
    ben_wins = 0
    jerry_wins = 0
    draws = 0
    game_ben_rewards = []
    game_jerry_rewards = []

    for e in range(test_episodes):
        state, _ = env.reset()
        all_game_weights.append(env.weights.copy())

        ep_ben_rew, ep_jerry_rew = 0, 0
        ep_ben_pos = [env._get_position_value(env.ben_pos)]
        ep_jerry_pos = [env._get_position_value(env.jerry_pos)]

        for step in range(steps_per_episode):
            # Greedy action selection
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

            # Track movement stats
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

        # Count wins
        if ep_ben_rew > ep_jerry_rew:
            ben_wins += 1
        elif ep_jerry_rew > ep_ben_rew:
            jerry_wins += 1
        else:
            draws += 1

    # 4. RESULTS
    total_ben = sum(game_ben_rewards)
    total_jerry = sum(game_jerry_rewards)
    avg_ben = total_ben / test_episodes
    avg_jerry = total_jerry / test_episodes

    print("\n" + "=" * 60)
    print(f"TEST RESULTS ({test_episodes} episodes):")
    print("=" * 60)
    print(f"Ben Wins:   {ben_wins:3d} ({ben_wins / test_episodes * 100:5.1f}%)")
    print(f"Jerry Wins: {jerry_wins:3d} ({jerry_wins / test_episodes * 100:5.1f}%)")
    print(f"Draws:      {draws:3d} ({draws / test_episodes * 100:5.1f}%)")
    print("-" * 60)
    print(f"Total Ben Reward:   {total_ben:8.2f} (Avg: {avg_ben:6.2f})")
    print(f"Total Jerry Reward: {total_jerry:8.2f} (Avg: {avg_jerry:6.2f})")
    print("=" * 60)

    # 5. VISUALIZATION
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

    # 6. RETURN RESULTS DICTIONARY
    results = {
        "ben_wins": ben_wins,
        "jerry_wins": jerry_wins,
        "draws": draws,
        "total_ben_reward": total_ben,
        "total_jerry_reward": total_jerry,
        "avg_ben_reward": avg_ben,
        "avg_jerry_reward": avg_jerry,
        "game_ben_rewards": game_ben_rewards,
        "game_jerry_rewards": game_jerry_rewards,
        "movement_stats": movement_stats,
    }

    return results


def compare_scenarios(
    scenario1_ben: str,
    scenario1_jerry: str,
    scenario2_ben: str,
    scenario2_jerry: str,
    cost_scaling1: float = 0.001,
    cost_scaling2: float = 0.20,
    test_episodes: int = 50,
):
    """Compare two different trained scenarios"""

    print("\n" + "=" * 70)
    print("COMPARING TWO SCENARIOS")
    print("=" * 70)

    print("\n>>> SCENARIO 1: LOW COST (Fluid Market)")
    results1 = test_agents(
        scenario1_ben,
        scenario1_jerry,
        test_episodes=test_episodes,
        cost_scaling=cost_scaling1,
        visualize=True,
    )

    print("\n>>> SCENARIO 2: HIGH COST (Rigid Market)")
    results2 = test_agents(
        scenario2_ben,
        scenario2_jerry,
        test_episodes=test_episodes,
        cost_scaling=cost_scaling2,
        visualize=True,
    )

    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'Fluid Market':>15} {'Rigid Market':>15}")
    print("-" * 70)
    print(
        f"{'Ben Win Rate':<30} {results1['ben_wins'] / test_episodes * 100:>14.1f}% {results2['ben_wins'] / test_episodes * 100:>14.1f}%"
    )
    print(
        f"{'Jerry Win Rate':<30} {results1['jerry_wins'] / test_episodes * 100:>14.1f}% {results2['jerry_wins'] / test_episodes * 100:>14.1f}%"
    )
    print(
        f"{'Avg Ben Reward':<30} {results1['avg_ben_reward']:>15.2f} {results2['avg_ben_reward']:>15.2f}"
    )
    print(
        f"{'Avg Jerry Reward':<30} {results1['avg_jerry_reward']:>15.2f} {results2['avg_jerry_reward']:>15.2f}"
    )
    print("=" * 70)

    return results1, results2


if __name__ == "__main__":
    # Example 1: Test a single scenario
    print("\nTesting single scenario...")
    results = test_agents(
        ben_filepath="fluid_market_ben.npy",
        jerry_filepath="fluid_market_jerry.npy",
        test_episodes=50,
        cost_scaling=0.001,
        visualize=True,
    )

    # Example 2: Compare two scenarios
    # Uncomment to run comparison
    # compare_scenarios(
    #     scenario1_ben="fluid_market_ben.npy",
    #     scenario1_jerry="fluid_market_jerry.npy",
    #     scenario2_ben="rigid_market_ben.npy",
    #     scenario2_jerry="rigid_market_jerry.npy",
    #     test_episodes=50,
    # )
