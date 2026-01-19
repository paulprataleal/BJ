"""
Visualization utilities for Hotelling Location Model
Provides comprehensive plotting including distributions, heatmaps, and reward tracking
"""

import numpy as np
import matplotlib.pyplot as plt


class HotellingVisualizer:
    """
    Complete visualization including:
    - Final distribution
    - Heatmap (crowd density evolution)
    - Cumulative rewards (profitability)
    """

    def __init__(self, env):
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
        game_weight_history: list,
    ):
        """Create comprehensive training summary visualization"""
        
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(5, 3, hspace=0.5, wspace=0.3)

        # --- PLOT 1: Final Weight Distribution ---
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
        
        # Average positions over final period
        avg_ben = np.mean(ben_positions[-50:]) if len(ben_positions) > 50 else np.mean(ben_positions)
        avg_jerry = np.mean(jerry_positions[-50:]) if len(jerry_positions) > 50 else np.mean(jerry_positions)
        max_weight = final_weights.max() if len(final_weights) > 0 else 0.1

        ax1.plot(avg_ben, max_weight * 1.1, "bv", markersize=15, label=f"Ben (avg: {avg_ben:.1f})", zorder=5)
        ax1.plot(avg_jerry, max_weight * 1.1, "rv", markersize=15, label=f"Jerry (avg: {avg_jerry:.1f})", zorder=5)
        ax1.set_title("Final Market Weight Distribution", fontsize=13, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # --- PLOT 2: Position Evolution ---
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(ben_positions, "b-", linewidth=1.5, alpha=0.8, label="Ben")
        ax2.plot(jerry_positions, "r-", linewidth=1.5, alpha=0.8, label="Jerry")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Position")
        ax2.set_title("Position Evolution (Game Phase)", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- PLOT 3: Rewards per Step ---
        ax3 = fig.add_subplot(gs[1, 1])
        # Smoothing for readability
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

        # --- PLOT 4: Movement Stats ---
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

        # --- PLOT 5: Position Occupancy Histogram ---
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.hist(ben_positions, bins=self.n_positions, alpha=0.5, color='blue', label='Ben Locations')
        ax5.hist(jerry_positions, bins=self.n_positions, alpha=0.5, color='red', label='Jerry Locations')
        ax5.set_title("Position Occupancy Histogram", fontweight="bold")
        ax5.legend()

        # --- PLOT 6: Market State Snapshot ---
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_market_state(ax6, ben_positions[-1], jerry_positions[-1])

        # --- PLOT 7: Cumulative Rewards (Profitability) ---
        ax_cum = fig.add_subplot(gs[3, :])
        ax_cum.plot(ben_cumulative, "b-", linewidth=2.5, label=f"Ben: {ben_name}")
        ax_cum.plot(jerry_cumulative, "r-", linewidth=2.5, label=f"Jerry: {jerry_name}")
        ax_cum.set_xlabel("Time Step")
        ax_cum.set_ylabel("Total Profit")
        ax_cum.set_title("Total Profitability (Cumulative)", fontsize=13, fontweight="bold")
        ax_cum.legend(loc="upper left")
        ax_cum.grid(True, alpha=0.3)

        # --- PLOT 8: HEATMAP (Weight Evolution) ---
        ax7 = fig.add_subplot(gs[4, :])
        if len(game_weight_history) > 0:
            # Transpose to have time on X-axis and positions on Y-axis
            weight_array = np.array(game_weight_history).T 
            im = ax7.imshow(
                weight_array, 
                aspect="auto", 
                cmap="YlOrRd", 
                interpolation="nearest",
                origin="lower"  # Position 0 at bottom
            )
            ax7.set_xlabel("Time Step (Game Phase)", fontsize=11)
            ax7.set_ylabel("Position Index", fontsize=11)
            ax7.set_title("Market Crowd Density Evolution (Heatmap)", fontsize=13, fontweight="bold")
            plt.colorbar(im, ax=ax7, label="Customer Density")
        
        plt.tight_layout()
        plt.show()

    def _plot_market_state(self, ax, ben_pos: float, jerry_pos: float):
        """Plot final market snapshot"""
        positions = np.linspace(0, self.beach_length, self.n_positions)
        weights = self.env.weights
        ax.plot([0, self.beach_length], [0, 0], "k-", linewidth=2, alpha=0.5)
        for i, (pos, weight) in enumerate(zip(positions, weights)):
            ax.plot([pos, pos], [0, weight * 100], "lightblue", linewidth=8, alpha=0.7)
        ax.plot(ben_pos, 0, "bo", markersize=15, label="Ben", zorder=10)
        ax.plot(jerry_pos, 0, "ro", markersize=15, label="Jerry", zorder=10)
        ax.set_title("Final Market Snapshot", fontweight="bold")
        ax.legend()
