# The Story of Ben & Jerry's Fight for the Biggest Segment of the Beach's Ice Cream Market
## A Dynamic Hotelling Model with Relocation Costs and Consumer Loyalty

## Overview
This project studies spatial competition between two firms using a reinforcement learning (RL) framework.  
We build on the classical Hotelling (1929) model of location choice and extend it to a dynamic, discrete, and stochastic environment.

Unlike the classical model, firms:
- Operate on a discrete spatial grid
- Face explicit relocation costs
- Compete over time rather than in a one-shot game
- Learn optimal behavior through reinforcement learning

The goal is to analyze how relocation frictions and consumer loyalty alter long-run location outcomes compared to the classical Hotelling prediction of minimum differentiation.

---

## Research Questions
- How do relocation costs affect spatial differentiation?
- Does consumer loyalty stabilize firm locations away from the market center?
- Can reinforcement learning reproduce or deviate from classical Hotelling equilibria?
- Which loyalty structure leads to higher long-run market shares?

---

## Baseline Economic Model (Hotelling)
- Two firms compete on a line market
- Consumers are uniformly distributed
- Prices are fixed and identical
- Consumers choose the nearest firm (linear transport cost)
- Demand is perfectly inelastic (one unit per consumer)

Classical equilibrium: both firms locate at the center (minimum differentiation).

This project investigates how this result changes in a dynamic learning environment.

---

## Key Extensions
### 1. Discrete Market
- The market consists of 11 discrete locations
- Locations are mapped to the interval [0,1]
- Firms occupy one position at each time step

### 2. Dynamic Competition
- Time is discrete: (t = 0, 1, 2, ...)
- Firms interact repeatedly
- Firms maximize discounted expected profits

### 3. Beam Relocation
- Firms can relocate directly to any grid position in a single step
- Movement is not restricted to incremental left/right moves
- Relocation decisions are therefore risky and strategic

### 4. Relocation Costs
- Moving is costly
- Costs increase quadratically with distance moved
- This captures risk, adjustment frictions, and organizational costs

### 5. Consumer Loyalty
- A fraction of consumers follows a firm after relocation
- Loyalty increases with amount of time, that a firm stays at a certain position
- Different firms may have different loyalty profiles
- Loyalty structures are compared based on accumulated reward

---

## Reinforcement Learning Framework
- Each firm is modeled as an independent RL agent
- Agents learn simultaneously in a shared environment
- Rewards are based on realized profits net of relocation costs
- The learning problem is non-stationary due to strategic interaction

### Why Reinforcement Learning?
Classical equilibrium concepts assume:
- Perfect rationality
- Static optimization
- Full knowledge of the environment

These assumptions break down in a dynamic setting with relocation frictions.  
Reinforcement learning allows firms to:
- Learn optimal strategies through experience
- Adapt to opponent behavior
- Solve intertemporal trade-offs naturally

---

## Project Structure
```text
BJ/
├── README.md
│
├── Hotelling_Final/
│   ├── hotelling_env.py          # Gymnasium environment (market dynamics, rewards, loyalty)
│   ├── hotelling_agent.py        # Q-learning agent implementation
│   ├── hotelling_train.py        # Training loop (fluid vs. rigid markets)
│   ├── hotelling_test.py         # Evaluation / testing of trained agents
│   ├── hotelling_viz.py          # Visualization utilities (positions, rewards, crowd density)
│   │
│   ├── fluid_market_ben.npy      # Trained Q-table (Ben, low relocation cost)
│   ├── fluid_market_jerry.npy   # Trained Q-table (Jerry, low relocation cost)
│   ├── rigid_market_ben.npy      # Trained Q-table (Ben, high relocation cost)
│   ├── rigid_market_jerry.npy   # Trained Q-table (Jerry, high relocation cost)
│   │
│   └── script.py                 # Auxiliary execution / debugging script
│
├── hotelling_env.py              # Development version of environment (root)
├── hotelling_agent.py            # Development version of agent
├── hotelling_train.py            # Development training script
├── hotelling_test.py             # Development testing script
├── hotelling_viz.py              # Development visualization script
│
├── readings/                     # Academic references
│   ├── Hotelling_1929_StabilityInCompetition.pdf
│   ├── BiscaiaMota_2012_ModelsOfSpatialCompetition.pdf
│   └── BalversSzerb_1996_LocationInHotelling.pdf
│
├── __pycache__/                  # Python cache files
└── script_corrected.py           # Legacy / corrected experimental script
```
---

## Output and Analysis
The project analyzes:
- Long-run firm locations
- Market share dynamics
- Oscillation versus convergence behavior
- Sensitivity to relocation costs
- Effectiveness of different loyalty structures

Visual output is generated during the testing phase and used directly in the accompanying report.

---

## Status
- Economic model: complete
- Reinforcement learning environment: implemented
- Training and testing pipeline: functional
- Analysis: completed
- Report: finalized

---

## Authors
- PIERRON, Marie
- BACHER, Quentin
- PRATA LEAL, Paul
- SKOVGAARD, Anémone
