#The Story of Ben & Jerry's Fight for the Biggest Segment of the Beach's Ice Cream Market
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

### 3. Stochastic Relocation
- Firms can relocate to new positions probabilistically
- Moves are not deterministic “left/right” steps
- Relocation introduces uncertainty and risk

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
(To be updated once implementation is finalized)

---

## Output and Analysis
The project analyzes:
- Long-run firm locations
- Market share dynamics
- Convergence vs. cycling behavior
- Sensitivity to relocation costs
- Effectiveness of different loyalty programs

---

## Status
- Economic model: defined
- Extensions: specified
- RL framework: in progress
- Code: under active development
- Report: being written in parallel

---

## Authors
- PIERRON, Marie
- BACHER, Quentin
- PRATA LEAL, Paul
- SKOVGAARD, Anémone