# Learnability and Actionability of the Catan Value Function Across Game Phases

Building on *Catan: Computational Assessment of Terrain and Nodes* (Kim et al. 2025), this project studies how the learnability of the Catan value function evolves across the phases of a game and compares linear vs. non-linear value function approximators in 2-player Settlers of Catan.

## Motivation

Settlers of Catan (2-player) is a turn-based strategy board game where players race to 15 victory points. Players must increase resource production to enable future actions and earn victory points, but face tradeoffs between scaling their setup and earning cheaper victory points in the short term. Each action also affects the opponent's board state. These dynamics mirror real-world decisions where agents weigh scalability, short-term benefits, and adversarial impact.

Kim et al. (2025) analyzed initial position strengths in 4-player Catan. We extend this value function exploration to ask:

1. Does the learnability of the Catan value function evolve over a game, and if so, by how much?
2. How do linear and non-linear value functions compare?

## Method

### Data Collection

We run roughly 15,000 two-player Catan games through [Catanatron](https://github.com/bcollazo/catanatron) (a Python game engine) using a mix of its built-in bots so the dataset covers a wide variety of game states. Each game produces ~100 snapshots, each labeled with who eventually won. We split 80/10/10 by game ID rather than by individual turn so that turns from the same game don't bleed across train and test.

A *game snapshot* is a numerical description of everything happening in the game at a given moment, capturing roughly 80 features (production counts, settlement counts, road length, etc.).

Instead of defining game phases by turn number, we bucket by the maximum victory points held by either player at that moment. Catan games vary a lot in length — turn 20 in a fast game looks nothing like turn 20 in a slow one. Bucketing by VP means "late game" actually corresponds to a state where someone is close to winning, which is what we care about.

### Model Training & Comparison

We train two models:

- **L2-regularized logistic regression** (linear baseline)
- **Gradient-boosted trees** (non-linear)

For each, we train both per-bucket models and a single unified model on all data. Comparing them tells us whether late-game states are genuinely more predictable or whether per-bucket models are just overfitting to phase-specific patterns.

We also run PCA and k-means on the raw state vectors to visualize how game states shift across phases and measure how decisive each natural cluster is via win rates.

For online evaluation, each trained model is wrapped in a greedy one-step lookahead agent that scores every legal move's resulting state and picks the best one.

## Experiments

**Offline.** For each estimator and VP bucket, we report accuracy, loss, and expected calibration error with bootstrap confidence intervals.

**Online.** We pit each estimator-driven agent against a random baseline and Catanatron's built-in heuristic bots.

**Interpretability.** We extract feature importance per VP bucket to identify which game-state features drive predictions across different game stages.

## Limitations

- **2-player Catan** removes the trading and negotiation dynamics of the standard 4-player game, so findings may not generalize.
- Games are simulated with Catanatron bots, not human play.
- A greedy one-step lookahead agent is a simplified strategy; real players plan further ahead.
- Logistic regression assumes independent observations and low multicollinearity, neither of which strictly holds for snapshots drawn from the same game.

## Team
George Weiksner, Anaya Shintre, and Derek Jain

## References

- Kim et al. (2025, CS229). *Catan: Computational Assessment of Terrain and Nodes.*
- Collazo (2021). Catanatron.
- Pfeiffer (2004).
- Gendre & Kaneko (2020).
- Silver et al. (2012, 2017).
- Baier & Kaisers (2021, IJCAI).
- Browne et al. (2012).
