# Problem statement

> Locked from PROJECT.md §1 — authoritative one-paragraph claim.

## One-sentence problem

Preference-based RL selects queries to reduce uncertainty about the reward parameters,
but much of that uncertainty never changes the optimal policy — so we waste limited human
feedback resolving decision-irrelevant disagreement.

## Contribution (novelty boundary)

**Not novel:** regret-based / decision-relevant elicitation (Regan & Boutilier, UAI 2009).

**Ours:** combine that criterion with (a) human trajectory-segment comparisons, (b) scalable
computation via a **learned population prior** over previous users (and later continuous
control). Cite Regan & Boutilier and Model-Free Preference Elicitation (IJCAI 2024) in ¶1.

## Falsifiable success criterion (simulation)

On the gridworld protocol in experiment 3, with ≥3 seeds and paired bootstrap:

1. **Primary (recommended):** VOI final-query regret lower than BALD with 95% paired CI
   excluding zero.
2. **Secondary:** VOI queries-to-threshold ≤ BALD (report tie rate + censoring rate).
3. **Ceiling:** keep the oracle; report the oracle gap honestly.

## Explicitly out of scope (for now)

- Deep nonlinear reward networks
- Claiming novelty for the decision-relevant *concept*
- Single-seed claims
- Retracted slogans (“BALD worse than random”; “factored latent beats VPL”)
