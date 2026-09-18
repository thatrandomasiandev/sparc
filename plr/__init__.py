"""Preference-based RL research package: decision-relevant elicitation.

See PROJECT.md for scientific context and AGENTS.md for invariants.
Paper algorithm: ``plr.algorithm.POPVOI``.
"""

from plr import acquisition, algorithm, encoder, likelihood, mdp, users

__all__ = ["acquisition", "algorithm", "encoder", "likelihood", "mdp", "users"]
__version__ = "0.1.0"
