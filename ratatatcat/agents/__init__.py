"""Playing agents, from trivial baselines to the ISMCTS solver."""

from .base import Agent, NeverKnockRandomAgent, RandomAgent
from .heuristic import HeuristicAgent, HeuristicParams
from .search import ISMCTSAgent, PIMCAgent

#: Name -> zero-argument constructor, for the CLI and the tournament runner.
REGISTRY = {
    "random": RandomAgent,
    "random-nocall": NeverKnockRandomAgent,
    "heuristic": HeuristicAgent,
    "pimc": PIMCAgent,
    "ismcts": ISMCTSAgent,
}

__all__ = [
    "Agent",
    "RandomAgent",
    "NeverKnockRandomAgent",
    "HeuristicAgent",
    "HeuristicParams",
    "PIMCAgent",
    "ISMCTSAgent",
    "REGISTRY",
]
