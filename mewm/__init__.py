"""MEWM-Agent: a multi-agent interactive emotional world model for micro-expression
understanding in long videos.

* **V** -- representation: deterministic motion quantisation, AU object slots,
  slow/fast/belief latent decomposition, evidence-token regulation.
* **M** -- rollout: emotion-conditioned AU dynamics, prediction-error spotting, the
  four service primitives, and the scheduling signal.
* **C** -- control: the perception / structure / reasoning / critic agents, driven by a
  deterministic orchestrator over a strictly acyclic evidence chain.
"""

__version__ = "1.0.0"

from .config import MEWMConfig, load_config  # noqa: F401

__all__ = ["MEWMConfig", "load_config", "__version__"]
