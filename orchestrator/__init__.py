"""Paper-style autonomous data iteration primitives."""

__all__ = ["default_policy", "validate_policy", "IterationController"]


def __getattr__(name):
    # Keep compatibility without importing the optional training loop at startup.
    if name in {"default_policy", "validate_policy"}:
        from .iteration import policy
        return getattr(policy, name)
    if name == "IterationController":
        from .iteration.controller import IterationController
        return IterationController
    raise AttributeError(name)
