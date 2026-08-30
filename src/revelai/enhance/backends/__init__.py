"""Pluggable restoration backends.

``local`` is the default and needs no account anywhere. Hosted backends exist
for machines that cannot run the models, and they upload the photographs, so
they are opt-in and the CLI says where the images are going before it sends
them.
"""

from __future__ import annotations

from revelai import RevelAIError
from revelai.enhance.backends.base import (
    MODEL_OPERATIONS,
    OPERATIONS,
    BackendUnavailable,
    EnhancerBackend,
    Operation,
)

__all__ = [
    "OPERATIONS",
    "MODEL_OPERATIONS",
    "BackendUnavailable",
    "EnhancerBackend",
    "Operation",
    "BACKENDS",
    "get_backend",
]

#: Backends the CLI will accept for ``--backend``.
BACKENDS = ("local", "replicate")


def get_backend(name: str, **kwargs) -> EnhancerBackend:
    """Build a backend by name."""
    key = (name or "local").strip().lower()
    if key == "local":
        from revelai.enhance.backends.local import LocalBackend

        return LocalBackend(**kwargs)
    if key == "replicate":
        from revelai.enhance.backends.hosted_replicate import ReplicateBackend

        return ReplicateBackend(**kwargs)
    raise RevelAIError(f"unknown backend {name!r}; choose one of {', '.join(BACKENDS)}")
