"""Thin adapter over the ``relval`` package.

The read-only API needs only the variable registry (to label SHAP contributions with their
category). The later use cases (valuing private companies, what-if scenarios) will load trained
models through this module, so nothing else in the backend imports ``relval`` directly.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache
def _registry() -> dict[str, str]:
    try:
        from relval.features.registry import CATEGORY_OF

        return dict(CATEGORY_OF)
    except ImportError:  # relval not installed in a slim image: categories are optional metadata
        return {}


def category_of(variable: str) -> str | None:
    return _registry().get(variable)
