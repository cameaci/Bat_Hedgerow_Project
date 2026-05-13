from __future__ import annotations

from importlib import import_module

from .exceptions import OptionalDependencyError


def require_module(module_name: str):
    try:
        return import_module(module_name)
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError(
            f"Optional dependency '{module_name}' is required for this operation."
        ) from exc


def require_geopandas():
    return require_module("geopandas")


def require_numpy():
    return require_module("numpy")


def require_rasterio():
    return require_module("rasterio")


def require_networkx():
    return require_module("networkx")


def require_astral():
    return require_module("astral")
