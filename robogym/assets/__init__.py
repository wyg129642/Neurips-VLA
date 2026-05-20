"""Reasoning-centric asset generation (Sec. 3.6)."""

from .generator import (
    AITextureBackend,
    ApiGeometryBackend,
    Asset,
    AssetGenerator,
    PhysicalProperties,
    ProceduralGeometryBackend,
    ProceduralTextureBackend,
    build_asset_library,
)

__all__ = [
    "AssetGenerator", "Asset", "PhysicalProperties",
    "ProceduralGeometryBackend", "ApiGeometryBackend",
    "ProceduralTextureBackend", "AITextureBackend",
    "build_asset_library",
]
