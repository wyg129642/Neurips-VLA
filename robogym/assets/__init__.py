"""RoboGym reasoning-centric asset generation (paper §3.6).

text-to-3D / image-to-3D / AI-texture pipelines with a
deterministic procedural fallback (runnable here). Appearance is heavily
randomized; physical properties (mass / friction / mass-distribution) are
sampled from *controlled, logged* ranges and varied at test time.
"""

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
