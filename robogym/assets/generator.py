"""Reasoning-centric asset generation (paper §3.6).

Pluggable generation backends (text-to-3D, image-to-3D, AI texture) with a
deterministic procedural fallback so the asset library and appearance
randomization can be built without the heavy simulation stack. Each asset
gets a MuJoCo MJCF and an OBJ mesh (geometry), a procedural or AI texture
PNG (appearance), and physical properties (mass, friction, mass
distribution) sampled from logged, reproducible ranges.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Physical properties: appearance is randomised, physics is controlled.
@dataclass
class PhysicalProperties:
    mass: float
    friction: float
    com_offset: tuple                 # mass-distribution (center-of-mass)
    inertia_scale: float

    @staticmethod
    def sample(rng: np.random.Generator,
               mass_range=(0.05, 1.2),
               friction_range=(0.3, 1.2)) -> "PhysicalProperties":
        """Programmatically vary properties at test time from controlled
        ranges so Physical-Intuition tasks remain physically faithful."""
        return PhysicalProperties(
            mass=float(rng.uniform(*mass_range)),
            friction=float(rng.uniform(*friction_range)),
            com_offset=tuple(np.round(rng.uniform(-0.01, 0.01, 3), 4)),
            inertia_scale=float(rng.uniform(0.8, 1.2)),
        )

# Geometry backends.
class ProceduralGeometryBackend:
    """Deterministic parametric primitives (offline default, always works)."""

    name = "procedural"

    _PRIMS = {"box": (0.03, 0.03, 0.03), "cylinder": (0.025, 0.04, 0.0),
              "sphere": (0.025, 0.0, 0.0), "ring": (0.03, 0.012, 0.0)}

    def make_obj(self, kind: str, rng) -> str:
        """Emit a tiny OBJ (unit primitive scaled by the MJCF)."""
        if kind == "box":
            v = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                 (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
            f = [(1, 2, 3), (1, 3, 4), (5, 6, 7), (5, 7, 8),
                 (1, 2, 6), (1, 6, 5), (3, 4, 8), (3, 8, 7),
                 (2, 3, 7), (2, 7, 6), (1, 4, 8), (1, 8, 5)]
        else:  # cheap polyhedral approximation for cyl/sphere/ring
            k = 12
            ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
            v = [(np.cos(a), np.sin(a), -1) for a in ang] + \
                [(np.cos(a), np.sin(a), 1) for a in ang]
            f = [(i + 1, (i + 1) % k + 1, k + (i + 1) % k + 1)
                 for i in range(k)]
        s = "\n".join(f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in v)
        s += "\n" + "\n".join("f " + " ".join(map(str, t)) for t in f)
        return s + "\n"

class ApiGeometryBackend:
    """text-to-3D / image-to-3D via an external generator callable.

    Configure with an endpoint or generator callable; falls back to the
    procedural backend transparently so the library always builds.
    """

    def __init__(self, mode: str = "text_to_3d", generator=None):
        self.name = mode
        self._gen = generator
        self._fallback = ProceduralGeometryBackend()

    def make_obj(self, kind: str, rng) -> str:
        if self._gen is not None:
            try:  # pragma: no cover - needs the real model/service
                return self._gen(kind)
            except Exception:
                pass
        return self._fallback.make_obj(kind, rng)

# Texture backends.
class ProceduralTextureBackend:
    name = "procedural_texture"

    def make_png(self, path: Path, rng) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        res = 128
        base = rng.random(3)
        kind = rng.integers(3)
        if kind == 0:                                  # noise
            img = rng.random((res, res, 3)) * 0.4 + base * 0.6
        elif kind == 1:                                # stripes
            xs = (np.sin(np.linspace(0, rng.uniform(6, 24), res))[:, None]
                  > 0).astype(float)
            img = xs[..., None] * base + (1 - xs[..., None]) * (1 - base)
        else:                                          # checker
            g = (np.indices((res, res)).sum(0) //
                 int(rng.integers(6, 20))) % 2
            img = g[..., None] * base + (1 - g[..., None]) * base[::-1]
        fig = plt.figure(figsize=(1, 1), dpi=res)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.imshow(np.clip(img, 0, 1))
        fig.savefig(path, dpi=res)
        plt.close(fig)

class AITextureBackend(ProceduralTextureBackend):
    """AI-driven texture generation (diffusion). Falls back to procedural."""

    name = "ai_texture"

    def __init__(self, generator=None):
        self._gen = generator

    def make_png(self, path: Path, rng) -> None:
        if self._gen is not None:
            try:  # pragma: no cover - needs a real image model
                self._gen(str(path))
                if path.exists():
                    return
            except Exception:
                pass
        super().make_png(path, rng)

# Asset + library.
_MJCF = """<mujoco model="{name}">
  <asset>
    <texture name="{name}_tex" type="2d" file="{tex}"/>
    <material name="{name}_mat" texture="{name}_tex"/>
    <mesh name="{name}_mesh" file="{obj}" scale="{sx} {sy} {sz}"/>
  </asset>
  <worldbody>
    <body name="{name}">
      <freejoint/>
      <geom type="mesh" mesh="{name}_mesh" material="{name}_mat"
            mass="{mass}" friction="{fr} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""

@dataclass
class Asset:
    name: str
    kind: str
    obj_path: str
    tex_path: str
    mjcf_path: str
    physics: dict
    geometry_backend: str
    texture_backend: str

class AssetGenerator:
    def __init__(self, geometry=None, texture=None, seed: int = 0):
        self.geometry = geometry or ProceduralGeometryBackend()
        self.texture = texture or ProceduralTextureBackend()
        self.rng = np.random.default_rng(seed)

    def generate(self, name: str, kind: str, out_dir: str | Path,
                 controlled_physics: PhysicalProperties | None = None) -> Asset:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        obj_p = out / f"{name}.obj"
        tex_p = out / f"{name}.png"
        mjcf_p = out / f"{name}.xml"

        obj_p.write_text(self.geometry.make_obj(kind, self.rng))
        self.texture.make_png(tex_p, self.rng)        # appearance randomized
        phys = controlled_physics or PhysicalProperties.sample(self.rng)
        sx, sy, sz = ProceduralGeometryBackend._PRIMS.get(
            kind, (0.03, 0.03, 0.03))
        mjcf_p.write_text(_MJCF.format(
            name=name, tex=tex_p.name, obj=obj_p.name,
            sx=sx, sy=sy or sx, sz=sz or sx,
            mass=round(phys.mass, 4), fr=round(phys.friction, 4)))
        return Asset(name, kind, str(obj_p), str(tex_p), str(mjcf_p),
                     asdict(phys), self.geometry.name, self.texture.name)

# reasoning-centric asset library: one diverse asset set per task family
_FAMILY_KINDS = {
    "maze_navigation": ["sphere"], "seesaw_weight": ["box", "box"],
    "tangram_assembly": ["box", "cylinder", "box"],
    "number_block": ["box"], "color_hanoi": ["ring", "ring", "ring"],
    "sequential_counting": ["box", "box", "box"],
}

def build_asset_library(out_dir: str | Path, seed: int = 0,
                        episodes: int = 2) -> dict:
    """Build the reasoning-centric library: ``episodes`` visually randomised
    variants per family with controlled physics (paper §3.6).

    Returns a manifest of every generated asset.
    """
    out = Path(out_dir)
    gen = AssetGenerator(seed=seed)
    manifest = {"assets": [], "spec": "paper §3.6 reasoning-centric library "
                "(appearance randomised, physics controlled)"}
    for fam, kinds in _FAMILY_KINDS.items():
        for ep in range(episodes):
            # Physics varied per episode from a logged range; appearance
            # fully randomised via the texture backend.
            for ki, kind in enumerate(kinds):
                phys = PhysicalProperties.sample(gen.rng)
                a = gen.generate(f"{fam}_e{ep}_{ki}_{kind}", kind,
                                 out / fam, controlled_physics=phys)
                manifest["assets"].append(asdict(a))
    (out / "asset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
