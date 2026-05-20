"""Expert-dataset export: LIBERO HDF5 schema plus RLDS and a manifest.

Writes the HDF5 layout the OpenVLA-OFT ``ExpertDataLoader`` reads:

    f["data"]["demo_i"]["actions"]   (T, 7)
    f["data"]["demo_i"]["states"]    (T+1, S)   # states[0] == init_state

so a generated dataset is a drop-in replacement for the LIBERO expert demos
in the OpenVLA-OFT runner or the unified runner with ``backend="libero"``.
Also emits a TFDS/RLDS-style sharded JSON-lines export (for VLA
fine-tuning frameworks that ingest RLDS) and a dataset manifest with the
per-domain balance.

``h5py`` is an optional dependency. When it is unavailable, a portable
``.npz`` fallback with the same logical schema is written so the pipeline
remains runnable and testable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

def _write_hdf5(path: Path, demos) -> bool:
    try:
        import h5py
    except ImportError:
        return False
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        for i, d in enumerate(demos):
            g = grp.create_group(f"demo_{i}")
            g.create_dataset("actions", data=np.asarray(d.actions, np.float32))
            g.create_dataset("states", data=np.asarray(d.states, np.float32))
            g.attrs["success"] = bool(d.success)
            g.attrs["seed"] = int(d.seed)
            g.attrs["task"] = d.description
        grp.attrs["task_name"] = demos[0].task_name if demos else ""
    return True

def _write_npz(path: Path, demos) -> None:
    """Portable fallback with the same logical schema as the HDF5 layout."""
    blob = {}
    for i, d in enumerate(demos):
        blob[f"demo_{i}/actions"] = np.asarray(d.actions, np.float32)
        blob[f"demo_{i}/states"] = np.asarray(d.states, np.float32)
        blob[f"demo_{i}/success"] = np.array([d.success])
        blob[f"demo_{i}/seed"] = np.array([d.seed])
    np.savez_compressed(path, **blob)

def write_task_dataset(out_dir: str | Path, task_name: str, demos) -> dict:
    """Write one task's demos as ``{task_name}_demo.hdf5`` (the standard naming)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5 = out_dir / f"{task_name}_demo.hdf5"
    if _write_hdf5(h5, demos):
        fmt, written = "hdf5", str(h5)
    else:
        npz = out_dir / f"{task_name}_demo.npz"
        _write_npz(npz, demos)
        fmt, written = "npz", str(npz)
    return {"task_name": task_name, "format": fmt, "path": written,
            "num_demos": len(demos)}

def write_rlds_shard(out_dir: str | Path, task_name: str, demos,
                     shard_size: int = 64) -> list[str]:
    """RLDS/TFDS-style JSON-lines episodes (one step per line) for VLA
    fine-tuning frameworks (OpenVLA RLDS, openpi LeRobot ingestion)."""
    out_dir = Path(out_dir) / "rlds" / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(0, len(demos), shard_size):
        shard = demos[s:s + shard_size]
        p = out_dir / f"shard_{s // shard_size:04d}.jsonl"
        with open(p, "w") as f:
            for d in shard:
                for t, act in enumerate(d.actions):
                    f.write(json.dumps({
                        "task": d.description,
                        "is_first": t == 0,
                        "is_last": t == len(d.actions) - 1,
                        "action": np.asarray(act, float).tolist(),
                        "state": np.asarray(
                            d.states[min(t, len(d.states) - 1)],
                            float).tolist(),
                    }) + "\n")
        paths.append(str(p))
    return paths

def write_manifest(out_dir: str | Path, report, cfg) -> Path:
    """Dataset manifest: totals, per-domain balance, quota shortfalls."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    man = {
        "spec": "paper §4.1: 500 successful demos/task, 25k balanced",
        "demos_per_task_target": cfg.demos_per_task,
        "total_demos": report.total_demos,
        "num_tasks": len(report.per_task),
        "per_category": report.per_category,
        "per_task": report.per_task,
        "shortfall": report.shortfall,
        "balanced": len(report.shortfall) == 0,
    }
    p = out_dir / "dataset_manifest.json"
    p.write_text(json.dumps(man, indent=2))
    return p
