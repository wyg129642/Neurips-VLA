"""HDF5 / RLDS / manifest exporters for the generated expert demos."""

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
    blob = {}
    for i, d in enumerate(demos):
        blob[f"demo_{i}/actions"] = np.asarray(d.actions, np.float32)
        blob[f"demo_{i}/states"] = np.asarray(d.states, np.float32)
        blob[f"demo_{i}/success"] = np.array([d.success])
        blob[f"demo_{i}/seed"] = np.array([d.seed])
    np.savez_compressed(path, **blob)


def write_task_dataset(out_dir: str | Path, task_name: str, demos) -> dict:
    """Write ``{task_name}_demo.hdf5`` (or npz fallback) for one task."""
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
    """RLDS / TFDS-style JSON-lines episodes (one step per line)."""
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    man = {
        "spec": "500 success-filtered demos/task across the 50-task suite",
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
