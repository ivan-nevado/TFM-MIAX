"""Configuration loading. One YAML file drives a run; experiment files extend the base file."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = "replication.yaml"


def find_root(start: Path | None = None) -> Path:
    """Repository root: ``RELVAL_ROOT`` if set, else the first parent containing ``ml/configs``."""
    env = os.environ.get("RELVAL_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(start or Path.cwd()).resolve()
    for cand in [here, *here.parents]:
        if (cand / "ml" / "configs").is_dir():
            return cand
    return here


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """Thin read-only wrapper around the YAML dictionary with path helpers."""

    def __init__(self, data: dict[str, Any], root: Path, source_file: Path | None = None):
        self.data = data
        self.root = root
        self.source_file = source_file

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # ---- paths -----------------------------------------------------------------
    def path(self, key: str) -> Path:
        return (self.root / self.data["paths"][key]).resolve()

    @property
    def run_name(self) -> str:
        return str(self.data["run_name"])

    def interim(self, name: str) -> Path:
        return self.path("interim") / self.data["source"] / name

    def panel_path(self) -> Path:
        return self.path("processed") / f"{self.data['panel_name']}_{self.data['source']}.parquet"

    def predictions_dir(self, model: str | None = None, seed: int | None = None) -> Path:
        base = self.path("predictions") / self.run_name
        if model is not None:
            base = base / f"model={model}"
        if seed is not None:
            base = base / f"seed={seed}"
        return base

    def artifacts_dir(self, kind: str, model: str | None = None, seed: int | None = None) -> Path:
        base = self.path("artifacts") / self.run_name / kind
        if model is not None:
            base = base / f"model={model}"
        if seed is not None:
            base = base / f"seed={seed}"
        return base

    def reports_dir(self, sub: str = "") -> Path:
        base = self.path("reports") / self.run_name
        return base / sub if sub else base

    # ---- common settings ---------------------------------------------------------
    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self.data["splits"]["seeds"]]

    @property
    def headline_seed(self) -> int:
        return self.seeds[0]

    @property
    def n_jobs(self) -> int:
        return int(os.environ.get("RELVAL_N_JOBS", self.data.get("n_jobs", -1)))

    def years(self) -> list[int]:
        start, end = self.data["period"]["start"], self.data["period"]["end"]
        return list(range(int(start[:4]), int(end[:4]) + 1))


def load_config(path: str | Path | None = None, root: Path | None = None) -> Config:
    """Load a config file (name inside ``ml/configs`` or a full path), resolving ``extends``."""
    root = (root or find_root()).resolve()
    cfg_dir = root / "ml" / "configs"
    file = Path(path) if path else cfg_dir / DEFAULT_CONFIG
    if not file.is_absolute() and not file.exists():
        file = cfg_dir / file
    with open(file, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("extends", None)
    if parent:
        base = load_config(file.parent / parent, root=root).data
        data = _deep_merge(base, data)
    return Config(data, root, file)
