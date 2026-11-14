"""Parquet helpers and run manifests (config, git commit, library versions, row counts, timings)."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Atomic parquet write: a crashed job never leaves a half-written file that looks finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def read_partitioned(directory: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Concatenate every ``*.parquet`` under a directory (year partitions of one model/seed)."""
    files = sorted(directory.rglob("*.parquet")) if directory.exists() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def write_table(df: pd.DataFrame, directory: Path, name: str, floatfmt: str = ".3f") -> None:
    """Write a result table as CSV and as a Markdown file with the same stem."""
    directory.mkdir(parents=True, exist_ok=True)
    df.to_csv(directory / f"{name}.csv", index=False)
    (directory / f"{name}.md").write_text(to_markdown(df, floatfmt) + "\n", encoding="utf-8")


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Small Markdown table writer (avoids the optional `tabulate` dependency)."""

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "" if pd.isna(v) else format(v, floatfmt)
        return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("lightgbm", "numpy", "pandas", "pyarrow", "sklearn", "scipy", "statsmodels"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - optional library
            versions[name] = "not installed"
    return versions


@contextmanager
def manifest(cfg, stage: str, extra: dict[str, Any] | None = None):
    """Context manager that writes ``artifacts/<run>/manifests/<stage>_<utc>.json`` when the stage ends."""
    info: dict[str, Any] = {
        "stage": stage,
        "run_name": cfg.run_name,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(cfg.root),
        "versions": library_versions(),
        "config_file": str(cfg.source_file) if cfg.source_file else None,
        "config": cfg.data,
        "rows": {},
    }
    info.update(extra or {})
    t0 = time.time()
    try:
        yield info
        info["status"] = "ok"
    except Exception as exc:
        info["status"] = f"failed: {exc!r}"
        raise
    finally:
        info["seconds"] = round(time.time() - t0, 1)
        out = cfg.artifacts_dir("manifests")
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (out / f"{stage}_{stamp}.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
