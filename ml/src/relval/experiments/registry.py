"""Discovery and validation of the experiment configs in ``ml/configs/experiments/``.

An experiment config has an ``experiment`` block:

    experiment:
      id: e1_l1                      # equal to the file stem; run_name must be experiments/<id>
      family: E1 robust loss         # groups experiments in the comparison tables
      title: GBM with L1 loss
      hypothesis: ...                # written before the run, never edited after seeing results
      baseline: replication          # run whose predictions it is compared with
      models: [ML_lnm2b, ML_lnv2a, ML_lnv2s]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from relval.config import Config, find_root, load_config
from relval.models.specs import BASE_MODELS, SPECS

EXPERIMENTS_SUBDIR = "experiments"
RUN_PREFIX = "experiments/"


@dataclass
class Experiment:
    id: str
    family: str
    title: str
    hypothesis: str
    baseline: str
    models: list[str]
    cfg: Config

    @property
    def config_name(self) -> str:
        return f"{EXPERIMENTS_SUBDIR}/{self.id}.yaml"


def is_experiment(cfg: Config) -> bool:
    return bool(cfg.get("experiment"))


def experiment_models(cfg: Config) -> list[str]:
    """Models an experiment trains (default: the three base GBMs)."""
    return list(cfg.get("experiment.models") or BASE_MODELS)


def validate(cfg: Config) -> Experiment:
    meta = cfg.get("experiment") or {}
    exp_id = str(meta.get("id", ""))
    problems = []
    if not exp_id:
        problems.append("experiment.id is missing")
    if cfg.source_file is not None and Path(cfg.source_file).stem != exp_id:
        problems.append(f"experiment.id {exp_id!r} differs from the file name {Path(cfg.source_file).name!r}")
    if cfg.run_name != RUN_PREFIX + exp_id:
        problems.append(f"run_name must be {RUN_PREFIX + exp_id!r} (is {cfg.run_name!r})")
    models = experiment_models(cfg)
    unknown = [m for m in models if m not in SPECS]
    if unknown:
        problems.append(f"unknown models {unknown}")
    if problems:
        raise ValueError(f"invalid experiment config {cfg.source_file}: " + "; ".join(problems))
    return Experiment(
        id=exp_id,
        family=str(meta.get("family", "")),
        title=str(meta.get("title", exp_id)),
        hypothesis=" ".join(str(meta.get("hypothesis", "")).split()),
        baseline=str(meta.get("baseline", "replication")),
        models=models,
        cfg=cfg,
    )


def discover(root: Path | None = None) -> list[Experiment]:
    """Every experiment config, sorted by id. Files starting with ``_`` are shared bases, not
    experiments. Invalid files raise (a typo must not go unnoticed)."""
    root = (root or find_root()).resolve()
    folder = root / "ml" / "configs" / EXPERIMENTS_SUBDIR
    files = sorted(f for f in folder.glob("*.yaml") if not f.name.startswith("_"))
    return [validate(load_config(f, root=root)) for f in files]


def baseline_config(exp: Experiment) -> Config:
    """Config of the baseline run (the replication unless the experiment names another file)."""
    name = exp.baseline if exp.baseline.endswith(".yaml") else f"{exp.baseline}.yaml"
    return load_config(name, root=exp.cfg.root)
