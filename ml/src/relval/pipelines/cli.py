"""Command-line entry points, one per stage. Every stage is idempotent and resumable and writes a
manifest (config, git commit, library versions, seeds, row counts, timings).

    relval external     download the public Ken French files
    relval ingest       vendor files -> standardised tables (data/interim)
    relval panel        firm-month panel with the 97 inputs and the targets (data/processed)
    relval train        models -> master predictions table (data/predictions)
    relval peers        peer weights for the configured months (data/artifacts)
    relval report       tables, figures and REPLICATION_REPORT.md
    relval replicate    everything above, in order
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from relval.config import load_config
from relval.data import french, schemas
from relval.data.adapter import get_adapter
from relval.features.industry import add_industries
from relval.io import manifest, read_parquet, write_parquet
from relval.models.specs import BASE_MODELS, SPECS, models_in_groups

GROUP_ORDER = ["base", "benchmarks", "recon", "robust", "sensitivity"]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def stage_external(cfg, args=None) -> None:
    with manifest(cfg, "external") as info:
        info["downloads"] = french.download(cfg.path("external"), force=bool(getattr(args, "force", False)))
    log(f"[done] external files in {cfg.path('external')}")


def stage_ingest(cfg, args=None) -> None:
    with manifest(cfg, "ingest") as info:
        raw = get_adapter(cfg["source"], cfg.path("raw")).load()
        info["rows"] = schemas.validate(raw)
        for name in ("fund_q", "fund_a", "market", "static"):
            write_parquet(getattr(raw, name), cfg.interim(f"{name}.parquet"))
    log(f"[done] ingest: {info['rows']}")


def _load_interim(cfg) -> schemas.RawTables:
    if not cfg.interim("market.parquet").exists():
        stage_ingest(cfg)
    return schemas.RawTables(
        *(read_parquet(cfg.interim(f"{n}.parquet")) for n in ("fund_q", "fund_a", "market", "static"))
    )


def stage_panel(cfg, args=None) -> None:
    from relval.panel.build import build_panel

    if getattr(args, "skip_existing", False) and cfg.panel_path().exists():
        log("[skip] panel exists")
        return
    with manifest(cfg, "panel") as info:
        raw = _load_interim(cfg)
        missing = [f for f in ("ff3", "sic49") if not (cfg.path("external") / french.FILES[f]).exists()]
        if missing or not (cfg.path("external") / french.FILES["sic12"]).exists():
            french.download(cfg.path("external"))
        factors = french.load_factors(cfg.path("external"))
        panel, controls = build_panel(raw, factors, cfg)
        write_parquet(panel, cfg.panel_path())
        firms = add_industries(raw.static, cfg.path("external"))
        write_parquet(firms, cfg.path("processed") / f"firms_{cfg['source']}.parquet")
        ctl_dir = cfg.path("processed") / "controls" / cfg["panel_name"]
        ctl_dir.mkdir(parents=True, exist_ok=True)
        for name, table in controls.items():
            table.to_csv(ctl_dir / f"{name}.csv", index=False)
        info["rows"] = {"panel": len(panel), "in_sample": int(panel["in_sample"].sum())}
    log(f"[done] panel: {info['rows']}\n{controls['table1'].to_string(index=False)}")


def _parse_years(text: str | None, cfg) -> list[int]:
    if not text:
        return cfg.years()
    if "-" in text:
        a, b = text.split("-")
        return [y for y in range(int(a), int(b) + 1) if y in cfg.years()]
    return [int(y) for y in text.split(",")]


def stage_train(cfg, args=None) -> None:
    from relval.models.runner import train_models

    groups = (getattr(args, "group", None) or "all").split(",")
    names = getattr(args, "models", None)
    models = names.split(",") if names else models_in_groups(GROUP_ORDER if "all" in groups else groups)
    unknown = [m for m in models if m not in SPECS]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}")
    models.sort(key=lambda m: (GROUP_ORDER.index(SPECS[m].group), m not in BASE_MODELS))
    seeds = [int(s) for s in args.seeds.split(",")] if getattr(args, "seeds", None) else None
    years = _parse_years(getattr(args, "years", None), cfg)
    if not cfg.panel_path().exists():
        stage_panel(cfg)
    with manifest(cfg, "train", {"models": models, "years": years, "seeds": seeds or cfg.seeds}) as info:
        info["rows"] = train_models(
            cfg, models, seeds, years, force=bool(getattr(args, "force", False)), log=log
        )
    log(f"[done] train: {len(info['rows'])} model-seed runs written")


def stage_peers(cfg, args=None) -> None:
    from relval.pipelines import explain

    with manifest(cfg, "peers") as info:
        info["rows"] = explain.run(cfg, log)


def stage_report(cfg, args=None) -> None:
    from relval.reporting.figures import build_figures
    from relval.reporting.report import write_report
    from relval.reporting.tables import Tables

    with manifest(cfg, "report") as info:
        tables = Tables(cfg, log).build_all()
        figures = build_figures(cfg, tables, log)
        write_report(cfg, tables, figures, log)
        info["rows"] = {"tables": len(tables), "figures": len(figures)}
    log(f"[done] report: {info['rows']}")


def stage_replicate(cfg, args=None) -> None:
    """Rebuild everything from data/raw: panel, every model, every table and figure."""
    t0 = time.time()
    stage_external(cfg, args)
    stage_ingest(cfg, args)
    stage_panel(cfg, args)
    stage_train(cfg, args)
    stage_peers(cfg, args)
    stage_report(cfg, args)
    log(f"[done] replicate in {(time.time() - t0) / 60:.1f} minutes")


def stage_status(cfg, args=None) -> None:
    """Which (model, seed) runs are complete."""
    rows = []
    for name in SPECS:
        for seed in cfg.seeds if name in BASE_MODELS else [cfg.headline_seed]:
            d = cfg.predictions_dir(name, seed)
            done = len(list(d.glob("year=*.parquet"))) if d.exists() else 0
            rows.append(
                {
                    "model": name,
                    "group": SPECS[name].group,
                    "seed": seed,
                    "years_done": done,
                    "years_total": len(cfg.years()),
                }
            )
    print(pd.DataFrame(rows).to_string(index=False))


STAGES = {
    "external": stage_external, "ingest": stage_ingest, "panel": stage_panel, "train": stage_train,
    "peers": stage_peers, "report": stage_report, "replicate": stage_replicate, "status": stage_status,
}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="relval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument(
        "--config", default=None, help="config file name in ml/configs or a path (default replication.yaml)"
    )
    parser.add_argument(
        "--group",
        default=None,
        help="train: comma-separated groups (base,benchmarks,recon,robust,sensitivity,all)",
    )
    parser.add_argument(
        "--models", default=None, help="train: comma-separated model names (overrides --group)"
    )
    parser.add_argument("--seeds", default=None, help="train: comma-separated block seeds (default: config)")
    parser.add_argument("--years", default=None, help="train: 1990-1999 or 1990,1991 (default: whole period)")
    parser.add_argument("--force", action="store_true", help="recompute finished work")
    parser.add_argument("--skip-existing", action="store_true", help="panel: keep an existing panel")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    STAGES[args.stage](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
