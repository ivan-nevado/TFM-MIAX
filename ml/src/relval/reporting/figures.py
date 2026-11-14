"""Figures 1-5 of the paper as static PNG files, drawn from the tables of ``reporting.tables``.

Colour is assigned by role and never cycled: in model comparisons the GBM is always blue and each
benchmark family keeps its own hue; in target comparisons m2b, v2a and v2s keep theirs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#e34948",
)
FAMILY_COLOR = {"GBM": BLUE, "BL": ORANGE, "IND": AQUA, "HP": YELLOW, "RRV": MAGENTA, "BG": GREEN}
TARGET_COLOR = {"m2b": BLUE, "v2a": ORANGE, "v2s": AQUA}
MULTS = ["m2b", "v2a", "v2s"]
FF12_ORDER = [
    "NoDur",
    "Durbl",
    "Manuf",
    "Enrgy",
    "Chems",
    "BusEq",
    "Telcm",
    "Utils",
    "Shops",
    "Hlth",
    "Money",
    "Other",
]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6, "lines.markersize": 5,
    }
)  # fmt: skip


def _family(model: str) -> str:
    return "GBM" if model.startswith("ML_") else model.split("_")[0]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figure1(monthly: pd.DataFrame, out: Path) -> Path | None:
    """Monthly median absolute valuation error. Panels A-C: GBM against BL, IND and HP for the
    same multiple; D: the three GBMs against RRV and BG; E: the three GBMs."""
    if monthly is None or monthly.empty:
        return None
    m = monthly.copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.set_index("date")
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    axes = axes.ravel()
    for ax, mult, letter in zip(axes[:3], MULTS, "ABC", strict=False):
        for model in (f"ML_ln{mult}", f"BL_{mult}", f"IND_{mult}", f"HP_{mult}"):
            if model in m:
                ax.plot(m.index, m[model], color=FAMILY_COLOR[_family(model)], label=model)
        ax.set_title(f"Panel {letter}: models targeting {mult}")
    for mult in MULTS:
        if f"ML_ln{mult}" in m:
            axes[3].plot(m.index, m[f"ML_ln{mult}"], color=TARGET_COLOR[mult], label=f"ML_ln{mult}")
            axes[4].plot(m.index, m[f"ML_ln{mult}"], color=TARGET_COLOR[mult], label=f"ML_ln{mult}")
    for model in ("RRV", "BG"):
        if model in m:
            axes[3].plot(m.index, m[model], color=FAMILY_COLOR[model], label=model)
    axes[3].set_title("Panel D: GBMs against RRV and BG")
    axes[4].set_title("Panel E: the three GBMs")
    for ax in axes[:5]:
        ax.set_ylabel("median |ε|")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", ncol=2)
    axes[5].axis("off")
    axes[5].text(0, 0.9, "Median absolute valuation error by month.\nε = M̂ / M − 1, out-of-sample predictions.",
                 color=INK2, va="top", transform=axes[5].transAxes)  # fmt: skip
    fig.suptitle(
        "Figure 1. Model valuation errors over time", x=0.01, ha="left", fontsize=12, fontweight="bold"
    )
    fig.tight_layout()
    return _save(fig, out)


def _group_panel(
    ax, data: pd.DataFrame, sort: str, models: list[str], value: str, colors: dict, title: str
) -> None:
    d = data[data["sort"] == sort]
    for model in models:
        s = d[d["model"] == model].sort_values("group")
        if s.empty:
            continue
        if sort == "ff12":
            ax.plot(range(len(s)), s[value], marker="o", color=colors[model], label=model)
            ax.set_xticks(range(len(s)))
            ax.set_xticklabels(s["group_label"], rotation=45, ha="right")
        else:
            ax.plot(s["group"].astype(int), s[value], marker="o", color=colors[model], label=model)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlabel("quintile (1 = low)")
    ax.set_title(title)


def figure2(by_group: pd.DataFrame, out: Path) -> Path | None:
    """Median absolute error by size, market-to-book and ROE quintile: ML_lnm2b against the
    traditional models (panel A: models targeting m2b; panel B: RRV and BG)."""
    if by_group is None or by_group.empty:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey="row")
    sets = [["ML_lnm2b", "BL_m2b", "IND_m2b", "HP_m2b"], ["ML_lnm2b", "RRV", "BG"]]
    names = {"size": "size", "m2b": "market-to-book", "roe": "ROE"}
    for r, models in enumerate(sets):
        colors = {mdl: FAMILY_COLOR[_family(mdl)] for mdl in models}
        for c, sort in enumerate(("size", "m2b", "roe")):
            _group_panel(
                axes[r, c],
                by_group,
                sort,
                models,
                "med_abs_eps",
                colors,
                f"Panel {'AB'[r]}.{c + 1}: by {names[sort]}",
            )
            axes[r, c].set_ylim(bottom=0)
        axes[r, 0].set_ylabel("median |ε|")
        axes[r, 0].legend(loc="upper right")
    fig.suptitle("Figure 2. Valuation errors across firms", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out)


def figure3(by_group: pd.DataFrame, ff12_names: dict[int, str], out: Path) -> Path | None:
    """The three GBMs: median absolute error (panel A) and median error (panel B) by size,
    market-to-book and ROE quintile and by Fama-French 12 industry."""
    if by_group is None or by_group.empty:
        return None
    data = by_group.copy()
    data["group_label"] = data["group"].map(lambda g: ff12_names.get(int(g), str(int(g))))
    models = [f"ML_ln{m}" for m in MULTS]
    colors = {f"ML_ln{m}": TARGET_COLOR[m] for m in MULTS}
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey="row")
    names = {"size": "size", "m2b": "market-to-book", "roe": "ROE", "ff12": "FF12 industry"}
    for r, value in enumerate(("med_abs_eps", "med_eps")):
        for c, sort in enumerate(("size", "m2b", "roe", "ff12")):
            _group_panel(
                axes[r, c], data, sort, models, value, colors, f"Panel {'AB'[r]}.{c + 1}: by {names[sort]}"
            )
            if value == "med_eps":
                axes[r, c].axhline(0, color=AXIS, linewidth=0.8)
        axes[r, 0].set_ylabel("median |ε|" if r == 0 else "median ε")
        axes[r, 0].legend(loc="best")
    fig.suptitle(
        "Figure 3. Valuation errors across firms for the ML models",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, out)


def figure4(weights: pd.DataFrame, title: str, out: Path) -> Path | None:
    """Peer weights of the focus firm, sorted in descending order."""
    if weights is None or weights.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(weights["rank"], weights["weight"], color=BLUE)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("training firms, sorted by peer weight")
    ax.set_ylabel("peer weight")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out)


def figure5(waterfall: pd.DataFrame, title: str, out: Path) -> Path | None:
    """Raw SHAP waterfall: training mean, top contributions, "other", prediction."""
    if waterfall is None or waterfall.empty:
        return None
    wf = waterfall.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    running = 0.0
    for i, row in wf.iterrows():
        if row["kind"] == "base":
            start, width, color = 0.0, row["value"], MUTED
            running = row["value"]
        elif row["kind"] == "total":
            start, width, color = 0.0, row["value"], INK2
        else:
            start, width = running, row["value"]
            color = BLUE if width >= 0 else RED
            running += width
        ax.barh(i, width, left=start, color=color, height=0.55)
        end = start + width
        label = f"{row['value']:.2f}" if row["kind"] != "contribution" else f"{row['value']:+.2f}"
        ax.text(max(end, start) + 0.01, i, label, va="center", color=INK2, fontsize=8)
    ax.set_yticks(range(len(wf)))
    ax.set_yticklabels(wf["label"])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("ln(m2b): positive contributions in blue, negative in red")
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out)


def build_figures(cfg, tables: dict[str, pd.DataFrame], log=print) -> list[Path]:
    out = cfg.reports_dir("figures")
    made: list[Path | None] = []
    made.append(figure1(tables.get("fig1_monthly_median_abs_error"), out / "figure1_errors_over_time.png"))
    groups = tables.get("fig2_fig3_errors_by_group")
    ff12 = {}
    firms_path = cfg.path("processed") / f"firms_{cfg['source']}.parquet"
    if firms_path.exists():
        firms = pd.read_parquet(firms_path, columns=["ff12", "ff12_name"]).dropna().drop_duplicates()
        ff12 = {int(k): v for k, v in zip(firms["ff12"], firms["ff12_name"], strict=False)}
    made.append(figure2(groups, out / "figure2_errors_across_firms.png"))
    made.append(figure3(groups, ff12, out / "figure3_ml_errors_across_firms.png"))
    ex = cfg["explain"]
    label = f"{ex['focus_firm']}, {ex['focus_month']}"
    made.append(
        figure4(
            tables.get("fig4_focus_peer_weights"),
            f"Figure 4. GBM peer weights of {label}",
            out / "figure4_peer_weights.png",
        )
    )
    made.append(
        figure5(
            tables.get("fig5_focus_shap_waterfall"),
            f"Figure 5. SHAP values of {label}",
            out / "figure5_shap_waterfall.png",
        )
    )
    done = [p for p in made if p is not None]
    log(f"[done] figures: {len(done)} written to {out}")
    return done


__all__ = ["build_figures", "figure1", "figure2", "figure3", "figure4", "figure5", "np"]
