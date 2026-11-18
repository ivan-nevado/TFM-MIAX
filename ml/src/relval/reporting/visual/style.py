"""Shared style of the visual report: palette roles, matplotlib defaults and the figure registry.

Colour is assigned by role and never cycled (same rules as ``reporting.figures``):

* targets: m2b blue, v2a orange, v2s aqua;
* model families: GBM blue, BL orange, IND aqua, RRV magenta, BG green, LASSO violet, CART yellow;
* data splits: train grey, validation blue, test orange;
* magnitude: one blue ramp; signed quantities: blue <-> red with a grey midpoint.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# importing figures applies the project's rcParams (surface, fonts, recessive grid)
from relval.reporting.figures import (  # noqa: E402,F401
    AQUA,
    AXIS,
    BLUE,
    GREEN,
    GRID,
    INK,
    INK2,
    MAGENTA,
    MUTED,
    ORANGE,
    RED,
    SURFACE,
    YELLOW,
)

VIOLET = "#6250d6"
TARGETS = ["lnm2b", "lnv2a", "lnv2s"]
MULT_OF = {"lnm2b": "m2b", "lnv2a": "v2a", "lnv2s": "v2s"}
TARGET_LABEL = {"lnm2b": "ln(M/B)", "lnv2a": "ln(V/A)", "lnv2s": "ln(V/S)"}
TARGET_COLOR = {"lnm2b": BLUE, "lnv2a": ORANGE, "lnv2s": AQUA}
FAMILY_COLOR = {
    "GBM": BLUE, "BL": ORANGE, "IND": AQUA, "RRV": MAGENTA, "BG": GREEN, "LASSO": VIOLET, "CART": YELLOW,
}  # fmt: skip
SPLIT_COLOR = {"train": MUTED, "validación": BLUE, "test": ORANGE}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", *BLUE_RAMP])
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b02e2d"]
)

CATEGORY_ES = {
    "Profitability": "Rentabilidad", "Growth": "Crecimiento", "Financial soundness": "Solidez financiera",
    "Efficiency": "Eficiencia", "Liquidity": "Liquidez", "Solvency": "Solvencia",
    "Capitalization": "Capitalización", "Size": "Tamaño", "Industry": "Sector", "Payout": "Payout",
    "CAPMbeta": "Beta CAPM",
}  # fmt: skip

plt.rcParams.update({"figure.dpi": 100, "axes.titlesize": 10.5, "font.size": 9})


def family(model: str) -> str:
    if model.startswith("ML") or model.startswith("GBM"):
        return "GBM"
    return model.split("_")[0]


@dataclass
class Figure:
    key: str
    section: str
    title: str
    caption: str
    path: Path

    def data_uri(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.path.read_bytes()).decode()


@dataclass
class FigureBook:
    """Saves figures as PNG (for the thesis) and keeps their titles and captions (for the HTML)."""

    out_dir: Path
    figures: list[Figure] = field(default_factory=list)

    def save(self, fig, key: str, section: str, title: str, caption: str) -> Figure:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{key}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        item = Figure(key, section, es_numbers(title), es_numbers(caption), path)
        self.figures = [f for f in self.figures if f.key != key] + [item]
        return item

    def section(self, name: str) -> list[Figure]:
        return [f for f in self.figures if f.section == name]


def es_numbers(text: str) -> str:
    """Spanish number style in prose: 1,234.5 -> 1.234,5. Thousands groups only after a non-zero lead, so
    Spanish decimals already in the text (0,001) are left alone."""
    text = re.sub(
        r"(?<![\d.,])([1-9]\d{0,2})((?:,\d{3})+)(?![\d,])",
        lambda m: m.group(1) + m.group(2).replace(",", "#THOUSANDS#"),
        text,
    )
    text = re.sub(r"(?<=\d)\.(?=\d)", ",", text)
    return text.replace("#THOUSANDS#", ".")


def suptitle(fig, text: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)


def label_end(ax, x, y, text: str, color: str = INK2) -> None:
    """Direct label at the end of a line (text in ink, not in the series colour)."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=color)
