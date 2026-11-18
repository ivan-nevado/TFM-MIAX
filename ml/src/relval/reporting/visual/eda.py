"""Exploratory data analysis of the panel: sample size, composition, targets, inputs and coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from relval.features.registry import CATEGORIES, CATEGORY_OF, FEATURES
from relval.reporting.visual.data import FF12_ES
from relval.reporting.visual.style import (
    AQUA,
    BLUE,
    CATEGORY_ES,
    DIV_CMAP,
    GRID,
    INK2,
    MUTED,
    ORANGE,
    RED,
    SEQ_CMAP,
    TARGET_COLOR,
    TARGET_LABEL,
    TARGETS,
    label_end,
    suptitle,
)

SECTION = "eda"
PAPER_TARGET_MEDIAN = {"lnm2b": 0.48, "lnv2a": -0.03, "lnv2s": 0.35}
PAPER_PER_MONTH = {"targets": 4505.2, "sample": 3774.6}


def build(data, book) -> dict:
    s = data.sample
    facts: dict = {}
    _sample_over_time(data, book, facts)
    _composition(s, book, facts)
    _sectors(s, book, facts)
    _targets(s, book, facts)
    _size_and_losses(s, book, facts)
    _coverage(data, book, facts)
    _missing_by_category(s, book)
    _correlations(s, book, facts)
    return facts


def _sample_over_time(data, book, facts):
    c = data.panel_counts
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(c["date"], c["targets"], color=MUTED, label="Con targets positivos")
    ax.plot(c["date"], c["sample"], color=BLUE, label="Muestra final (filtro de tamaño)")
    label_end(ax, c["date"].iloc[-1], c["sample"].iloc[-1], f"{int(c['sample'].iloc[-1]):,}")
    label_end(ax, c["date"].iloc[0], c["sample"].iloc[0], f"{int(c['sample'].iloc[0]):,}")
    ax.set_ylabel("empresas por mes")
    ax.set_ylim(0)
    ax.legend(loc="upper left")
    suptitle(fig, "Empresas por mes en el panel")
    fig.tight_layout()
    facts["firms_first"], facts["firms_last"] = int(c["sample"].iloc[0]), int(c["sample"].iloc[-1])
    facts["firms_mean"] = float(c["sample"].mean())
    book.save(
        fig, "eda_01_muestra_mensual", SECTION, "Tamaño de la muestra mes a mes",
        f"La muestra crece de {facts['firms_first']:,} empresas en enero de 1990 a {facts['firms_last']:,} en "
        f"diciembre de 2019 (media {facts['firms_mean']:,.0f}; el paper tiene {PAPER_PER_MONTH['sample']:,.0f}). "
        "El crecimiento no es real: LSEG solo trae empresas que cotizan hoy, así que las que existían en 1990 "
        "y desaparecieron faltan (sesgo de supervivencia).",
    )  # fmt: skip


def _composition(s, book, facts):
    ex = s.groupby(["date", "exchange"]).size().unstack(fill_value=0)
    ex = ex[[c for c in ("NYSE", "NASDAQ", "AMEX") if c in ex]]
    share = ex.div(ex.sum(axis=1), axis=0) * 100
    org = s.groupby(["date", "origen"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = {"NYSE": BLUE, "NASDAQ": ORANGE, "AMEX": AQUA}
    ax[0].stackplot(share.index, share.T.values, colors=[colors[c] for c in share], edgecolor="white",
                    linewidth=0.4, labels=share.columns)  # fmt: skip
    ax[0].set_ylim(0, 100)
    ax[0].set_ylabel("% de la muestra")
    ax[0].set_title("Mercado de cotización")
    ax[0].legend(loc="lower left", ncol=3)
    delisted = org.get("delisted", org.drop(columns=["final"], errors="ignore").sum(axis=1))
    ax[1].plot(org.index, delisted, color=ORANGE)
    ax[1].set_title("Empresas ya deslistadas en la muestra")
    ax[1].set_ylabel("empresas por mes")
    ax[1].set_ylim(0)
    suptitle(fig, "Composición de la muestra")
    fig.tight_layout()
    facts["share_nasdaq_last"] = float(share.get("NASDAQ", pd.Series([np.nan])).iloc[-1])
    facts["delisted_firms"] = int(s.loc[s["origen"] != "final", "id"].nunique())
    book.save(
        fig, "eda_02_composicion", SECTION, "Mercado de cotización y empresas deslistadas",
        "No hay regiones geográficas en los datos: todas las empresas son estadounidenses (es un filtro del "
        "paper), así que el desglose disponible es el mercado (NYSE, NASDAQ, AMEX). Es el mercado de hoy, no el "
        f"histórico. Solo {facts['delisted_firms']} empresas de la muestra ya no cotizan: es la medida directa "
        "del sesgo de supervivencia.",
    )  # fmt: skip


def _sectors(s, book, facts):
    lab = s["ff12_name"].map(FF12_ES).fillna("Sin clasificar")
    dec = (s["date"].dt.year // 10 * 10).rename("dec")
    sh = pd.crosstab(lab, dec, normalize="columns") * 100
    sh = sh.sort_values(sh.columns[-1])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    y = np.arange(len(sh))
    a, b = sh.columns[0], sh.columns[-1]
    ax[0].hlines(y, sh[a], sh[b], color=GRID, linewidth=2)
    ax[0].scatter(sh[a], y, color=MUTED, s=36, zorder=3, label=f"{a}s")
    ax[0].scatter(sh[b], y, color=BLUE, s=36, zorder=3, label=f"{b}s")
    ax[0].set_yticks(y, sh.index)
    ax[0].set_xlabel("% de empresa-mes")
    ax[0].set_title("Sector Fama-French 12: década inicial frente a final")
    ax[0].legend(loc="lower right")
    ax[0].grid(axis="x")
    tr = s["trbc"].fillna("Sin dato").value_counts(normalize=True).sort_values() * 100
    ax[1].barh(tr.index, tr.values, color=BLUE, height=0.6)
    ax[1].set_xlabel("% de empresa-mes")
    ax[1].set_title("Sector económico de LSEG (TRBC), toda la muestra")
    ax[1].grid(axis="x")
    ax[1].grid(axis="y", visible=False)
    suptitle(fig, "Composición sectorial")
    fig.tight_layout()
    facts["unclassified_ff12"] = float((lab == "Sin clasificar").mean() * 100)
    chg = (sh[b] - sh[a]).drop("Sin clasificar", errors="ignore").sort_values()
    up, down = list(chg.index[-2:][::-1]), list(chg.index[:2])
    book.save(
        fig, "eda_03_sectores", SECTION, "Composición sectorial",
        f"El sector Fama-French sale del SIC, que solo existe para parte de las empresas: el "
        f"{facts['unclassified_ff12']:.0f} % de las empresa-mes queda sin clasificar (en el paper, ~1 %). "
        "El sector de LSEG (TRBC) cubre a casi todas y se usa como alternativa descriptiva. "
        f"De los años 90 a los 2010 ganan peso {up[0].lower()} y {up[1].lower()}; pierden {down[0].lower()} y "
        f"{down[1].lower()}.",
    )  # fmt: skip


def _targets(s, book, facts):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for a, t in zip(ax, TARGETS, strict=True):
        v = s[t].dropna()
        lo, hi = v.quantile([0.005, 0.995])
        a.hist(
            v[v.between(lo, hi)], bins=70, color=TARGET_COLOR[t], alpha=0.85, edgecolor="white", linewidth=0.3
        )
        a.axvline(v.median(), color=INK2, linewidth=1.2)
        a.axvline(PAPER_TARGET_MEDIAN[t], color=INK2, linewidth=1.2, linestyle=":")
        a.set_title(f"{TARGET_LABEL[t]}  ·  mediana {v.median():.2f} (paper {PAPER_TARGET_MEDIAN[t]:.2f})")
        a.set_xlabel("valor del target")
        facts[f"median_{t}"] = float(v.median())
    ax[0].set_ylabel("empresa-mes")
    ax[0].legend(
        handles=[
            plt.Line2D([], [], color=INK2, label="mediana nuestra"),
            plt.Line2D([], [], color=INK2, ls=":", label="mediana del paper"),
        ],
        loc="upper right",
    )
    suptitle(fig, "Distribución de los tres múltiplos (en logaritmos)")
    fig.tight_layout()
    book.save(
        fig, "eda_04_targets", SECTION, "Distribución de los targets",
        "Los tres múltiplos en logaritmo tienen forma casi normal, que es la razón por la que el paper los "
        "modeliza en logaritmos. Nuestras medianas están por encima de las del paper: una muestra de "
        "supervivientes está sesgada hacia empresas grandes y con éxito, que cotizan con múltiplos más altos.",
    )  # fmt: skip

    yr = s.groupby(s["date"].dt.year)[TARGETS].quantile([0.25, 0.5, 0.75]).unstack()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3), sharex=True)
    for a, t in zip(ax, TARGETS, strict=True):
        q = yr[t]
        a.fill_between(q.index, q[0.25], q[0.75], color=TARGET_COLOR[t], alpha=0.18, linewidth=0)
        a.plot(q.index, q[0.5], color=TARGET_COLOR[t])
        a.set_title(TARGET_LABEL[t])
    ax[0].set_ylabel("mediana y rango intercuartílico")
    suptitle(fig, "Evolución anual de los múltiplos")
    fig.tight_layout()
    book.save(
        fig, "eda_05_targets_tiempo", SECTION, "Múltiplos a lo largo del tiempo",
        "Mediana (línea) y percentiles 25–75 (banda) por año. Se ven la burbuja tecnológica (1999–2000), la "
        "caída de 2008 y la recuperación posterior: el nivel de los múltiplos cambia con el ciclo, y por eso el "
        "paper entrena un modelo distinto cada mes en lugar de uno para todo el periodo.",
    )  # fmt: skip

    lab = s["ff12_name"].map(FF12_ES)
    order = s.groupby(lab)["lnm2b"].median().sort_values().index
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    for a, t in zip(ax, TARGETS, strict=True):
        groups = [s.loc[lab == g, t].dropna().values for g in order]
        bp = a.boxplot(groups, vert=False, showfliers=False, widths=0.55, patch_artist=True,
                       medianprops={"color": "white", "linewidth": 1.4},
                       whiskerprops={"color": MUTED}, capprops={"color": MUTED})  # fmt: skip
        for box in bp["boxes"]:
            box.set(facecolor=TARGET_COLOR[t], edgecolor=TARGET_COLOR[t])
        a.set_yticks(range(1, len(order) + 1), order)
        a.set_title(TARGET_LABEL[t])
        a.grid(axis="x")
        a.grid(axis="y", visible=False)
    meds = {t: s.groupby(lab)[t].median().sort_values() for t in TARGETS}
    suptitle(fig, "Múltiplos por sector (sin las empresas sin sector)")
    fig.tight_layout()
    book.save(
        fig, "eda_06_targets_sector", SECTION, "Múltiplos por sector",
        "Cajas = percentiles 25–75 y mediana; bigotes = 1,5 veces el rango intercuartílico. "
        + " ".join(f"{TARGET_LABEL[t]}: más alto en {m.index[-1].lower()}, más bajo en {m.index[0].lower()}."
                   for t, m in meds.items())
        + " El orden cambia según el múltiplo (finanzas, con balances enormes, tiene el V/A más bajo), y esa "
        "variación entre sectores es la que explota el benchmark IND y que el GBM tiene que superar.",
    )  # fmt: skip


def _size_and_losses(s, book, facts):
    lm = np.log10(s["M"].where(s["M"] > 0))
    q = lm.groupby(s["date"].dt.year).quantile([0.1, 0.5, 0.9]).unstack()
    loss = (s["roe_wr"] < 0).groupby(s["date"].dt.year).mean() * 100
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].fill_between(q.index, q[0.1], q[0.9], color=BLUE, alpha=0.15, linewidth=0)
    ax[0].plot(q.index, q[0.5], color=BLUE)
    ticks = [1, 2, 3, 4, 5]
    ax[0].set_yticks(ticks, ["10 M$", "100 M$", "1.000 M$", "10.000 M$", "100.000 M$"])
    ax[0].set_title("Capitalización: mediana y percentiles 10–90")
    ax[1].plot(loss.index, loss.values, color=RED)
    ax[1].set_ylim(0)
    ax[1].set_ylabel("% de empresas con ROE negativo")
    ax[1].set_title("Empresas en pérdidas")
    suptitle(fig, "Tamaño y rentabilidad de la muestra")
    fig.tight_layout()
    facts["median_mcap_last"] = float(10 ** q[0.5].iloc[-1])
    facts["loss_share_mean"] = float(loss.mean())
    book.save(
        fig, "eda_07_tamano_perdidas", SECTION, "Tamaño y empresas en pérdidas",
        f"La empresa mediana de 2019 vale unos {facts['median_mcap_last']:,.0f} M$. De media, el "
        f"{facts['loss_share_mean']:.0f} % de las empresa-mes tiene ROE negativo. El paper muestra que el GBM "
        "valora peor a las empresas pequeñas y a las que pierden dinero: por eso se analizan los errores por "
        "tamaño y por rentabilidad en la sección de resultados.",
    )  # fmt: skip


def _coverage(data, book, facts):
    t = data.table("ia4_summary_statistics")
    if t.empty:
        return
    t["cat_order"] = t["category"].map({c: i for i, c in enumerate(CATEGORIES)})
    t = t.sort_values(["cat_order", "coverage"], ascending=[False, True]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 17))
    y = np.arange(len(t))
    ax.hlines(y, t["coverage"] * 100, t["paper_coverage"] * 100, color=GRID, linewidth=2)
    ax.scatter(t["paper_coverage"] * 100, y, color=MUTED, s=22, zorder=3, label="paper")
    ax.scatter(t["coverage"] * 100, y, color=BLUE, s=22, zorder=3, label="nuestro")
    ax.set_yticks(y, [f"{v}  ({CATEGORY_ES[c]})" for v, c in zip(t["variable"], t["category"], strict=True)],
                  fontsize=6.5)  # fmt: skip
    ax.set_xlabel("% de empresa-mes con dato")
    ax.set_xlim(max(0, min(t["coverage"].min(), t["paper_coverage"].min()) * 100 - 5), 101)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left")
    suptitle(fig, "Cobertura de las 97 variables frente a la Tabla IA.4 del paper")
    fig.tight_layout()
    d = (t["coverage"] - t["paper_coverage"]) * 100
    facts["coverage_within_5pp"] = int((d.abs() <= 5).sum())
    facts["coverage_worst"] = t.loc[d.idxmin(), "variable"]
    book.save(
        fig, "eda_08_cobertura", SECTION, "Cobertura de las variables de entrada",
        f"{facts['coverage_within_5pp']} de las 97 variables tienen una cobertura a menos de 5 puntos de la del "
        f"paper. Las mayores diferencias (p. ej. {facts['coverage_worst']}, el sector y los márgenes brutos) "
        "vienen de datos que LSEG no trae (SIC, coste de ventas de las financieras) y están en DEVIATIONS.md. "
        "El GBM admite datos ausentes, así que una cobertura menor resta información pero no rompe el modelo.",
    )  # fmt: skip


def _missing_by_category(s, book):
    miss = pd.DataFrame({cat: s[cols].isna().mean(axis=1) for cat, cols in CATEGORIES.items()})
    by_year = miss.groupby(s["date"].dt.year).mean().T * 100
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(by_year.values, aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=max(40, by_year.values.max()))
    ax.set_yticks(range(len(by_year)), [CATEGORY_ES[c] for c in by_year.index])
    ax.set_xticks(range(0, by_year.shape[1], 3), by_year.columns[::3])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("% de valores ausentes")
    suptitle(fig, "Datos ausentes por categoría de variables y año")
    fig.tight_layout()
    book.save(
        fig, "eda_09_ausentes", SECTION, "Datos ausentes por categoría y año",
        "Cada celda es la media de valores ausentes de las variables de esa categoría en ese año (más oscuro = "
        "más huecos). Los primeros años tienen más huecos porque faltan los trimestres previos que necesitan las "
        "variables de crecimiento, y la beta exige 60 meses de historia.",
    )  # fmt: skip


def _correlations(s, book, facts):
    dec = s[s["date"].dt.month == 12]
    X = dec[FEATURES].astype(float)
    rho = X.rank().corr(method="pearson").values  # Spearman = Pearson on ranks
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(rho, cmap=DIV_CMAP, vmin=-1, vmax=1)
    bounds = np.cumsum([len(v) for v in CATEGORIES.values()])
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="white", linewidth=1)
        ax.axvline(b - 0.5, color="white", linewidth=1)
    mids = bounds - np.array([len(v) for v in CATEGORIES.values()]) / 2
    ax.set_yticks(mids - 0.5, [CATEGORY_ES[c] for c in CATEGORIES], fontsize=7.5)
    ax.set_xticks(mids - 0.5, [CATEGORY_ES[c] for c in CATEGORIES], rotation=60, ha="right", fontsize=7.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.8)
    cb.set_label("correlación de Spearman")
    suptitle(fig, "Correlación entre las 97 variables (diciembres de 1990–2019)")
    fig.tight_layout()
    upper = rho[np.triu_indices_from(rho, 1)]
    facts["pairs_high_corr"] = int((np.abs(upper) > 0.9).sum())
    book.save(
        fig, "eda_10_correlacion", SECTION, "Correlación entre variables",
        f"Variables ordenadas por categoría. Hay {facts['pairs_high_corr']} pares con |ρ| > 0,9, entre ellos los "
        "cuatro pares duplicados del paper. Para un árbol la colinealidad no es un problema de predicción, pero "
        "sí de interpretación: SHAP reparte la importancia entre variables casi iguales de forma arbitraria.",
    )  # fmt: skip

    rows = []
    for t in TARGETS:
        for f in FEATURES:
            ok = dec[[f, t]].dropna()
            r = spearmanr(ok[f], ok[t]).statistic if len(ok) > 50 else np.nan
            rows.append({"target": t, "variable": f, "rho": r})
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.8))
    for a, t in zip(ax, TARGETS, strict=True):
        top = r[r.target == t].assign(a=lambda d: d.rho.abs()).nlargest(12, "a").sort_values("rho")
        a.barh(top["variable"], top["rho"], color=[RED if v < 0 else BLUE for v in top["rho"]], height=0.6)
        a.axvline(0, color=INK2, linewidth=0.8)
        a.set_title(TARGET_LABEL[t])
        a.set_xlim(-1, 1)
        a.grid(axis="x")
        a.grid(axis="y", visible=False)
        a.tick_params(axis="y", labelsize=7.5)
    ax[0].legend(handles=[Patch(color=BLUE, label="relación positiva"), Patch(color=RED, label="relación negativa")],
                 loc="lower right", fontsize=7)  # fmt: skip
    suptitle(fig, "Las 12 variables más correlacionadas con cada múltiplo (Spearman)")
    fig.tight_layout()
    book.save(
        fig, "eda_11_correlacion_target", SECTION, "Relación univariante con los targets",
        "Correlación de rangos de cada variable con el múltiplo, sin modelo. Rentabilidad y crecimiento suben "
        "el M/B; para V/S manda la eficiencia (rotación de activos con signo negativo). Es una primera pista "
        "de lo que SHAP confirmará con el modelo completo, que además capta no linealidades e interacciones.",
    )  # fmt: skip
    facts["corr_by_category"] = (
        r.assign(cat=r.variable.map(CATEGORY_OF))
        .groupby(["target", "cat"])
        .rho.apply(lambda v: v.abs().mean())
    )
