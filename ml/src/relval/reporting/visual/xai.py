"""Explainability: SHAP (global, by category, over time, dependence, by sector) and peer weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon

from relval.features.registry import CATEGORIES, CATEGORY_OF, FEATURES
from relval.reporting.visual.data import FF12_ES
from relval.reporting.visual.style import (
    BLUE,
    CATEGORY_ES,
    DIV_CMAP,
    GRID,
    INK2,
    MUTED,
    ORANGE,
    RED,
    SURFACE,
    TARGET_COLOR,
    TARGET_LABEL,
    TARGETS,
    suptitle,
)

SECTION = "xai"
TOP_CATS = 5


def build(data, book) -> dict:
    facts: dict = {}
    shares, yearly, samples = {}, {}, {}
    rng = np.random.default_rng(0)
    for t in TARGETS:
        sh = data.shap(f"ML_{t}")
        if sh.empty:
            continue
        sh["date"] = pd.to_datetime(sh["date"])
        a = sh[FEATURES].abs()
        shares[t] = a.mean() / a.mean().sum() * 100
        yr = a.groupby(sh["date"].dt.year).mean()
        yearly[t] = yr.div(yr.sum(axis=1), axis=0) * 100
        idx = rng.choice(len(sh), size=min(40_000, len(sh)), replace=False)
        samples[t] = sh.iloc[idx].reset_index(drop=True)
        del sh, a
    if not shares:
        return facts
    facts["shares"] = shares
    _global(shares, book, facts)
    _categories(data, shares, book, facts)
    _categories_time(yearly, book)
    feats = data.sample[["id", "date", "ff12_name", *FEATURES]]
    joined = {t: s.merge(feats, on=["id", "date"], suffixes=("", "_x")) for t, s in samples.items()}
    _beeswarm(shares, joined, book)
    _dependence(shares, joined, book)
    _industry(joined, book, facts)
    _peers(data, book, facts)
    return facts


def _global(shares, book, facts):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.2))
    for a, t in zip(ax, TARGETS, strict=True):
        top = shares[t].nlargest(15)[::-1]
        a.barh([f"{v}  ·  {CATEGORY_ES[CATEGORY_OF[v]]}" for v in top.index], top.values, color=TARGET_COLOR[t],
               height=0.62)  # fmt: skip
        a.set_title(TARGET_LABEL[t])
        a.set_xlabel("% de la importancia total (media de |SHAP|)")
        a.tick_params(axis="y", labelsize=7.5)
        a.grid(axis="x")
        a.grid(axis="y", visible=False)
        facts[f"top3_{t}"] = list(shares[t].nlargest(3).index)
    suptitle(fig, "Las 15 variables más importantes de cada modelo (Tabla 7 del paper)")
    fig.tight_layout()
    book.save(
        fig, "xai_01_importancia", SECTION, "Importancia global de las variables",
        "SHAP reparte cada predicción entre las variables: cuánto sube o baja el múltiplo estimado por el valor "
        "de cada una frente a la predicción media. La importancia global es la media de su valor absoluto, en % "
        f"del total. Para M/B mandan {', '.join(facts['top3_lnm2b'])}; para V/S, la eficiencia "
        f"({', '.join(facts['top3_lnv2s'][:2])}). Coincide con el paper en las primeras posiciones.",
    )  # fmt: skip


def _categories(data, shares, book, facts):
    t8 = data.table("table8_vs_paper")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    order = list(CATEGORIES)[::-1]
    y = np.arange(len(order))
    for a, t in zip(ax, TARGETS, strict=True):
        ours = shares[t].groupby(lambda v: CATEGORY_OF[v]).sum().reindex(order)
        paper = (
            t8[t8["model"] == f"ML_{t}"].set_index("category")["paper_shap"].reindex(order)
            if len(t8)
            else None
        )
        if paper is not None:
            a.hlines(y, ours, paper, color=GRID, linewidth=2)
            a.scatter(paper, y, color=MUTED, s=30, zorder=3, label="paper")
        a.scatter(
            ours, y, color=TARGET_COLOR[t], s=36, zorder=3, edgecolor=SURFACE, linewidth=1, label="nuestro"
        )
        a.set_title(TARGET_LABEL[t])
        a.set_xlabel("% de la importancia total")
        a.grid(axis="x")
        a.grid(axis="y", visible=False)
    ax[0].set_yticks(y, [CATEGORY_ES[c] for c in order])
    ax[0].legend(loc="lower right")
    suptitle(fig, "Importancia por categoría frente al paper (Tabla 8)")
    fig.tight_layout()
    if len(t8):
        rows = []
        for t in TARGETS:
            ours = shares[t].groupby(lambda v: CATEGORY_OF[v]).sum()
            paper = t8[t8["model"] == f"ML_{t}"].set_index("category")["paper_shap"]
            both = pd.concat([ours.rename("o"), paper.rename("p")], axis=1).dropna()
            rows.append(both)
        allc = pd.concat(rows)
        facts["category_corr"] = float(np.corrcoef(allc["o"], allc["p"])[0, 1])
    book.save(
        fig, "xai_02_categorias", SECTION, "Importancia por categoría",
        "La suma de la importancia de las variables de cada categoría. El reparto es muy parecido al del paper "
        f"(correlación {facts.get('category_corr', float('nan')):.2f} entre nuestras 33 cifras y las suyas). La "
        "diferencia más visible es el sector: pesa mucho menos que en el paper porque falta para el 27 % de "
        "nuestras empresas. En V/A la solvencia ocupa el segundo puesto que el paper da al crecimiento.",
    )  # fmt: skip


def _categories_time(yearly, book):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    colors = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4"]
    for a, t in zip(ax, TARGETS, strict=True):
        cat = yearly[t].T.groupby(lambda v: CATEGORY_OF[v]).sum().T
        top = cat.mean().nlargest(TOP_CATS).index
        for c, col in zip(top, colors, strict=False):
            a.plot(cat.index, cat[c], color=col, label=CATEGORY_ES[c])
        a.plot(cat.index, cat.drop(columns=top).sum(axis=1), color=MUTED, linestyle="--", label="Resto")
        a.set_title(TARGET_LABEL[t])
        a.legend(loc="upper right", fontsize=7)
        a.set_ylim(0)
    ax[0].set_ylabel("% de la importancia en el año")
    suptitle(fig, "Cómo cambia la importancia de las categorías con el tiempo (Tabla 9)")
    fig.tight_layout()
    book.save(
        fig, "xai_03_categorias_tiempo", SECTION, "Importancia por categoría a lo largo del tiempo",
        "Cada año se recalcula el reparto de la importancia. La rentabilidad domina M/B en todo el periodo; en "
        "V/S la eficiencia gana peso tras 2000. Que el reparto cambie con el tiempo justifica entrenar un "
        "modelo por mes: la forma de valorar del mercado no es fija.",
    )  # fmt: skip


def _beeswarm(shares, joined, book):
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 5.4), gridspec_kw={"wspace": 0.62})
    rng = np.random.default_rng(1)
    for a, t in zip(ax, TARGETS, strict=True):
        d = joined[t]
        top = list(shares[t].nlargest(10).index)[::-1]
        for i, f in enumerate(top):
            sv = d[f].to_numpy(float)
            xv = d[f + "_x"] if f + "_x" in d else d[f]
            pct = pd.Series(np.asarray(xv, float)).rank(pct=True).to_numpy()
            ok = np.isfinite(sv)
            miss = ok & np.isnan(pct)
            a.scatter(sv[miss], i + rng.normal(0, 0.12, miss.sum()), color="#b9b7b0", s=2.5, alpha=0.5,
                      linewidths=0, rasterized=True)  # fmt: skip
            have = ok & ~np.isnan(pct)
            sc = a.scatter(sv[have], i + rng.normal(0, 0.12, have.sum()), c=pct[have], cmap=DIV_CMAP, s=2.5,
                           alpha=0.5, vmin=0, vmax=1, linewidths=0, rasterized=True)  # fmt: skip
        a.axvline(0, color=INK2, linewidth=0.8)
        a.set_yticks(range(len(top)), top, fontsize=7.5)
        a.set_xlabel("valor SHAP (efecto sobre el log-múltiplo)")
        a.set_title(TARGET_LABEL[t])
        a.grid(axis="x")
        a.grid(axis="y", visible=False)
        lo, hi = np.nanpercentile(d[top].to_numpy(float), [0.5, 99.5])
        a.set_xlim(lo, hi)
    cb = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.8)
    cb.set_label("valor de la variable (percentil: azul bajo, rojo alto; gris = dato ausente)")
    suptitle(fig, "Dirección de los efectos: cada punto es una empresa-mes")
    book.save(
        fig, "xai_04_beeswarm", SECTION, "Dirección y forma de los efectos",
        "Para las 10 variables más importantes, cada punto es una empresa-mes: su posición horizontal es lo que "
        "esa variable suma o resta al log-múltiplo y el color, si la variable es alta (rojo) o baja (azul). En "
        "rentabilidad, rojo a la derecha: más rentable, más múltiplo. En V/S, la rotación de activos y el "
        "apalancamiento operativo van al revés (rojo a la izquierda), como encuentra el paper. Los puntos grises "
        "son empresas sin dato en esa variable: el GBM también aprende de la ausencia (en debt_capital_wr, las "
        "empresas sin dato reciben casi siempre un efecto negativo en V/A).",
    )  # fmt: skip


def _dependence(shares, joined, book):
    fig, ax = plt.subplots(3, 3, figsize=(13, 9.5))
    for r, t in enumerate(TARGETS):
        d = joined[t]
        for c, f in enumerate(list(shares[t].nlargest(3).index)):
            a = ax[r, c]
            x = pd.to_numeric(d[f + "_x"] if f + "_x" in d else d[f], errors="coerce")
            y = d[f].astype(float)
            ok = x.notna() & y.notna()
            lo, hi = x[ok].quantile([0.01, 0.99])
            m = ok & x.between(lo, hi)
            a.scatter(x[m], y[m], s=2, alpha=0.18, color=TARGET_COLOR[t], linewidths=0, rasterized=True)
            bins = pd.qcut(x[m], 20, duplicates="drop")
            med = y[m].groupby(bins, observed=True).median()
            a.plot([iv.mid for iv in med.index], med.values, color=INK2, linewidth=1.6)
            a.axhline(0, color=GRID, linewidth=0.8)
            a.set_title(f"{TARGET_LABEL[t]} · {f}", fontsize=9)
            a.set_xlabel("valor de la variable (p1–p99)")
            if c == 0:
                a.set_ylabel("SHAP")
    suptitle(fig, "Curvas de dependencia: cómo cambia el efecto con el valor de la variable")
    fig.tight_layout()
    book.save(
        fig, "xai_05_dependencia", SECTION, "Curvas de dependencia",
        "Las tres variables más importantes de cada modelo. Cada punto es una empresa-mes y la línea, la mediana "
        "por tramos. Las relaciones son claramente no lineales (saturan o cambian de pendiente alrededor de "
        "cero, por ejemplo en las pérdidas), que es lo que una regresión lineal como BL no puede captar y una "
        "de las razones de la ventaja del GBM.",
    )  # fmt: skip


def _industry(joined, book, facts):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    order = None
    for a, t in zip(ax, TARGETS, strict=True):
        d = joined[t]
        g = d.groupby(d["ff12_name"].map(FF12_ES).fillna("Sin clasificar"))["industry"].mean()
        if order is None:
            order = g.sort_values().index
        g = g.reindex(order)
        a.barh(g.index, g.values, color=[RED if v < 0 else BLUE for v in g.values], height=0.6)
        a.axvline(0, color=INK2, linewidth=0.8)
        a.set_title(TARGET_LABEL[t])
        a.set_xlabel("SHAP medio de la variable sector")
        a.grid(axis="x")
        a.grid(axis="y", visible=False)
    suptitle(fig, "Qué aporta el sector a la valoración")
    fig.tight_layout()
    book.save(
        fig, "xai_06_sector", SECTION, "Efecto del sector",
        "Media del SHAP de la variable sector (Fama-French 49, tratada como categórica) por sector Fama-French 12. "
        "Azul = el sector sube el múltiplo estimado; rojo = lo baja. Es la prima o el descuento sectorial que "
        "aprende el modelo por encima de lo que ya explican las cifras contables. Las empresas sin sector no "
        "reciben este efecto.",
    )  # fmt: skip


def _peers(data, book, facts):
    focus = {t: data.peers("focus", t) for t in TARGETS}
    top = {t: data.peers("top", t) for t in TARGETS}
    if all(f.empty for f in focus.values()):
        return
    firms = data.firms.set_index("id")
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    for t in TARGETS:
        f = focus[t].sort_values("rank")
        if f.empty:
            continue
        ax[0].plot(f["rank"], f["cumweight"], color=TARGET_COLOR[t], label=TARGET_LABEL[t])
    firm = focus["lnm2b"]["id"].iloc[0] if len(focus["lnm2b"]) else ""
    ax[0].set_xscale("log")
    ax[0].axhline(1, color=INK2, linewidth=0.8)
    ax[0].set_xlabel("número de comparables, ordenados por peso (log)")
    ax[0].set_ylabel("peso acumulado")
    ax[0].set_title(f"{firms['name'].get(firm, firm)}, diciembre de 2019")
    ax[0].legend(loc="lower right")
    conc, same = {}, {}
    for t in TARGETS:
        p = top[t]
        if p.empty:
            continue
        t10 = p[p["rank"] <= 10]
        conc[t] = t10.groupby("id")["weight"].sum()
        ind = firms["ff49_name"]
        t10 = t10.assign(own=t10["id"].map(ind), peer=t10["peer_id"].map(ind)).dropna(subset=["own"])
        obs = t10.assign(hit=t10["own"] == t10["peer"]).groupby("id")["hit"].mean()
        pool = ind[ind.index.isin(set(p["peer_id"]) | set(p["id"]))].dropna()
        freq = pool.value_counts(normalize=True)
        exp = t10.groupby("id")["own"].first().map(freq)
        same[t] = pd.DataFrame({"observado": obs, "aleatorio": exp}).dropna()
    for t in TARGETS:
        if t in conc:
            ax[1].hist(conc[t].clip(-0.5, 1.5), bins=40, histtype="step", color=TARGET_COLOR[t], linewidth=1.4,
                       label=TARGET_LABEL[t])  # fmt: skip
    ax[1].set_xlabel("peso de los 10 comparables principales")
    ax[1].set_ylabel("empresas")
    ax[1].set_title("Concentración de los pesos (todas las empresas)")
    labs = [TARGET_LABEL[t] for t in same]
    pos = np.arange(len(same))
    ax[2].bar(pos - 0.18, [same[t]["observado"].mean() * 100 for t in same], width=0.34, color=BLUE,
              label="entre los 10 comparables")  # fmt: skip
    ax[2].bar(pos + 0.18, [same[t]["aleatorio"].mean() * 100 for t in same], width=0.34, color=MUTED,
              label="esperado al azar")  # fmt: skip
    ax[2].set_xticks(pos, labs)
    ax[2].set_ylabel("% del mismo sector (FF49)")
    ax[2].set_title("¿Los comparables son del mismo sector?")
    ax[2].legend(loc="upper right", fontsize=7.5)
    suptitle(fig, "Pesos de comparables: la predicción del GBM es una media ponderada de empresas reales")
    fig.tight_layout()
    tests = {}
    for t, d in same.items():
        stat = wilcoxon(d["observado"], d["aleatorio"], alternative="greater")
        tests[t] = {"obs": float(d["observado"].mean()), "exp": float(d["aleatorio"].mean()),
                    "n": int(len(d)), "p": float(stat.pvalue)}  # fmt: skip
    facts["peers_same_sector"] = tests
    facts["peers_top10_median"] = {t: float(c.median()) for t, c in conc.items()}
    book.save(
        fig, "xai_07_comparables", SECTION, "Pesos de comparables",
        "Cada predicción del GBM se puede escribir exactamente como una media ponderada de los múltiplos de las "
        "empresas de entrenamiento (la descomposición del paper). Izquierda: para la empresa del caso de estudio, "
        "el peso acumulado sube por encima de 1 con los comparables de peso positivo y vuelve a 1 con los de "
        "peso negativo (el modelo también resta empresas: corrige hacia abajo con las que se parecen en lo que "
        "no se debe parecer). Centro: peso de los 10 principales para todas las empresas de "
        "diciembre de 2019 (mediana de los 10 primeros: "
        + ", ".join(f"{v:.2f}" for v in facts["peers_top10_median"].values())
        + "). Derecha: los 10 comparables principales son del mismo sector (FF49) en torno al 37–39 % de los "
        "casos, frente al 6,5 % que se esperaría al azar. Nadie le da al modelo una lista de comparables: los "
        "construye a partir de las 97 variables, y el sector es solo una de ellas.",
    )  # fmt: skip
