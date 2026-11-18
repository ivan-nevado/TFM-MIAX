"""Out-of-sample results: Table 2 against the paper, errors over time and by group, calibration,
reconciliation, robustness, portfolios and the diagnosis of the BL benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from relval.models.benchmarks.bl import BL_FEATURES, bl_features
from relval.reporting.visual.data import FF12_ES
from relval.reporting.visual.style import (
    BLUE,
    FAMILY_COLOR,
    GRID,
    INK2,
    MUTED,
    ORANGE,
    SEQ_CMAP,
    SURFACE,
    TARGET_COLOR,
    TARGET_LABEL,
    TARGETS,
    family,
    label_end,
    suptitle,
)
from relval.splits.blocks import add_blocks

SECTION = "resultados"
MULT = {"lnm2b": "m2b", "lnv2a": "v2a", "lnv2s": "v2s"}


def build(data, book) -> dict:
    facts: dict = {}
    _table2(data, book, facts)
    _monthly(data, book, facts)
    _decades(data, book)
    _distribution_and_calibration(data, book, facts)
    _groups(data, book, facts)
    _by_year(data, book)
    _reconciliation(data, book)
    _robustness(data, book, facts)
    _portfolios(data, book, facts)
    facts["bl_diagnosis"] = _bl_diagnosis(data, book)
    data.drop_cache()
    return facts


# ---- Table 2 --------------------------------------------------------------------------------
def _table2(data, book, facts):
    t = data.table("table2_vs_paper")
    if t.empty:
        return
    order = ["ML_lnm2b", "ML_lnv2a", "ML_lnv2s", "IND_m2b", "IND_v2a", "IND_v2s", "BL_m2b", "BL_v2a", "BL_v2s",
             "RRV", "BG"]  # fmt: skip
    t = t.set_index("model")
    t = t.reindex([m for m in order if m in t.index])[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    y = np.arange(len(t))
    for a, col, lab in ((ax[0], "med_abs_eps", "mediana de |ε| (menor es mejor)"),
                        (ax[1], "r2oos_y", "R² fuera de muestra del log-múltiplo (mayor es mejor)")):  # fmt: skip
        ours, paper = t[col].clip(-0.5, None), t[f"{col}_paper"]
        a.hlines(y, ours, paper, color=GRID, linewidth=2)
        a.scatter(paper, y, color=MUTED, s=34, zorder=3, label="paper")
        a.scatter(ours, y, color=[FAMILY_COLOR[family(m)] for m in t.index], s=40, zorder=3,
                  edgecolor=SURFACE, linewidth=1, label="nuestro")  # fmt: skip
        a.set_xlabel(lab)
        a.grid(axis="x")
        a.grid(axis="y", visible=False)
    ax[0].set_yticks(y, t.index)
    ax[1].set_xlim(-0.5, 1)
    ax[0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=MUTED, label="paper"),
                          plt.Line2D([], [], marker="o", ls="", color=BLUE, label="nuestro (color = familia)")],
                 loc="upper right")  # fmt: skip
    ax[1].annotate("BL: R² < −0,5 (recortado)", (-0.48, 4.35), fontsize=7.5, color=INK2)
    suptitle(fig, "Tabla 2: nuestra réplica frente al paper")
    fig.tight_layout()
    g = t.loc[[m for m in ("ML_lnm2b", "ML_lnv2a", "ML_lnv2s") if m in t.index]]
    facts["gbm_med_abs"] = g["med_abs_eps"].to_dict()
    facts["gbm_med_abs_paper"] = g["med_abs_eps_paper"].to_dict()
    book.save(
        fig, "res_01_tabla2", SECTION, "Tabla 2 frente al paper",
        "Cada fila es un modelo; gris = paper, color = nuestro. Los tres GBM (azul) caen a 0,01–0,02 del paper "
        "en mediana del error y en R². Los benchmarks tradicionales quedan en el mismo orden que en el paper. "
        "El R² de BL y del OLS en logaritmos se sale por la izquierda: unas pocas predicciones explotan "
        "(se diagnostica al final de esta sección). RRV parece muy bueno, pero valora solo una parte de la "
        "muestra; la comparación justa es la de la muestra común (contraste de hipótesis).",
    )  # fmt: skip


# ---- errors over time -----------------------------------------------------------------------
def _monthly(data, book, facts):
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 7), sharex=True)
    ax = ax.ravel()
    for a, t in zip(ax[:3], TARGETS, strict=True):
        m = MULT[t]
        for model, color in (
            (f"ML_{t}", BLUE),
            (f"IND_{m}", FAMILY_COLOR["IND"]),
            (f"BL_{m}", FAMILY_COLOR["BL"]),
        ):
            p = data.preds(model, None, ["date", "eps"])
            s = p.assign(a=p["eps"].abs()).groupby("date")["a"].median()
            a.plot(s.index, s.values, color=color, linewidth=0.7, alpha=0.35)
            r = s.rolling(12, min_periods=6).mean()
            a.plot(r.index, r.values, color=color, label=model.split("_")[0].replace("ML", "GBM"))
            label_end(a, r.index[-1], r.iloc[-1], f"{r.iloc[-1]:.2f}")
        a.set_title(f"Modelos que estiman {TARGET_LABEL[t]}")
        a.set_ylim(0)
        a.legend(loc="upper left", ncol=3)
    for t in TARGETS:
        p = data.preds(f"ML_{t}", None, ["date", "eps"])
        s = p.assign(a=p["eps"].abs()).groupby("date")["a"].median().rolling(12, min_periods=6).mean()
        ax[3].plot(s.index, s.values, color=TARGET_COLOR[t], label=f"GBM {TARGET_LABEL[t]}")
    for model in ("RRV", "BG"):
        p = data.preds(model, None, ["date", "eps"])
        s = p.assign(a=p["eps"].abs()).groupby("date")["a"].median().rolling(12, min_periods=6).mean()
        ax[3].plot(s.index, s.values, color=FAMILY_COLOR[model], label=model, linestyle="--")
    ax[3].set_title("Los tres GBM frente a RRV y BG")
    ax[3].set_ylim(0)
    ax[3].legend(loc="upper right", ncol=2)
    for a in ax:
        a.set_ylabel("mediana de |ε|")
    suptitle(
        fig,
        "Figura 1 del paper: error de valoración mes a mes (línea fina = mes; gruesa = media móvil 12 meses)",
    )
    fig.tight_layout()
    book.save(
        fig, "res_02_error_mensual", SECTION, "Error de valoración a lo largo del tiempo",
        "El GBM (azul) queda por debajo de IND y BL en prácticamente todos los meses y es muy estable "
        "(entre 0,25 y 0,30 casi todo el periodo). Los picos comunes (burbuja de 2000, crisis de 2008–2009) son "
        "meses en que los precios se separan de los fundamentales y todos los modelos fallan más; la regresión "
        "BL es la que más sufre. BG (regresión sobre partidas contables en dólares) queda muy lejos, como en el "
        "paper, y RRV compite con el GBM pero valora solo una parte de las empresas.",
    )  # fmt: skip


def _decades(data, book):
    t = data.table("table2_by_decade")
    if t.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    for a, tg in zip(ax, TARGETS, strict=True):
        m = MULT[tg]
        models = [f"ML_{tg}", f"IND_{m}", f"BL_{m}"]
        d = t[t["model"].isin(models)].pivot(index="period", columns="model", values="med_abs_eps")[models]
        pos = np.arange(len(d))
        for i, mdl in enumerate(models):
            a.bar(pos + (i - 1) * 0.26, d[mdl], width=0.24, color=FAMILY_COLOR[family(mdl)],
                  label=mdl.split("_")[0].replace("ML", "GBM"))  # fmt: skip
        a.set_xticks(pos, [f"{int(p)}s" for p in d.index])
        a.set_title(TARGET_LABEL[tg])
    ax[0].set_ylabel("mediana de |ε|")
    ax[0].legend(loc="upper left")
    suptitle(fig, "Error por década")
    fig.tight_layout()
    book.save(
        fig, "res_03_decadas", SECTION, "Error por década",
        "La ventaja del GBM sobre los benchmarks tradicionales se mantiene en las tres décadas. El paper tiene "
        "además los años 80, que nuestros datos no cubren.",
    )  # fmt: skip


# ---- error distribution and calibration -----------------------------------------------------------
def _distribution_and_calibration(data, book, facts):
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5), sharey=True)
    bins = np.linspace(-1, 2, 90)
    for a, t in zip(ax, TARGETS, strict=True):
        g = data.preds(f"ML_{t}", None, ["id", "date", "eps"])
        i = data.preds(f"IND_{MULT[t]}", None, ["id", "date", "eps"])
        both = g.merge(i, on=["id", "date"], suffixes=("_g", "_i"))
        a.hist(both["eps_i"].clip(-1, 2), bins=bins, color=FAMILY_COLOR["IND"], alpha=0.45, label="IND")
        a.hist(both["eps_g"].clip(-1, 2), bins=bins, histtype="step", color=BLUE, linewidth=1.5, label="GBM")
        a.axvline(0, color=INK2, linewidth=0.8)
        a.set_title(f"{TARGET_LABEL[t]} · mismas empresas")
        a.set_xlabel("ε = valor estimado / valor de mercado − 1")
    ax[0].set_ylabel("empresa-mes")
    ax[0].legend(loc="upper right")
    suptitle(fig, "Distribución del error de valoración: GBM frente a la mediana del sector")
    fig.tight_layout()
    book.save(
        fig, "res_04_distribucion_error", SECTION, "Distribución del error",
        "Sobre las mismas empresa-mes, el error del GBM está mucho más concentrado alrededor de 0. IND "
        "(asignar la mediana del sector) está sesgado a la baja: infravalora de forma sistemática, porque la "
        "mediana del sector ignora que las empresas grandes y rentables valen más. ε está acotado en −1 (valor "
        "estimado cero) y tiene una cola larga a la derecha, por eso el paper usa medianas.",
    )  # fmt: skip

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for a, t in zip(ax, TARGETS, strict=True):
        g = data.preds(f"ML_{t}", None, ["y", "yhat"])
        lo, hi = g["y"].quantile([0.005, 0.995])
        a.hexbin(g["yhat"].clip(lo, hi), g["y"].clip(lo, hi), gridsize=55, cmap=SEQ_CMAP, bins="log", mincnt=1,
                 linewidths=0)  # fmt: skip
        a.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.9, linestyle=":")
        r = np.corrcoef(g["yhat"], g["y"])[0, 1]
        a.set_title(f"{TARGET_LABEL[t]} · correlación {r:.2f}")
        a.set_xlabel("predicción del GBM")
        a.grid(False)
        facts[f"corr_{t}"] = float(r)
    ax[0].set_ylabel("valor real")
    suptitle(fig, "Calibración: predicción frente a valor real (densidad en escala log)")
    fig.tight_layout()
    book.save(
        fig, "res_05_calibracion", SECTION, "Predicción frente a valor real",
        "Cada celda cuenta empresa-mes (más oscuro = más). La nube sigue la diagonal, sin curvatura: el modelo "
        "no está descalibrado en ninguna zona. Es algo más plana que la diagonal en los extremos, el "
        "comportamiento normal de un modelo que promedia (hacia la media en los múltiplos muy altos o muy bajos).",
    )  # fmt: skip


# ---- errors by group -----------------------------------------------------------------------
def _quintile(s: pd.Series, by: pd.Series) -> pd.Series:
    return s.groupby(by).transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False) + 1)


def _groups(data, book, facts):
    s = data.sample[["id", "date", "M", "roe_wr", "lnm2b", "ff12_name"]].copy()
    s["size_q"] = _quintile(s["M"], s["date"])
    s["m2b_q"] = _quintile(s["lnm2b"], s["date"])
    s["roe_g"] = np.where(s["roe_wr"] < 0, 0, np.nan)
    pos = s["roe_wr"] >= 0
    s.loc[pos, "roe_g"] = _quintile(s.loc[pos, "roe_wr"], s.loc[pos, "date"])
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    sorts = (("size_q", "Tamaño (quintil de capitalización)", None),
             ("roe_g", "Rentabilidad (0 = pérdidas; 1–5 quintiles de ROE positivo)", None),
             ("m2b_q", "Quintil de M/B", None))  # fmt: skip
    res = {}
    for t in TARGETS:
        p = data.preds(f"ML_{t}", None, ["id", "date", "eps"]).merge(s, on=["id", "date"])
        p["a"] = p["eps"].abs()
        for a, (col, title, _) in zip(ax, sorts, strict=True):
            v = p.groupby(col)["a"].median()
            a.plot(v.index, v.values, marker="o", color=TARGET_COLOR[t], label=f"GBM {TARGET_LABEL[t]}")
            a.set_title(title, fontsize=9.5)
            a.set_xticks(sorted(v.index))
            res[(t, col)] = v
    p = data.preds("IND_m2b", None, ["id", "date", "eps"]).merge(s, on=["id", "date"])
    p["a"] = p["eps"].abs()
    for a, (col, _, _) in zip(ax, sorts, strict=True):
        v = p.groupby(col)["a"].median()
        a.plot(v.index, v.values, marker="o", color=MUTED, linestyle="--", label="IND M/B (referencia)")
    ax[0].set_ylabel("mediana de |ε|")
    ax[0].legend(loc="upper right", fontsize=7.5)
    for a in ax:
        a.set_ylim(0)
    suptitle(fig, "¿Dónde se equivoca más el modelo? (Figuras 2 y 3 del paper)")
    fig.tight_layout()
    facts["small_vs_large"] = {t: (float(res[(t, "size_q")].iloc[0]), float(res[(t, "size_q")].iloc[-1]))
                               for t in TARGETS}  # fmt: skip
    book.save(
        fig, "res_06_grupos", SECTION, "Error por tamaño, rentabilidad y M/B",
        "Como en el paper, el GBM se equivoca más con las empresas pequeñas (quintil 1: "
        f"{facts['small_vs_large']['lnm2b'][0]:.2f} frente a {facts['small_vs_large']['lnm2b'][1]:.2f} en el "
        "quintil 5 para M/B), con las que pierden dinero (grupo 0) y, sobre todo, con las de M/B más bajo "
        "(quintil 1). En todos los grupos queda por debajo de IND (gris). Son justo los grupos donde el plan del "
        "TFM espera que las variables cualitativas del LLM aporten más.",
    )  # fmt: skip

    fig, ax = plt.subplots(figsize=(9, 4.4))
    rows = []
    for t in TARGETS:
        p = data.preds(f"ML_{t}", None, ["id", "date", "eps"]).merge(s, on=["id", "date"])
        rows.append(p.assign(a=p["eps"].abs()).groupby("ff12_name")["a"].median().rename(t))
    d = pd.concat(rows, axis=1).rename(index=FF12_ES).sort_values("lnm2b")
    y = np.arange(len(d))
    for i, t in enumerate(TARGETS):
        ax.scatter(d[t], y + (i - 1) * 0.22, color=TARGET_COLOR[t], s=30, label=TARGET_LABEL[t], zorder=3)
    ax.set_yticks(y, d.index)
    ax.set_xlabel("mediana de |ε| del GBM")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    suptitle(fig, "Error del GBM por sector")
    fig.tight_layout()
    rank = d.mean(axis=1).sort_values()
    easy, hard = list(rank.index[:2]), list(rank.index[-2:][::-1])
    book.save(
        fig, "res_07_sectores", SECTION, "Error por sector",
        f"Los sectores más fáciles de valorar son {easy[0].lower()} y {easy[1].lower()} (negocios estables y "
        f"regulados, con múltiplos parecidos entre sí); los más difíciles, {hard[0].lower()} y "
        f"{hard[1].lower()}, seguidos de salud y tecnología, donde buena parte del valor está en intangibles, "
        "materias primas o crecimiento futuro que la contabilidad no recoge.",
    )  # fmt: skip


def _by_year(data, book):
    t = data.table("table2_by_year")
    if t.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.6))
    for tg in TARGETS:
        d = t[t["model"] == f"ML_{tg}"].sort_values("period")
        ax[0].plot(d["period"], d["r2oos_y"], color=TARGET_COLOR[tg], label=TARGET_LABEL[tg])
        ax[1].plot(d["period"], d["med_eps"], color=TARGET_COLOR[tg], label=TARGET_LABEL[tg])
    ax[0].set_title("R² fuera de muestra del log-múltiplo, por año")
    ax[0].set_ylim(0, 1)
    ax[1].axhline(0, color=INK2, linewidth=0.8)
    ax[1].set_title("Sesgo: mediana de ε por año")
    ax[0].legend(loc="lower right")
    suptitle(fig, "Estabilidad del GBM año a año")
    fig.tight_layout()
    book.save(
        fig, "res_08_por_anio", SECTION, "R² y sesgo por año",
        "El R² de M/B sube con los años (de ≈ 0,4 a ≈ 0,63, con más empresas por modelo); el de V/A y V/S se "
        "mantiene en ≈ 0,75–0,8, con un bache en 1997–1999. El sesgo se queda dentro de ±0,03 todos los años: el "
        "modelo no infravalora ni sobrevalora de forma sistemática en ningún periodo.",
    )  # fmt: skip


# ---- reconciliation and robustness ---------------------------------------------------------------
def _reconciliation(data, book):
    t = data.table("table4_vs_paper")
    if t.empty:
        return
    steps = ["OLS niveles", "OLS logs", "GBM 10 var.", "GBM 97 var.", "LASSO (completos)", "GBM (completos)"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for a, (panel, g) in zip(ax, t.groupby("panel", sort=False), strict=False):
        x = np.arange(len(g))
        a.plot(x[:4], g["med_abs_eps"].values[:4], marker="o", color=BLUE, label="nuestro")
        a.plot(
            x[:4], g["med_abs_eps_paper"].values[:4], marker="o", color=MUTED, linestyle="--", label="paper"
        )
        a.plot(x[4:], g["med_abs_eps"].values[4:], marker="o", color=BLUE)
        a.plot(x[4:], g["med_abs_eps_paper"].values[4:], marker="o", color=MUTED, linestyle="--")
        a.set_xticks(x, steps[: len(g)], rotation=35, ha="right", fontsize=8)
        a.set_title(f"Panel {panel}")
    ax[0].set_ylabel("mediana de |ε|")
    ax[0].legend(loc="upper right")
    suptitle(fig, "Tabla 4: de dónde viene la mejora (cada paso cambia una sola cosa)")
    fig.tight_layout()
    book.save(
        fig, "res_09_reconciliacion", SECTION, "Reconciliación paso a paso",
        "De izquierda a derecha: regresión lineal en niveles con las 10 variables de Bhojraj y Lee → la misma en "
        "logaritmos → un GBM con esas 10 variables → el GBM con las 97 (mismas empresas en los cuatro pasos). "
        "Los dos puntos sueltos comparan LASSO y GBM en las empresas sin ningún dato ausente. Reproducimos el "
        "orden del paper: los logaritmos y el GBM aportan la mayor parte de la mejora, las 97 variables "
        "aportan sobre todo en V/S.",
    )  # fmt: skip


def _robustness(data, book, facts):
    t = data.table("ia6_vs_paper")
    sens = data.table("sensitivity_runs")
    if t.empty or sens.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    groups = [
        ("CART", "un solo árbol"),
        ("ML8020", "80:20 sin validación"),
        ("GBMALT", "GBM sobre el múltiplo"),
    ]
    rows = []
    for tg in TARGETS:
        base = t.loc[t["model"] == f"ML_{tg}", "med_abs_eps"].iloc[0]
        for pre, lab in groups:
            name = f"{pre}_{tg}" if pre != "GBMALT" else f"GBMALT_{MULT[tg]}"
            r = t[t["model"] == name]
            if len(r):
                rows.append({"target": tg, "variante": lab, "dif": r["med_abs_eps"].iloc[0] - base,
                             "dif_paper": r["med_abs_eps_paper"].iloc[0] - 0.0})  # fmt: skip
    r = pd.DataFrame(rows)
    labs = [g[1] for g in groups]
    y = np.arange(len(labs))
    for i, tg in enumerate(TARGETS):
        g = r[r.target == tg].set_index("variante").reindex(labs)
        ax[0].barh(y + (i - 1) * 0.26, g["dif"], height=0.24, color=TARGET_COLOR[tg], label=TARGET_LABEL[tg])
    ax[0].set_yticks(y, labs)
    ax[0].axvline(0, color=INK2, linewidth=0.8)
    ax[0].set_xlabel("diferencia de mediana de |ε| frente al GBM base (positivo = peor)")
    ax[0].set_title("Robustez del método (apéndice IA.6)")
    ax[0].legend(loc="lower right")
    ax[0].grid(axis="x")
    ax[0].grid(axis="y", visible=False)
    sv = sens[sens["variant"] != "base"]
    labs2 = list(dict.fromkeys(sv["variant"]))
    names_es = {"industry as numeric": "sector numérico", "listed-today firms only": "solo cotizadas hoy",
                "beta min 24 months": "beta con 24 meses", "beta min 36 months": "beta con 36 meses",
                "beta min 48 months": "beta con 48 meses"}  # fmt: skip
    y2 = np.arange(len(labs2))
    for i, tg in enumerate(TARGETS):
        g = sv[sv["target"] == tg].set_index("variant").reindex(labs2)
        ax[1].barh(
            y2 + (i - 1) * 0.26, g["med_abs_eps_minus_base"] * 1000, height=0.24, color=TARGET_COLOR[tg]
        )
    ax[1].set_yticks(y2, [names_es.get(v, v) for v in labs2])
    ax[1].axvline(0, color=INK2, linewidth=0.8)
    ax[1].set_xlabel("diferencia frente al base (milésimas de |ε|)")
    ax[1].set_title("Sensibilidad a decisiones de datos")
    ax[1].grid(axis="x")
    ax[1].grid(axis="y", visible=False)
    suptitle(fig, "Robustez y sensibilidad")
    fig.tight_layout()
    facts["sens_max_abs"] = float(sv["med_abs_eps_minus_base"].abs().max())
    book.save(
        fig, "res_10_robustez", SECTION, "Robustez y sensibilidad",
        "Izquierda (escala en |ε|): un solo árbol, entrenar sin validación con 20 árboles fijos o predecir el "
        "múltiplo sin logaritmo empeoran el error, como en el paper; confirma que cada pieza del método aporta. "
        f"Derecha (escala en milésimas): las decisiones de datos que no fija el paper mueven el error como mucho "
        f"{facts['sens_max_abs'] * 1000:.1f} milésimas: los resultados no dependen de ellas.",
    )  # fmt: skip


# ---- portfolios -----------------------------------------------------------------------------
def _portfolios(data, book, facts):
    vw = data.table("portfolio_returns_vw")
    svw, sew = data.table("portfolio_stats_vw"), data.table("portfolio_stats_ew")
    if vw.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1.4, 1]})
    vw["date"] = pd.to_datetime(vw["date"])
    for t in TARGETS:
        r = vw.set_index("date")[f"ML_{t}"].fillna(0)
        ax[0].plot(r.index, (1 + r).cumprod(), color=TARGET_COLOR[t], label=f"GBM {TARGET_LABEL[t]}")
    ax[0].axhline(1, color=INK2, linewidth=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("valor de 1 $ invertido (log)")
    ax[0].set_title("Long-short ponderada por valor: comprar infravaloradas, vender sobrevaloradas")
    ax[0].legend(loc="upper left")
    models = [f"ML_{t}" for t in TARGETS] + ["IND_m2b", "BL_m2b", "RRV", "BG"]
    y = np.arange(len(models))
    for i, (st, lab, color) in enumerate(
        ((svw, "ponderada por valor", BLUE), (sew, "equiponderada", ORANGE))
    ):
        s = st.set_index("model").reindex(models)
        se = s["mean"] / s["mean_t"]
        ax[1].errorbar(s["mean"], y + (i - 0.5) * 0.3, xerr=1.96 * se, fmt="o", color=color, ms=5,
                       elinewidth=1.2, capsize=0, label=lab)  # fmt: skip
    ax[1].axvline(0, color=INK2, linewidth=0.8)
    ax[1].set_yticks(y, models)
    ax[1].set_xlabel("rentabilidad media mensual (%) e IC 95 %")
    ax[1].set_title("Rentabilidad media de la cartera long-short")
    ax[1].legend(loc="lower right", fontsize=7.5)
    ax[1].grid(axis="x")
    ax[1].grid(axis="y", visible=False)
    suptitle(fig, "¿El error de valoración predice rentabilidades? (Tabla 3 del paper)")
    fig.tight_layout()
    facts["ls_vw"] = svw.set_index("model")[["mean", "mean_p", "alpha", "alpha_p"]].to_dict("index")
    facts["ls_ew"] = sew.set_index("model")[["mean", "mean_p", "alpha", "alpha_p"]].to_dict("index")
    book.save(
        fig, "res_11_carteras", SECTION, "Carteras long-short",
        "Cada mes se compran las empresas que el modelo ve más infravaloradas (quintil de ε más bajo) y se venden "
        "las más sobrevaloradas. Con los GBM la cartera gana dinero de media, pero en la versión ponderada por "
        "valor el intervalo cruza el cero (no significativo); en la equiponderada sí es significativo. En el "
        "paper ambas lo son. La explicación más probable es la muestra de supervivientes: faltan justo las "
        "empresas sobrevaloradas que acabaron quebrando, que son las que hacen ganar al lado corto.",
    )  # fmt: skip


# ---- BL diagnosis ---------------------------------------------------------------------------
def _bl_diagnosis(data, book) -> pd.DataFrame:
    g = data.preds("BL_m2b", None, ["id", "date", "block", "y", "yhat"])
    if g.empty:
        return pd.DataFrame()
    g["err"] = (g["yhat"] - g["y"]).abs()
    worst = g.nlargest(8, "err")
    cfg = data.cfg.data
    rows = []
    for _, w in worst.iterrows():
        month = data.sample[data.sample["date"] == w["date"]]
        df = add_blocks(month, data.cfg.headline_seed, int(cfg["splits"]["n_blocks"]))
        fit = df["block"].to_numpy() != w["block"]
        feats = bl_features(df, fit, int(cfg["benchmarks"]["ind_min_firms"]))
        row = feats[df["id"] == w["id"]].iloc[0]
        ref = feats[fit].dropna()
        out = {
            "empresa": w["id"],
            "mes": w["date"].strftime("%Y-%m"),
            "real": w["y"],
            "predicción": w["yhat"],
        }
        worst_var, worst_ratio = None, 0.0
        for f in BL_FEATURES:
            lo, hi = ref[f].quantile([0.01, 0.99])
            span = hi - lo if hi > lo else 1.0
            dist = max(lo - row[f], row[f] - hi, 0) / span
            if dist > worst_ratio:
                worst_var, worst_ratio = f, dist
        out["variable extrema"] = worst_var
        out["valor"] = row[worst_var] if worst_var else np.nan
        out["rango p1–p99 del ajuste"] = (f"{ref[worst_var].quantile(0.01):.2f} a {ref[worst_var].quantile(0.99):.2f}"
                                         if worst_var else "")  # fmt: skip
        rows.append(out)
    diag = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    for model, color in (("BL_m2b", FAMILY_COLOR["BL"]), ("OLSLOG_BL_lnm2b", "#c98500"), ("ML_lnm2b", BLUE)):
        p = data.preds(model, None, ["y", "yhat"])
        e = np.sort((p["yhat"] - p["y"]).abs().to_numpy())[::-1]
        ax.plot(np.arange(1, len(e) + 1), e, color=color, label=model)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("posición de la predicción, de peor a mejor (log)")
    ax.set_ylabel("|predicción − real| (log)")
    ax.legend(loc="upper right")
    suptitle(fig, "Diagnóstico de BL: unas pocas predicciones explotan")
    fig.tight_layout()
    book.save(
        fig, "res_12_bl_diagnostico", SECTION, "Diagnóstico del benchmark BL",
        "Errores ordenados de mayor a menor. Las regresiones lineales con las variables de Bhojraj y Lee tienen "
        "unas pocas decenas de predicciones con errores de cientos o miles de veces el múltiplo; el GBM no, "
        "porque un árbol nunca predice fuera del rango que ha visto. La tabla de abajo muestra que en esas filas "
        "alguna variable de entrada está muy fuera del rango del resto del mes: la regresión extrapola. La "
        "mediana del error no se ve afectada, pero sí el RMSE y el R².",
    )  # fmt: skip
    return diag
