"""What was trained and how it converged.

* Inventory: every (model, seed) with the number of month-block fits behind it.
* Early stopping: the number of trees chosen on the validation block, by target, month and seed.
* Learning curves: RMSE on train, validation and test as trees are added, rebuilt from the stored
  models of a sample of months (the stored model keeps the trees up to the chosen one, so the curve
  ends at the stopping point). No model is refitted: the stored boosters are only evaluated.
* Choices of LASSO (penalty) and CART (depth) on the validation block.
* Seeds: how much the result moves with the random allocation of firms to blocks.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from relval.models.runner import gbm_inputs, sample_mask
from relval.models.specs import BASE_MODELS, SPECS
from relval.reporting.visual.style import (
    BLUE,
    FAMILY_COLOR,
    GRID,
    INK2,
    MUTED,
    ORANGE,
    SPLIT_COLOR,
    SURFACE,
    TARGET_COLOR,
    TARGET_LABEL,
    TARGETS,
    suptitle,
)
from relval.splits.blocks import add_blocks, roles
from relval.targets.multiples import target_values

SECTION = "convergencia"
PAPER_TRAIN_ROWS = 2265


def build(data, book) -> dict:
    facts: dict = {}
    facts["inventory"] = inventory(data)
    _block_diagram(book)
    _trees(data, book, facts)
    _learning_curves(data, book, facts)
    _lasso_cart(data, book, facts)
    _seeds(data, book, facts)
    return facts


# ---- inventory ------------------------------------------------------------------------------
def inventory(data) -> pd.DataFrame:
    rows = []
    for m in data.models_available():
        spec = SPECS[m]
        for seed in data.seeds_available(m):
            p = data.preds(m, seed, ["date", "block"])
            if p.empty:
                continue
            fits = p.drop_duplicates(["date", "block"])
            rows.append({
                "modelo": m, "grupo": spec.group, "tipo": spec.kind, "target": spec.target, "semilla": seed,
                "meses": int(p["date"].nunique()), "ajustes (mes × bloque)": int(len(fits)), "predicciones": int(len(p)),
            })  # fmt: skip
    data.drop_cache()
    return pd.DataFrame(rows)


# ---- how the five blocks rotate --------------------------------------------------------------
def _block_diagram(book):
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    colors = {"train": "#cfcdc6", "validación": BLUE, "test": ORANGE}
    for k in range(5):
        tr, va, te = roles(np.arange(5), k)
        for b in range(5):
            role = "test" if te[b] else ("validación" if va[b] else "train")
            ax.add_patch(plt.Rectangle((b + 0.04, 4 - k + 0.08), 0.92, 0.84, color=colors[role], linewidth=0))
            ax.text(b + 0.5, 4 - k + 0.5, {"train": "train", "validación": "valid.", "test": "test"}[role],
                    ha="center", va="center", fontsize=8, color="white" if role != "train" else INK2)  # fmt: skip
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks(np.arange(5) + 0.5, [f"bloque {b + 1}" for b in range(5)])
    ax.set_yticks(np.arange(5) + 0.5, [f"modelo {5 - k}" for k in range(5)])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    suptitle(fig, "Cada mes: 5 bloques, 5 modelos (60 % train, 20 % validación, 20 % test)")
    fig.tight_layout()
    book.save(
        fig, "conv_01_bloques", SECTION, "Cómo se reparten los datos cada mes",
        "Cada mes las empresas se reparten al azar en 5 bloques y se entrenan 5 modelos. Cada bloque es test "
        "una vez; el siguiente hace de validación y los otros tres son el train. La validación solo decide "
        "cuántos árboles tiene el modelo (parada anticipada); el test solo se predice. Así cada empresa recibe "
        "exactamente una predicción fuera de muestra, hecha por un modelo que nunca la vio.",
    )  # fmt: skip


# ---- number of trees chosen on validation -----------------------------------------------------
def _fits(data, model, seed=None):
    p = data.preds(model, seed, ["date", "block", "best_iteration", "n_train"])
    return p.drop_duplicates(["date", "block"])


def _trees(data, book, facts):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    bins = np.logspace(np.log10(10), np.log10(3000), 45)
    stats = {}
    for m, t in zip(BASE_MODELS, TARGETS, strict=True):
        allseeds = pd.concat([_fits(data, m, s) for s in data.seeds_available(m)], ignore_index=True)
        b = allseeds["best_iteration"].dropna()
        stats[t] = {"median": float(b.median()), "p90": float(b.quantile(0.9)), "max": float(b.max()),
                    "fits": int(len(b)), "capped": int((b > 10_000 - 50).sum())}  # fmt: skip
        ax[0].hist(b, bins=bins, histtype="step", color=TARGET_COLOR[t], linewidth=1.6,
                   label=f"{TARGET_LABEL[t]} · mediana {b.median():.0f}")  # fmt: skip
        h = _fits(data, m)
        q = h.groupby(h["date"].dt.year)["best_iteration"].quantile([0.25, 0.5, 0.75]).unstack()
        ax[1].fill_between(q.index, q[0.25], q[0.75], color=TARGET_COLOR[t], alpha=0.12, linewidth=0)
        ax[1].plot(q.index, q[0.5], color=TARGET_COLOR[t], label=TARGET_LABEL[t])
    ax[0].set_xscale("log")
    ax[0].set_xlabel("árboles elegidos por la validación (escala log)")
    ax[0].set_ylabel("modelos")
    ax[0].set_title("Todos los modelos de las 5 semillas")
    ax[0].legend()
    ax[1].set_ylabel("árboles (mediana y p25–p75)")
    ax[1].set_title("Por año (semilla principal)")
    ax[1].legend(loc="upper left")
    suptitle(fig, "Convergencia: número de árboles que elige la parada anticipada")
    fig.tight_layout()
    facts["trees"] = stats
    total = sum(v["fits"] for v in stats.values())
    capped = sum(v["capped"] for v in stats.values())
    book.save(
        fig, "conv_02_arboles", SECTION, "Árboles elegidos por la validación",
        f"Cada uno de los {total:,} modelos GBM base (3 targets × 360 meses × 5 bloques × 5 semillas) añade "
        "árboles mientras el error en su bloque de validación baja y para cuando lleva 50 árboles sin mejorar. "
        f"Mediana: {stats['lnm2b']['median']:.0f} / {stats['lnv2a']['median']:.0f} / {stats['lnv2s']['median']:.0f} "
        f"árboles para M/B, V/A y V/S. {total - capped:,} modelos ({(1 - capped / total) * 100:.2f} %) pararon "
        f"solos; {capped} llegaron al tope de 10.000 árboles sin que la validación dejara de mejorar (casi todos "
        "de V/S y de los años 90, con pocas empresas): en ellos el error de validación sigue bajando muy despacio "
        "y el tope no cambia el resultado de forma apreciable. Se comenta en las mejoras propuestas.",
    )  # fmt: skip

    fig, ax = plt.subplots(figsize=(10, 3.4))
    for m, t in zip(BASE_MODELS, TARGETS, strict=True):
        h = _fits(data, m)
        n = h.groupby("date")["n_train"].median()
        ax.plot(n.index, n.values, color=TARGET_COLOR[t], label=TARGET_LABEL[t], linewidth=1.2)
    ax.axhline(PAPER_TRAIN_ROWS, color=INK2, linestyle=":", linewidth=1)
    ax.annotate(f"paper ≈ {PAPER_TRAIN_ROWS:,}", (ax.get_xlim()[0], PAPER_TRAIN_ROWS), xytext=(4, 4),
                textcoords="offset points", fontsize=8, color=INK2)  # fmt: skip
    ax.set_ylabel("filas de entrenamiento por modelo")
    ax.set_ylim(0)
    ax.legend(loc="upper left")
    suptitle(fig, "Datos con los que aprende cada modelo mensual")
    fig.tight_layout()
    book.save(
        fig, "conv_03_filas_train", SECTION, "Filas de entrenamiento por modelo",
        "Cada modelo aprende con el 60 % de las empresas de su mes. Con datos de supervivientes son unas 300 en "
        "1990 y unas 1.200 en 2019, frente a unas 2.265 de media en el paper: menos datos por modelo es la razón "
        "principal de que nuestros errores de los años 90 sean mayores.",
    )  # fmt: skip


# ---- learning curves from the stored models ---------------------------------------------------
def _block_data(data, spec, month: pd.DataFrame, seed: int, k: int):
    cfg = data.cfg.data
    df = add_blocks(month, seed, int(cfg["splits"]["n_blocks"]))
    y = target_values(df, spec.target).to_numpy(float)
    train, val, test = roles(df["block"].to_numpy(), k, int(cfg["splits"]["n_blocks"]))
    Xdf, _, _ = gbm_inputs(spec, df, ~test, cfg)
    mask = ~np.isnan(y) & sample_mask(spec, df, ~test, cfg)
    X = Xdf.to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    return {
        name: (X[r & mask], y[r & mask])
        for name, r in (("train", train), ("validación", val), ("test", test))
    }


def _staged_rmse(booster: lgb.Booster, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = booster.num_trees()
    contrib = np.column_stack([booster.predict(X, start_iteration=i, num_iteration=1) for i in range(n)])
    pred = np.cumsum(contrib, axis=1)
    return np.sqrt(np.mean((pred - y[:, None]) ** 2, axis=0))


def learning_curves(data, months: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = data.cfg.headline_seed
    sample = data.sample
    curves, ends = [], []
    for m, t in zip(BASE_MODELS, TARGETS, strict=True):
        models = data.saved_models(m)
        models["date"] = pd.to_datetime(models["date"])
        stored = data.preds(m, None, ["id", "date", "block", "yhat"])
        for date in months:
            month = sample[sample["date"] == date]
            for _, row in models[models["date"] == date].iterrows():
                booster = lgb.Booster(model_str=row["model_text"])
                sets = _block_data(data, SPECS[m], month, seed, int(row["block"]))
                # sanity check: the rebuilt test set reproduces the stored predictions exactly
                st = stored[(stored["date"] == date) & (stored["block"] == row["block"])]
                Xte = sets["test"][0]
                diff = float(np.max(np.abs(np.sort(booster.predict(Xte)) - np.sort(st["yhat"].to_numpy())))) \
                    if len(st) == len(Xte) else np.nan  # fmt: skip
                for name, (X, y) in sets.items():
                    r = _staged_rmse(booster, X, y)
                    curves.append(pd.DataFrame({"target": t, "date": date, "block": int(row["block"]), "set": name,
                                                "tree": np.arange(1, len(r) + 1), "rmse": r}))  # fmt: skip
                    ends.append({"target": t, "date": date, "block": int(row["block"]), "set": name,
                                 "rmse_first": float(r[0]), "rmse_best": float(r[-1]), "trees": len(r),
                                 "rmse_mean_only": float(np.sqrt(np.mean((y - sets["train"][1].mean()) ** 2))),
                                 "max_diff_stored": diff})  # fmt: skip
        data.drop_cache()
    return pd.concat(curves, ignore_index=True), pd.DataFrame(ends)


def _learning_curves(data, book, facts):
    dates = sorted(data.sample["date"].unique())
    pick = [d for d in dates if pd.Timestamp(d).month == 12 and (pd.Timestamp(d).year - 1990) % 3 == 0]
    pick = sorted(set(pick) | {dates[-1]})
    curves, ends = learning_curves(data, [pd.Timestamp(d) for d in pick])
    facts["curve_months"] = [pd.Timestamp(d).strftime("%Y-%m") for d in pick]
    facts["curve_models"] = int(ends.drop_duplicates(["target", "date", "block"]).shape[0])
    facts["curve_max_diff"] = float(ends["max_diff_stored"].max())

    last = pd.Timestamp(dates[-1])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for a, t in zip(ax, TARGETS, strict=True):
        c = curves[(curves.target == t) & (curves.date == last)]
        for (_blk, name), g in c.groupby(["block", "set"]):
            a.plot(g["tree"], g["rmse"], color=SPLIT_COLOR[name], linewidth=1.0, alpha=0.85)
            a.scatter(g["tree"].iloc[-1], g["rmse"].iloc[-1], color=SPLIT_COLOR[name], s=14, zorder=3)
        a.set_title(f"{TARGET_LABEL[t]} · diciembre de {last.year}, 5 bloques")
        a.set_xlabel("número de árboles")
    ax[0].set_ylabel("RMSE del target (log)")
    ax[0].legend(handles=[Patch(color=SPLIT_COLOR[k], label=k) for k in SPLIT_COLOR], loc="upper right")
    suptitle(fig, "Curvas de aprendizaje: el error baja al añadir árboles; la validación decide dónde parar")
    fig.tight_layout()
    book.save(
        fig, "conv_04_curvas", SECTION, "Curvas de aprendizaje de un mes",
        "Error (RMSE en logaritmos) de los 5 modelos de diciembre de 2019 según se añaden árboles. El de train "
        "baja siempre; el de validación baja deprisa y se aplana, y el modelo se corta en su mínimo (punto). "
        "El test, que no interviene, sigue a la validación: la elección hecha en validación generaliza. Las "
        "curvas se reconstruyen evaluando los modelos guardados, sin reentrenar; por eso terminan en el árbol "
        "elegido (los 50 árboles de paciencia posteriores no se guardaron).",
    )  # fmt: skip

    e = ends.pivot_table(index=["target", "date", "block"], columns="set", values="rmse_best").reset_index()
    base = ends[ends.set == "test"].set_index(["target", "date", "block"])["rmse_mean_only"]
    e = e.join(base, on=["target", "date", "block"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for t in TARGETS:
        g = e[e.target == t]
        ax[0].scatter(g["validación"], g["test"], s=16, color=TARGET_COLOR[t], alpha=0.75, label=TARGET_LABEL[t],
                      edgecolor=SURFACE, linewidth=0.4)  # fmt: skip
    lo, hi = e[["validación", "test"]].min().min() * 0.9, e[["validación", "test"]].max().max() * 1.05
    ax[0].plot([lo, hi], [lo, hi], color=INK2, linewidth=0.8, linestyle=":")
    ax[0].set_xlabel("RMSE en validación (árbol elegido)")
    ax[0].set_ylabel("RMSE en test")
    ax[0].set_title("¿Predice la validación el error de test?")
    ax[0].legend(loc="upper left")
    gain = pd.DataFrame({
        "train": 1 - e["train"] / e["rmse_mean_only"], "validación": 1 - e["validación"] / e["rmse_mean_only"],
        "test": 1 - e["test"] / e["rmse_mean_only"], "target": e["target"],
    })  # fmt: skip
    pos = np.arange(3)
    for i, t in enumerate(TARGETS):
        g = gain[gain.target == t]
        means = [g[s].median() * 100 for s in ("train", "validación", "test")]
        ax[1].bar(pos + (i - 1) * 0.26, means, width=0.24, color=TARGET_COLOR[t], label=TARGET_LABEL[t])
    ax[1].set_xticks(pos, ["train", "validación", "test"])
    ax[1].set_ylabel("% de reducción del RMSE frente a predecir la media")
    ax[1].set_title("Mejora sobre un modelo que predice la media")
    ax[1].legend(loc="upper right")
    suptitle(fig, f"Convergencia en {facts['curve_models']} modelos de {len(pick)} meses (1990–2019)")
    fig.tight_layout()
    gap = (e["test"] - e["validación"]) / e["validación"] * 100
    corr = float(np.corrcoef(e["validación"], e["test"])[0, 1])
    facts["val_test_corr"], facts["val_test_gap_median"] = corr, float(gap.median())
    facts["gain_median"] = {s: float(gain[s].median() * 100) for s in ("train", "validación", "test")}
    book.save(
        fig, "conv_05_validacion_test", SECTION, "La validación como guía del test",
        f"Cada punto es un modelo (mes × bloque × target). La correlación entre el error de validación y el de "
        f"test es {corr:.2f} y la diferencia mediana es {facts['val_test_gap_median']:+.1f} %: el error que se "
        "usa para decidir es un estimador casi insesgado del error real. A la derecha, la mejora frente a "
        f"predecir la media de train: {facts['gain_median']['train']:.0f} % en train frente a "
        f"{facts['gain_median']['test']:.0f} % en test. Los árboles memorizan buena parte del train (con unos "
        "cientos de empresas por modelo es inevitable); por eso es imprescindible parar con la validación, que "
        "da prácticamente la misma mejora que el test.",
    )  # fmt: skip


# ---- LASSO and CART choices -------------------------------------------------------------------
def _lasso_cart(data, book, facts):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for t in TARGETS:
        p = data.preds(f"LASSO_NM_{t}", None, ["date", "block", "best_iteration"]).drop_duplicates(
            ["date", "block"]
        )
        if p.empty:
            continue
        a = p.groupby(p["date"].dt.year)["best_iteration"].median()
        ax[0].plot(a.index, a.values, color=TARGET_COLOR[t], label=TARGET_LABEL[t])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("penalización α elegida (mediana, log)")
    ax[0].set_title("LASSO: penalización elegida en validación")
    ax[0].legend()
    depths = []
    for t in TARGETS:
        p = data.preds(f"CART_{t}", None, ["date", "block", "best_iteration"]).drop_duplicates(
            ["date", "block"]
        )
        if p.empty:
            continue
        depths.append(p["best_iteration"].value_counts(normalize=True).rename(t) * 100)
    if depths:
        d = pd.concat(depths, axis=1).fillna(0).sort_index()
        pos = np.arange(len(d))
        for i, t in enumerate(d.columns):
            ax[1].bar(pos + (i - 1) * 0.26, d[t], width=0.24, color=TARGET_COLOR[t], label=TARGET_LABEL[t])
        ax[1].set_xticks(pos, [f"{int(x)}" for x in d.index])
        ax[1].set_xlabel("profundidad del árbol")
        ax[1].set_ylabel("% de los modelos")
        ax[1].set_title("CART: profundidad elegida en validación")
        facts["cart_depth"] = d.idxmax().to_dict()
    suptitle(fig, "Otras decisiones tomadas con el bloque de validación")
    fig.tight_layout()
    data.drop_cache()
    book.save(
        fig, "conv_06_lasso_cart", SECTION, "Hiperparámetros de LASSO y CART",
        "Igual que el número de árboles del GBM, la penalización del LASSO (50 valores) y la profundidad del "
        "CART (3 a 6) se eligen en cada mes y bloque minimizando el error en el bloque de validación, nunca en "
        "el de test. La profundidad más elegida es "
        + ", ".join(f"{int(v)} para {TARGET_LABEL[k]}" for k, v in facts.get("cart_depth", {}).items())
        + ". La penalización del LASSO baja con los años: con más empresas por mes el modelo admite más "
        "variables sin sobreajustar.",
    )  # fmt: skip


# ---- seeds ----------------------------------------------------------------------------------
def seed_table(data) -> pd.DataFrame:
    rows = []
    for m, t in zip(BASE_MODELS, TARGETS, strict=True):
        for s in data.seeds_available(m):
            p = data.preds(m, s, ["date", "eps"])
            rows.append({"target": t, "seed": s, "med_abs_eps": float(p["eps"].abs().median()),
                         "med_eps": float(p["eps"].median())})  # fmt: skip
        data.drop_cache()
    return pd.DataFrame(rows)


def _seeds(data, book, facts):
    st = seed_table(data)
    facts["seeds"] = st
    paper = {"lnm2b": 0.29, "lnv2a": 0.30, "lnv2s": 0.32}
    fig, ax = plt.subplots(figsize=(9, 3.0))
    for i, t in enumerate(TARGETS):
        g = st[st.target == t]
        ax.scatter(g["med_abs_eps"], np.full(len(g), i), color=TARGET_COLOR[t], s=46, zorder=3,
                   edgecolor=SURFACE, linewidth=1)  # fmt: skip
        ax.scatter([paper[t]], [i], marker="|", s=300, color=INK2, zorder=3)
        ax.annotate(f"rango {g.med_abs_eps.max() - g.med_abs_eps.min():.4f}", (g.med_abs_eps.max(), i),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)  # fmt: skip
    ax.set_yticks(range(3), [TARGET_LABEL[t] for t in TARGETS])
    ax.set_xlabel("mediana del error absoluto |ε|")
    ax.set_xlim(0.26, 0.33)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=MUTED, label="cada una de las 5 semillas"),
                       plt.Line2D([], [], marker="|", ls="", color=INK2, markersize=12, label="paper")],
              loc="lower right")  # fmt: skip
    suptitle(fig, "Las 5 semillas dan el mismo resultado")
    fig.tight_layout()
    book.save(
        fig, "conv_07_semillas", SECTION, "Robustez a la semilla",
        "Cada semilla cambia el reparto aleatorio de las empresas en los 5 bloques de cada mes. Los cinco "
        "resultados caen prácticamente en el mismo punto (rango menor de 0,001), muy por debajo de la distancia "
        "al valor del paper: la conclusión no depende de la suerte del reparto.",
    )  # fmt: skip
    _ = (GRID, FAMILY_COLOR, MUTED)
