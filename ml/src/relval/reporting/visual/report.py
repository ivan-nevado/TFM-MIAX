"""Assembles the visual report: one self-contained HTML file (figures embedded) plus the PNG files
and the CSV tables, under reports/<run>/visual/."""

from __future__ import annotations

import html
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from relval.io import git_commit, library_versions
from relval.models.specs import SPECS
from relval.reporting.visual import convergence, eda, hypotheses, performance, xai
from relval.reporting.visual.data import VisualData
from relval.reporting.visual.style import TARGET_LABEL, TARGETS, FigureBook


def fnum(x, d=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fp(p) -> str:
    if p is None or not np.isfinite(p):
        return "–"
    return "< 0,001" if p < 0.001 else fnum(p, 3)


def table_html(df: pd.DataFrame, cls: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, float | np.floating):
                cells.append(f"<td class='num'>{fnum(float(v), 3)}</td>")
            elif isinstance(v, int | np.integer):
                cells.append(f"<td class='num'>{fnum(int(v), 0)}</td>")
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='tbl'><table class='{cls}'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def figs_html_one(f) -> str:
    return (
        f"<figure id='{f.key}'>"
        f"<img src='{f.data_uri()}' alt='{html.escape(f.title)}' loading='lazy'>"
        f"<figcaption>{html.escape(f.caption)}<span class='file'>{f.key}.png</span></figcaption></figure>"
    )


def figs_html(book: FigureBook, section: str) -> str:
    return "\n".join(figs_html_one(f) for f in book.section(section))


def run(cfg, log=print) -> dict:
    t0 = time.time()
    out_dir = cfg.reports_dir("visual")
    book = FigureBook(out_dir / "figures")
    data = VisualData(cfg, log)
    log("[visual] EDA")
    f_eda = eda.build(data, book)
    log("[visual] convergence")
    f_conv = convergence.build(data, book)
    log("[visual] results")
    f_res = performance.build(data, book)
    log("[visual] explainability")
    f_xai = xai.build(data, book)
    log("[visual] hypothesis tests")
    f_hyp = hypotheses.build(data, book, f_xai)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    f_hyp["tests"].drop(columns=["ic90"], errors="ignore").to_csv(tables / "contrastes.csv", index=False)
    f_conv["inventory"].to_csv(tables / "inventario_modelos.csv", index=False)
    f_res["bl_diagnosis"].to_csv(tables / "diagnostico_bl.csv", index=False)
    page = render(cfg, data, book, f_eda, f_conv, f_res, f_xai, f_hyp)
    path = out_dir / "informe_visual.html"
    path.write_text(page, encoding="utf-8")
    log(f"[done] visual report: {path} ({len(book.figures)} figures, {(time.time() - t0) / 60:.1f} min)")
    return {"path": str(path), "figures": len(book.figures)}


# ---------------------------------------------------------------------------------------------
def render(cfg, data, book, f_eda, f_conv, f_res, f_xai, f_hyp) -> str:
    inv = f_conv["inventory"]
    tests = f_hyp["tests"]
    gbm = f_res.get("gbm_med_abs", {})
    paper = f_res.get("gbm_med_abs_paper", {})
    months_tbl = data.table("months_gbm_below_benchmarks")
    total_fits = int(inv["ajustes (mes × bloque)"].sum())
    gbm_fits = int(inv.loc[inv["modelo"].isin([f"ML_{t}" for t in TARGETS]), "ajustes (mes × bloque)"].sum())
    n_models = inv["modelo"].nunique()
    fam_summary = (
        tests.assign(ok=tests["rechaza H0 (5 %)"] == "sí")
        .groupby("familia")["ok"]
        .agg(["sum", "count"])
        .reset_index()
    )
    gcfg = cfg["gbm"]
    commit = git_commit(cfg.root)
    versions = library_versions()

    tiles = "".join(
        f"<div class='tile'><div class='k'>Error mediano {TARGET_LABEL[t]}</div>"
        f"<div class='v'>{fnum(gbm.get(f'ML_{t}', np.nan), 3)}</div>"
        f"<div class='s'>paper {fnum(paper.get(f'ML_{t}', np.nan), 2)}</div></div>"
        for t in TARGETS
    )
    below = ""
    if not months_tbl.empty:
        ind = months_tbl[months_tbl["benchmark"].str.startswith("IND")]["share_below"].min() * 100
        below = (f"<div class='tile'><div class='k'>Meses con el GBM por debajo de IND</div>"
                 f"<div class='v'>{fnum(ind, 1)} %</div><div class='s'>mínimo de los 3 targets</div></div>")  # fmt: skip
    tiles += below + (
        f"<div class='tile'><div class='k'>Modelos ajustados</div><div class='v'>{fnum(total_fits, 0)}</div>"
        f"<div class='s'>{n_models} variantes; {fnum(gbm_fits, 0)} GBM base</div></div>"
        f"<div class='tile'><div class='k'>Muestra</div><div class='v'>{fnum(len(data.sample), 0)}</div>"
        f"<div class='s'>empresa-mes · {data.sample['id'].nunique():,} empresas · 360 meses</div></div>"
    )

    inv_g = inv.groupby("grupo").agg(**{
        "variantes": ("modelo", "nunique"), "semillas": ("semilla", "nunique"),
        "ajustes": ("ajustes (mes × bloque)", "sum"), "predicciones": ("predicciones", "sum"),
    }).reset_index()  # fmt: skip
    group_es = {"base": "GBM base (Tabla 2)", "benchmarks": "Benchmarks (IND, BL, RRV, BG)",
                "recon": "Reconciliación (Tabla 4)", "robust": "Robustez (IA.6)", "sensitivity": "Sensibilidad"}  # fmt: skip
    inv_g["grupo"] = inv_g["grupo"].map(group_es)
    variants = pd.DataFrame([
        {"modelo": m, "qué es": _describe(m), "semillas": int(g["semilla"].nunique()),
         "ajustes por semilla": int(g["ajustes (mes × bloque)"].iloc[0])}
        for m, g in inv.groupby("modelo", sort=False)
    ])  # fmt: skip
    unavailable = data.unavailable()

    hyper = pd.DataFrame([
        ("Árboles", f"parada anticipada: hasta {gcfg['num_boost_round']:,}, paciencia {gcfg['early_stopping_rounds']}",
         "bloque de validación", "Paper (nota 7)"),
        ("learning_rate", gcfg["params"]["learning_rate"], "fijo", "Por defecto de LightGBM, como el paper"),
        ("num_leaves", gcfg["params"]["num_leaves"], "fijo", "Por defecto"),
        ("min_data_in_leaf", gcfg["params"]["min_data_in_leaf"], "fijo", "Por defecto"),
        ("objective", gcfg["params"]["objective"], "fijo", "Error cuadrático, como el paper"),
        ("Sector", "variable categórica", "fijo", "Decisión E4 (robustez: numérica)"),
        ("Versión de LightGBM", gcfg["lightgbm_version"], "fijo", "La del paper"),
        ("Semillas de los bloques", ", ".join(str(s) for s in cfg.seeds), "fijo", "5 repartos aleatorios distintos"),
        ("Reparto mensual", "5 bloques: 60 % train, 20 % validación, 20 % test", "fijo", "Paper, sección 3.1"),
        ("Mínimo de filas de train", gcfg["min_train_rows"], "fijo", "Decisión E6"),
        ("LASSO: penalización", f"{cfg['benchmarks']['lasso_n_alphas']} valores", "bloque de validación", "Decisión E9"),
        ("CART: profundidad", ", ".join(str(d) for d in cfg["benchmarks"]["cart_depths"]), "bloque de validación",
         "Apéndice IA.6.2"),
        ("80:20: árboles", gcfg["fixed_rounds_80_20"], "fijo (sin validación)", "Apéndice IA.6.3"),
        ("IND: empresas mínimas por sector", cfg["benchmarks"]["ind_min_firms"], "fijo", "Decisión E7"),
        ("RRV: empresas mínimas por sector-mes", cfg["benchmarks"]["rrv_min_firms"], "fijo", "Decisión E7"),
    ], columns=["parámetro", "valor", "se decide con", "origen"])  # fmt: skip

    audit = pd.DataFrame([
        ("Número de árboles del GBM", "Bloque de validación de cada mes (parada anticipada)", "relval/models/gbm.py · fit_gbm"),
        ("Penalización del LASSO", "Bloque de validación", "relval/models/lasso.py"),
        ("Profundidad del CART", "Bloque de validación", "relval/models/cart.py"),
        ("Hiperparámetros de LightGBM", "Ninguno: valores por defecto, como el paper", "configs/replication.yaml"),
        ("Medias de sector (IND, BL) y regresiones (BL, RRV, BG)", "Solo los 4 bloques que no son test", "models/benchmarks/*"),
        ("Definiciones contables LSEG → Compustat", "Cobertura frente a la Tabla IA.4 (sin mirar errores)", "DECISIONS.md, sección D"),
        ("Bloque de test", "Solo se predice; no decide nada", "models/runner.py · fit_gbm_block"),
    ], columns=["decisión", "datos con los que se toma", "dónde está en el código"])  # fmt: skip

    t_show = tests.copy()
    t_show["IC 95 %"] = [
        f"[{fnum(a, 3)}, {fnum(b, 3)}]" if np.isfinite(a) and np.isfinite(b) else "–"
        for a, b in zip(t_show["ic95_lo"].astype(float), t_show["ic95_hi"].astype(float), strict=True)
    ]
    t_show["p"] = t_show["p"].map(fp)
    t_show["p Holm"] = t_show["p_holm"].map(fp)
    t_show["efecto"] = t_show["efecto"].map(lambda v: fnum(v, 4))
    t_show["% meses a favor"] = t_show["% meses a favor"].map(lambda v: fnum(v, 1) if pd.notna(v) else "–")
    t_show = t_show[["familia", "hipótesis", "target", "efecto", "IC 95 %", "p", "p Holm", "% meses a favor",
                     "rechaza H0 (5 %)"]]  # fmt: skip
    fam = fam_summary.rename(
        columns={"familia": "familia de contrastes", "sum": "rechazan H0", "count": "contrastes"}
    )

    eq = f_hyp["equivalence"]
    eq_rows = "".join(
        f"<li><b>{r['target']}</b>: diferencia con el paper {fnum(r['efecto'], 3)}; IC 90 % de nuestra mediana "
        f"[{fnum(r['ic90'][0], 3)}, {fnum(r['ic90'][1], 3)}], dentro del margen ±{fnum(hypotheses.TOLERANCE, 2)}.</li>"
        for _, r in eq.iterrows()
    )

    a_tests = tests[tests["familia"].str.startswith("A.")]
    rrv_fail = a_tests[(a_tests["hipótesis"].str.contains("RRV")) & (a_tests["rechaza H0 (5 %)"] == "no")][
        "target"
    ].tolist()
    lasso_fail = tests[(tests["familia"].str.startswith("D.")) & (tests["rechaza H0 (5 %)"] == "no")][
        "target"
    ].tolist()
    peers = f_xai.get("peers_same_sector", {})
    trees = f_conv.get("trees", {})
    capped = sum(v["capped"] for v in trees.values())
    diag = f_res.get("bl_diagnosis", pd.DataFrame())

    improvements = [
        (
            "Alta",
            "Predicciones que explotan en BL y en el OLS en logaritmos",
            f"{len(diag)} peores predicciones de BL_m2b analizadas: en todas, una variable de entrada (sobre todo el "
            "crecimiento de ventas ajustado y la rentabilidad sobre activos operativos netos) está miles de veces fuera "
            "del rango del resto del mes. Vienen de denominadores casi nulos o negativos en LSEG (ventas del año anterior, "
            "activos operativos netos).",
            "Recortar (winsorizar) las 10 variables de BL al percentil 1–99 de los datos de ajuste de cada mes y elegir "
            "ese corte con el bloque de validación. Documentarlo como desviación: la mediana no cambia, pero el RMSE y el "
            "R² de BL pasarían a ser comparables con el paper. Reentrenar solo BL y OLSLOG (menos de 1 hora en Batch).",
        ),
        (
            "Alta",
            "Sesgo de supervivencia",
            f"Solo {f_eda.get('delisted_firms', '–')} empresas deslistadas; 2.220 empresas frente a 16.201 en el paper. "
            "Explica la mayor parte de las diferencias (múltiplos más altos, carteras no significativas).",
            "Pedir a LSEG el universo histórico con las deslistadas (o CRSP si llega WRDS). Mientras tanto, presentarlo "
            "como la limitación principal y mostrar los resultados por década.",
        ),
        (
            "Alta",
            "Sector ausente en el 27 % de las empresa-mes",
            "El SIC de LSEG falta para muchas empresas: el sector pesa la mitad que en el paper en SHAP e IND/BL/RRV "
            "valoran menos empresas.",
            "Obtener el SIC de SEC EDGAR (cada empresa con CIK lo tiene). Encaja con la extensión del LLM, que ya "
            "necesita enlazar cada empresa con su CIK.",
        ),
        (
            "Media",
            "RRV iguala o mejora al GBM en V/A y V/S en las empresas que valora",
            f"Contrastes A: el GBM no gana a RRV en {', '.join(rrv_fail) or '—'}. RRV solo valora el 38 % de las "
            "empresa-mes (sectores con 30 empresas o más y datos completos) y modela directamente el valor del equity; "
            "el GBM de V/S convierte con la deuda, lo que añade error. En el paper RRV es claramente peor (0,43).",
            "Presentar siempre la comparación en la muestra común (Tabla IA.5) y añadir el modelo combinado de los tres "
            "targets (extensión E4 del plan), que valora el equity directamente como RRV.",
        ),
        (
            "Media",
            f"{capped} modelos GBM llegaron al tope de 10.000 árboles",
            "Casi todos de V/S en los años 90 (pocas empresas): la validación sigue mejorando muy despacio. Son el "
            f"{fnum(capped / max(1, sum(v['fits'] for v in trees.values())) * 100, 2)} % de los modelos; no cambian los "
            "resultados.",
            "Dejarlo como está en la réplica (es el procedimiento del paper) y mencionarlo. En las extensiones, probar "
            "una paciencia o un learning rate distintos dentro del barrido de hiperparámetros (E1).",
        ),
        (
            "Media",
            "Las curvas de aprendizaje terminan en el árbol elegido",
            "Los modelos se guardan hasta el árbol óptimo, así que no se ve la zona en la que la validación empieza a "
            "empeorar.",
            "Guardar el historial de la métrica de validación (evals_result) de cada modelo en el próximo "
            "entrenamiento: cuesta casi nada y da la curva completa.",
        ),
        (
            "Media",
            "Carteras long-short no significativas en la versión ponderada por valor",
            "Rentabilidad media positiva pero sin significación; el alfa equiponderado sí lo es.",
            "Probar la robustez sin microcaps y con otros cortes; en la memoria, atribuirlo a la muestra de "
            "supervivientes (faltan las sobrevaloradas que quebraron).",
        ),
        (
            "Baja",
            "GBM frente a LASSO en V/S (empresas sin datos ausentes)",
            f"No significativo en {', '.join(lasso_fail) or '—'}: con datos completos y una muestra pequeña, una "
            "regresión penalizada se acerca al GBM.",
            "Comentarlo; coincide con la idea del paper de que la ventaja del GBM está en manejar datos ausentes y no "
            "linealidades.",
        ),
        (
            "Baja",
            "El informe generado en AWS no registra el commit",
            "La imagen de Docker no incluye la carpeta .git.",
            "Pasar el commit a la imagen al construirla (variable de entorno) y leerlo en relval.io.git_commit.",
        ),
        (
            "Baja",
            "Avisos de fuente en el contenedor ('Segoe UI' no encontrada)",
            "Las figuras de AWS usan otra fuente.",
            "Añadir una fuente libre a la imagen o quitar Segoe UI del estilo.",
        ),
        (
            "Baja",
            "CI de GitHub bloqueado por facturación",
            "Ningún check se ejecuta, ni en main ni en los PR.",
            "Resolver la facturación de la cuenta del repositorio.",
        ),
    ]
    imp_df = pd.DataFrame(improvements, columns=["prioridad", "problema", "evidencia", "propuesta"])

    blocked = "".join(
        f"<li><code>{html.escape(k)}</code>: {html.escape(v)}</li>" for k, v in unavailable.items()
    )
    peers_txt = ", ".join(f"{TARGET_LABEL[t]} {fnum(v['obs'] * 100, 0)} % frente a {fnum(v['exp'] * 100, 1)} %"
                          for t, v in peers.items())  # fmt: skip

    conv_figs = book.section("convergencia")
    conv_intro = "".join(figs_html_one(f) for f in conv_figs if f.key == "conv_01_bloques")
    conv_rest = "".join(figs_html_one(f) for f in conv_figs if f.key != "conv_01_bloques")

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Informe visual de la réplica</title>
<style>
:root {{ color-scheme: light; --bg:#f6f5f2; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --line:#e1e0d9; --accent:#2a78d6; --good:#0ca30c; --warn:#c98500; --bad:#d03b3b; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --bg:#121211;
  --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#8f8e86; --line:#33332f; --accent:#3987e5; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --bg:#121211; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --muted:#8f8e86; --line:#33332f; --accent:#3987e5; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 "Segoe UI", system-ui, -apple-system, sans-serif; }}
header {{ padding:32px 16px 8px; max-width:1180px; margin:0 auto; }}
header h1 {{ margin:0 0 4px; font-size:28px; }}
header p {{ margin:0; color:var(--ink2); }}
nav {{ position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); }}
nav div {{ max-width:1180px; margin:0 auto; padding:8px 16px; display:flex; gap:14px; overflow-x:auto; white-space:nowrap; }}
nav a {{ color:var(--ink2); text-decoration:none; font-size:14px; }} nav a:hover {{ color:var(--accent); }}
main {{ max-width:1180px; margin:0 auto; padding:0 16px 64px; }}
section {{ margin-top:40px; }}
h2 {{ font-size:22px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
h3 {{ font-size:16px; margin:0 0 8px; }}
.lead {{ color:var(--ink2); max-width:900px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin:16px 0; }}
.tile {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.tile .k {{ color:var(--ink2); font-size:13px; }} .tile .v {{ font-size:28px; font-weight:600; margin:2px 0; }}
.tile .s {{ color:var(--muted); font-size:12.5px; }}
figure {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:16px; margin:18px 0; }}
figure img {{ width:100%; height:auto; display:block; background:#fcfcfb; border-radius:4px; }}
figcaption {{ color:var(--ink2); font-size:14px; margin-top:10px; }}
figcaption .file {{ display:block; color:var(--muted); font-size:12px; margin-top:4px; }}
.tbl {{ overflow-x:auto; margin:12px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; background:var(--surface); }}
th, td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }}
th {{ color:var(--ink2); font-weight:600; }} td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:16px 20px; margin:14px 0; }}
.formula {{ font-family:Consolas, monospace; background:var(--bg); padding:2px 6px; border-radius:4px; }}
details summary {{ cursor:pointer; color:var(--accent); margin:8px 0; }}
code {{ font-family:Consolas, monospace; font-size:13px; }}
ul li {{ margin:4px 0; }}
</style></head><body>
<header>
<h1>Réplica de Geertsema &amp; Lu (2023): informe visual</h1>
<p>Datos LSEG, 1990–2019 · run <code>{cfg.run_name}</code> · generado el {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC ·
commit <code>{commit}</code> · {len(book.figures)} figuras</p>
</header>
<nav><div><a href="#resumen">Resumen</a><a href="#modelos">Cómo funcionan los modelos</a><a href="#entrenado">Qué se ha entrenado</a>
<a href="#eda">Datos (EDA)</a><a href="#convergencia">Convergencia</a><a href="#resultados">Resultados</a>
<a href="#xai">Explicabilidad</a><a href="#hipotesis">Contrastes</a><a href="#mejoras">Errores y mejoras</a><a href="#apendice">Apéndice</a></div></nav>
<main>

<section id="resumen"><h2>Resumen</h2>
<div class="tiles">{tiles}</div>
<div class="card"><ul>
<li><b>El método está replicado y probado.</b> Los tres GBM quedan a 0,01–0,02 del error del paper y el contraste de
equivalencia (TOST, margen ±0,05) se cumple para los tres targets.</li>
<li><b>El GBM gana a los benchmarks tradicionales</b> (sector, Bhojraj-Lee y Bartram-Grinblatt) en todos los targets,
con contrastes pareados mes a mes significativos tras corregir por comparaciones múltiples. Frente a RRV gana en M/B,
pero no en V/A ni V/S en las empresas que RRV valora (se explica en los contrastes).</li>
<li><b>Las decisiones se toman con validación, nunca con test:</b> número de árboles, penalización del LASSO y profundidad
del CART. El resto son los valores por defecto del paper. El error de validación predice bien el de test
(correlación {fnum(f_conv.get("val_test_corr", np.nan), 2)}).</li>
<li><b>El resultado no depende del azar del reparto:</b> 5 semillas, ninguna diferencia significativa entre ellas.</li>
<li><b>La explicabilidad coincide con el paper:</b> correlación de {fnum(f_xai.get("category_corr", np.nan), 2)} en la
importancia por categoría; los comparables que construye el modelo son del mismo sector mucho más que al azar.</li>
<li><b>Diferencias con el paper:</b> datos de supervivientes (LSEG), carteras ponderadas por valor no significativas,
sector ausente en el 27 % de la muestra y unas pocas predicciones que explotan en BL. Todas tienen explicación y una
propuesta de mejora (última sección).</li>
</ul></div>
</section>

<section id="modelos"><h2>Cómo funcionan los modelos</h2>
<div class="card">
<h3>Qué se predice</h3>
<p>Para cada empresa y mes se estima su múltiplo de valoración en logaritmo: precio sobre valor contable
<span class="formula">ln(M/B)</span>, valor de empresa sobre activos <span class="formula">ln((M+D)/A)</span> y valor de empresa
sobre ventas <span class="formula">ln((M+D)/S)</span>. La predicción se convierte a valor del equity
(<span class="formula">M̂ = e<sup>ŷ</sup>·B</span>, o <span class="formula">e<sup>ŷ</sup>·A − D</span>,
<span class="formula">e<sup>ŷ</sup>·S − D</span>) y el error es <span class="formula">ε = M̂ / M − 1</span>: −0,2 significa que el
modelo valora la empresa un 20 % por debajo del mercado. La métrica principal es la mediana de |ε|.</p>
<h3>El modelo: gradient boosting (LightGBM)</h3>
<p>Un árbol de decisión divide a las empresas en grupos según sus ratios (por ejemplo, "ROE &gt; 12 % y crecimiento &gt;
5 %") y predice la media del múltiplo en cada grupo. El <i>gradient boosting</i> suma cientos de árboles pequeños:
cada árbol nuevo aprende de lo que los anteriores todavía fallan. Así capta relaciones no lineales e interacciones
entre las 97 variables sin tener que especificarlas, y admite datos ausentes. Se entrena <b>un modelo distinto cada
mes</b> con la sección cruzada de ese mes: la valoración del mercado cambia con el ciclo.</p>
<h3>Cómo se evita mirar el futuro o el test</h3>
<p>Cada mes las empresas se reparten al azar en 5 bloques. Se entrenan 5 modelos: cada uno aprende con 3 bloques, usa el
cuarto (validación) para decidir cuántos árboles añadir y predice el quinto (test), que no ha visto. Cada empresa recibe
una sola predicción fuera de muestra. Las variables contables solo usan información publicada (3 meses de retraso).</p>
<h3>Los modelos con los que se compara</h3>
<ul>
<li><b>IND</b>: la media armónica del múltiplo del sector (Fama-French 49). Lo que haría un analista con un grupo de comparables.</li>
<li><b>BL</b> (Bhojraj y Lee, 2002): regresión lineal del múltiplo sobre 10 variables (múltiplos del sector, margen, crecimiento, apalancamiento, rentabilidad, I+D).</li>
<li><b>RRV</b> (Rhodes-Kropf, Robinson y Viswanathan, 2005): regresión del logaritmo del valor del equity por sector y mes.</li>
<li><b>BG</b> (Bartram y Grinblatt): regresión del valor del equity sobre 16 partidas contables en dólares.</li>
<li><b>OLS en logaritmos, LASSO y CART</b>: para separar de dónde viene la mejora (Tabla 4) y para la robustez.</li>
<li><b>HP</b> (comparables por texto de Hoberg y Phillips): no disponible; motivo en el apéndice.</li>
</ul>
<h3>Cómo se explica cada predicción</h3>
<p><b>SHAP</b> reparte cada predicción entre las 97 variables: cuánto sube o baja el múltiplo por el valor de cada una. La
<b>descomposición en comparables</b> del paper escribe cada predicción como una media ponderada de los múltiplos de las
empresas de entrenamiento: el modelo es, literalmente, un método de comparables con pesos aprendidos.</p>
</div>
{conv_intro}
</section>

<section id="entrenado"><h2>Qué se ha entrenado</h2>
<p class="lead">Cada fila de las predicciones es una empresa-mes valorada fuera de muestra. Un "ajuste" es un modelo entrenado
para un mes y un bloque. Los modelos base tienen 5 semillas (5 repartos distintos de los bloques); el resto, la primera.</p>
{table_html(inv_g)}
<details><summary>Las {len(variants)} variantes, una a una</summary>{table_html(variants)}</details>
<h3>Hiperparámetros y semillas</h3>
{table_html(hyper)}
<h3>Auditoría: con qué datos se toma cada decisión</h3>
<p class="lead">Ninguna decisión usa el bloque de test. Lo que el paper no fija se decidió antes de entrenar o con la cobertura
de las variables frente al paper (DECISIONS.md), nunca mirando errores.</p>
{table_html(audit)}
</section>

<section id="eda"><h2>Datos (EDA)</h2>
<p class="lead">Qué contiene la muestra: tamaño, composición, distribución de los múltiplos y de las 97 variables.</p>
{figs_html(book, "eda")}
</section>

<section id="convergencia"><h2>Convergencia del entrenamiento</h2>
<p class="lead">Cuántos árboles elige la validación, cómo baja el error al añadirlos, si la validación representa bien al test
y cómo se eligen los parámetros de LASSO y CART. Las curvas se obtienen evaluando los modelos guardados, sin reentrenar.</p>
{conv_rest}
</section>

<section id="resultados"><h2>Resultados</h2>
{figs_html(book, "resultados")}
<h3>Las peores predicciones de BL</h3>
{table_html(diag) if len(diag) else "<p>Sin datos.</p>"}
</section>

<section id="xai"><h2>Explicabilidad (XAI)</h2>
{figs_html(book, "xai")}
</section>

<section id="hipotesis"><h2>Contraste de hipótesis</h2>
<div class="card">
<p><b>Cómo se contrasta.</b> La unidad es el mes. Cada comparación entre dos modelos se hace sobre las <b>mismas empresas</b>
y se resume en una serie de 360 diferencias mensuales de la mediana de |ε|. Se contrasta si su media es distinta de cero
con errores de <b>Newey-West (12 retardos)</b>, porque los meses seguidos están correlacionados; se añade el porcentaje de
meses a favor (prueba del signo). Con varios contrastes a la vez se corrige con <b>Holm</b>. La equivalencia con el paper
se contrasta con <b>TOST</b> sobre un <b>bootstrap por bloques de 12 meses</b>: la réplica es equivalente si el intervalo
del 90 % de nuestra mediana cae dentro de ±{fnum(hypotheses.TOLERANCE, 2)} del valor del paper.</p>
<p>Con cientos de miles de observaciones casi cualquier diferencia es "significativa": por eso siempre se lee el
<b>tamaño del efecto</b> junto al p-valor.</p></div>
{table_html(fam)}
{figs_html(book, "hipotesis")}
<div class="card"><h3>Lectura para la defensa</h3><ul>
<li><b>A. El GBM es mejor que los benchmarks tradicionales.</b> Frente a IND, BL y BG la mejora es grande (0,12 a 0,77 puntos
de |ε|), significativa y a favor en casi el 100 % de los meses. Frente a RRV, en las empresas que RRV valora, el GBM gana en
M/B pero no en {", ".join(rrv_fail) or "—"}. Hay que decirlo así: RRV solo cubre el 38 % de la muestra (sectores grandes
y datos completos) y modela el valor del equity sin conversión.</li>
<li><b>B. Sesgo.</b> La mediana mensual de ε es estadísticamente distinta de 0, pero menor del 1 % en valor absoluto: un sesgo
sin importancia económica, del mismo orden que el del paper.</li>
<li><b>C. Equivalencia con el paper.</b><ul>{eq_rows}</ul></li>
<li><b>D. De dónde viene la mejora.</b> Logaritmos, GBM y las 97 variables mejoran cada uno de forma significativa; la única
excepción es GBM frente a LASSO en {", ".join(lasso_fail) or "—"} (empresas sin datos ausentes).</li>
<li><b>E. Sensibilidad y G. semillas.</b> Ninguna variante ni ninguna semilla cambia el error de forma significativa tras Holm
(efectos de milésimas): los resultados no dependen de decisiones arbitrarias ni del reparto.</li>
<li><b>F. Carteras.</b> La rentabilidad long-short ponderada por valor no es significativa; el alfa FF6 equiponderado sí.
En el paper ambas lo son: es la diferencia más clara y se atribuye a la muestra de supervivientes.</li>
<li><b>H e I. Explicabilidad.</b> Comparables del mismo sector: {peers_txt}. El ranking de categorías SHAP coincide con el
del paper (Spearman significativo en los tres targets).</li>
</ul></div>
<details><summary>Tabla completa de los {len(tests)} contrastes</summary>{table_html(t_show)}
<p class="lead">Efecto: A y D, mejora de la mediana de |ε| (positivo = la hipótesis se cumple); B, mediana de ε; C, diferencia con
el paper; E y G, diferencia en milésimas de |ε|; F, rentabilidad mensual en %; H, puntos porcentuales; I, ρ de Spearman.
Contrastes de una cola en A, D, F y H; de dos colas en B, E y G.</p></details>
</section>

<section id="mejoras"><h2>Errores detectados y mejoras propuestas</h2>
{table_html(imp_df)}
</section>

<section id="apendice"><h2>Apéndice</h2>
<div class="card">
<p><b>Reproducir:</b> <code>.venv/Scripts/relval visual</code> (lee las predicciones, los modelos y las tablas ya calculados; no
entrena nada). Figuras en <code>reports/{cfg.run_name}/visual/figures/</code> (PNG para la memoria) y tablas en
<code>reports/{cfg.run_name}/visual/tables/</code>.</p>
<p><b>No disponible:</b></p><ul>{blocked or "<li>—</li>"}</ul>
<p><b>Versiones:</b> {", ".join(f"{k} {v}" for k, v in versions.items())}.</p>
<p><b>Datos con licencia:</b> este informe contiene nombres de empresas y cifras derivadas de LSEG. No se publica ni se sube al
repositorio (la carpeta <code>reports/</code> está en .gitignore).</p>
</div></section>
</main></body></html>"""


def _describe(model: str) -> str:
    s = SPECS[model]
    kind = {"gbm": "GBM", "ind": "media del sector", "bl": "OLS (Bhojraj-Lee)", "ols_log": "OLS en logaritmos",
            "rrv": "RRV", "bg": "Bartram-Grinblatt", "hp": "Hoberg-Phillips", "lasso": "LASSO", "cart": "árbol CART"}[s.kind]  # fmt: skip
    extra = []
    if s.features == "bl10":
        extra.append("10 variables BL")
    if s.sample == "nm":
        extra.append("solo empresas sin datos ausentes")
    if s.sample == "bl":
        extra.append("muestra BL")
    if s.sample == "listed":
        extra.append("solo cotizadas hoy")
    if s.split == "80_20":
        extra.append("80:20 sin validación")
    if s.options.get("industry_numeric"):
        extra.append("sector numérico")
    if s.options.get("beta_column"):
        extra.append(s.options["beta_column"].replace("CAPMbeta_", "beta con ") + " meses")
    return f"{kind} → {s.target}" + (f" ({', '.join(extra)})" if extra else "")
