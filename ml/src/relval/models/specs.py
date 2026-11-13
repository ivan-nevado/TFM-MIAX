"""Registry of every model of the replication: base GBMs, benchmarks, reconciliation steps,
robustness variants and the sensitivity runs required by the data situation."""

from __future__ import annotations

from dataclasses import dataclass, field

LOGS = ["lnm2b", "lnv2a", "lnv2s"]
MULTS = ["m2b", "v2a", "v2s"]


@dataclass
class ModelSpec:
    name: str
    kind: str  # gbm | ind | bl | ols_log | rrv | bg | hp | lasso | cart
    target: str  # lnm2b lnv2a lnv2s | m2b v2a v2s | lnM | M
    group: str  # base | benchmarks | recon | robust | sensitivity
    features: str = "ml97"  # ml97 | bl10
    sample: str = "all"  # all | bl (10 BL variables present) | nm (all 97 present) | listed
    split: str = "60_20_20"  # 60_20_20 | 80_20 (no validation set, 20 trees) | fit4 (benchmarks)
    options: dict = field(default_factory=dict)
    label: str = ""  # row label in the paper's tables


def build_specs() -> dict[str, ModelSpec]:
    specs: list[ModelSpec] = []
    for t, m in zip(LOGS, MULTS, strict=True):
        # ---- base models (Table 2) -----------------------------------------------------
        specs.append(ModelSpec(f"ML_{t}", "gbm", t, "base", options={"shap": True, "save": True}))
        # ---- benchmarks (Table 2) ------------------------------------------------------
        specs.append(
            ModelSpec(f"BL_{m}", "bl", m, "benchmarks", "bl10", "bl", "fit4", label=f"OLS : BL vars -> {m}")
        )
        specs.append(ModelSpec(f"IND_{m}", "ind", m, "benchmarks", split="fit4"))
        specs.append(ModelSpec(f"HP_{m}", "hp", m, "benchmarks", split="fit4"))
        # ---- reconciliation (Table 4) --------------------------------------------------
        specs.append(
            ModelSpec(
                f"OLSLOG_BL_{t}",
                "ols_log",
                t,
                "recon",
                "bl10",
                "bl",
                "fit4",
                label=f"OLS : BL vars -> ln({m})",
            )
        )
        specs.append(
            ModelSpec(f"GBM_BL_{t}", "gbm", t, "recon", "bl10", "bl", label=f"GBM : BL vars -> ln({m})")
        )
        specs.append(
            ModelSpec(f"GBM_ML_BLS_{t}", "gbm", t, "recon", "ml97", "bl", label=f"GBM : ML vars -> ln({m})")
        )
        specs.append(
            ModelSpec(
                f"LASSO_NM_{t}", "lasso", t, "recon", "ml97", "nm", label=f"LASSO : ML vars (nm) -> ln({m})"
            )
        )
        specs.append(
            ModelSpec(f"GBM_NM_{t}", "gbm", t, "recon", "ml97", "nm", label=f"GBM : ML vars (nm) -> ln({m})")
        )
        # ---- robustness (online appendix IA.6) -----------------------------------------
        specs.append(ModelSpec(f"CART_{t}", "cart", t, "robust"))
        specs.append(ModelSpec(f"ML8020_{t}", "gbm", t, "robust", split="80_20"))
        specs.append(ModelSpec(f"GBMALT_{m}", "gbm", m, "robust"))  # GBM on the multiple in levels
        # ---- sensitivity runs (section 8 of the master prompt) -------------------------
        specs.append(ModelSpec(f"ML_{t}_indnum", "gbm", t, "sensitivity", options={"industry_numeric": True}))
        specs.append(ModelSpec(f"ML_{t}_listed", "gbm", t, "sensitivity", sample="listed"))
        for months in (24, 36, 48):
            specs.append(
                ModelSpec(
                    f"ML_{t}_beta{months}",
                    "gbm",
                    t,
                    "sensitivity",
                    options={"beta_column": f"CAPMbeta_{months}"},
                )
            )
    specs.append(ModelSpec("RRV", "rrv", "lnM", "benchmarks", split="fit4"))
    specs.append(ModelSpec("BG", "bg", "M", "benchmarks", split="fit4"))
    specs.append(ModelSpec("GBMALT_lnM", "gbm", "lnM", "robust"))
    specs.append(ModelSpec("GBMALT_M", "gbm", "M", "robust"))
    return {s.name: s for s in specs}


SPECS = build_specs()
BASE_MODELS = [f"ML_{t}" for t in LOGS]
TABLE2_MODELS = [
    *BASE_MODELS, *[f"BL_{m}" for m in MULTS], *[f"IND_{m}" for m in MULTS], *[f"HP_{m}" for m in MULTS], "RRV", "BG",
]  # fmt: skip
TABLE4_ROWS = {
    m: [
        f"BL_{m}",
        f"OLSLOG_BL_ln{m}",
        f"GBM_BL_ln{m}",
        f"GBM_ML_BLS_ln{m}",
        f"LASSO_NM_ln{m}",
        f"GBM_NM_ln{m}",
    ]
    for m in MULTS
}


def models_in_groups(groups: list[str]) -> list[str]:
    if "all" in groups:
        return list(SPECS)
    return [name for name, spec in SPECS.items() if spec.group in groups]
