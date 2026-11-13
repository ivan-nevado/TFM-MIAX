"""Paper targets: machine-readable copies of the published tables, and a check against the PDF.

    python scripts/paper_targets.py extract   # writes docs/replication/paper_targets/*.csv
    python scripts/paper_targets.py verify    # every number must appear on the PDF page of its table

Sources (no number is typed here):
* Tables 2, 3, 4, 5, 7, 8, 9: the CSV blocks of section 6 of ``docs/PROMPT_REPLICACION.md``.
* Table 1: the three rows of the paper, parsed from the PDF text.
* Tables IA.4 (coverage) and IA.8 (SHAP of all 97 variables): Annex A.2 of the team's plan
  (``docs/paper/Replica_paper1_y_extensiones_revisado_v3.docx``).
* If the paper's public online appendix has been downloaded to
  ``data/external/online_appendix/gl-online-appendix.pdf`` (``appendix`` command), Table IA.4
  (N, mean, median of every input), IA.5 (common sample) and IA.6 (alternative models) are parsed
  from it as well.

    python scripts/paper_targets.py appendix  # download the online appendix and parse IA.4-IA.6
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "replication" / "paper_targets"
PROMPT = ROOT / "docs" / "PROMPT_REPLICACION.md"
PDF = next((ROOT / "docs" / "paper").glob("*GEERTSEMA*.pdf"), None)
DOCX = next((ROOT / "docs" / "paper").glob("Replica_paper1*.docx"), None)
BLOCK_ORDER = ["table2", "table3", "table4", "table7", "table8", "table9", "table5"]
# 1-based PDF pages holding each table (journal pages 329-376 -> PDF pages 1-48)
PDF_PAGES = {"table1": [8], "table2": [11], "table3": [18, 19], "table4": [21, 22], "table5": [25],
             "table7": [30, 31], "table8": [34], "table9": [36, 37]}  # fmt: skip


def extract() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = PROMPT.read_text(encoding="utf-8")
    section6 = text[text.index("## 6. Numbers to reproduce") : text.index("## 7. How to judge the results")]
    blocks = re.findall(r"```csv\n(.*?)```", section6, flags=re.S)
    assert len(blocks) == len(BLOCK_ORDER), f"expected {len(BLOCK_ORDER)} CSV blocks, found {len(blocks)}"
    for name, block in zip(BLOCK_ORDER, blocks, strict=True):
        df = pd.read_csv(io.StringIO(block), dtype=str, keep_default_na=False)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows")

    import pymupdf as fitz

    page = fitz.open(PDF)[PDF_PAGES["table1"][0] - 1].get_text()
    rows = re.findall(
        r"(Common domestic stocks[^\n]*|Positive and nonmissing targets[^\n]*|All scale variables[^\n]*)\n\s*([\d,]+)\n\s*([\d,]+)",
        page,
    )
    t1 = pd.DataFrame(
        [(a.strip(), int(b.replace(",", "")), int(c.replace(",", ""))) for a, b, c in rows],
        columns=["step", "firm_months", "firms"],
    )
    assert len(t1) == 3, "Table 1 not found in the PDF text"
    t1.to_csv(OUT / "table1.csv", index=False)
    print(f"table1: {len(t1)} rows")

    import docx

    sys.path.insert(0, str(ROOT / "ml" / "src"))
    from relval.features.registry import CATEGORY_OF, FEATURES

    recs = []
    for table in docx.Document(DOCX).tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if len(cells) >= 5 and cells[0] in FEATURES:
                shap = [float(x.replace(",", ".")) for x in re.split(r"\s*/\s*", cells[4])]
                cov = float(re.sub(r"[^\d,]", "", cells[3]).replace(",", ".")) / 100.0
                recs.append({"variable": cells[0], "category": CATEGORY_OF[cells[0]], "formula": cells[2], "coverage": cov,
                             "shap_lnm2b": shap[0], "shap_lnv2a": shap[1], "shap_lnv2s": shap[2]})  # fmt: skip
    ia = pd.DataFrame(recs).drop_duplicates("variable")
    missing = sorted(set(FEATURES) - set(ia["variable"]))
    assert not missing, f"variables missing from the plan's annex: {missing}"
    ia.to_csv(OUT / "ia4_ia8.csv", index=False)
    print(
        f"ia4_ia8: {len(ia)} rows; SHAP column sums {ia[['shap_lnm2b', 'shap_lnv2a', 'shap_lnv2s']].sum().round(1).to_dict()}"
    )


APPENDIX_URL = (
    "https://www.chicagobooth.edu/-/media/research/arc/docs/journal/online-supplements/gl-online-appendix.pdf"
)
APPENDIX = ROOT / "data" / "external" / "online_appendix" / "gl-online-appendix.pdf"
METRICS = [
    "N",
    "med_abs_eps",
    "avg_abs_eps",
    "med_eps",
    "avg_eps",
    "sd_eps",
    "rmse_y",
    "rmse_M",
    "r2oos_y",
    "r2oos_M",
    "rho",
]
IA6_NAMES = {"ML_m": "GBMALT_M", "ML_lnm": "GBMALT_lnM", "ML_m2b": "GBMALT_m2b", "Tree_lnm2b": "CART_lnm2b",
             "Tree_lnv2a": "CART_lnv2a", "Tree_lnv2s": "CART_lnv2s", "AS_lnm2b": "ML8020_lnm2b",
             "AS_lnv2a": "ML8020_lnv2a", "AS_lnv2s": "ML8020_lnv2s"}  # fmt: skip


def _num(token: str) -> float | None:
    token = token.replace("−", "-").replace(",", "")
    try:
        return float(token)
    except ValueError:
        return None


def _model_rows(text: str, names: list[str]) -> pd.DataFrame:
    """Rows ``<model> v1 ... v11`` of a performance table, read token by token."""
    tokens = text.split()
    rows = []
    for i, tok in enumerate(tokens):
        if tok in names:
            values = [_num(t) for t in tokens[i + 1 : i + 1 + len(METRICS)]]
            if len(values) == len(METRICS) and all(v is not None for v in values):
                rows.append({"paper_model": tok, **dict(zip(METRICS, values, strict=True))})
    return pd.DataFrame(rows).drop_duplicates("paper_model")


def appendix() -> None:
    """Download the public online appendix (if missing) and parse Tables IA.4, IA.5 and IA.6."""
    import pymupdf as fitz

    sys.path.insert(0, str(ROOT / "ml" / "src"))
    from relval.features.registry import FEATURES

    if not APPENDIX.exists():
        import requests

        APPENDIX.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(APPENDIX_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        APPENDIX.write_bytes(resp.content)
    pages = [p.get_text() for p in fitz.open(APPENDIX)]

    # Table IA.4: variable, N, mean, p1, p10, p50, p90, p99 (one value per line)
    recs = {}
    for text in pages:
        if "Table IA.4" not in text:
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            if ln in FEATURES or ln in ("lnm2b", "lnv2a", "lnv2s"):
                # N and the mean share a line when the mean is very large
                vals = [_num(x) for x in " ".join(lines[i + 1 : i + 8]).split()[:7]]
                if len(vals) == 7 and all(v is not None for v in vals):
                    recs[ln] = {
                        "variable": ln,
                        "paper_n": int(vals[0]),
                        "paper_mean": vals[1],
                        "paper_median": vals[4],
                    }
    stats = pd.DataFrame(recs.values())
    stats.to_csv(OUT / "ia4_statistics.csv", index=False)
    print(
        f"ia4_statistics: {len(stats)} rows ({len(set(FEATURES) - set(stats['variable']))} inputs not parsed)"
    )

    names = [
        "ML_lnm2b",
        "ML_lnv2a",
        "ML_lnv2s",
        "BL_m2b",
        "BL_v2a",
        "BL_v2s",
        "IND_m2b",
        "IND_v2a",
        "IND_v2s",
        "RRV",
        "BG",
    ]
    ia5_page = next(t for t in pages if "Table IA.5: Model performance comparison" in t)
    ia5 = _model_rows(ia5_page, names).rename(columns={"paper_model": "model"})
    ia5.to_csv(OUT / "ia5.csv", index=False)
    print(f"ia5: {len(ia5)} rows")
    ia6_page = next(t for t in pages if "Table IA.6: Performance of alternative machine learning models" in t)
    ia6 = _model_rows(ia6_page, list(IA6_NAMES))
    ia6.insert(0, "model", ia6["paper_model"].map(IA6_NAMES))
    ia6.to_csv(OUT / "ia6.csv", index=False)
    print(f"ia6: {len(ia6)} rows")


def _numbers(text: str) -> set[str]:
    text = text.replace("−", "-").replace("–", "-").replace(",", "")
    return set(re.findall(r"-?\d+\.\d+|-?\d+", text))


def verify() -> int:
    import pymupdf as fitz

    doc = fitz.open(PDF)
    failures = 0
    for name, pages in PDF_PAGES.items():
        path = OUT / f"{name}.csv"
        if not path.exists():
            print(f"{name}: missing, run extract first")
            failures += 1
            continue
        on_page = set()
        for p in pages:
            on_page |= _numbers(doc[p - 1].get_text())
        on_page |= {n.lstrip("-") for n in on_page}
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        wanted = set()
        for col in df.columns:
            for value in df[col]:
                value = re.sub(r"\*+$", "", value.strip())
                if re.fullmatch(r"-?\d+(\.\d+)?", value):
                    wanted.add(value)
        missing = sorted(v for v in wanted if v not in on_page and v.lstrip("-") not in on_page
                         and v.rstrip("0").rstrip(".") not in on_page)  # fmt: skip
        status = (
            "OK" if not missing else f"{len(missing)} values not found on PDF pages {pages}: {missing[:15]}"
        )
        print(f"{name}: {len(wanted)} distinct numbers, {status}")
        failures += bool(missing)
    return failures


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "extract"
    if command == "extract":
        extract()
    elif command == "verify":
        sys.exit(1 if verify() else 0)
    elif command == "appendix":
        appendix()
    else:
        sys.exit(__doc__)
