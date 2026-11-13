"""API tests on a tiny synthetic copy of the replication outputs (no licensed data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

SEED = 20230301
DATE = pd.Timestamp("2019-12-31")


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    root = tmp_path_factory.mktemp("outputs")
    data, reports = root / "data", root / "reports"
    (reports / "replication" / "tables").mkdir(parents=True)
    pd.DataFrame(
        {"model": ["ML_lnm2b"], "med_abs_eps": [0.3], "med_abs_eps_paper": [0.29], "empty": [np.nan]}
    ).to_csv(reports / "replication" / "tables" / "table2_vs_paper.csv", index=False)
    (data / "processed").mkdir(parents=True)
    firms = pd.DataFrame(
        {"id": ["AAA.N", "BBB.OQ", "CCC.OQ"], "name": ["Alpha Corp", "Beta Inc", None], "exchange": ["NYSE", "NASDAQ", "NASDAQ"],
         "sector": ["Technology"] * 3, "ff49_name": ["Softw", "Drugs", None]}
    )  # fmt: skip
    firms.to_parquet(data / "processed" / "firms_lseg.parquet")
    for target in ("lnm2b", "lnv2a", "lnv2s"):
        d = data / "predictions" / "replication" / f"model=ML_{target}" / f"seed={SEED}"
        d.mkdir(parents=True)
        pd.DataFrame(
            {"model": f"ML_{target}", "target": target, "id": ["AAA.N", "BBB.OQ"], "date": DATE, "block": [0, 1], "seed": SEED,
             "y": [1.0, 0.5], "yhat": [0.9, 0.7], "M": [100.0, 50.0], "Mhat": [90.0, 61.0], "eps": [-0.1, 0.22],
             "n_train": [600.0, 600.0], "train_mean_y": 0.6, "train_mean_M": 80.0, "best_iteration": [80.0, 90.0]}
        ).to_parquet(d / "year=2019.parquet")  # fmt: skip
    s = data / "artifacts" / "replication" / "shap" / "model=ML_lnm2b" / f"seed={SEED}"
    s.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["AAA.N"],
            "date": DATE,
            "block": 0,
            "roe_wr": 0.2,
            "rd_sale_wr": -0.1,
            "industry": 0.05,
            "npm_wr": 0.01,
            "base_value": 0.74,
        }
    ).to_parquet(s / "year=2019.parquet")
    p = data / "artifacts" / "replication" / "peers"
    p.mkdir(parents=True)
    pd.DataFrame(
        {"date": DATE, "target": "lnm2b", "id": "AAA.N", "peer_id": ["BBB.OQ", "CCC.OQ"], "weight": [0.6, 0.4],
         "cumweight": [0.6, 1.0], "rank": [1, 2], "n_peers": 2, "peer_y": [0.5, 0.2]}
    ).to_parquet(p / "peers_top_lnm2b_201912.parquet")  # fmt: skip

    from app.api import deps
    from app.core import settings as settings_module
    from app.main import create_app

    settings_module.get_settings.cache_clear()
    deps.get_store.cache_clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("RELVAL_DATA_ROOT", str(data))
        mp.setenv("RELVAL_REPORTS_ROOT", str(reports))
        yield TestClient(create_app())
    settings_module.get_settings.cache_clear()
    deps.get_store.cache_clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["tables_available"] == 1


def test_tables(client):
    assert client.get("/api/v1/replication/tables").json() == ["table2_vs_paper"]
    table = client.get("/api/v1/replication/tables/table2_vs_paper").json()
    assert table["columns"][0] == "model" and table["rows"][0]["med_abs_eps_paper"] == 0.29
    assert table["rows"][0]["empty"] is None  # NaN is serialised as null
    assert client.get("/api/v1/replication/tables/nope").status_code == 404
    assert client.get("/api/v1/replication/tables/..%2Fsecret").status_code == 404


def test_firm_search(client):
    hits = client.get("/api/v1/firms", params={"query": "beta"}).json()
    assert [h["id"] for h in hits] == ["BBB.OQ"]
    assert len(client.get("/api/v1/firms", params={"limit": 2}).json()) == 2


def test_valuations(client):
    body = client.get("/api/v1/valuations/AAA.N", params={"date": "2019-12"}).json()
    assert body["name"] == "Alpha Corp" and len(body["valuations"]) == 3
    v = body["valuations"][0]
    assert v["target"] == "lnm2b" and v["valuation_error"] == pytest.approx(-0.1)
    assert v["predicted_multiple"] == pytest.approx(np.exp(0.9))
    assert client.get("/api/v1/valuations/AAA.N", params={"date": "2018-01"}).status_code == 404
    assert client.get("/api/v1/valuations/ZZZ.N").status_code == 404


def test_shap_waterfall_adds_up(client):
    body = client.get("/api/v1/valuations/AAA.N/shap", params={"top": 2}).json()
    labels = [i["label"] for i in body["waterfall"]]
    assert labels == ["Training mean", "roe_wr", "rd_sale_wr", "other", "Prediction"]
    assert body["prediction"] == pytest.approx(0.74 + 0.2 - 0.1 + 0.05 + 0.01)
    assert (
        client.get("/api/v1/valuations/AAA.N/shap", params={"target": "lnv2a"}).status_code == 503
    )  # not produced


def test_peers(client):
    body = client.get("/api/v1/valuations/AAA.N/peers").json()
    assert body["n_peers"] == 2 and body["most_comparable"][0]["peer_id"] == "BBB.OQ"
    assert body["most_comparable"][1]["name"] is None
    assert client.get("/api/v1/valuations/BBB.OQ/peers").status_code == 404


def test_scenarios_returns_the_planned_contract(client):
    r = client.post("/api/v1/scenarios", json={"id": "AAA.N", "date": "2019-12", "changes": {"sale": -0.2}})
    assert r.status_code == 501 and "planned_response" in r.json()
