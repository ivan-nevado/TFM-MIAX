"""Valuations, SHAP explanations and peers of a firm-month, read from the replication outputs."""

from __future__ import annotations

import math

import pandas as pd

from app.core.errors import NotFoundError
from app.ml.adapter import category_of
from app.repositories.store import TARGETS, OutputStore
from app.schemas.models import Peer, PeersResponse, ShapItem, ShapResponse, Valuation, ValuationSet
from app.services.firms import FirmService

MULTIPLE = {"lnm2b": "m2b", "lnv2a": "v2a", "lnv2s": "v2s"}
SHAP_META = {"id", "date", "block", "base_value"}


def parse_month(date: str | None) -> pd.Timestamp | None:
    if not date:
        return None
    try:
        return pd.Timestamp(date[:7]) + pd.offsets.MonthEnd(0)
    except ValueError as exc:
        raise NotFoundError(f"invalid date {date!r}, expected YYYY-MM") from exc


class ValuationService:
    def __init__(self, store: OutputStore):
        self.store = store
        self.firms = FirmService(store)

    def _row(self, target: str, firm_id: str, date: pd.Timestamp | None) -> pd.Series:
        preds = self.store.predictions(target, firm_id, date.year if date is not None else None)
        if date is not None and len(preds):
            preds = preds[preds["date"] == date]
        if preds.empty:
            when = f" in {date:%Y-%m}" if date is not None else ""
            raise NotFoundError(f"no out-of-sample prediction for {firm_id!r}{when}")
        return preds.iloc[-1]

    def valuations(self, firm_id: str, date: str | None, target: str | None) -> ValuationSet:
        firm = self.firms.get(firm_id)
        month = parse_month(date)
        history = self.store.predictions("lnm2b", firm_id)
        available = [f"{d:%Y-%m}" for d in history["date"]] if len(history) else []
        out = []
        for t in [target] if target else TARGETS:
            r = self._row(t, firm_id, month)
            month = pd.Timestamp(r["date"])
            trees = None if pd.isna(r["best_iteration"]) else int(r["best_iteration"])
            out.append(
                Valuation(
                    id=firm_id,
                    date=f"{month:%Y-%m}",
                    target=t,
                    multiple=MULTIPLE[t],
                    actual_log_multiple=float(r["y"]),
                    predicted_log_multiple=float(r["yhat"]),
                    actual_multiple=math.exp(r["y"]),
                    predicted_multiple=math.exp(r["yhat"]),
                    market_value=float(r["M"]),
                    predicted_equity_value=float(r["Mhat"]),
                    valuation_error=float(r["eps"]),
                    block=int(r["block"]),
                    n_train=int(r["n_train"]),
                    trees=trees,
                )
            )
        return ValuationSet(
            id=firm_id, name=firm.name, date=f"{month:%Y-%m}", available_dates=available, valuations=out
        )

    def shap(self, firm_id: str, date: str | None, target: str, top: int) -> ShapResponse:
        month = pd.Timestamp(self._row(target, firm_id, parse_month(date))["date"])
        raw = self.store.shap(target, firm_id, month)
        if raw.empty:
            raise NotFoundError(f"no SHAP values for {firm_id!r} in {month:%Y-%m}")
        row = raw.iloc[0]
        contrib = row[[c for c in raw.columns if c not in SHAP_META]].astype(float)
        order = contrib.abs().sort_values(ascending=False).index
        base = float(row["base_value"])
        items = [ShapItem(label="Training mean", value=base, kind="base")]
        items += [
            ShapItem(label=v, value=float(contrib[v]), kind="contribution", category=category_of(v))
            for v in order[:top]
        ]
        items.append(ShapItem(label="other", value=float(contrib[order[top:]].sum()), kind="contribution"))
        prediction = base + float(contrib.sum())
        items.append(ShapItem(label="Prediction", value=prediction, kind="total"))
        return ShapResponse(
            id=firm_id,
            date=f"{month:%Y-%m}",
            target=target,
            base_value=base,
            prediction=prediction,
            waterfall=items,
        )

    def peers(self, firm_id: str, date: str | None, target: str, top: int) -> PeersResponse:
        months = self.store.peer_months()
        month = parse_month(date)
        if month is None and months:
            month = pd.Timestamp(f"{months[-1][:4]}-{months[-1][4:]}") + pd.offsets.MonthEnd(0)
        if month is None:
            raise NotFoundError("no peer weights have been produced")
        df = self.store.peers(target, firm_id, month)
        if df.empty:
            raise NotFoundError(f"no peer weights for {firm_id!r} in {month:%Y-%m} (stored months: {months})")
        firms = self.store.firms().set_index("id")
        df = df.sort_values("rank")

        def to_peer(r) -> Peer:
            name = firms["name"].get(r.peer_id)
            industry = firms["ff49_name"].get(r.peer_id)
            return Peer(
                peer_id=r.peer_id,
                name=None if pd.isna(name) else name,
                industry=None if pd.isna(industry) else industry,
                weight=float(r.weight),
                cumulative_weight=float(r.cumweight),
                rank=int(r.rank),
                peer_log_multiple=float(r.peer_y),
            )

        n = int(df["n_peers"].iloc[0])
        most = [to_peer(r) for r in df.head(top).itertuples()]
        least = [to_peer(r) for r in df[df["rank"] > n - top].itertuples()]
        return PeersResponse(
            id=firm_id,
            date=f"{month:%Y-%m}",
            target=target,
            n_peers=n,
            most_comparable=most,
            least_comparable=least,
        )
