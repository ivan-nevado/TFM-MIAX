"""Firm search."""

from __future__ import annotations

import pandas as pd

from app.core.errors import NotFoundError
from app.repositories.store import OutputStore
from app.schemas.models import Firm


def _opt(value) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


class FirmService:
    def __init__(self, store: OutputStore):
        self.store = store

    @staticmethod
    def to_schema(row) -> Firm:
        return Firm(
            id=row["id"],
            name=_opt(row.get("name")),
            exchange=_opt(row.get("exchange")),
            sector=_opt(row.get("sector")),
            industry=_opt(row.get("ff49_name")),
        )

    def search(self, query: str, limit: int) -> list[Firm]:
        firms = self.store.firms()
        q = query.strip().lower()
        if q:
            ids = firms["id"].str.lower()
            names = firms["name"].fillna("").str.lower()
            firms = firms[ids.str.contains(q, regex=False) | names.str.contains(q, regex=False)]
            starts = firms["id"].str.lower().str.startswith(q)
            firms = pd.concat([firms[starts], firms[~starts]])
        return [self.to_schema(r) for _, r in firms.head(limit).iterrows()]

    def get(self, firm_id: str) -> Firm:
        firms = self.store.firms()
        row = firms[firms["id"] == firm_id]
        if row.empty:
            raise NotFoundError(f"firm {firm_id!r} not found")
        return self.to_schema(row.iloc[0])
