"""Replication tables (paper versus ours)."""

from __future__ import annotations

import math

from app.repositories.store import OutputStore
from app.schemas.models import TableResponse


def _clean(value):
    return None if isinstance(value, float) and (math.isnan(value) or math.isinf(value)) else value


class ReplicationService:
    def __init__(self, store: OutputStore):
        self.store = store

    def list_tables(self) -> list[str]:
        return self.store.list_tables()

    def table(self, table_id: str) -> TableResponse:
        df = self.store.table(table_id)
        rows = [{k: _clean(v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]
        return TableResponse(table_id=table_id, columns=list(df.columns), rows=rows)
