"""Request and response models of the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Target = Literal["lnm2b", "lnv2a", "lnv2s"]


class Health(BaseModel):
    status: str
    run_name: str
    tables_available: int


class TableResponse(BaseModel):
    table_id: str
    columns: list[str]
    rows: list[dict[str, Any]]


class Firm(BaseModel):
    id: str
    name: str | None = None
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = Field(None, description="Fama-French 49 industry")


class Valuation(BaseModel):
    id: str
    date: str
    target: Target
    multiple: str
    actual_log_multiple: float
    predicted_log_multiple: float
    actual_multiple: float
    predicted_multiple: float
    market_value: float = Field(description="USD millions")
    predicted_equity_value: float = Field(description="USD millions")
    valuation_error: float = Field(description="Mhat / M - 1; negative = expensive according to the model")
    block: int
    n_train: int
    trees: int | None


class ValuationSet(BaseModel):
    id: str
    name: str | None
    date: str
    available_dates: list[str]
    valuations: list[Valuation]


class ShapItem(BaseModel):
    label: str
    value: float
    kind: Literal["base", "contribution", "total"]
    category: str | None = None


class ShapResponse(BaseModel):
    id: str
    date: str
    target: Target
    base_value: float
    prediction: float
    waterfall: list[ShapItem]


class Peer(BaseModel):
    peer_id: str
    name: str | None
    industry: str | None
    weight: float
    cumulative_weight: float
    rank: int
    peer_log_multiple: float


class PeersResponse(BaseModel):
    id: str
    date: str
    target: Target
    n_peers: int
    most_comparable: list[Peer]
    least_comparable: list[Peer]


class ScenarioRequest(BaseModel):
    """Planned contract of the what-if use case (not implemented yet)."""

    id: str
    date: str
    target: Target = "lnm2b"
    changes: dict[str, float] = Field(
        description="relative changes of accounting items, for example sale: -0.20"
    )


class ScenarioPlanned(BaseModel):
    detail: str
    planned_request: dict[str, Any]
    planned_response: dict[str, Any]
