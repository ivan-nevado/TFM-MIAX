from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import valuation_service
from app.schemas.models import PeersResponse, Target
from app.services.valuation import ValuationService

router = APIRouter(prefix="/valuations", tags=["peers"])


@router.get("/{firm_id}/peers", response_model=PeersResponse)
def get_peers(
    firm_id: str,
    date: str | None = Query(
        None, description="month YYYY-MM with stored peer weights (default: the latest)"
    ),
    target: Target = Query("lnm2b"),
    top: int = Query(10, ge=1, le=20),
    service: ValuationService = Depends(valuation_service),
) -> PeersResponse:
    """Most and least comparable training firms according to the GBM peer weights."""
    return service.peers(firm_id, date, target, top)
