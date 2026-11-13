from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import valuation_service
from app.schemas.models import Target, ValuationSet
from app.services.valuation import ValuationService

router = APIRouter(prefix="/valuations", tags=["valuations"])


@router.get("/{firm_id}", response_model=ValuationSet)
def get_valuations(
    firm_id: str,
    date: str | None = Query(None, description="valuation month YYYY-MM (default: the latest)"),
    target: Target | None = Query(None, description="one target (default: the three)"),
    service: ValuationService = Depends(valuation_service),
) -> ValuationSet:
    """Actual and predicted multiple, predicted equity value and valuation error."""
    return service.valuations(firm_id, date, target)
