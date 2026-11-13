from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import valuation_service
from app.schemas.models import ShapResponse, Target
from app.services.valuation import ValuationService

router = APIRouter(prefix="/valuations", tags=["explanations"])


@router.get("/{firm_id}/shap", response_model=ShapResponse)
def get_shap(
    firm_id: str,
    date: str | None = Query(None, description="valuation month YYYY-MM (default: the latest)"),
    target: Target = Query("lnm2b"),
    top: int = Query(5, ge=1, le=30),
    service: ValuationService = Depends(valuation_service),
) -> ShapResponse:
    """SHAP waterfall: training mean, the largest contributions, the rest, the prediction."""
    return service.shap(firm_id, date, target, top)
