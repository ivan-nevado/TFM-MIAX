from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import firm_service
from app.schemas.models import Firm
from app.services.firms import FirmService

router = APIRouter(prefix="/firms", tags=["firms"])


@router.get("", response_model=list[Firm])
def search_firms(
    query: str = Query("", description="part of the identifier (RIC) or of the company name"),
    limit: int = Query(20, ge=1, le=200),
    service: FirmService = Depends(firm_service),
) -> list[Firm]:
    return service.search(query, limit)
