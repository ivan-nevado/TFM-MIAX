from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import replication_service
from app.core.settings import get_settings
from app.schemas.models import Health
from app.services.replication import ReplicationService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health)
def health(service: ReplicationService = Depends(replication_service)) -> Health:
    return Health(status="ok", run_name=get_settings().run_name, tables_available=len(service.list_tables()))
