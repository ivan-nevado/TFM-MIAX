from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import replication_service
from app.schemas.models import TableResponse
from app.services.replication import ReplicationService

router = APIRouter(prefix="/replication", tags=["replication"])


@router.get("/tables", response_model=list[str])
def list_tables(service: ReplicationService = Depends(replication_service)) -> list[str]:
    """Identifiers of every table produced by the replication (paper versus ours)."""
    return service.list_tables()


@router.get("/tables/{table_id}", response_model=TableResponse)
def get_table(table_id: str, service: ReplicationService = Depends(replication_service)) -> TableResponse:
    return service.table(table_id)
