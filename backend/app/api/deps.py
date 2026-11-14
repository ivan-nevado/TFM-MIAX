"""Dependency wiring: settings -> repository -> services."""

from __future__ import annotations

from functools import lru_cache

from app.core.settings import get_settings
from app.repositories.store import OutputStore
from app.services.firms import FirmService
from app.services.replication import ReplicationService
from app.services.valuation import ValuationService


@lru_cache
def get_store() -> OutputStore:
    return OutputStore(get_settings())


def replication_service() -> ReplicationService:
    return ReplicationService(get_store())


def firm_service() -> FirmService:
    return FirmService(get_store())


def valuation_service() -> ValuationService:
    return ValuationService(get_store())
