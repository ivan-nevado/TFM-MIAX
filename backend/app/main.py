"""FastAPI application: a read-only API over the replication outputs.

Routers contain no logic, services contain no I/O details, repositories hide where the data live.
Run locally with ``uvicorn app.main:app --reload`` from the ``backend`` folder.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.routers import explanations, firms, health, peers, replication, scenarios, valuations
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.core.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="Relative valuation with machine learning",
        description="Replication of Geertsema & Lu (2023): tables, valuations, SHAP explanations and peers.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(health.router)
    for module in (replication, firms, valuations, explanations, peers, scenarios):
        app.include_router(module.router, prefix="/api/v1")
    return app


app = create_app()
