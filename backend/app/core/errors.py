"""Domain errors and their HTTP mapping. Services raise these; routers never build responses for errors."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class NotFoundError(Exception):
    """The requested firm, month, table or artifact does not exist in the replication outputs."""


class NotAvailableError(Exception):
    """The output exists in principle but has not been produced by the pipeline yet."""


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(NotAvailableError)
    async def _not_available(_: Request, exc: NotAvailableError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
