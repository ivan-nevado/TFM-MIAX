from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.models import ScenarioPlanned, ScenarioRequest

router = APIRouter(prefix="/scenarios", tags=["scenarios"])

PLANNED_RESPONSE = {
    "id": "string",
    "date": "YYYY-MM",
    "target": "lnm2b | lnv2a | lnv2s",
    "baseline": {"predicted_multiple": 0.0, "predicted_equity_value": 0.0},
    "scenario": {"predicted_multiple": 0.0, "predicted_equity_value": 0.0},
    "changed_inputs": [{"variable": "string", "baseline": 0.0, "scenario": 0.0}],
    "shap_delta": [{"variable": "string", "delta": 0.0}],
}


@router.post("", response_model=ScenarioPlanned, status_code=501)
def run_scenario(request: ScenarioRequest) -> JSONResponse:
    """What-if scenarios (for example "sales drop 20% with constant costs") are a later use case.
    The endpoint exists so clients can integrate against the planned contract."""
    body = ScenarioPlanned(
        detail="Scenario analysis is not implemented yet; this is the planned contract.",
        planned_request=ScenarioRequest.model_json_schema(),
        planned_response=PLANNED_RESPONSE,
    )
    return JSONResponse(status_code=501, content=body.model_dump())
