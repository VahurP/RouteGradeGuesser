from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.ml.model import GradeModel

router = APIRouter(tags=["v1"])


class PredictRequest(BaseModel):
    moves: List[str]


class PredictResponse(BaseModel):
    grade: str
    sorted_moves: List[str]


def get_model() -> GradeModel:
    if not hasattr(get_model, "_instance"):
        get_model._instance = GradeModel()  # type: ignore[attr-defined]
    return get_model._instance  # type: ignore[attr-defined]


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    model: GradeModel = Depends(get_model),
) -> PredictResponse:
    if not req.moves:
        return PredictResponse(grade="N/A", sorted_moves=[])

    grade, sorted_moves = model.predict(req.moves)
    return PredictResponse(grade=grade, sorted_moves=sorted_moves)
