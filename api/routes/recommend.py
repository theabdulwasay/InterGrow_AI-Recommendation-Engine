from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_engine
from api.schemas import RecommendationResponse
from config import HISTORY_DB_PATH
from src.history.history_manager import save_history

router = APIRouter(tags=["recommendations"])


@router.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(
    user_id: int, limit: int = Query(default=10, ge=1, le=100)
) -> RecommendationResponse:
    try:
        recommendations = get_engine().recommend(user_id, limit)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    save_history(HISTORY_DB_PATH, user_id, recommendations)
    return RecommendationResponse(user_id=user_id, recommendations=recommendations)
