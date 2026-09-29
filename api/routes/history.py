from fastapi import APIRouter, HTTPException, Query

from api.schemas import ClearHistoryResponse, HistoryResponse
from config import HISTORY_DB_PATH
from src.history.history_manager import clear_history, get_history

router = APIRouter(tags=["history"])


@router.get("/history/{user_id}", response_model=HistoryResponse)
def history(
    user_id: int, limit: int = Query(default=20, ge=1, le=100)
) -> HistoryResponse:
    return HistoryResponse(user_id=user_id, entries=get_history(HISTORY_DB_PATH, user_id, limit))


@router.delete("/history/{user_id}", response_model=ClearHistoryResponse)
def delete_history(user_id: int) -> ClearHistoryResponse:
    deleted = clear_history(HISTORY_DB_PATH, user_id)
    return ClearHistoryResponse(user_id=user_id, deleted_entries=deleted)
