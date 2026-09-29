from fastapi import APIRouter, HTTPException

from api.dependencies import get_engine
from api.schemas import UserProfileResponse

router = APIRouter(tags=["profiles"])


@router.get("/profile/{user_id}", response_model=UserProfileResponse)
def profile(user_id: int) -> UserProfileResponse:
    try:
        result = get_engine().profile(user_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return UserProfileResponse(**result)
