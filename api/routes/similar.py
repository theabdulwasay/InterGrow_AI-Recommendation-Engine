from fastapi import APIRouter, HTTPException, Query

from api.dependencies import get_engine
from api.schemas import SimilarItemsResponse

router = APIRouter(tags=["similar items"])


@router.get("/similar/{movie_id}", response_model=SimilarItemsResponse)
def similar(
    movie_id: int, limit: int = Query(default=10, ge=1, le=100)
) -> SimilarItemsResponse:
    try:
        results = get_engine().similar(movie_id, limit)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return SimilarItemsResponse(movie_id=movie_id, similar_items=results)
