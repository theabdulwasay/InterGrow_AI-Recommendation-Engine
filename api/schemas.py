from pydantic import BaseModel, Field


class MovieRecommendation(BaseModel):
    movie_id: int
    title: str
    genres: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[MovieRecommendation]


class SimilarItemsResponse(BaseModel):
    movie_id: int
    similar_items: list[MovieRecommendation]


class UserProfileResponse(BaseModel):
    user_id: int
    rating_count: int
    average_rating: float | None
    activity_level: str
    favorite_genres: list[str]
    age: int | None = None
    gender: str | None = None
    occupation: str | None = None
    zipcode: str | None = None


class HistoryEntry(BaseModel):
    id: int
    user_id: int
    created_at: str
    recommendations: list[MovieRecommendation]


class HistoryResponse(BaseModel):
    user_id: int
    entries: list[HistoryEntry]


class ClearHistoryResponse(BaseModel):
    user_id: int
    deleted_entries: int


class HealthResponse(BaseModel):
    status: str
    dataset_ready: bool
