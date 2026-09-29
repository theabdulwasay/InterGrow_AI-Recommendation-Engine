from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from api.dependencies import get_engine, initialize_service
from api.routes import history, profile, recommend, similar
from api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_service()
    yield


app = FastAPI(
    title="Movie Recommendation Engine",
    description="Hybrid movie recommendations, user profiles, similar movies, and history.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(recommend.router)
app.include_router(similar.router)
app.include_router(profile.router)
app.include_router(history.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    try:
        get_engine()
    except FileNotFoundError:
        return HealthResponse(status="dataset required", dataset_ready=False)
    except ValueError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    return HealthResponse(status="ok", dataset_ready=True)
