"""FastAPI application for the planning approach comparison dashboard."""

import itertools
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-cache aggregate data for all approach pairs on startup."""
    approaches = data_loader.get_approaches()
    pairs = list(itertools.combinations(approaches, 2))
    logger.info(f"Pre-caching aggregate data for {len(pairs)} approach pairs...")

    for a1, a2 in pairs:
        logger.info(f"  Caching {a1} vs {a2}...")
        data_loader.get_aggregate_data(a1, a2)
        # Also cache reverse order
        data_loader.get_aggregate_data(a2, a1)

    logger.info("Pre-caching complete.")
    yield


app = FastAPI(title="Planning Approach Comparison Dashboard", lifespan=lifespan)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    """Serve the main dashboard page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/approaches")
async def get_approaches():
    """List available approaches."""
    return {"approaches": data_loader.get_approaches()}


@app.get("/api/trials")
async def get_trials(
    approach1: str = Query(..., description="First approach"),
    approach2: str = Query(..., description="Second approach"),
):
    """Get common trials between two approaches."""
    common = data_loader.get_common_trials(approach1, approach2)
    return {"trials": common}


@app.get("/api/trial-data")
async def get_trial_data(
    approach: str = Query(..., description="Approach name"),
    trial: str = Query(..., description="Trial name"),
):
    """Load debug_info.pkl data for a trial."""
    data = data_loader.load_trial_data(approach, trial)
    if data is None:
        raise HTTPException(status_code=404, detail="Trial data not found")

    costmap_count = data_loader.get_costmap_count(approach, trial)

    return {
        "accumulated_cost": data["accumulated_cost"],
        "costmap_count": costmap_count,
    }


@app.get("/api/costmap-image")
async def get_costmap_image(
    approach: str = Query(..., description="Approach name"),
    trial: str = Query(..., description="Trial name"),
    index: int = Query(..., description="Image index"),
):
    """Serve a costmap PNG image."""
    image_path = data_loader.get_costmap_path(approach, trial, index)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Costmap image not found")

    return FileResponse(image_path, media_type="image/png")


@app.get("/api/aggregate-data")
async def get_aggregate_data(
    approach1: str = Query(..., description="First approach"),
    approach2: str = Query(..., description="Second approach"),
):
    """Get final costs for all common trials between two approaches."""
    data = data_loader.get_aggregate_data(approach1, approach2)
    return {"data": list(data)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
