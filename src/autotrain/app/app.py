import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from autotrain import __version__, logger
from autotrain.app.api_routes import api_router
from autotrain.app.oauth import attach_oauth
from autotrain.app.ui_routes import ui_router


logger.info("Starting AutoTrain...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()
if "SPACE_ID" in os.environ:
    attach_oauth(app)
else:
    session_secret = os.environ.get("AUTOTRAIN_SESSION_SECRET", "autotrain-local-session")
    app.add_middleware(
        SessionMiddleware,
        secret_key=session_secret,
        same_site="lax",
        https_only=False,
    )

# Mount UI router at root for cleaner URLs in chat mode
app.include_router(ui_router, include_in_schema=False)
app.include_router(api_router, prefix="/api")
static_path = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
logger.info(f"AutoTrain version: {__version__}")
logger.info("AutoTrain started successfully")
