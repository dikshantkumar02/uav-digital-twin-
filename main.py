"""Root FastAPI entry point delegating to app.main:app.

Allows running:
    uvicorn main:app --reload
or:
    python main.py
"""

from app.config import get_settings
from app.main import app

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=(settings.app_env.lower() == "development"),
        log_level=settings.log_level.lower(),
    )
