from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.memory.service import MemoryEngine


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Long-Form Memory System",
        version="0.1.0",
        description="Hybrid long-form memory architecture for 1,000+ turn conversations.",
    )

    app.state.memory_engine = MemoryEngine(settings)
    app.include_router(router)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        app.state.memory_engine.shutdown()

    return app


app = create_app()
