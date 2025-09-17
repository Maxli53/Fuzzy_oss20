"""
Main FastAPI application entry point.
Configures the application, middleware, routes, and WebSocket connections.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import structlog
from prometheus_client import make_asgi_app

from app.core.config import settings
from app.api.endpoints import health, data, validation, collection, backfill
from app.websocket.manager import WebSocketManager
from app.middleware.logging import LoggingMiddleware
from app.middleware.metrics import MetricsMiddleware


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json"
        else structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    Initialize resources on startup and cleanup on shutdown.
    """
    # Startup
    logger.info("Starting Fuzzy OSS20 Trading Platform", version=settings.APP_VERSION)

    # Initialize WebSocket manager
    app.state.ws_manager = WebSocketManager()
    await app.state.ws_manager.initialize()

    # Initialize database connections
    # This will be implemented when we add database services

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application")

    # Cleanup WebSocket connections
    await app.state.ws_manager.shutdown()

    # Close database connections
    # This will be implemented when we add database services

    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# Mount Prometheus metrics endpoint
if settings.ENABLE_METRICS:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

# Include API routers
app.include_router(
    health.router,
    prefix=f"{settings.API_V1_STR}/health",
    tags=["health"]
)

app.include_router(
    data.router,
    prefix=f"{settings.API_V1_STR}/data",
    tags=["data"]
)

app.include_router(
    validation.router,
    prefix=f"{settings.API_V1_STR}/validation",
    tags=["validation"]
)

app.include_router(
    collection.router,
    prefix=f"{settings.API_V1_STR}/collection",
    tags=["collection"]
)

app.include_router(
    backfill.router,
    prefix=f"{settings.API_V1_STR}/backfill",
    tags=["backfill"]
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "api_docs": f"{settings.API_V1_STR}/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )