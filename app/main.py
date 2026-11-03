from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import Base, SessionLocal, engine, ensure_schema
from .admin_routes import router as admin_router
from .routers import router
from .hospital_routes import router as hospital_router
from .seed import seed_if_empty
from . import supabase_storage


@asynccontextmanager
async def lifespan(_app: FastAPI):
    supabase_storage.ensure_bucket()
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    if settings.SEED_ON_START:
        db = SessionLocal()
        try:
            seed_if_empty(db)
        finally:
            db.close()
    yield


app = FastAPI(
    title="Aarogya Drishti API",
    description="Synthetic-data oral screening workflow for AAROHAN Hackathon 2026. NOT FOR CLINICAL USE.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(admin_router)
app.include_router(hospital_router)
