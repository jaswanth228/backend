from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL only: local Postgres, or Supabase cloud (paste its connection string into DATABASE_URL).
    DATABASE_URL: str = ""
    JWT_SECRET: str = "local-demo-secret-not-for-production"
    JWT_EXPIRE_MINUTES: int = 480
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://localhost:8080"
    # Images are stored in Supabase Storage (private bucket). STORAGE_DIR is only used by migrate_local_images.py.
    STORAGE_DIR: str = str(Path(__file__).resolve().parents[1] / "storage")
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    SUPABASE_BUCKET: str = "patient-images"
    AI_FORCE_FAIL: bool = False
    SEED_ON_START: bool = True
    # Also insert synthetic cases at every workflow stage so the medical officer, specialist and navigator pages
    # have data straight from the database without waiting for a health worker to create it.
    SEED_DEMO_DATA: bool = True
    PORT: int = 8000
    # Bootstrap super admin, created on start-up when no super admin exists. CHANGE THE PASSWORD in production.
    SUPER_ADMIN_EMAIL: str = "superadmin@demo.local"
    SUPER_ADMIN_PASSWORD: str = "Demo@123"
    SUPER_ADMIN_NAME: str = "Super Admin"

    # AI image analysis. AI_PROVIDER: "auto" (Ollama Cloud if OLLAMA_API_KEY is set, else mock),
    # "ollama" (always Ollama Cloud) or "mock" (offline demo adapter).
    AI_PROVIDER: str = "auto"
    # When false (default) there is no local blur/size/type gate: the AI judges photo quality itself.
    LOCAL_QUALITY_CHECK: bool = False
    OLLAMA_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "https://ollama.com"
    OLLAMA_MODEL: str = "gemma4"
    OLLAMA_FALLBACK_MODEL: str = ""
    OLLAMA_TIMEOUT_SECONDS: int = 90
    OLLAMA_MAX_IMAGE_PX: int = 1024

    @property
    def use_ollama(self) -> bool:
        provider = self.AI_PROVIDER.strip().lower()
        if provider == "mock":
            return False
        if provider == "ollama":
            return True
        return bool(self.OLLAMA_API_KEY.strip())

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
