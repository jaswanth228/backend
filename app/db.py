from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


def normalize_database_url(raw: str) -> str:
    """Accept any PostgreSQL URL, including the ones Supabase / Heroku-style hosts hand out.

    - postgres:// and postgresql:// are switched to the psycopg2 driver
    - Supabase hosts (…supabase.co, …pooler.supabase.com) get sslmode=require when it is not set
    """
    raw = (raw or "").strip()
    if raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://"):]
    if raw.startswith("postgresql://"):
        raw = "postgresql+psycopg2://" + raw[len("postgresql://"):]
    if not raw.startswith("postgresql"):
        raise RuntimeError("DATABASE_URL must be a PostgreSQL URL (postgresql+psycopg2://user:pass@host:5432/db)")
    url = make_url(raw)
    host = url.host or ""
    if host.endswith("supabase.co") or host.endswith("supabase.com"):
        if "sslmode" not in url.query:
            url = url.update_query_dict({"sslmode": "require"})
    return url.render_as_string(hide_password=False)


DATABASE_URL = normalize_database_url(settings.DATABASE_URL)

# pool_pre_ping + pool_recycle keep connections healthy through cloud poolers (Supabase pgbouncer drops idle ones).
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def ensure_schema() -> None:
    """Tiny forward-only migration so an existing database picks up new columns.

    Base.metadata.create_all() creates missing tables but never alters existing ones.
    """
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE ai_analyses ADD COLUMN IF NOT EXISTS feedback TEXT"))
        # privileges / cross-district assignment (CorrectedRoles)
        conn.execute(text("ALTER TABLE patients ADD COLUMN IF NOT EXISTS video_allowed BOOLEAN"))
        conn.execute(text("UPDATE patients SET video_allowed = TRUE WHERE video_allowed IS NULL"))  # patients registered before the privilege existed keep video
        for col, typ in (("cross_mandal", "BOOLEAN DEFAULT FALSE"), ("cross_reason", "VARCHAR(255)"), ("approval_status", "VARCHAR(20) DEFAULT 'NOT_REQUIRED'"), ("approval_note", "VARCHAR(255)"), ("decided_by", "VARCHAR(36)")):
            conn.execute(text(f"ALTER TABLE referrals ADD COLUMN IF NOT EXISTS {col} {typ}"))
        conn.execute(text("UPDATE referrals SET approval_status = 'NOT_REQUIRED' WHERE approval_status IS NULL"))
        conn.execute(text("UPDATE referrals SET cross_mandal = FALSE WHERE cross_mandal IS NULL"))
        conn.execute(text("ALTER TABLE appointments ADD COLUMN IF NOT EXISTS video_provider VARCHAR(20) DEFAULT 'google_meet'"))
        conn.execute(text("ALTER TABLE appointments ADD COLUMN IF NOT EXISTS video_url VARCHAR(300)"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS district VARCHAR(120)"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS mandal VARCHAR(120)"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS suspended_at TIMESTAMPTZ"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS suspension_reason VARCHAR(255)"))
        conn.execute(text("ALTER TABLE patients ADD COLUMN IF NOT EXISTS district VARCHAR(120)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_district ON users (district)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_patients_district ON patients (district)"))
        for col, typ in (
            ("hospital_id", "VARCHAR(36)"), ("specialist_id", "VARCHAR(36)"), ("booked_by", "VARCHAR(36)"),
            ("cross_mandal", "BOOLEAN DEFAULT FALSE"), ("cross_reason", "VARCHAR(255)"),
            ("approval_status", "VARCHAR(20) DEFAULT 'NOT_REQUIRED'"), ("approval_note", "VARCHAR(255)"), ("decided_by", "VARCHAR(36)"),
        ):
            conn.execute(text(f"ALTER TABLE appointments ADD COLUMN IF NOT EXISTS {col} {typ}"))
        for col, typ in (("hospital_id", "VARCHAR(36)"), ("specialty", "VARCHAR(80)"), ("phone", "VARCHAR(40)")):
            conn.execute(text(f"ALTER TABLE users ADD COLUMN IF NOT EXISTS {col} {typ}"))
        for col in ("hospital_id", "specialist_id"):
            conn.execute(text(f"ALTER TABLE referrals ADD COLUMN IF NOT EXISTS {col} VARCHAR(36)"))
        conn.execute(text("ALTER TABLE referrals ADD COLUMN IF NOT EXISTS specialist_analysis TEXT"))
        conn.execute(text("ALTER TABLE referrals ADD COLUMN IF NOT EXISTS analysis_outcome VARCHAR(40)"))
        conn.execute(text("ALTER TABLE referrals ADD COLUMN IF NOT EXISTS analysed_at TIMESTAMPTZ"))
        conn.execute(text("ALTER TABLE notifications ADD COLUMN IF NOT EXISTS ref VARCHAR(80)"))
        conn.execute(text("ALTER TABLE notifications ADD COLUMN IF NOT EXISTS kind VARCHAR(12) DEFAULT 'info'"))
        conn.execute(text("ALTER TABLE notifications ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_notifications_ref ON notifications (ref)"))
        conn.execute(text("UPDATE appointments SET approval_status = 'NOT_REQUIRED' WHERE approval_status IS NULL"))
        conn.execute(text("UPDATE appointments SET cross_mandal = FALSE WHERE cross_mandal IS NULL"))
        conn.execute(text("UPDATE notifications SET kind = 'info' WHERE kind IS NULL"))
        conn.execute(text("UPDATE users SET role = 'medical_officer' WHERE role = 'health_officer'"))  # merged into one role
        conn.execute(text("UPDATE appointments SET video_provider = 'google_meet' WHERE video_provider IS NULL OR video_provider = 'jitsi'"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
