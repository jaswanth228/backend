import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def uid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    email: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(160))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(40), index=True)
    assigned_facility: Mapped[str] = mapped_column(String(120), default="PHC G. Madugula")
    status: Mapped[str] = mapped_column(String(20), default="active")
    district: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    mandal: Mapped[str | None] = mapped_column(String(120), nullable=True)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    suspension_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # specialists only: the hospital they are registered in (set by the district admin) + contact details
    hospital_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    specialty: Mapped[str | None] = mapped_column(String(80), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class District(Base):
    """Master list of districts, maintained by district admins / super admins."""
    __tablename__ = "districts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Mandal(Base):
    __tablename__ = "mandals"
    __table_args__ = (UniqueConstraint("district_id", "name", name="uq_mandal_district_name"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    district_id: Mapped[str] = mapped_column(String(36), ForeignKey("districts.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Hospital(Base):
    """Hospital that patients can be booked into. Added and managed by the medical officer."""
    __tablename__ = "hospitals"
    __table_args__ = (UniqueConstraint("district", "mandal", "name", name="uq_hospital_place_name"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(160), index=True)
    district: Mapped[str] = mapped_column(String(120), index=True)
    mandal: Mapped[str] = mapped_column(String(120), index=True)
    address: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contact_phone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    contact_email: Mapped[str | None] = mapped_column(String(120), nullable=True)
    available: Mapped[bool] = mapped_column(Boolean, default=True)  # False = hidden from booking
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    full_name: Mapped[str] = mapped_column(String(160), index=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(20), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(20), index=True, nullable=True)
    email: Mapped[str | None] = mapped_column(String(120), nullable=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    village: Mapped[str | None] = mapped_column(String(120), nullable=True)
    habitation: Mapped[str | None] = mapped_column(String(120), nullable=True)
    gram_panchayat: Mapped[str | None] = mapped_column(String(120), nullable=True)
    district: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    mandal: Mapped[str | None] = mapped_column(String(120), nullable=True)
    phc: Mapped[str | None] = mapped_column(String(120), nullable=True)
    tobacco_use: Mapped[str | None] = mapped_column(String(80), nullable=True)
    symptoms: Mapped[str | None] = mapped_column(Text, nullable=True)
    medical_history: Mapped[str | None] = mapped_column(Text, nullable=True)
    duplicate_hint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # set by the health worker at registration: only then may a specialist hold a video consultation with this patient
    video_allowed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[str | None] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    facility: Mapped[str] = mapped_column(String(120), default="PHC G. Madugula")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Consent(Base):
    __tablename__ = "consents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    patient_id: Mapped[str] = mapped_column(String(36), ForeignKey("patients.id"), index=True)
    purpose: Mapped[str] = mapped_column(String(80))
    language: Mapped[str] = mapped_column(String(20))
    method: Mapped[str] = mapped_column(String(40), default="verbal_plus_checkbox")
    version: Mapped[str] = mapped_column(String(20), default="v1")
    granted: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Screening(Base):
    __tablename__ = "screenings"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    patient_id: Mapped[str] = mapped_column(String(36), ForeignKey("patients.id"), index=True)
    examiner_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(40), default="DRAFT", index=True)
    symptoms: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration: Mapped[str | None] = mapped_column(String(80), nullable=True)
    tobacco: Mapped[str | None] = mapped_column(String(80), nullable=True)
    exam_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    sync_status: Mapped[str] = mapped_column(String(20), default="confirmed")
    idempotency_key: Mapped[str | None] = mapped_column(String(80), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ImageRecord(Base):
    __tablename__ = "images"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    screening_id: Mapped[str] = mapped_column(String(36), ForeignKey("screenings.id"), index=True)
    view: Mapped[str] = mapped_column(String(40), default="unspecified")
    checksum: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str] = mapped_column(String(255))
    quality_status: Mapped[str] = mapped_column(String(20), default="pending")
    quality_reason: Mapped[str | None] = mapped_column(String(80), nullable=True)
    blur_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AiAnalysis(Base):
    __tablename__ = "ai_analyses"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    image_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("images.id"), nullable=True)
    screening_id: Mapped[str] = mapped_column(String(36), ForeignKey("screenings.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(40))
    provider: Mapped[str] = mapped_column(String(40))
    research_label: Mapped[str | None] = mapped_column(String(80), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="completed")
    warning: Mapped[str] = mapped_column(String(120), default="NOT FOR CLINICAL USE")
    clinical_use: Mapped[bool] = mapped_column(Boolean, default=False)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON from the vision model
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ClinicalReview(Base):
    __tablename__ = "clinical_reviews"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    screening_id: Mapped[str] = mapped_column(String(36), ForeignKey("screenings.id"), index=True)
    doctor_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"))
    finding: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(String(80))
    ai_agreement: Mapped[str] = mapped_column(String(40), default="deferred")
    signed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    version: Mapped[int] = mapped_column(Integer, default=1)


class Referral(Base):
    __tablename__ = "referrals"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    review_id: Mapped[str] = mapped_column(String(36), ForeignKey("clinical_reviews.id"))
    patient_id: Mapped[str] = mapped_column(String(36), ForeignKey("patients.id"), index=True)
    facility: Mapped[str] = mapped_column(String(160))
    specialty: Mapped[str] = mapped_column(String(80), default="Oral surgery / ENT")
    priority: Mapped[str] = mapped_column(String(20), default="routine")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="OPEN")
    # chosen by the medical officer: the hospital, and optionally one specialist there (blank = any specialist of that hospital)
    hospital_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    specialist_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    # written by the specialist after the patient has been seen
    specialist_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_outcome: Mapped[str | None] = mapped_column(String(40), nullable=True)
    analysed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # cross-mandal / cross-district assignment: the medical officers of the hospital's district must accept it first
    cross_mandal: Mapped[bool] = mapped_column(Boolean, default=False)
    cross_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    approval_status: Mapped[str] = mapped_column(String(20), default="NOT_REQUIRED")  # NOT_REQUIRED | PENDING | APPROVED | REJECTED
    approval_note: Mapped[str | None] = mapped_column(String(255), nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Appointment(Base):
    __tablename__ = "appointments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    referral_id: Mapped[str] = mapped_column(String(36), ForeignKey("referrals.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20), default="physical")
    slot: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(40), default="REQUESTED")
    video_ref: Mapped[str | None] = mapped_column(String(160), nullable=True)
    video_provider: Mapped[str] = mapped_column(String(20), default="google_meet")
    video_url: Mapped[str | None] = mapped_column(String(300), nullable=True)
    # hospital chosen for the patient at booking time (physical visits)
    hospital_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("hospitals.id"), nullable=True, index=True)
    specialist_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    booked_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    # cross-mandal: hospital is outside the patient's own mandal -> a medical officer must accept first
    cross_mandal: Mapped[bool] = mapped_column(Boolean, default=False)
    cross_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    approval_status: Mapped[str] = mapped_column(String(20), default="NOT_REQUIRED")  # NOT_REQUIRED | PENDING | APPROVED | REJECTED
    approval_note: Mapped[str | None] = mapped_column(String(255), nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    attendance_outcome: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Investigation(Base):
    __tablename__ = "investigations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    referral_id: Mapped[str] = mapped_column(String(36), ForeignKey("referrals.id"), index=True)
    kind: Mapped[str] = mapped_column(String(80), default="biopsy")
    state: Mapped[str] = mapped_column(String(40), default="ADVISED")
    report_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    clinician_result: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class NavigationTask(Base):
    __tablename__ = "navigation_tasks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    referral_id: Mapped[str] = mapped_column(String(36), ForeignKey("referrals.id"), index=True)
    owner_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    due: Mapped[str | None] = mapped_column(String(40), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="OPEN")
    contact_log: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    actor_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    event: Mapped[str] = mapped_column(String(80), index=True)
    resource: Mapped[str] = mapped_column(String(120))
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Notification(Base):
    __tablename__ = "notifications"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(160))
    body: Mapped[str] = mapped_column(String(255))
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    # ref groups alerts about one thing ("appt:<id>", "xmandal:<id>", "user:<id>", "referral:<id>") so they can be cleared together
    ref: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    # "info" alerts are cleared once their action is taken; "meet" alerts (video links) expire after the scheduled day
    kind: Mapped[str] = mapped_column(String(12), default="info")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_keys"
    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    resource_type: Mapped[str] = mapped_column(String(40))
    resource_id: Mapped[str] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
