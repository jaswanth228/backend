import hashlib
import json
import re
from pathlib import Path

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from .access import district_officers, appointment_in_area, referral_in_area, assert_patient_access, patient_area_clause, referral_area_clause, same_area, user_area, users_in_area, geo_scoped, assert_screening_access, audit, clear_notifications, get_patient_or_404, notify, referral_visible, specialist_referral_filter
from .config import settings
from .db import get_db
from .geo import GeoScope, clean, geo_tree, validate_geo
from .models import (
    AiAnalysis,
    Appointment,
    AuditEvent,
    ClinicalReview,
    Consent,
    Hospital,
    IdempotencyRecord,
    ImageRecord,
    Investigation,
    NavigationTask,
    Notification,
    Patient,
    Referral,
    Screening,
    User,
    uid,
)
from .security import ADMIN_ROLES, bearer, create_token, current_user, decode_token, hash_password, require_roles, verify_password

from .ai.mock_adapter import analyze as mock_analyze
from .ai.ollama_adapter import OllamaError, analyze as ollama_analyze
from .ai.quality import MAX_BYTES, QualityResult, check_quality

router = APIRouter()
from . import supabase_storage

REQUIRED_CONSENTS = {"examination", "imaging"}


class LoginIn(BaseModel):
    email: str
    password: str


class PatientIn(BaseModel):
    full_name: str
    age: int | None = None
    sex: str | None = None
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    village: str | None = None
    habitation: str | None = None
    gram_panchayat: str | None = None
    district: str | None = None
    mandal: str | None = None
    phc: str | None = None
    tobacco_use: str | None = None
    symptoms: str | None = None
    medical_history: str | None = None
    video_allowed: bool = False  # health worker grants this at registration: lets the specialist hold a video consultation


class ConsentIn(BaseModel):
    purpose: str
    language: str = "en"
    method: str = "checkbox"
    granted: bool = True


class ScreeningIn(BaseModel):
    patient_id: str
    symptoms: str | None = None
    duration: str | None = None
    tobacco: str | None = None
    exam_notes: str | None = None


class ReviewIn(BaseModel):
    screening_id: str
    finding: str
    outcome: str
    ai_agreement: str = "deferred"


class ReferralIn(BaseModel):
    review_id: str
    hospital_id: str  # chosen by the medical officer from the list the district admin maintains
    specialist_id: str | None = None  # optional: blank = any available specialist of that hospital
    facility: str | None = None
    specialty: str | None = None
    priority: str = "urgent"
    reason: str | None = None
    cross_reason: str | None = None  # required when the hospital is in another mandal or district


class AppointmentIn(BaseModel):
    referral_id: str
    kind: str = "physical"
    slot: str
    status: str = "REQUESTED"
    meet_url: str | None = None  # Google Meet link, required for video consultations
    start_now: bool = False  # specialists only: start the Meet immediately instead of scheduling
    hospital_id: str | None = None  # physical visits: the hospital the patient chose (district -> mandal -> hospital)
    specialist_id: str | None = None  # optional: the specialist the patient chose
    cross_reason: str | None = None  # required when the hospital is outside the patient's own mandal


class AppointmentPatch(BaseModel):
    status: str
    attendance_outcome: str | None = None
    slot: str | None = None  # new time when rescheduling
    meet_url: str | None = None  # add or replace the Google Meet link


class InvestigationIn(BaseModel):
    referral_id: str
    kind: str = "biopsy"


class InvestigationPatch(BaseModel):
    state: str
    report_note: str | None = None
    clinician_result: str | None = None


class NavPatch(BaseModel):
    status: str | None = None
    contact_log: str | None = None
    due: str | None = None


class RegisterIn(BaseModel):
    full_name: str
    email: str
    password: str
    role: str
    assigned_facility: str | None = None
    district: str | None = None
    mandal: str | None = None
    hospital_id: str | None = None  # specialists: the hospital they work in (chosen from the admin's list)
    specialty: str | None = None
    phone: str | None = None


class SuspendIn(BaseModel):
    reason: str | None = None


class AnalyzeIn(BaseModel):
    screening_id: str
    image_id: str | None = None


def user_out(u: User) -> dict:
    return {
        "id": u.id,
        "email": u.email,
        "full_name": u.full_name,
        "role": u.role,
        "assigned_facility": u.assigned_facility,
        "status": u.status,
        "district": u.district,
        "mandal": u.mandal,
        "hospital_id": u.hospital_id,
        "specialty": u.specialty,
        "phone": u.phone,
        "suspended_at": u.suspended_at.isoformat() if u.suspended_at else None,
        "suspension_reason": u.suspension_reason,
        "created_at": u.created_at.isoformat() if u.created_at else None,
    }


def patient_out(p: Patient) -> dict:
    return {
        "id": p.id,
        "full_name": p.full_name,
        "age": p.age,
        "sex": p.sex,
        "phone": p.phone,
        "email": p.email,
        "address": p.address,
        "village": p.village,
        "habitation": p.habitation,
        "gram_panchayat": p.gram_panchayat,
        "district": p.district,
        "mandal": p.mandal,
        "phc": p.phc,
        "tobacco_use": p.tobacco_use,
        "symptoms": p.symptoms,
        "medical_history": p.medical_history,
        "duplicate_hint": p.duplicate_hint,
        "facility": p.facility,
        "video_allowed": bool(p.video_allowed),
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


def has_consent(db: Session, patient_id: str, purpose: str) -> bool:
    return (
        db.query(Consent)
        .filter(Consent.patient_id == patient_id, Consent.purpose == purpose, Consent.granted.is_(True))
        .first()
        is not None
    )


def parse_feedback(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


@router.get("/healthz")
def healthz():
    return {"ok": True, "service": "aarogya-drishti-api", "clinical_use": False}


@router.post("/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email.lower()).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    if user.status == "pending":
        raise HTTPException(403, "Your registration is awaiting admin approval. You can sign in once it is approved.")
    if user.status == "rejected":
        raise HTTPException(403, "Your registration was not approved. Please contact the district admin.")
    if user.status == "suspended":
        raise HTTPException(403, "Your account has been suspended. Please contact the district admin to restore access.")
    if user.status != "active":
        raise HTTPException(403, "This account is not active.")
    audit(db, user.id, "login", f"user:{user.id}")
    db.commit()
    return {"token": create_token(user), "user": user_out(user)}


SELF_REGISTER_ROLES = {"health_worker", "navigator", "medical_officer", "specialist"}  # specialists pick their hospital; the district admin approves (or adds them on the Specialists page)


@router.get("/public/hospitals")
def public_hospitals(district: str | None = None, mandal: str | None = None, db: Session = Depends(get_db)):
    """Hospital list for the registration form (no login yet): names and places only."""
    q = db.query(Hospital).filter(Hospital.available.is_(True))
    if clean(district):
        q = q.filter(Hospital.district == clean(district))
    if clean(mandal):
        q = q.filter(Hospital.mandal == clean(mandal))
    return [{"id": h.id, "name": h.name, "district": h.district, "mandal": h.mandal} for h in q.order_by(Hospital.name).all()]


@router.post("/auth/register", status_code=201)
def register_account(body: RegisterIn, db: Session = Depends(get_db)):
    """Public self-registration. The account stays `pending` until an admin approves it."""
    email = body.email.strip().lower()
    name = body.full_name.strip()
    if body.role not in SELF_REGISTER_ROLES:
        raise HTTPException(422, "Choose a valid designation")
    if len(name) < 2 or "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(422, "Enter your full name and a valid email")
    if len(body.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(409, "An account with this email already exists")
    district, mandal = validate_geo(db, body.district, body.mandal)
    facility = (body.assigned_facility or "").strip() or "PHC G. Madugula"
    hospital_id = specialty = phone = None
    if body.role == "specialist":
        hospital = db.get(Hospital, body.hospital_id) if body.hospital_id else None
        if not hospital or not hospital.available:
            raise HTTPException(422, "Select the hospital you work in from the list")
        hospital_id, facility, district, mandal = hospital.id, hospital.name, hospital.district, hospital.mandal
        specialty, phone = clean(body.specialty), clean(body.phone)
    user = User(
        email=email,
        full_name=name,
        password_hash=hash_password(body.password),
        role=body.role,
        assigned_facility=facility,
        district=district,
        mandal=mandal,
        hospital_id=hospital_id,
        specialty=specialty,
        phone=phone,
        status="pending",
    )
    db.add(user)
    db.flush()
    label = body.role.replace("_", " ")
    for admin in db.query(User).filter(User.role.in_(ADMIN_ROLES), User.status == "active").all():
        notify(db, admin.id, "New registration awaiting approval", f"{name} applied as {label}. Review it under Pending approvals.", ref=f"user:{user.id}")
    audit(db, user.id, "user.register", f"user:{user.id}", json.dumps({"role": body.role}))
    db.commit()
    return {"ok": True, "status": "pending", "message": "Registration received. An admin must approve it before you can sign in."}


@router.get("/auth/me")
def me(user: User = Depends(current_user)):
    return user_out(user)


@router.post("/patients")
def create_patient(
    body: PatientIn,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("health_worker")),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    if idempotency_key:
        existing = db.get(IdempotencyRecord, idempotency_key)
        if existing:
            return patient_out(db.get(Patient, existing.resource_id))
    dup = None
    if body.phone:
        dup = db.query(Patient).filter(Patient.phone == body.phone).first()
    if not dup and body.full_name and body.village:
        dup = (
            db.query(Patient)
            .filter(Patient.full_name == body.full_name, Patient.village == body.village)
            .first()
        )
    district = clean(body.district) or user.district
    mandal = clean(body.mandal) or (user.mandal if district == user.district else None)
    district, mandal = validate_geo(db, district, mandal)
    data = body.model_dump()
    data["district"], data["mandal"] = district, mandal
    patient = Patient(
        **data,
        created_by=user.id,
        facility=user.assigned_facility,
        duplicate_hint=f"Possible duplicate of {dup.id}" if dup else None,
    )
    db.add(patient)
    db.flush()
    if idempotency_key:
        db.add(IdempotencyRecord(key=idempotency_key, resource_type="patient", resource_id=patient.id))
    audit(db, user.id, "patient.create", f"patient:{patient.id}")
    db.commit()
    db.refresh(patient)
    return patient_out(patient)


@router.get("/patients")
def list_patients(q: str | None = None, db: Session = Depends(get_db), user: User = Depends(current_user)):
    if user.role == "admin":
        raise HTTPException(403, "Administrators receive aggregate data only")
    query = db.query(Patient)
    if user.role in {"health_worker", "medical_officer"}:
        query = query.filter(Patient.facility == user.assigned_facility)
    area = patient_area_clause(user)
    if area is not None and user.role in {"health_worker", "medical_officer"}:
        query = query.filter(area)  # own district + mandal only
    if user.role in {"specialist", "navigator"}:
        referred = db.query(Referral.patient_id)
        if user.role == "specialist":
            referred = referred.filter(specialist_referral_filter(user))
        else:
            rc = referral_area_clause(user)
            if rc is not None:
                referred = referred.filter(rc)
        query = query.filter(Patient.id.in_(referred))
    if q:
        like = f"%{q}%"
        query = query.filter((Patient.full_name.ilike(like)) | (Patient.phone.ilike(like)))
    print([patient_out(p) for p in query.order_by(Patient.created_at.desc()).limit(100).all()])
    return [patient_out(p) for p in query.order_by(Patient.created_at.desc()).limit(100).all()]


@router.get("/patients/{patient_id}")
def get_patient(patient_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    patient = get_patient_or_404(db, patient_id)
    assert_patient_access(user, patient, db)
    audit(db, user.id, "patient.view", f"patient:{patient.id}")
    db.commit()
    return patient_out(patient)


@router.post("/patients/{patient_id}/consents")
def add_consent(
    patient_id: str,
    body: ConsentIn,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("health_worker")),
):
    patient = get_patient_or_404(db, patient_id)
    assert_patient_access(user, patient, db)
    rec = Consent(
        patient_id=patient_id,
        purpose=body.purpose,
        language=body.language,
        method=body.method,
        granted=body.granted,
        created_by=user.id,
    )
    db.add(rec)
    audit(db, user.id, "consent.create", f"patient:{patient_id}", body.purpose)
    db.commit()
    db.refresh(rec)
    return {
        "id": rec.id,
        "patient_id": rec.patient_id,
        "purpose": rec.purpose,
        "language": rec.language,
        "granted": rec.granted,
        "version": rec.version,
        "created_at": rec.created_at.isoformat(),
    }


@router.get("/patients/{patient_id}/consents")
def list_consents(patient_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    patient = get_patient_or_404(db, patient_id)
    assert_patient_access(user, patient, db)
    rows = db.query(Consent).filter(Consent.patient_id == patient_id).all()
    return [
        {
            "id": c.id,
            "purpose": c.purpose,
            "language": c.language,
            "granted": c.granted,
            "version": c.version,
            "created_at": c.created_at.isoformat(),
        }
        for c in rows
    ]


@router.post("/screenings")
def create_screening(
    body: ScreeningIn,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("health_worker")),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    if idempotency_key:
        existing = db.get(IdempotencyRecord, idempotency_key)
        if existing:
            row = db.get(Screening, existing.resource_id)
            return {"id": row.id, "status": row.status, "sync_status": row.sync_status}
    patient = get_patient_or_404(db, body.patient_id)
    assert_patient_access(user, patient, db)
    if not has_consent(db, patient.id, "examination"):
        raise HTTPException(400, "Examination consent is required before screening")
    rec = Screening(
        patient_id=patient.id,
        examiner_id=user.id,
        status="SCREENED",
        symptoms=body.symptoms,
        duration=body.duration,
        tobacco=body.tobacco,
        exam_notes=body.exam_notes,
        sync_status="confirmed",
        idempotency_key=idempotency_key,
    )
    db.add(rec)
    db.flush()
    if idempotency_key:
        db.add(IdempotencyRecord(key=idempotency_key, resource_type="screening", resource_id=rec.id))
    audit(db, user.id, "screening.create", f"screening:{rec.id}")
    db.commit()
    return {"id": rec.id, "status": rec.status, "sync_status": rec.sync_status, "patient_id": rec.patient_id}


@router.get("/screenings")
def list_screenings(
    status: str | None = None,
    awaiting_review: bool = False,
    patient_id: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    """Screenings read straight from the database.

    awaiting_review=true returns every screening of the user's scope that has no signed review yet, whatever stage
    the health worker left it in (the medical officer never waits for a particular status).
    """
    if user.role == "admin":
        raise HTTPException(403, "Administrators receive aggregate data only")
    if user.role == "navigator":
        raise HTTPException(403, "Navigators cannot list clinical screenings")
    q = db.query(Screening)
    if patient_id:
        q = q.filter(Screening.patient_id == patient_id)
    if status:
        q = q.filter(Screening.status == status)
    if awaiting_review:
        reviewed = db.query(ClinicalReview.screening_id)
        q = q.filter(~Screening.id.in_(reviewed))
    if user.role in {"health_worker", "medical_officer"}:
        patients = db.query(Patient.id).filter(Patient.facility == user.assigned_facility)
        area = patient_area_clause(user)
        if area is not None:
            patients = patients.filter(area)
        q = q.filter(Screening.patient_id.in_(patients))
    if user.role == "specialist":
        referred = db.query(Referral.patient_id).filter(specialist_referral_filter(user))
        q = q.filter(Screening.patient_id.in_(referred))
    rows = q.order_by(Screening.created_at.desc()).limit(200).all()
    patient_ids = {s.patient_id for s in rows}
    people = {p.id: p for p in db.query(Patient).filter(Patient.id.in_(patient_ids)).all()} if patient_ids else {}
    analysed = {r[0] for r in db.query(AiAnalysis.screening_id).filter(AiAnalysis.screening_id.in_([s.id for s in rows])).all()} if rows else set()
    return [
        {
            "id": s.id,
            "patient_id": s.patient_id,
            "patient_name": people[s.patient_id].full_name if s.patient_id in people else None,
            "village": people[s.patient_id].village if s.patient_id in people else None,
            "patient_district": people[s.patient_id].district if s.patient_id in people else None,
            "patient_mandal": people[s.patient_id].mandal if s.patient_id in people else None,
            "status": s.status,
            "symptoms": s.symptoms,
            "tobacco": s.tobacco,
            "exam_notes": s.exam_notes,
            "has_ai": s.id in analysed,
            "created_at": s.created_at.isoformat(),
        }
        for s in rows
    ]


@router.get("/screenings/{screening_id}")
def get_screening(screening_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    s = db.get(Screening, screening_id)
    if not s:
        raise HTTPException(404, "Screening not found")
    assert_screening_access(db, user, s)
    images = db.query(ImageRecord).filter(ImageRecord.screening_id == s.id).order_by(ImageRecord.created_at).all()
    ai = db.query(AiAnalysis).filter(AiAnalysis.screening_id == s.id).order_by(AiAnalysis.created_at).all()
    reviews = db.query(ClinicalReview).filter(ClinicalReview.screening_id == s.id).all()
    return {
        "id": s.id,
        "patient_id": s.patient_id,
        "status": s.status,
        "symptoms": s.symptoms,
        "duration": s.duration,
        "tobacco": s.tobacco,
        "exam_notes": s.exam_notes,
        "images": [
            {
                "id": i.id,
                "view": i.view,
                "quality_status": i.quality_status,
                "quality_reason": i.quality_reason,
            }
            for i in images
        ],
        "ai_analyses": [
            {
                "id": a.id,
                "image_id": a.image_id,
                "status": a.status,
                "research_label": a.research_label,
                "model_version": a.model_version,
                "provider": a.provider,
                "warning": a.warning,
                "clinical_use": a.clinical_use,
                "feedback": parse_feedback(a.feedback),
            }
            for a in ai
        ],
        "reviews": [
            {
                "id": r.id,
                "finding": r.finding,
                "outcome": r.outcome,
                "ai_agreement": r.ai_agreement,
                "signed_at": r.signed_at.isoformat(),
            }
            for r in reviews
        ],
    }


@router.post("/screenings/{screening_id}/images")
def upload_image(
    screening_id: str,
    view: str = Form("unspecified"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("health_worker")),
):
    s = db.get(Screening, screening_id)
    if not s:
        raise HTTPException(404, "Screening not found")
    assert_screening_access(db, user, s)
    if not has_consent(db, s.patient_id, "imaging"):
        raise HTTPException(400, "Imaging consent is required before photograph upload")
    data = file.file.read()
    if settings.LOCAL_QUALITY_CHECK:
        quality = check_quality(data, file.content_type)
    else:
        # No local gate: the AI decides whether the photo is good enough. Only reject files that cannot be processed at all.
        if not data:
            raise HTTPException(400, "The uploaded file is empty")
        if len(data) > MAX_BYTES:
            raise HTTPException(400, "The photo is larger than 8 MB; please upload a smaller one")
        quality = QualityResult("pending_ai", None)
    checksum = hashlib.sha256(data).hexdigest()
    image_id = uid()
    ext, mime = supabase_storage.detect_type(data)
    storage_key = f"{s.id}/{image_id}{ext}"  # grouped per screening inside the Supabase bucket
    try:
        supabase_storage.upload(storage_key, data, mime)
    except Exception as exc:
        raise HTTPException(502, f"Image storage unavailable: {exc}")
    rec = ImageRecord(
        id=image_id,
        screening_id=s.id,
        view=view,
        checksum=checksum,
        storage_key=storage_key,
        quality_status=quality.status,
        quality_reason=quality.reason,
        blur_score=quality.blur_score,
        created_by=user.id,
    )
    db.add(rec)
    audit(db, user.id, "image.upload", f"image:{image_id}")
    db.commit()
    return {
        "id": rec.id,
        "quality_status": rec.quality_status,
        "quality_reason": rec.quality_reason,
        "blur_score": rec.blur_score,
        "warning": "Photographs supplement examination and are not a diagnosis.",
    }


@router.get("/screenings/{screening_id}/images/{image_id}/file")
def download_image(
    screening_id: str,
    image_id: str,
    access_token: str | None = Query(default=None),
    db: Session = Depends(get_db),
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
):
    token = creds.credentials if creds else access_token
    if not token:
        raise HTTPException(401, "Authentication required")
    data = decode_token(token)
    user = db.get(User, data.get("sub"))
    if user is None:
        raise HTTPException(401, "Authentication required")
    s = db.get(Screening, screening_id)
    if not s:
        raise HTTPException(404, "Screening not found")
    if user.role == "navigator":
        raise HTTPException(403, "Photograph access is restricted")
    assert_screening_access(db, user, s)
    rec = db.get(ImageRecord, image_id)
    if not rec or rec.screening_id != screening_id:
        raise HTTPException(404, "Image not found")
    try:
        blob = supabase_storage.download(rec.storage_key)
    except Exception as exc:
        raise HTTPException(502, f"Image storage unavailable: {exc}")
    if blob is None:
        raise HTTPException(404, "Stored file missing")
    audit(db, user.id, "image.view", f"image:{image_id}")
    db.commit()
    return Response(content=blob, media_type=supabase_storage.mime_for_key(rec.storage_key), headers={"X-Content-Type-Options": "nosniff"})


def _record_failed_analysis(db: Session, s: Screening, image_id: str | None, provider: str, model: str, user: User, event: str, reason: str | None = None) -> AiAnalysis:
    rec = AiAnalysis(
        feedback=json.dumps({"error": reason}, ensure_ascii=False) if reason else None,
        screening_id=s.id,
        image_id=image_id,
        model_version=model[:40],
        provider=provider,
        status="failed",
        research_label=None,
    )
    db.add(rec)
    db.flush()
    s.status = "AWAITING_CLINICIAN"
    audit(db, user.id, event, f"screening:{s.id}")
    db.commit()
    return rec


@router.post("/ai/analyze")
def ai_analyze(body: AnalyzeIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    if user.role not in {"health_worker", "medical_officer"}:
        raise HTTPException(403, "Not permitted")
    s = db.get(Screening, body.screening_id)
    if not s:
        raise HTTPException(404, "Screening not found")
    assert_screening_access(db, user, s)
    use_ollama = settings.use_ollama
    provider_name = "ollama" if use_ollama else "mock"
    model_name = settings.OLLAMA_MODEL if use_ollama else "mock-v1"
    if settings.AI_FORCE_FAIL:
        rec = _record_failed_analysis(db, s, body.image_id, provider_name, model_name, user, "ai.fail")
        raise HTTPException(
            503,
            detail={
                "message": "AI adapter unavailable. Screening is preserved for manual clinician review.",
                "analysis_id": rec.id,
                "status": "failed",
                "clinical_use": False,
            },
        )
    image = None
    if body.image_id:
        image = db.get(ImageRecord, body.image_id)
        if image and image.screening_id != s.id:
            raise HTTPException(404, "Image not found for this screening")
    else:
        q = db.query(ImageRecord).filter(ImageRecord.screening_id == s.id)
        if settings.LOCAL_QUALITY_CHECK:
            q = q.filter(ImageRecord.quality_status == "acceptable")
        image = q.order_by(ImageRecord.created_at.desc()).first()
    if not image:
        raise HTTPException(400, "No photo available for analysis. Upload a photo first.")
    if settings.LOCAL_QUALITY_CHECK and image.quality_status != "acceptable":
        rec = AiAnalysis(
            screening_id=s.id,
            image_id=image.id,
            model_version=model_name[:40],
            provider=provider_name,
            status="image_insufficient",
            research_label=None,
        )
        db.add(rec)
        db.commit()
        return {
            "analysis_id": rec.id,
            "status": "image_insufficient",
            "image_quality": image.quality_status,
            "image_quality_reason": image.quality_reason,
            "clinical_use": False,
            "warning": "NOT FOR CLINICAL USE",
        }
    try:
        raw = supabase_storage.download(image.storage_key)
    except Exception as exc:
        raise HTTPException(502, f"Image storage unavailable: {exc}")
    if raw is None:
        raise HTTPException(404, "Stored file missing")
    if hashlib.sha256(raw).hexdigest() != image.checksum:
        raise HTTPException(409, "The stored photo does not match its upload record. Please upload the photo again.")

    if use_ollama:
        patient = db.get(Patient, s.patient_id)
        context = {
            "view": image.view,
            "age": patient.age if patient else None,
            "sex": patient.sex if patient else None,
            "symptoms": s.symptoms,
            "duration": s.duration,
            "tobacco": s.tobacco,
            "exam_notes": s.exam_notes,
        }
        try:
            result = ollama_analyze(raw, s.id, context)
        except OllamaError as exc:
            rec = _record_failed_analysis(db, s, image.id, "ollama", model_name, user, "ai.fail", reason=str(exc))
            raise HTTPException(
                503,
                detail={
                    "message": f"AI analysis unavailable: {exc}. Screening is preserved for manual clinician review.",
                    "analysis_id": rec.id,
                    "status": "failed",
                    "clinical_use": False,
                },
            )
    else:
        result = mock_analyze(raw, s.id)
        result.setdefault("status", "completed")
        result.setdefault("feedback", None)

    rec = AiAnalysis(
        screening_id=s.id,
        image_id=image.id,
        model_version=result["model_version"],
        provider=result["provider"],
        research_label=result["research_label"],
        status=result["status"],
        warning=result["warning"],
        clinical_use=False,
        feedback=json.dumps(result["feedback"], ensure_ascii=False) if result.get("feedback") else None,
    )
    db.add(rec)
    db.flush()
    fb = result.get("feedback") or {}
    if not settings.LOCAL_QUALITY_CHECK and fb.get("image_assessment"):
        # Show the AI's quality verdict on the photo instead of a local blur/size result.
        assess = fb["image_assessment"]
        image.quality_status = "acceptable" if assess.get("usable", True) else "recapture"
        issues = assess.get("issues") or []
        image.quality_reason = str(issues[0])[:80] if (issues and not assess.get("usable", True)) else None
    s.status = "AWAITING_CLINICIAN"
    audit(db, user.id, "ai.analyze", f"analysis:{rec.id}")
    db.commit()
    return {
        "analysis_id": rec.id,
        "screening_id": s.id,
        "image_id": image.id,
        "status": rec.status,
        "image_quality": image.quality_status,
        "provider": rec.provider,
        "model_version": rec.model_version,
        "research_label": rec.research_label,
        "score": None,
        "clinical_use": False,
        "requires_clinician_review": True,
        "warning": rec.warning,
        "feedback": result.get("feedback"),
    }


@router.post("/reviews")
def create_review(body: ReviewIn, db: Session = Depends(get_db), user: User = Depends(require_roles("medical_officer"))):
    s = db.get(Screening, body.screening_id)
    if not s:
        raise HTTPException(404, "Screening not found")
    assert_screening_access(db, user, s)
    existing = db.query(ClinicalReview).filter(ClinicalReview.screening_id == s.id).first()
    if existing:
        raise HTTPException(409, "A signed review already exists and cannot be silently overwritten")
    rec = ClinicalReview(
        screening_id=s.id,
        doctor_id=user.id,
        finding=body.finding,
        outcome=body.outcome,
        ai_agreement=body.ai_agreement,
    )
    db.add(rec)
    s.status = "REVIEWED"
    audit(db, user.id, "review.sign", f"review:{rec.id}")
    db.commit()
    db.refresh(rec)
    return {
        "id": rec.id,
        "screening_id": s.id,
        "finding": rec.finding,
        "outcome": rec.outcome,
        "ai_agreement": rec.ai_agreement,
        "signed_at": rec.signed_at.isoformat(),
    }


def _activate_referral(db: Session, rec: Referral, hospital: Hospital, specialist: User | None, patient: Patient | None) -> None:
    """Open the referral for work: navigation task, navigator and specialist alerts (after any cross-district approval)."""
    db.add(NavigationTask(referral_id=rec.id, status="OPEN", due="7d"))
    for nav in users_in_area(db, "navigator", (patient.district if patient else None, patient.mandal if patient else None), (hospital.district, hospital.mandal)):
        notify(db, nav.id, "New referral", f"A patient was referred to {hospital.name}. Guide them to the hospital.", ref=f"referral:{rec.id}")
    targets = [specialist] if specialist else db.query(User).filter(User.role == "specialist", User.status == "active", User.hospital_id == hospital.id).all()
    for sp in targets:
        notify(db, sp.id, "New referred case", f"A patient was referred to you at {hospital.name}.", ref=f"referral:{rec.id}")


@router.post("/referrals")
def create_referral(body: ReferralIn, db: Session = Depends(get_db), user: User = Depends(require_roles("medical_officer"))):
    """The medical officer assigns a hospital (+ optional specialist). Same mandal: assigned at once.
    Another mandal or district: the request goes to the medical officers of the hospital's district first."""
    review = db.get(ClinicalReview, body.review_id)
    if not review:
        raise HTTPException(404, "Review not found")
    hospital = db.get(Hospital, body.hospital_id)
    if not hospital or not hospital.available:
        raise HTTPException(422, "Select an available hospital for this referral")
    specialist: User | None = None
    if body.specialist_id:
        specialist = db.get(User, body.specialist_id)
        if not specialist or specialist.role != "specialist" or specialist.status != "active":
            raise HTTPException(422, "Choose an active specialist")
        if specialist.hospital_id != hospital.id:
            raise HTTPException(422, "That specialist is not registered at the selected hospital")
    s = db.get(Screening, review.screening_id)
    _p = db.get(Patient, s.patient_id)
    if _p is not None and not same_area(user, _p.district, _p.mandal):
        raise HTTPException(403, "This patient belongs to another mandal")
    if db.query(Referral.id).filter(Referral.review_id == review.id, Referral.status != "CANCELLED").first():
        raise HTTPException(409, "A specialist has already been assigned for this review")
    norm = lambda v: (v or "").strip().lower()
    cross = bool(_p and (norm(_p.mandal) != norm(hospital.mandal) or norm(_p.district) != norm(hospital.district)))
    reason = clean(body.cross_reason)
    if cross and (not reason or len(reason) < 5):
        raise HTTPException(422, "This hospital is outside the patient's mandal or district. Write the reason; the medical officers of that district will decide")
    rec = Referral(
        review_id=review.id,
        patient_id=s.patient_id,
        facility=hospital.name,
        hospital_id=hospital.id,
        specialist_id=specialist.id if specialist else None,
        specialty=(body.specialty or (specialist.specialty if specialist else None) or "Oral surgery / ENT"),
        priority=body.priority,
        reason=body.reason,
        created_by=user.id,
        cross_mandal=cross,
        cross_reason=reason[:255] if cross and reason else None,
        approval_status="PENDING" if cross else "NOT_REQUIRED",
        status="PENDING_APPROVAL" if cross else "OPEN",
    )
    db.add(rec)
    db.flush()
    s.status = "REFERRAL_PLANNED"
    if cross:
        text = f"A patient of {_p.mandal}, {_p.district} needs {hospital.name} ({hospital.mandal}, {hospital.district}). Reason: {reason}"
        tag = f" [xref:{rec.id}]"
        deciders = [o for o in district_officers(db, hospital.district) if o.id != user.id] or db.query(User).filter(User.role == "super_admin", User.status == "active").all()
        for o in deciders:
            notify(db, o.id, "Cross-mandal / district request", text[: 255 - len(tag)] + tag, ref=f"xref:{rec.id}")
        audit(db, user.id, "referral.cross_request", f"referral:{rec.id}", json.dumps({"hospital": hospital.id}))
    else:
        _activate_referral(db, rec, hospital, specialist, _p)
        audit(db, user.id, "referral.create", f"referral:{rec.id}", json.dumps({"hospital": hospital.id, "specialist": specialist.id if specialist else None}))
    db.commit()
    return referral_out(rec, db.get(Patient, rec.patient_id), db)


@router.get("/patients/{patient_id}/assignment-options")
def assignment_options(patient_id: str, db: Session = Depends(get_db), user: User = Depends(require_roles("medical_officer"))):
    """Page data for 'Assign specialist': the patient, signed 'refer' reviews still waiting for a specialist, and the
    hospitals of the patient's own mandal with their available specialists."""
    patient = get_patient_or_404(db, patient_id)
    assert_patient_access(user, patient, db)
    reviews = []
    for sc in db.query(Screening).filter(Screening.patient_id == patient.id).order_by(Screening.created_at.desc()).all():
        for r in db.query(ClinicalReview).filter(ClinicalReview.screening_id == sc.id).all():
            if r.outcome != "refer":
                continue
            ref = db.query(Referral).filter(Referral.review_id == r.id, Referral.status != "CANCELLED").first()
            reviews.append({"id": r.id, "finding": r.finding, "signed_at": r.signed_at.isoformat() if r.signed_at else None,
                            "referral_id": ref.id if ref else None, "referral_status": ref.status if ref else None})
    open_states = ("ASSESSED", "CARE_LINKED", "CANCELLED")
    hospitals = []
    q = db.query(Hospital).filter(Hospital.available.is_(True))
    for h in q.order_by(Hospital.name).all():
        if (h.district or "").strip().lower() != (patient.district or "").strip().lower() or (h.mandal or "").strip().lower() != (patient.mandal or "").strip().lower():
            continue
        specs = db.query(User).filter(User.role == "specialist", User.status == "active", User.hospital_id == h.id).order_by(User.full_name).all()
        hospitals.append({
            "id": h.id, "name": h.name, "mandal": h.mandal, "district": h.district, "address": h.address, "contact_phone": h.contact_phone,
            "specialists": [{
                "id": sp.id, "full_name": sp.full_name, "specialty": sp.specialty, "phone": sp.phone,
                "open_cases": db.query(Referral.id).filter(Referral.specialist_id == sp.id, Referral.status.notin_(open_states)).count(),
            } for sp in specs],
        })
    return {"patient": patient_out(patient), "reviews": reviews, "hospitals": hospitals}


def _active_appointment(db: Session | None, referral_id: str) -> dict | None:
    if not db:
        return None
    a = (
        db.query(Appointment)
        .filter(Appointment.referral_id == referral_id, Appointment.status != "CANCELLED")
        .order_by(Appointment.created_at.desc())
        .first()
    )
    if not a:
        return None
    return {"id": a.id, "kind": a.kind, "slot": a.slot, "status": a.status, "has_meet_link": bool(a.video_url), "approval_status": a.approval_status or "NOT_REQUIRED"}


class AnalysisIn(BaseModel):
    analysis: str
    outcome: str = "follow_up"


@router.post("/referrals/{referral_id}/analysis")
def save_analysis(referral_id: str, body: AnalysisIn, db: Session = Depends(get_db), user: User = Depends(require_roles("specialist"))):
    """The specialist records their analysis after the patient has visited (or had the video consult)."""
    ref = db.get(Referral, referral_id)
    if not ref or not referral_visible(user, ref, db):
        raise HTTPException(404, "Referral not found")
    if ref.specialist_id and ref.specialist_id != user.id:
        raise HTTPException(403, "This patient was accepted by another specialist")
    if body.outcome not in {"treatment_started", "biopsy_advised", "follow_up", "no_issue", "referred_higher"}:
        raise HTTPException(422, "Choose an outcome")
    text_ = (body.analysis or "").strip()
    if len(text_) < 5:
        raise HTTPException(422, "Write your analysis of the patient")
    appt = db.query(Appointment).filter(Appointment.referral_id == ref.id, Appointment.status != "CANCELLED").order_by(Appointment.created_at.desc()).first()
    if not appt:
        raise HTTPException(409, "Accept the patient (hospital visit or video consult) before recording the analysis")
    ref.specialist_id = ref.specialist_id or user.id
    ref.specialist_analysis, ref.analysis_outcome, ref.analysed_at, ref.status = text_, body.outcome, datetime.now(timezone.utc), "ASSESSED"
    if appt.status != "COMPLETED":
        appt.status = "COMPLETED"
        notify_appointment(db, appt, user, "Specialist consultation completed", "The specialist has recorded the analysis. No clinical details included.", roles={"health_worker", "medical_officer", "navigator"})
    clear_notifications(db, f"referral:{ref.id}")
    audit(db, user.id, "referral.analysis", f"referral:{ref.id}", body.outcome)
    db.commit()
    return referral_out(ref, db.get(Patient, ref.patient_id), db)


def referral_out(r: Referral, patient: Patient | None = None, db: Session | None = None) -> dict:
    name = patient.full_name if patient else None
    village = patient.village if patient else None
    hospital = db.get(Hospital, r.hospital_id) if db and r.hospital_id else None
    specialist = db.get(User, r.specialist_id) if db and r.specialist_id else None
    return {
        "id": r.id,
        "patient_id": r.patient_id,
        "patient_name": name,
        "village": village,
        "district": patient.district if patient else None,
        "mandal": patient.mandal if patient else None,
        "facility": r.facility,
        "hospital_id": r.hospital_id,
        "hospital_name": hospital.name if hospital else r.facility,
        "hospital_mandal": hospital.mandal if hospital else None,
        "hospital_district": hospital.district if hospital else None,
        "specialist_id": r.specialist_id,
        "specialist_name": specialist.full_name if specialist else None,
        "specialty": r.specialty,
        "priority": r.priority,
        "status": r.status,
        "reason": r.reason,
        "cross_mandal": bool(r.cross_mandal),
        "cross_reason": r.cross_reason,
        "approval_status": r.approval_status or "NOT_REQUIRED",
        "approval_note": r.approval_note,
        "video_allowed": bool(patient.video_allowed) if patient else False,
        "patient_age": patient.age if patient else None,
        "patient_sex": patient.sex if patient else None,
        "patient_symptoms": patient.symptoms if patient else None,
        "specialist_analysis": r.specialist_analysis,
        "analysis_outcome": r.analysis_outcome,
        "analysed_at": r.analysed_at.isoformat() if r.analysed_at else None,
        "appointment": _active_appointment(db, r.id),
        "slip": {
            "en": f"Referral {r.id[:8]} to {r.facility} ({r.specialty}). Priority: {r.priority}.",
            "te": f"రెఫరల్ {r.id[:8]} — {r.facility} ({r.specialty}). ప్రాధాన్యం: {r.priority}.",
        },
    }


@router.get("/referrals")
def list_referrals(db: Session = Depends(get_db), user: User = Depends(require_roles("medical_officer", "specialist", "navigator", "super_admin"))):
    q = db.query(Referral)
    if user.role == "navigator":
        q = q.filter(Referral.approval_status.notin_(["PENDING", "REJECTED"]))
    if user.role == "specialist":
        q = q.filter(specialist_referral_filter(user))  # only patients the medical officer promoted to this specialist / hospital
    rc = referral_area_clause(user)
    if rc is not None:
        q = q.filter(rc)  # medical officers / navigators: own district + mandal only
    out = []
    for r in q.order_by(Referral.created_at.desc()).all():
        out.append(referral_out(r, db.get(Patient, r.patient_id), db))
    return out


@router.get("/referrals/{referral_id}")
def get_referral(referral_id: str, db: Session = Depends(get_db), user: User = Depends(require_roles("medical_officer", "specialist", "navigator", "super_admin"))):
    rec = db.get(Referral, referral_id)
    if not rec or not referral_visible(user, rec, db):
        raise HTTPException(404, "Referral not found")
    return referral_out(rec, db.get(Patient, rec.patient_id), db)


MEET_RE = re.compile(r"^https://meet\.google\.com/(?:lookup/)?[A-Za-z0-9_-]{3,40}(?:-[A-Za-z0-9_-]{2,40}){0,3}(?:\?[^\s#]*)?$")


def clean_meet_url(url: str | None) -> str:
    url = (url or "").strip()
    if not MEET_RE.match(url):
        raise HTTPException(422, "Enter a valid Google Meet link, for example https://meet.google.com/abc-defg-hij")
    return url


# ---------- appointment / video-call notifications ----------

def parse_slot(slot: str | None) -> datetime | None:
    """Slots are ISO strings from the scheduler. Older free-text slots return None."""
    if not slot:
        return None
    try:
        dt = datetime.fromisoformat(slot.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fmt_slot(slot: str | None) -> str:
    """Human text with the weekday, kept in the time zone the scheduler chose, e.g. 'Mon, 02 Nov 2026, 10:00 (UTC+05:30)'."""
    dt = parse_slot(slot)
    if not dt:
        return slot or "time to be confirmed"
    off = dt.strftime("%z")
    return f"{dt.strftime('%a, %d %b %Y, %H:%M')} (UTC{off[:3]}:{off[3:]})"


IST = timezone(timedelta(hours=5, minutes=30))


def end_of_slot_day(slot: str | None) -> datetime:
    """23:59:59 of the day the appointment is scheduled for (UTC slots are read in IST, the service area's time zone)."""
    dt = parse_slot(slot) or datetime.now(IST)
    if dt.utcoffset() == timedelta(0):
        dt = dt.astimezone(IST)
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)


def is_meet_alert(rec: Appointment, title: str) -> bool:
    """Video-link alerts stay until the scheduled day is over; cancelled / missed / completed notices are plain alerts."""
    return rec.kind == "video" and not any(w in title.lower() for w in ("cancelled", "completed", "missed"))


def officers_for(db: Session, *places: tuple[str | None, str | None]) -> list[User]:
    """Active medical officers whose own district + mandal matches one of the given (district, mandal) places.

    Falls back to super admins (never to every officer) so a request is not exposed to unrelated mandals.
    """
    rows = users_in_area(db, "medical_officer", *places)
    return rows or db.query(User).filter(User.role == "super_admin", User.status == "active").all()


def appointment_parties(db: Session, ref: Referral) -> dict[str, set[str]]:
    """Who should hear about an appointment, grouped by role. Names/clinical details are never sent."""
    parties: dict[str, set[str]] = {"health_worker": set(), "medical_officer": set(), "navigator": set(), "specialist": set()}
    review = db.get(ClinicalReview, ref.review_id)
    screening = db.get(Screening, review.screening_id) if review else None
    patient = db.get(Patient, ref.patient_id)
    if screening and screening.examiner_id:
        parties["health_worker"].add(screening.examiner_id)
    if patient and patient.created_by:
        parties["health_worker"].add(patient.created_by)
    if review and review.doctor_id:
        parties["medical_officer"].add(review.doctor_id)
    _hosp = db.get(Hospital, ref.hospital_id) if ref.hospital_id else None
    _places = [(patient.district, patient.mandal)] if patient else []
    if _hosp:
        _places.append((_hosp.district, _hosp.mandal))
    for u in users_in_area(db, "navigator", *_places):
        parties["navigator"].add(u.id)
    if ref.specialist_id:
        parties["specialist"].add(ref.specialist_id)
    elif ref.hospital_id:
        for u in db.query(User).filter(User.role == "specialist", User.status == "active", User.hospital_id == ref.hospital_id).all():
            parties["specialist"].add(u.id)
    if not parties["health_worker"] and patient:
        for u in db.query(User).filter(User.role == "health_worker", User.status == "active", User.assigned_facility == patient.facility).all():
            parties["health_worker"].add(u.id)
    return parties


def notify_appointment(db: Session, rec: Appointment, actor: User, title: str, body: str, roles: set[str] | None = None, dedupe_minutes: int = 0, extra_ids: set[str] | None = None) -> int:
    ref = db.get(Referral, rec.referral_id)
    if not ref:
        return 0
    tag = f" [appt:{rec.id}]"
    sent = 0
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=dedupe_minutes)
    seen: set[str] = set()
    meet = is_meet_alert(rec, title)
    expires = end_of_slot_day(rec.slot) if meet else None
    groups = [ids for role, ids in appointment_parties(db, ref).items() if roles is None or role in roles]
    if extra_ids:
        groups.append(extra_ids)
    for ids in groups:
        for uid_ in ids:
            if uid_ == actor.id or uid_ in seen:
                continue
            seen.add(uid_)
            if dedupe_minutes and (
                db.query(Notification)
                .filter(Notification.user_id == uid_, Notification.title == title, Notification.body.like(f"%{rec.id}%"), Notification.created_at >= cutoff)
                .first()
            ):
                continue
            notify(db, uid_, title, (body[: 255 - len(tag)] + tag), ref=f"appt:{rec.id}", kind="meet" if meet else "info", expires_at=expires)
            sent += 1
    return sent


def rollback_appointment_notifications(db: Session, appointment_id: str) -> int:
    """Remove every alert that was sent about this appointment (scheduled, live, joined, reminder...).

    Alerts carry an [appt:<id>] tag, so ending or deleting a call takes them away from every user's list.
    """
    return db.query(Notification).filter(
        or_(Notification.body.like(f"%[appt:{appointment_id}]%"), Notification.ref.in_([f"appt:{appointment_id}", f"xmandal:{appointment_id}"]))
    ).delete(synchronize_session=False)


def make_call_reminders(db: Session, user: User) -> None:
    """Lazy 'starting soon' reminders: created when the user's notifications are fetched."""
    now = datetime.now(timezone.utc)
    rows = db.query(Appointment).filter(Appointment.kind == "video", Appointment.status.in_(["REQUESTED", "CONFIRMED", "RESCHEDULED"])).all()
    changed = False
    for a in rows:
        dt = parse_slot(a.slot)
        if not dt or not (now - timedelta(minutes=15) <= dt <= now + timedelta(minutes=30)):
            continue
        ref = db.get(Referral, a.referral_id)
        if not ref or user.id not in set().union(*appointment_parties(db, ref).values()):
            continue
        title = "Video consultation starting soon"
        exists = db.query(Notification).filter(Notification.user_id == user.id, Notification.title == title, Notification.body.like(f"%{a.id}%")).first()
        if not exists:
            notify(db, user.id, title, f"Scheduled for {fmt_slot(a.slot)}. Be ready to join. [appt:{a.id}]", ref=f"appt:{a.id}", kind="meet", expires_at=end_of_slot_day(a.slot))
            changed = True
    if changed:
        db.commit()


@router.post("/appointments")
def create_appointment(
    body: AppointmentIn,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("navigator", "specialist", "medical_officer")),
):
    ref = db.get(Referral, body.referral_id)
    if not ref or not referral_visible(user, ref, db):
        raise HTTPException(404, "Referral not found")
    if body.kind not in {"video", "physical"}:
        raise HTTPException(422, "Unknown appointment type")
    if body.kind == "video":
        if user.role != "specialist":
            raise HTTPException(403, "Video consultations are held by the specialist only")
        _vp = db.get(Patient, ref.patient_id)
        if _vp is not None and not _vp.video_allowed:
            raise HTTPException(403, "Video consultation is not enabled for this patient. The health worker grants it when registering the patient")
    if body.start_now:
        if user.role != "specialist":
            raise HTTPException(403, "Only a specialist can start a video consultation immediately")
        if body.kind != "video":
            raise HTTPException(422, "Only video consultations can be started immediately")
        slot = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        status = "CONFIRMED"
    else:
        slot = body.slot
        status = body.status
        if parse_slot(slot) is None:
            raise HTTPException(422, "Pick a valid date and time")
    meet_url = clean_meet_url(body.meet_url) if body.kind == "video" else None

    # --- hospital + specialist chosen for the patient (patient picks district -> mandal -> hospital -> specialist)
    patient = db.get(Patient, ref.patient_id)
    hospital: Hospital | None = None
    cross, cross_reason = False, None
    hospital_id = body.hospital_id or ref.hospital_id  # defaults to the hospital the medical officer chose
    if body.kind == "physical":
        if not hospital_id:
            raise HTTPException(422, "Select the hospital the patient chose (district, mandal, then hospital)")
        hospital = db.get(Hospital, hospital_id)
        if not hospital or not hospital.available:
            raise HTTPException(422, "That hospital is not available for booking. Pick another one")
        if patient and patient.mandal and hospital.id != ref.hospital_id and (
            (patient.mandal or "").strip().lower() != hospital.mandal.strip().lower()
            or (patient.district or hospital.district).strip().lower() != hospital.district.strip().lower()
        ):
            cross = True
            cross_reason = clean(body.cross_reason)
            if not cross_reason or len(cross_reason) < 5:
                raise HTTPException(422, "This hospital is outside the patient's mandal. Give the special reason so the medical officer can decide")
    specialist: User | None = None
    specialist_id = body.specialist_id or (ref.specialist_id if not body.hospital_id or body.hospital_id == ref.hospital_id else None)
    if user.role == "specialist" and not specialist_id:
        specialist_id = user.id
    if specialist_id:
        specialist = db.get(User, specialist_id)
        if not specialist or specialist.role != "specialist" or specialist.status != "active":
            raise HTTPException(422, "Choose an active specialist")
        if hospital and specialist.hospital_id != hospital.id:
            raise HTTPException(422, "That specialist is not registered at the selected hospital")
    if user.role == "specialist" and not ref.specialist_id:
        ref.specialist_id = user.id  # accepting a hospital-level referral claims it for this doctor
    pending = cross and not body.start_now
    rec = Appointment(
        referral_id=ref.id,
        kind=body.kind,
        slot=slot,
        status="PENDING_APPROVAL" if pending else status,
        video_provider="google_meet",
        video_url=meet_url,
        video_ref=meet_url.rsplit("/", 1)[-1].split("?")[0] if meet_url else None,
        hospital_id=hospital.id if hospital else None,
        specialist_id=specialist.id if specialist else None,
        booked_by=user.id,
        cross_mandal=cross,
        cross_reason=cross_reason[:255] if cross_reason else None,
        approval_status="PENDING" if pending else "NOT_REQUIRED",
    )
    db.add(rec)
    s = db.query(Screening).join(ClinicalReview, ClinicalReview.screening_id == Screening.id).filter(
        ClinicalReview.id == ref.review_id
    ).first()
    if s and not pending:
        s.status = "APPOINTED"
    db.flush()
    clear_notifications(db, f"referral:{ref.id}")  # the "new referral" alerts are answered once an appointment is booked
    label = "Video consultation" if rec.kind == "video" else "Hospital appointment"
    where = f" at {hospital.name}, {hospital.mandal}" if hospital else ""
    who = f" with {specialist.full_name}" if specialist else ""
    if pending:
        text = (
            f"Patient from {patient.mandal}, {patient.district or '—'} wants {hospital.name} ({hospital.mandal}, {hospital.district}) "
            f"on {fmt_slot(rec.slot)}. Reason: {cross_reason}"
        )
        tag = f" [xmandal:{rec.id}]"
        for officer in officers_for(db, (hospital.district, hospital.mandal), (patient.district, patient.mandal)):
            if officer.id == user.id:
                continue  # never ask the person who made the request to approve it
            notify(db, officer.id, "Cross-mandal appointment request", text[: 255 - len(tag)] + tag, ref=f"xmandal:{rec.id}")
        audit(db, user.id, "appointment.cross_mandal_request", f"appointment:{rec.id}", json.dumps({"hospital": hospital.id}))
    elif body.start_now:
        notify_appointment(db, rec, user, "Video consultation is live", "The specialist has started the Google Meet call. Join now with the patient.", roles={"health_worker", "navigator"})
        audit(db, user.id, "appointment.start_now", f"appointment:{rec.id}")
    else:
        via = " on Google Meet" if rec.kind == "video" else ""
        notify_appointment(db, rec, user, f"{label} scheduled", f"{label}{via}{where}{who} booked for {fmt_slot(rec.slot)}. No clinical details included.")
        audit(db, user.id, "appointment.create", f"appointment:{rec.id}")
    db.commit()
    db.refresh(rec)
    return appointment_out(rec, db)


@router.patch("/appointments/{appointment_id}")
def patch_appointment(
    appointment_id: str,
    body: AppointmentPatch,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("navigator", "specialist", "medical_officer")),
):
    rec = db.get(Appointment, appointment_id)
    if not rec:
        raise HTTPException(404, "Appointment not found")
    if not appointment_in_area(db, user, rec):
        raise HTTPException(404, "Appointment not found")
    if user.role == "specialist":
        _ref = db.get(Referral, rec.referral_id)
        if rec.specialist_id != user.id and not (_ref and referral_visible(user, _ref, db)):
            raise HTTPException(403, "This appointment belongs to another specialist")
    allowed = {"REQUESTED", "CONFIRMED", "COMPLETED", "MISSED", "CANCELLED", "RESCHEDULED"}
    if body.status not in allowed:
        raise HTTPException(400, "Invalid appointment status")
    if rec.approval_status == "PENDING" and body.status != "CANCELLED":
        raise HTTPException(409, "This cross-mandal request is waiting for the medical officer's decision. You can only withdraw it")
    if rec.approval_status == "PENDING":  # withdrawn before the officer answered
        rec.approval_status = "WITHDRAWN"
        rec.approval_note = "Withdrawn by the person who booked it"
    rec.status = body.status
    rec.attendance_outcome = body.attendance_outcome
    if body.slot:
        rec.slot = body.slot
        # video-link alerts now expire after the NEW scheduled day
        db.query(Notification).filter(Notification.ref == f"appt:{rec.id}", Notification.kind == "meet").update({"expires_at": end_of_slot_day(rec.slot)}, synchronize_session=False)
    clear_notifications(db, f"appt:{rec.id}")  # an action was taken: earlier plain alerts about it are superseded (video links stay)
    if body.meet_url and rec.kind == "video":
        rec.video_url = clean_meet_url(body.meet_url)
        rec.video_provider = "google_meet"
        rec.video_ref = rec.video_url.rsplit("/", 1)[-1].split("?")[0]
    label = "Video consultation" if rec.kind == "video" else "Appointment"
    if body.status == "MISSED":
        task = NavigationTask(referral_id=rec.referral_id, status="OVERDUE", contact_log="Missed appointment follow-up")
        db.add(task)
        notify_appointment(db, rec, user, "Missed appointment", "Follow-up required. No clinical details included.")
    elif body.status == "RESCHEDULED":
        notify_appointment(db, rec, user, f"{label} rescheduled", f"New time: {fmt_slot(rec.slot)}.")
    elif body.status == "CANCELLED":
        rollback_appointment_notifications(db, rec.id)  # drop the earlier scheduled / live / reminder alerts first
        notify_appointment(db, rec, user, f"{label} cancelled", f"The booking for {fmt_slot(rec.slot)} was cancelled.")
    elif body.status == "CONFIRMED":
        notify_appointment(db, rec, user, f"{label} confirmed", f"Confirmed for {fmt_slot(rec.slot)}.")
    elif body.status == "COMPLETED":
        notify_appointment(db, rec, user, f"{label} completed", "The consultation was marked completed.", roles={"health_worker", "medical_officer", "navigator"})
    if body.status == "COMPLETED":
        ref = db.get(Referral, rec.referral_id)
        if ref:
            ref.status = "ASSESSED"
    audit(db, user.id, "appointment.status", f"appointment:{rec.id}", body.status)
    db.commit()
    return {"id": rec.id, "status": rec.status, "attendance_outcome": rec.attendance_outcome}


@router.post("/appointments/{appointment_id}/end")
def end_call(
    appointment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("navigator", "specialist", "medical_officer")),
):
    """Host stops the Google Meet: the link is removed, the booking is cancelled and every earlier alert about it is rolled back."""
    rec = db.get(Appointment, appointment_id)
    if not rec:
        raise HTTPException(404, "Appointment not found")
    if not appointment_in_area(db, user, rec):
        raise HTTPException(404, "Appointment not found")
    if rec.kind != "video":
        raise HTTPException(422, "Only video consultations have a call to end")
    rec.video_url = None
    rec.video_ref = None
    rec.status = "CANCELLED"
    removed = rollback_appointment_notifications(db, rec.id)
    audit(db, user.id, "appointment.end_call", f"appointment:{rec.id}", f"alerts_removed={removed}")
    db.commit()
    return {"id": rec.id, "status": rec.status, "has_meet_link": False, "alerts_removed": removed}


@router.delete("/appointments/{appointment_id}")
def delete_appointment(
    appointment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("navigator", "specialist", "medical_officer")),
):
    """Delete the appointment (and its Meet link) and roll back every alert about it."""
    rec = db.get(Appointment, appointment_id)
    if not rec:
        raise HTTPException(404, "Appointment not found")
    if not appointment_in_area(db, user, rec):
        raise HTTPException(404, "Appointment not found")
    referral_id = rec.referral_id
    removed = rollback_appointment_notifications(db, rec.id)
    db.delete(rec)
    db.flush()
    ref = db.get(Referral, referral_id)
    if ref and db.query(Appointment).filter(Appointment.referral_id == referral_id).count() == 0:
        s = db.query(Screening).join(ClinicalReview, ClinicalReview.screening_id == Screening.id).filter(ClinicalReview.id == ref.review_id).first()
        if s and s.status == "APPOINTED":
            s.status = "REFERRAL_PLANNED"
    audit(db, user.id, "appointment.delete", f"appointment:{appointment_id}", f"alerts_removed={removed}")
    db.commit()
    return {"deleted": True, "id": appointment_id, "alerts_removed": removed}


def navigation_details(db: Session, ref: Referral | None, appt: Appointment | None = None) -> dict:
    """Everything a navigator needs to guide the patient: patient contact + the specialist and hospital they will meet."""
    patient = db.get(Patient, ref.patient_id) if ref else None
    hid = (appt.hospital_id if appt and appt.hospital_id else None) or (ref.hospital_id if ref else None)
    sid = (appt.specialist_id if appt and appt.specialist_id else None) or (ref.specialist_id if ref else None)
    h = db.get(Hospital, hid) if hid else None
    sp = db.get(User, sid) if sid else None
    return {
        "patient_name": patient.full_name if patient else None,
        "patient_age": patient.age if patient else None,
        "patient_sex": patient.sex if patient else None,
        "patient_phone": patient.phone if patient else None,
        "patient_address": ", ".join(x for x in [patient.address, patient.village, patient.mandal, patient.district] if x) if patient else None,
        "patient_village": patient.village if patient else None,
        "nav_specialist_name": sp.full_name if sp else None,
        "nav_specialist_phone": sp.phone if sp else None,
        "nav_specialist_specialty": sp.specialty if sp else (ref.specialty if ref else None),
        "nav_hospital_name": h.name if h else (ref.facility if ref else None),
        "nav_hospital_address": h.address if h else None,
        "nav_hospital_phone": h.contact_phone if h else None,
        "nav_hospital_place": f"{h.mandal}, {h.district}" if h else None,
    }


def appointment_out(a: Appointment, db: Session | None = None, detail: bool = False) -> dict:
    hospital = db.get(Hospital, a.hospital_id) if db and a.hospital_id else None
    specialist = db.get(User, a.specialist_id) if db and a.specialist_id else None
    extra: dict = {}
    if detail and db:  # navigators / specialists also get who the patient is and who they will meet
        extra = navigation_details(db, db.get(Referral, a.referral_id), a)
    return {
        **extra,
        "hospital_id": a.hospital_id,
        "hospital_name": hospital.name if hospital else None,
        "hospital_mandal": hospital.mandal if hospital else None,
        "hospital_district": hospital.district if hospital else None,
        "hospital_address": hospital.address if hospital else None,
        "hospital_phone": hospital.contact_phone if hospital else None,
        "specialist_id": a.specialist_id,
        "specialist_name": specialist.full_name if specialist else None,
        "cross_mandal": bool(a.cross_mandal),
        "cross_reason": a.cross_reason,
        "approval_status": a.approval_status or "NOT_REQUIRED",
        "approval_note": a.approval_note,
        "id": a.id,
        "referral_id": a.referral_id,
        "kind": a.kind,
        "slot": a.slot,
        "status": a.status,
        "video_ref": a.video_ref,
        "has_meet_link": bool(a.video_url),
        "attendance_outcome": a.attendance_outcome,
    }


@router.get("/appointments")
def list_appointments(db: Session = Depends(get_db), user: User = Depends(require_roles("navigator", "specialist", "medical_officer", "health_worker", "super_admin"))):
    rows = db.query(Appointment).order_by(Appointment.created_at.desc()).all()
    if geo_scoped(user):
        rows = [a for a in rows if appointment_in_area(db, user, a)]
    if user.role == "specialist":
        mine = {r.id for r in db.query(Referral).filter(specialist_referral_filter(user)).all()}
        rows = [a for a in rows if a.referral_id in mine or a.specialist_id == user.id]
    return [appointment_out(a, db, detail=user.role in {"navigator", "specialist"}) for a in rows]


@router.get("/appointments/{appointment_id}")
def get_appointment(
    appointment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("navigator", "specialist", "medical_officer", "health_worker", "super_admin")),
):
    rec = db.get(Appointment, appointment_id)
    if not rec:
        raise HTTPException(404, "Appointment not found")
    if not appointment_in_area(db, user, rec):
        raise HTTPException(404, "Appointment not found")
    if user.role == "specialist":
        ref = db.get(Referral, rec.referral_id)
        if rec.specialist_id != user.id and not (ref and referral_visible(user, ref, db)):
            raise HTTPException(404, "Appointment not found")
    return appointment_out(rec, db, detail=user.role in {"navigator", "specialist"})


@router.get("/appointments/{appointment_id}/video")
def video_token(
    appointment_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("specialist", "health_worker")),  # video belongs to the specialist (the health worker sits with the patient)
):
    rec = db.get(Appointment, appointment_id)
    if not rec or rec.kind != "video":
        raise HTTPException(404, "Video appointment not found")
    if rec.status in {"CANCELLED"}:
        raise HTTPException(403, "Meeting link is not active")
    audit(db, user.id, "video.join", f"appointment:{rec.id}")
    if user.role == "health_worker":
        notify_appointment(db, rec, user, "Health worker joined the video call", "The patient side is ready. Join the consultation.", roles={"specialist"}, dedupe_minutes=10)
    else:
        notify_appointment(db, rec, user, "Video consultation is live", "The doctor has started the call. Join now with the patient.", roles={"health_worker", "navigator"}, dedupe_minutes=10)
    db.commit()
    if not rec.video_url:
        raise HTTPException(409, "No Google Meet link has been added to this appointment yet")
    return {
        "provider": "google_meet",
        "room": rec.video_ref,
        "url": rec.video_url,
        "recording_default": False,
        "warning": "Synthetic demo consultation only.",
    }


@router.post("/investigations")
def create_investigation(
    body: InvestigationIn,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("specialist", "medical_officer")),
):
    ref = db.get(Referral, body.referral_id)
    if not ref or not referral_visible(user, ref, db):
        raise HTTPException(404, "Referral not found")
    rec = Investigation(referral_id=ref.id, kind=body.kind, state="ADVISED")
    db.add(rec)
    audit(db, user.id, "investigation.create", f"investigation:{rec.id}")
    db.commit()
    db.refresh(rec)
    return {"id": rec.id, "kind": rec.kind, "state": rec.state, "referral_id": rec.referral_id}


@router.patch("/investigations/{investigation_id}")
def patch_investigation(
    investigation_id: str,
    body: InvestigationPatch,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("specialist", "medical_officer")),
):
    rec = db.get(Investigation, investigation_id)
    if not rec:
        raise HTTPException(404, "Investigation not found")
    if user.role == "specialist":
        _ref = db.get(Referral, rec.referral_id)
        if not (_ref and referral_visible(user, _ref, db)):
            raise HTTPException(403, "This case belongs to another specialist")
    allowed = {
        "ADVISED",
        "SCHEDULED",
        "SAMPLE_COLLECTED",
        "REPORT_PENDING",
        "REPORT_RECEIVED",
        "CLINICIAN_REVIEWED",
    }
    if body.state not in allowed:
        raise HTTPException(400, "Invalid investigation state")
    rec.state = body.state
    if body.report_note is not None:
        rec.report_note = body.report_note
    if body.clinician_result is not None:
        rec.clinician_result = body.clinician_result
    if body.state == "CLINICIAN_REVIEWED":
        ref = db.get(Referral, rec.referral_id)
        if ref:
            s = (
                db.query(Screening)
                .join(ClinicalReview, ClinicalReview.screening_id == Screening.id)
                .filter(ClinicalReview.id == ref.review_id)
                .first()
            )
            if s:
                s.status = "RESULT_REVIEWED"
    audit(db, user.id, "investigation.status", f"investigation:{rec.id}", body.state)
    db.commit()
    return {
        "id": rec.id,
        "state": rec.state,
        "report_note": rec.report_note,
        "clinician_result": rec.clinician_result,
    }


@router.get("/investigations")
def list_investigations(db: Session = Depends(get_db), user: User = Depends(current_user)):
    if user.role == "admin":
        raise HTTPException(403, "Administrators receive aggregate data only")
    rows = db.query(Investigation).order_by(Investigation.created_at.desc()).all()
    rc = referral_area_clause(user)
    if rc is not None:
        mine = {x.id for x in db.query(Referral).filter(rc).all()}
        rows = [i for i in rows if i.referral_id in mine]
    if user.role == "specialist":
        mine = {r.id for r in db.query(Referral).filter(specialist_referral_filter(user)).all()}
        rows = [i for i in rows if i.referral_id in mine]
    return [{"id": i.id, "referral_id": i.referral_id, "kind": i.kind, "state": i.state, "report_note": i.report_note, "clinician_result": i.clinician_result} for i in rows]


@router.get("/navigation")
def list_nav(db: Session = Depends(get_db), user: User = Depends(require_roles("navigator", "medical_officer", "super_admin"))):
    rows = db.query(NavigationTask).order_by(NavigationTask.created_at.desc()).all()
    rc = referral_area_clause(user)
    if rc is not None:
        ok = {x.id for x in db.query(Referral).filter(rc).all()}
        rows = [t for t in rows if t.referral_id in ok]
    out = []
    for t in rows:
        ref = db.get(Referral, t.referral_id)
        patient = db.get(Patient, ref.patient_id) if ref else None
        appt = db.query(Appointment).filter(Appointment.referral_id == t.referral_id, Appointment.status != "CANCELLED").order_by(Appointment.created_at.desc()).first()
        out.append(
            {
                **navigation_details(db, ref, appt),
                "id": t.id,
                "referral_id": t.referral_id,
                "village": patient.village if patient else None,
                "facility": ref.facility if ref else None,
                "priority": ref.priority if ref else None,
                "appointment_slot": appt.slot if appt else None,
                "appointment_status": appt.status if appt else None,
                "appointment_kind": appt.kind if appt else None,
                "status": t.status,
                "due": t.due,
                "contact_log": t.contact_log,
            }
        )
    return out


@router.patch("/navigation/{task_id}")
def patch_nav(task_id: str, body: NavPatch, db: Session = Depends(get_db), user: User = Depends(require_roles("navigator"))):
    rec = db.get(NavigationTask, task_id)
    if not rec:
        raise HTTPException(404, "Task not found")
    _rf = db.get(Referral, rec.referral_id)
    if _rf is not None and not referral_in_area(db, user, _rf):
        raise HTTPException(404, "Task not found")
    if body.status:
        rec.status = body.status
    if body.contact_log:
        rec.contact_log = (rec.contact_log or "") + "\n" + body.contact_log
    if body.due:
        rec.due = body.due
    rec.owner_id = user.id
    if rec.status == "CLOSED":
        ref = db.get(Referral, rec.referral_id)
        if ref:
            ref.status = "CARE_LINKED"
            s = (
                db.query(Screening)
                .join(ClinicalReview, ClinicalReview.screening_id == Screening.id)
                .filter(ClinicalReview.id == ref.review_id)
                .first()
            )
            if s:
                s.status = "CARE_LINKED"
    audit(db, user.id, "navigation.update", f"nav:{rec.id}")
    db.commit()
    return {"id": rec.id, "status": rec.status, "contact_log": rec.contact_log}


@router.get("/patients/{patient_id}/timeline")
def timeline(patient_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    patient = get_patient_or_404(db, patient_id)
    assert_patient_access(user, patient, db)
    events = []
    events.append({"type": "registration", "at": patient.created_at.isoformat(), "summary": "Patient registered"})
    for c in db.query(Consent).filter(Consent.patient_id == patient_id):
        events.append({"type": "consent", "at": c.created_at.isoformat(), "summary": f"Consent {c.purpose}"})
    for s in db.query(Screening).filter(Screening.patient_id == patient_id):
        events.append({"type": "screening", "at": s.created_at.isoformat(), "summary": f"Screening {s.status}", "id": s.id})
        for a in db.query(AiAnalysis).filter(AiAnalysis.screening_id == s.id):
            events.append({"type": "ai", "at": a.created_at.isoformat(), "summary": f"AI {a.status} ({a.research_label or 'n/a'}) — NOT CLINICAL"})
        for r in db.query(ClinicalReview).filter(ClinicalReview.screening_id == s.id):
            events.append({"type": "review", "at": r.signed_at.isoformat(), "summary": f"Signed review: {r.outcome}"})
    for ref in db.query(Referral).filter(Referral.patient_id == patient_id):
        events.append({"type": "referral", "at": ref.created_at.isoformat(), "summary": f"Referral to {ref.facility}", "id": ref.id})
        for ap in db.query(Appointment).filter(Appointment.referral_id == ref.id):
            events.append({"type": "appointment", "at": ap.created_at.isoformat(), "summary": f"{ap.kind} {ap.status}", "id": ap.id})
        for inv in db.query(Investigation).filter(Investigation.referral_id == ref.id):
            events.append({"type": "investigation", "at": inv.created_at.isoformat(), "summary": f"{inv.kind} {inv.state}"})
    events.sort(key=lambda e: e["at"])
    return {"patient": patient_out(patient), "events": events}


def _admin_district(user: User, requested: str | None) -> str | None:
    """A district admin is locked to their own district; only a super admin can look at other districts or all of them."""
    if user.role == "admin" and user.district:
        return user.district
    return requested


@router.get("/dashboard/summary")
def dashboard(
    district: str | None = None,
    mandal: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(*ADMIN_ROLES, "medical_officer")),
):
    """Programme indicators. District admins / super admins may narrow them to one district and/or mandal."""
    S = GeoScope(db, _admin_district(user, district), mandal) if user.role in ADMIN_ROLES else GeoScope(db, user.district, user.mandal)
    payload = {
        "screenings": S.q(Screening).count(),
        "reviews": S.q(ClinicalReview).count(),
        "referrals": S.q(Referral).count(),
        "appointments_completed": S.q(Appointment).filter(Appointment.status == "COMPLETED").count(),
        "appointments_overdue": S.q(Appointment).filter(Appointment.status.in_(["MISSED", "REQUESTED"])).count(),
        "investigations_pending": S.q(Investigation).filter(Investigation.state != "CLINICIAN_REVIEWED").count(),
        "investigations_done": S.q(Investigation).filter(Investigation.state == "CLINICIAN_REVIEWED").count(),
        "navigation_open": S.q(NavigationTask).filter(NavigationTask.status.in_(["OPEN", "OVERDUE"])).count(),
        "ai_suspicious_demo": S.q(AiAnalysis).filter(AiAnalysis.research_label == "suspicious_example").count(),
        "scope": {"district": S.district, "mandal": S.mandal, "level": S.level},
        "note": "Aggregated synthetic programme indicators only. Admin accounts cannot open identifiable records.",
    }
    audit(db, user.id, "dashboard.view", "dashboard:summary", json.dumps(payload["scope"]))
    db.commit()
    return payload


def _group_count(S: GeoScope, model, col) -> dict:
    return {(k or "unknown"): n for k, n in S.q(model, col, func.count()).group_by(col).all()}


def _area_rows(S: GeoScope, col) -> list[dict]:
    """Patients / screenings / referrals grouped by a patient location column (district or mandal)."""
    areas: dict[str, dict] = {}
    pmap: dict[str, str] = {}
    for pid, key in S.q(Patient, Patient.id, col).all():
        name = (key or "").strip() or "Unspecified"
        pmap[pid] = name
        areas.setdefault(name, {"area": name, "patients": 0, "screenings": 0, "referrals": 0})["patients"] += 1
    for (pid,) in S.q(Screening, Screening.patient_id).all():
        a = areas.get(pmap.get(pid, ""))
        if a:
            a["screenings"] += 1
    for (pid,) in S.q(Referral, Referral.patient_id).all():
        a = areas.get(pmap.get(pid, ""))
        if a:
            a["referrals"] += 1
    return sorted(areas.values(), key=lambda a: (-a["patients"], a["area"]))


@router.get("/admin/monitoring")
def admin_monitoring(
    district: str | None = None,
    mandal: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(*ADMIN_ROLES)),
):
    """Aggregated monitoring for the whole programme, one district, or one mandal. No identifiable patient data."""
    S = GeoScope(db, _admin_district(user, district), mandal)
    now = datetime.now(timezone.utc)
    n_patients = S.q(Patient).count()
    n_screen = S.q(Screening).count()
    n_review = S.q(ClinicalReview).count()
    n_ref = S.q(Referral).count()
    n_appt_done = S.q(Appointment).filter(Appointment.status == "COMPLETED").count()
    n_inv_done = S.q(Investigation).filter(Investigation.state == "CLINICIAN_REVIEWED").count()

    funnel = [
        {"stage": "Patients registered", "value": n_patients},
        {"stage": "Screenings", "value": n_screen},
        {"stage": "Clinical reviews", "value": n_review},
        {"stage": "Referrals", "value": n_ref},
        {"stage": "Appointments completed", "value": n_appt_done},
        {"stage": "Investigations reviewed", "value": n_inv_done},
    ]

    pending_reviews = S.q(Screening).filter(Screening.status.in_(["SUBMITTED", "AWAITING_REVIEW", "PENDING_REVIEW"])).count()
    alerts = [
        {"label": "Screenings awaiting review", "value": pending_reviews, "kind": "warn"},
        {"label": "Urgent referrals open", "value": S.q(Referral).filter(Referral.priority == "urgent", Referral.status != "CLOSED").count(), "kind": "bad"},
        {"label": "Missed appointments", "value": S.q(Appointment).filter(Appointment.status == "MISSED").count(), "kind": "bad"},
        {"label": "Appointments awaiting confirmation", "value": S.q(Appointment).filter(Appointment.status == "REQUESTED").count(), "kind": "warn"},
        {"label": "Investigations pending", "value": S.q(Investigation).filter(Investigation.state != "CLINICIAN_REVIEWED").count(), "kind": "warn"},
        {"label": "Navigation tasks open/overdue", "value": S.q(NavigationTask).filter(NavigationTask.status.in_(["OPEN", "OVERDUE"])).count(), "kind": "warn"},
    ]

    # 14-day screening trend
    start = (now - timedelta(days=13)).replace(hour=0, minute=0, second=0, microsecond=0)
    per_day: dict[str, int] = {}
    for (created,) in S.q(Screening, Screening.created_at).filter(Screening.created_at >= start).all():
        if created is None:
            continue
        key = created.date().isoformat()
        per_day[key] = per_day.get(key, 0) + 1
    trend = []
    for i in range(14):
        d = (start + timedelta(days=i)).date().isoformat()
        trend.append({"date": d, "count": per_day.get(d, 0)})

    # Staff, narrowed by the district / mandal they registered under
    def staff_q(*cols):
        q = db.query(*cols).filter(User.role.notin_(ADMIN_ROLES))
        if S.district:
            q = q.filter(User.district == S.district)
        if S.mandal:
            q = q.filter(User.mandal == S.mandal)
        return q

    active_roles = {r: n for r, n in staff_q(User.role, func.count()).filter(User.status == "active").group_by(User.role).all()}
    staff = {
        "by_role": active_roles,
        "pending": staff_q(User.id).filter(User.status == "pending").count(),
        "suspended": staff_q(User.id).filter(User.status == "suspended").count(),
    }

    payload = {
        "scope": {"district": S.district, "mandal": S.mandal, "level": S.level},
        "funnel": funnel,
        "alerts": alerts,
        "trend": trend,
        "staff": staff,
        "by_district": _area_rows(S, Patient.district),
        "by_area": _area_rows(S, Patient.mandal),
        "screening_status": _group_count(S, Screening, Screening.status),
        "referral_status": _group_count(S, Referral, Referral.status),
        "referral_priority": _group_count(S, Referral, Referral.priority),
        "appointment_status": _group_count(S, Appointment, Appointment.status),
        "investigation_state": _group_count(S, Investigation, Investigation.state),
        "navigation_status": _group_count(S, NavigationTask, NavigationTask.status),
        "ai_labels": _group_count(S, AiAnalysis, AiAnalysis.research_label),
        "generated_at": now.isoformat(),
        "note": "Aggregated programme indicators only. Admin accounts cannot open identifiable records.",
    }
    audit(db, user.id, "admin.monitoring.view", "admin:monitoring", json.dumps(payload["scope"]))
    db.commit()
    return payload


@router.get("/audit")
def list_audit(db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES, "medical_officer"))):
    q = db.query(AuditEvent)
    actors = {u.id: u for u in db.query(User).all()}
    if user.role == "medical_officer" and any(user_area(user)):
        ids = [u.id for u in actors.values() if u.role not in ADMIN_ROLES and same_area(user, u.district, u.mandal)]
        q = q.filter(AuditEvent.actor_id.in_(ids))
    rows = q.order_by(AuditEvent.created_at.desc()).limit(500).all()
    return [
        {
            "id": a.id,
            "actor_id": a.actor_id,
            "actor_name": actors[a.actor_id].full_name if a.actor_id in actors else None,
            "actor_role": actors[a.actor_id].role if a.actor_id in actors else None,
            "event": a.event,
            "resource": a.resource,
            "created_at": a.created_at.isoformat(),
        }
        for a in rows
    ]


@router.get("/notifications")
def list_notes(db: Session = Depends(get_db), user: User = Depends(current_user)):
    # video-link alerts vanish once their scheduled day is over
    db.query(Notification).filter(Notification.expires_at.isnot(None), Notification.expires_at < datetime.now(timezone.utc)).delete(synchronize_session=False)
    db.commit()
    make_call_reminders(db, user)
    rows = db.query(Notification).filter(Notification.user_id == user.id).order_by(Notification.created_at.desc()).all()
    return [
        {
            "id": n.id, "title": n.title, "body": n.body, "read": n.read, "created_at": n.created_at.isoformat(),
            "kind": "meet" if _is_meet_note(n) else "info",
            "action_pending": _awaits_action(n),
            "expires_at": n.expires_at.isoformat() if n.expires_at else None,
        }
        for n in rows
    ]


def _awaits_action(n: Notification) -> bool:
    """Requests that stay until someone answers them: cross-mandal requests, registrations, new referrals."""
    return bool(n.ref) and n.ref.split(":", 1)[0] in {"xmandal", "xref", "user", "referral"}


def _is_meet_note(n: Notification) -> bool:
    """Alerts made before the kind column existed are recognised by their video title and appointment tag."""
    if n.kind == "meet":
        return True
    return n.ref is None and "video" in (n.title or "").lower() and "[appt:" in (n.body or "")


@router.patch("/notifications/{note_id}")
def read_note(note_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)):
    rec = db.get(Notification, note_id)
    if not rec or rec.user_id != user.id:
        raise HTTPException(404, "Notification not found")
    if _is_meet_note(rec) or _awaits_action(rec):  # video links wait for their day to end; requests wait for the decision
        rec.read = True
        db.commit()
        return {"id": rec.id, "read": True, "cleared": False}
    db.delete(rec)  # plain alerts are cleared as soon as the person acts on / opens them
    db.commit()
    return {"id": note_id, "read": True, "cleared": True}


def _work_status(db: Session, u: User) -> dict:
    """Role-specific work figures shown on the admin user page. Counts only, no patient identities."""
    def n(q):
        return q.count()

    out: dict = {}
    if u.role == "health_worker":
        out = {
            "patients_registered": n(db.query(Patient).filter(Patient.created_by == u.id)),
            "screenings_done": n(db.query(Screening).filter(Screening.examiner_id == u.id)),
            "photos_uploaded": n(db.query(ImageRecord).filter(ImageRecord.created_by == u.id)),
            "awaiting_clinician": n(db.query(Screening).filter(Screening.examiner_id == u.id, Screening.status == "AWAITING_CLINICIAN")),
        }
    elif u.role == "medical_officer":
        out = {
            "reviews_signed": n(db.query(ClinicalReview).filter(ClinicalReview.doctor_id == u.id)),
            "referrals_created": n(db.query(Referral).filter(Referral.created_by == u.id)),
            "cases_waiting_review": n(db.query(Screening).filter(Screening.status.in_(["AWAITING_CLINICIAN", "SCREENED"]))),
        }
    elif u.role == "specialist":
        out = {
            "referrals_in_system": n(db.query(Referral)),
            "appointments_completed": n(db.query(Appointment).filter(Appointment.status == "COMPLETED")),
            "appointments_open": n(db.query(Appointment).filter(Appointment.status.in_(["REQUESTED", "CONFIRMED", "SCHEDULED", "RESCHEDULED"]))),
            "investigations_reviewed": n(db.query(Investigation).filter(Investigation.state == "CLINICIAN_REVIEWED")),
            "investigations_pending": n(db.query(Investigation).filter(Investigation.state != "CLINICIAN_REVIEWED")),
        }
    elif u.role == "navigator":
        out = {
            "tasks_assigned": n(db.query(NavigationTask).filter(NavigationTask.owner_id == u.id)),
            "tasks_open": n(db.query(NavigationTask).filter(NavigationTask.owner_id == u.id, NavigationTask.status.in_(["OPEN", "OVERDUE"]))),
            "tasks_closed": n(db.query(NavigationTask).filter(NavigationTask.owner_id == u.id, NavigationTask.status.notin_(["OPEN", "OVERDUE"]))),
            "unassigned_tasks": n(db.query(NavigationTask).filter(NavigationTask.owner_id.is_(None))),
        }
    events = db.query(AuditEvent).filter(AuditEvent.actor_id == u.id).order_by(AuditEvent.created_at.desc())
    out["audit_events"] = events.count()
    last = events.first()
    return {
        "metrics": out,
        "last_activity": last.created_at.isoformat() if last else None,
        "recent_activity": [
            {"event": e.event, "resource": e.resource, "created_at": e.created_at.isoformat()} for e in events.limit(8).all()
        ],
    }


def _manageable(actor: User, target: User) -> bool:
    """District admins manage staff only; super admins can manage every account (including admins)."""
    if actor.role == "super_admin":
        return True
    if target.role in ADMIN_ROLES:
        return False
    return not actor.district or (target.district or "").strip().lower() == actor.district.strip().lower()


@router.get("/admin/users")
def admin_list_users(
    status: str | None = None,
    role: str | None = None,
    district: str | None = None,
    mandal: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(*ADMIN_ROLES)),
):
    q = db.query(User)
    if user.role != "super_admin":
        q = q.filter(User.role.notin_(ADMIN_ROLES))
        if user.district:
            q = q.filter(User.district == user.district)  # a district admin handles their own district's staff
    if status:
        q = q.filter(User.status == status)
    if role:
        q = q.filter(User.role == role)
    if clean(district):
        q = q.filter(User.district == clean(district))
    if clean(mandal):
        q = q.filter(User.mandal == clean(mandal))
    rows = q.order_by(User.created_at.desc()).all()
    return [user_out(u) for u in rows]


@router.get("/admin/users/{user_id}")
def admin_get_user(user_id: str, db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES))):
    u = db.get(User, user_id)
    if not u or not _manageable(user, u):
        raise HTTPException(404, "User not found")
    audit(db, user.id, "admin.user.view", f"user:{u.id}")
    db.commit()
    return {"user": user_out(u), "work": _work_status(db, u)}


def _decide_user(user_id: str, decision: str, db: Session, admin: User) -> dict:
    u = db.get(User, user_id)
    if not u or not _manageable(admin, u):
        raise HTTPException(404, "User not found")
    if u.status != "pending":
        raise HTTPException(409, f"Registration already {u.status}")
    u.status = "active" if decision == "approve" else "rejected"
    clear_notifications(db, f"user:{u.id}")  # the approval request is answered: remove it from every admin's list
    notify(db, u.id, "Registration " + ("approved" if decision == "approve" else "declined"), "Your account was reviewed by the district admin.")
    audit(db, admin.id, f"admin.user.{decision}", f"user:{u.id}", json.dumps({"role": u.role}))
    db.commit()
    return user_out(u)


@router.post("/admin/users/{user_id}/approve")
def approve_user(user_id: str, db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES))):
    return _decide_user(user_id, "approve", db, user)


@router.post("/admin/users/{user_id}/reject")
def reject_user(user_id: str, db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES))):
    return _decide_user(user_id, "reject", db, user)


@router.post("/admin/users/{user_id}/suspend")
def suspend_user(user_id: str, body: SuspendIn | None = None, db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES))):
    """Soft-delete: the person can no longer sign in, but the account and every record linked to it stay in the database."""
    u = db.get(User, user_id)
    if not u or not _manageable(user, u):
        raise HTTPException(404, "User not found")
    if u.id == user.id:
        raise HTTPException(400, "You cannot suspend your own account")
    if u.status != "active":
        raise HTTPException(409, f"Only active accounts can be suspended (this one is {u.status})")
    if u.role == "super_admin" and db.query(User).filter(User.role == "super_admin", User.status == "active", User.id != u.id).count() == 0:
        raise HTTPException(409, "At least one active super admin must remain")
    reason = clean(body.reason if body else None)
    u.status = "suspended"
    u.suspended_at = datetime.now(timezone.utc)
    u.suspension_reason = reason[:255] if reason else None
    audit(db, user.id, "admin.user.suspend", f"user:{u.id}", json.dumps({"role": u.role, "reason": u.suspension_reason}))
    db.commit()
    return user_out(u)


@router.post("/admin/users/{user_id}/reactivate")
def reactivate_user(user_id: str, db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES))):
    u = db.get(User, user_id)
    if not u or not _manageable(user, u):
        raise HTTPException(404, "User not found")
    if u.status != "suspended":
        raise HTTPException(409, f"Only suspended accounts can be reactivated (this one is {u.status})")
    u.status = "active"
    u.suspended_at = None
    u.suspension_reason = None
    notify(db, u.id, "Account restored", "Your account was reactivated by the district admin.")
    audit(db, user.id, "admin.user.reactivate", f"user:{u.id}", json.dumps({"role": u.role}))
    db.commit()
    return user_out(u)


@router.post("/admin/reset-demo")
def reset_demo(db: Session = Depends(get_db), user: User = Depends(require_roles("super_admin"))):
    for model in (
        Notification,
        AuditEvent,
        NavigationTask,
        Investigation,
        Appointment,
        Referral,
        ClinicalReview,
        AiAnalysis,
        ImageRecord,
        Screening,
        Consent,
        IdempotencyRecord,
        Patient,
    ):
        db.query(model).delete()
    db.commit()
    from .seed import seed_if_empty

    seed_if_empty(db)
    audit(db, user.id, "demo.reset", "system")
    db.commit()
    return {"ok": True, "message": "Synthetic demo data reset"}
