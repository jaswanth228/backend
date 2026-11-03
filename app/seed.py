import hashlib
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from .config import settings
from .models import (
    AiAnalysis,
    Appointment,
    ClinicalReview,
    Consent,
    District,
    Hospital,
    ImageRecord,
    Investigation,
    Mandal,
    NavigationTask,
    Patient,
    Referral,
    Screening,
    User,
    uid,
)
from .security import hash_password

DEMO_PASSWORD = "Demo@123"
FACILITY = "PHC G. Madugula"
DEFAULT_DISTRICT = "Alluri Sitharama Raju"  # starter entry so the demo works; admins add the rest in the app
DEFAULT_MANDAL = "G. Madugula"
PIPELINE_MARK = "[demo-pipeline]"  # stored in Patient.medical_history so the seed runs only once per database

USERS = [
    ("hw@demo.local", "Lakshmi Health Worker", "health_worker"),
    ("mo@demo.local", "Dr. Rao Medical Officer", "medical_officer"),
    ("spec@demo.local", "Dr. Padma Specialist", "specialist"),
    ("nav@demo.local", "Ravi Navigator", "navigator"),
    ("admin@demo.local", "District Admin", "admin"),
]


def seed_geography(db: Session) -> None:
    """Starter district/mandal so registration forms are usable on a fresh database."""
    if db.query(District).first():
        return
    d = District(name=DEFAULT_DISTRICT)
    db.add(d)
    db.flush()
    db.add(Mandal(district_id=d.id, name=DEFAULT_MANDAL))
    db.flush()


DEMO_HOSPITALS = [
    # (name, mandal, address, phone)
    ("CHC G. Madugula", "G. Madugula", "Main road, G. Madugula", "08935-250001"),
    ("PHC G. Madugula", "G. Madugula", "PHC campus, G. Madugula", "08935-250002"),
    ("Area Hospital Paderu", "Paderu", "Hospital road, Paderu", "08935-250003"),
]


def seed_demo_hospitals(db: Session) -> None:
    """Starter hospitals (two in G. Madugula, one in Paderu) so booking and cross-mandal requests can be tried out."""
    if db.query(Hospital).first():
        return
    d = db.query(District).filter(District.name == DEFAULT_DISTRICT).first()
    if not d:
        return
    officer = db.query(User).filter(User.role == "medical_officer").first()
    for name, mandal, address, phone in DEMO_HOSPITALS:
        if not db.query(Mandal).filter(Mandal.district_id == d.id, Mandal.name == mandal).first():
            db.add(Mandal(district_id=d.id, name=mandal))
            db.flush()
        db.add(Hospital(name=name, district=DEFAULT_DISTRICT, mandal=mandal, address=address, contact_phone=phone, created_by=officer.id if officer else None))
    db.flush()


def seed_demo_specialists(db: Session) -> None:
    """Demo specialists are registered in a hospital (normally done by the district admin) and old referrals get their hospital."""
    by_name = {h.name.lower(): h for h in db.query(Hospital).all()}
    paderu, chc = by_name.get("area hospital paderu"), by_name.get("chc g. madugula")
    spec = db.query(User).filter(User.email == "spec@demo.local").first()
    if spec and not spec.hospital_id and paderu:
        spec.hospital_id, spec.assigned_facility, spec.specialty, spec.phone = paderu.id, paderu.name, "Oral surgery / ENT", "9000000001"
        spec.district, spec.mandal = paderu.district, paderu.mandal
    if chc and not db.query(User).filter(User.email == "spec2@demo.local").first():
        db.add(User(email="spec2@demo.local", full_name="Dr. Suresh Specialist", password_hash=hash_password(DEMO_PASSWORD), role="specialist",
                    assigned_facility=chc.name, district=chc.district, mandal=chc.mandal, hospital_id=chc.id, specialty="Oral medicine", phone="9000000002"))
    for r in db.query(Referral).filter(Referral.hospital_id.is_(None)).all():  # referrals made before hospitals were linked
        h = by_name.get((r.facility or "").lower())
        if h:
            r.hospital_id = h.id
    db.flush()


def ensure_super_admin(db: Session) -> None:
    """Make sure a super admin exists (also on databases created before this role was added)."""
    if db.query(User).filter(User.role == "super_admin").first():
        return
    email = settings.SUPER_ADMIN_EMAIL.strip().lower()
    if db.query(User).filter(User.email == email).first():
        return
    db.add(
        User(
            email=email,
            full_name=settings.SUPER_ADMIN_NAME,
            password_hash=hash_password(settings.SUPER_ADMIN_PASSWORD),
            role="super_admin",
            assigned_facility=FACILITY,
        )
    )
    db.flush()


def backfill_geography(db: Session) -> None:
    """Give records created before districts existed a district, using their mandal name when it is unambiguous."""
    by_mandal: dict[str, set[str]] = {}
    for dname, mname in db.query(District.name, Mandal.name).join(Mandal, Mandal.district_id == District.id).all():
        by_mandal.setdefault(mname.lower(), set()).add(dname)
    for p in db.query(Patient).filter(Patient.district.is_(None)).all():
        match = by_mandal.get((p.mandal or "").strip().lower())
        if match and len(match) == 1:
            p.district = next(iter(match))
    if DEFAULT_MANDAL.lower() in by_mandal and len(by_mandal[DEFAULT_MANDAL.lower()]) == 1:
        d = next(iter(by_mandal[DEFAULT_MANDAL.lower()]))
        for u in db.query(User).filter(User.district.is_(None), User.assigned_facility == FACILITY, User.role != "super_admin").all():
            u.district, u.mandal = d, DEFAULT_MANDAL
    db.commit()


def seed_if_empty(db: Session) -> None:
    seed_geography(db)
    if not db.query(User).first():
        for email, name, role in USERS:
            db.add(
                User(
                    email=email,
                    full_name=name,
                    password_hash=hash_password(DEMO_PASSWORD),
                    role=role,
                    assigned_facility=FACILITY,
                    district=DEFAULT_DISTRICT,  # a district admin is tied to one district
                    mandal=None if role == "admin" else DEFAULT_MANDAL,
                )
            )
        db.flush()
    ensure_super_admin(db)
    demo_admin = db.query(User).filter(User.email == "admin@demo.local", User.role == "admin").first()
    if demo_admin and not demo_admin.district:
        demo_admin.district = DEFAULT_DISTRICT  # district admins see their own district only
    seed_demo_hospitals(db)
    seed_demo_specialists(db)
    if not db.query(Patient).first():
        hw = db.query(User).filter(User.email == "hw@demo.local").first()
        db.add(
            Patient(video_allowed=True, 
                full_name="Synthetic Demo Patient — G. Madugula",
                age=46,
                sex="female",
                phone="9000000001",
                village="G. Madugula",
                habitation="Demo habitation",
                gram_panchayat="G. Madugula",
                district=DEFAULT_DISTRICT,
                mandal="G. Madugula",
                phc=FACILITY,
                tobacco_use="gutkha daily",
                symptoms="white patch inner cheek, 3 weeks",
                medical_history="No prior oral screening on record (synthetic)",
                facility=FACILITY,
                created_by=hw.id if hw else None,
                duplicate_hint=None,
            )
        )
    db.commit()
    if settings.SEED_DEMO_DATA:
        seed_demo_pipeline(db)
    backfill_geography(db)


def _synthetic_photo() -> bytes:
    """A plain synthetic placeholder picture (not a real patient photograph)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (480, 360), (214, 140, 140))
    d = ImageDraw.Draw(img)
    d.ellipse((140, 90, 340, 270), fill=(236, 170, 170))
    d.ellipse((220, 150, 290, 200), fill=(245, 245, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _feedback(name: str, label_urgency: str, other: list[str]) -> dict:
    return {
        "image_assessment": {"is_oral_photo": True, "usable": True, "issues": []},
        "predicted_condition": {"disease_detected": True, "name": name, "likelihood": "medium", "other_possibilities": other},
        "analysis": "Synthetic demo assessment based on the described symptoms and habits. Not a diagnosis.",
        "confidence": "medium",
        "suggested_urgency": label_urgency,
        "health_suggestions": ["Avoid tobacco and betel quid", "Maintain gentle oral hygiene"],
        "warning_signs": ["Non-healing ulcer", "Increasing size or pain", "Difficulty opening the mouth"],
        "consult_advice": "Please see a doctor for an in-person examination.",
        "health_worker_guidance": ["Arrange a medical officer review"],
        "recapture_tips": [],
        "clinician_summary": "Synthetic demo case. Examine the lesion, palpate for nodes and consider biopsy if it persists.",
        "patient_message": {
            "en": "A change was seen in your mouth. Please visit the doctor soon so it can be checked properly.",
            "te": "మీ నోటిలో ఒక మార్పు కనిపించింది. దయచేసి సరిగ్గా పరీక్ష చేయించుకోవడానికి త్వరలో డాక్టర్‌ను కలవండి.",
        },
        "limitations": "Advisory only. A doctor must confirm any diagnosis.",
    }


def seed_demo_pipeline(db: Session) -> None:
    """Insert synthetic cases at every workflow stage, once per database.

    Medical officer, specialist and navigator pages read these straight from the database, so they are not empty
    until a health worker creates something. All names and photos are synthetic.
    """
    if db.query(Patient).filter(Patient.medical_history.like(f"%{PIPELINE_MARK}%")).first():
        return
    users = {u.role: u for u in db.query(User).filter(User.role.in_(["health_worker", "medical_officer", "specialist", "navigator"])).all()}
    hw, mo = users.get("health_worker"), users.get("medical_officer")
    if not hw or not mo:
        return
    now = datetime.now(timezone.utc)
    from . import supabase_storage

    def patient(name, age, sex, phone, village, symptoms, tobacco, days_ago):
        p = Patient(video_allowed=True, 
            full_name=name, age=age, sex=sex, phone=phone, village=village, habitation="Demo habitation",
            gram_panchayat=village, district=DEFAULT_DISTRICT, mandal="G. Madugula", phc=FACILITY, tobacco_use=tobacco, symptoms=symptoms,
            medical_history=f"Synthetic record {PIPELINE_MARK}", facility=FACILITY, created_by=hw.id,
            created_at=now - timedelta(days=days_ago),
        )
        db.add(p)
        db.flush()
        for purpose in ("examination", "imaging", "teleconsultation"):
            db.add(Consent(patient_id=p.id, purpose=purpose, language="en", granted=True, created_by=hw.id, created_at=p.created_at))
        return p

    def screening(p, status, symptoms, exam, days_ago, photo=True, ai=None):
        s = Screening(
            patient_id=p.id, examiner_id=hw.id, status=status, symptoms=symptoms, duration="3 weeks",
            tobacco=p.tobacco_use, exam_notes=exam, created_at=now - timedelta(days=days_ago),
        )
        db.add(s)
        db.flush()
        image_id = None
        if photo:
            data = _synthetic_photo()
            image_id = uid()
            supabase_storage.upload(f"{s.id}/{image_id}.jpg", data, "image/jpeg")
            db.add(ImageRecord(
                id=image_id, screening_id=s.id, view="cheek", checksum=hashlib.sha256(data).hexdigest(),
                storage_key=f"{s.id}/{image_id}.jpg", quality_status="acceptable" if ai else "pending_ai",
                created_by=hw.id, created_at=s.created_at,
            ))
            db.flush()  # models declare no relationships, so write the image row before anything that references it
        if ai:
            label, name, urgency, other = ai
            db.add(AiAnalysis(
                image_id=image_id, screening_id=s.id, model_version="seed-demo", provider="mock", research_label=label,
                status="completed", feedback=json.dumps(_feedback(name, urgency, other), ensure_ascii=False), created_at=s.created_at,
            ))
            db.flush()
        return s

    # 1) Waiting for the medical officer: photo + AI result
    p1 = patient("Synthetic Patient A — Ramulu", 52, "male", "9000000011", "Pedakota", "ulcer on tongue edge, 4 weeks", "betel quid daily", 2)
    screening(p1, "AWAITING_CLINICIAN", p1.symptoms, "Single ulcer, firm edge", 2,
              ai=("suspicious_example", "Oral ulcer, needs evaluation (most likely)", "urgent", ["Traumatic ulcer"]))
    # 2) Waiting for the medical officer: photo, AI not run yet
    p2 = patient("Synthetic Patient B — Savitri", 38, "female", "9000000012", "Chintapalle", "burning sensation with spicy food", "none", 1)
    screening(p2, "SCREENED", p2.symptoms, "Mild redness of the cheek lining", 1)
    # 3) Reviewed and referred, appointment booked, investigation advised (specialist + navigator)
    p3 = patient("Synthetic Patient C — Appa Rao", 61, "male", "9000000013", "Lothugedda", "white patch inner cheek, 2 months", "gutkha daily", 9)
    s3 = screening(p3, "REFERRAL_PLANNED", p3.symptoms, "Non-scrapable white patch, 2 cm", 9,
                   ai=("condition_possible_example", "Oral leukoplakia (most likely)", "soon", ["Oral lichen planus"]))
    review = ClinicalReview(screening_id=s3.id, doctor_id=mo.id, finding="Persistent leukoplakic patch; specialist opinion and biopsy advised.",
                            outcome="refer", ai_agreement="agree", signed_at=now - timedelta(days=8))
    db.add(review)
    db.flush()
    _ph = db.query(Hospital).filter(Hospital.name == "Area Hospital Paderu").first()
    ref = Referral(review_id=review.id, patient_id=p3.id, facility="Area Hospital Paderu", hospital_id=(_ph.id if _ph else None), specialty="Oral surgery / ENT",
                   priority="soon", reason=review.finding, status="OPEN", created_by=mo.id, created_at=now - timedelta(days=8))
    db.add(ref)
    db.flush()
    db.add(NavigationTask(referral_id=ref.id, status="OPEN", due="7d", created_at=ref.created_at))
    db.add(Appointment(referral_id=ref.id, kind="physical", slot=(now + timedelta(days=3)).replace(hour=10, minute=0, second=0, microsecond=0).isoformat(),
                       status="CONFIRMED", created_at=now - timedelta(days=7)))
    db.add(Investigation(referral_id=ref.id, kind="biopsy", state="ADVISED", created_at=now - timedelta(days=7)))
    # 4) Fully closed loop
    p4 = patient("Synthetic Patient D — Kamala", 44, "female", "9000000014", "G. Madugula", "sore mouth, 6 weeks", "gutkha occasionally", 20)
    s4 = screening(p4, "CARE_LINKED", p4.symptoms, "Erythroplakia suspected", 20,
                   ai=("suspicious_example", "Possible erythroplakia (most likely)", "urgent", ["Candidiasis"]))
    review4 = ClinicalReview(screening_id=s4.id, doctor_id=mo.id, finding="Red patch on the floor of the mouth, referred.",
                             outcome="refer", ai_agreement="agree", signed_at=now - timedelta(days=19))
    db.add(review4)
    db.flush()
    ref4 = Referral(review_id=review4.id, patient_id=p4.id, facility="Area Hospital Paderu", hospital_id=(_ph.id if _ph else None), specialty="Oral surgery / ENT",
                    priority="urgent", reason=review4.finding, status="CARE_LINKED", created_by=mo.id, created_at=now - timedelta(days=19))
    db.add(ref4)
    db.flush()
    db.add(NavigationTask(referral_id=ref4.id, status="CLOSED", due="7d", contact_log="Patient reached the hospital (synthetic).", created_at=ref4.created_at))
    db.add(Appointment(referral_id=ref4.id, kind="physical", slot=(now - timedelta(days=15)).isoformat(), status="COMPLETED", created_at=now - timedelta(days=18)))
    db.add(Investigation(referral_id=ref4.id, kind="biopsy", state="CLINICIAN_REVIEWED", clinician_result="Synthetic result for demo.", created_at=now - timedelta(days=17)))

    db.commit()
