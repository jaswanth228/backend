from fastapi import HTTPException
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from .models import Appointment, Hospital, Patient, Referral, Screening, User


CLINICAL_ROLES = {"health_worker", "medical_officer", "specialist", "navigator"}


# ---------- mandal / district data segregation ----------
# Health workers, medical officers and navigators only work with data of their own district + mandal.
# Specialists are scoped by their hospital (see specialist_referral_filter). Admins / super admins are never scoped here.
# Cross-mandal / cross-district bookings are the single exception and go through the medical officer's decision.
GEO_SCOPED_ROLES = {"health_worker", "medical_officer", "navigator"}


def _norm(v: str | None) -> str:
    return (v or "").strip().lower()


def user_area(user: User) -> tuple[str, str]:
    return _norm(user.district), _norm(user.mandal)


def same_area(user: User, district: str | None, mandal: str | None) -> bool:
    """True when (district, mandal) lies inside the user's own area. A user with no area configured is not restricted."""
    ud, um = user_area(user)
    if ud and _norm(district) != ud:
        return False
    if um and _norm(mandal) != um:
        return False
    return True


def geo_scoped(user: User) -> bool:
    return user.role in GEO_SCOPED_ROLES and any(user_area(user))


def patient_area_clause(user: User):
    """SQL condition on Patient for the user's own district / mandal (None when the user is not scoped)."""
    if not geo_scoped(user):
        return None
    ud, um = user_area(user)
    conds = []
    if ud:
        conds.append(func.lower(func.trim(Patient.district)) == ud)
    if um:
        conds.append(func.lower(func.trim(Patient.mandal)) == um)
    return and_(*conds)


def hospital_area_clause(user: User):
    ud, um = user_area(user)
    conds = []
    if ud:
        conds.append(func.lower(func.trim(Hospital.district)) == ud)
    if um:
        conds.append(func.lower(func.trim(Hospital.mandal)) == um)
    return and_(*conds) if conds else None


def referral_area_clause(user: User):
    """SQL condition on Referral: patient lives in the user's area (navigators also see referrals into a hospital of their area)."""
    pc = patient_area_clause(user)
    if pc is None:
        return None
    clause = Referral.patient_id.in_(select(Patient.id).where(pc))
    if user.role == "navigator":
        hc = hospital_area_clause(user)
        if hc is not None:
            clause = clause | Referral.hospital_id.in_(select(Hospital.id).where(hc))
    return clause


def referral_in_area(db: Session, user: User, r: Referral) -> bool:
    clause = referral_area_clause(user)
    if clause is None:
        return True
    return db.query(Referral.id).filter(Referral.id == r.id, clause).first() is not None


def appointment_in_area(db: Session, user: User, a: Appointment) -> bool:
    if not geo_scoped(user):
        return True
    ref = db.get(Referral, a.referral_id)
    return bool(ref) and referral_in_area(db, user, ref)


def users_in_area(db: Session, role: str, *places: tuple[str | None, str | None]) -> list[User]:
    """Active users of `role` whose own area contains at least one of the given (district, mandal) places."""
    rows = db.query(User).filter(User.role == role, User.status == "active").all()
    return [u for u in rows if not any(user_area(u)) or any(same_area(u, d, m) for d, m in places)]


def district_officers(db: Session, district: str | None) -> list[User]:
    """Active medical officers of a district (any mandal): they decide cross-mandal / cross-district requests into it."""
    rows = db.query(User).filter(User.role == "medical_officer", User.status == "active").all()
    return [u for u in rows if _norm(u.district) == _norm(district)]


def audit(db: Session, actor_id: str | None, event: str, resource: str, meta: str | None = None):
    from .models import AuditEvent

    db.add(AuditEvent(actor_id=actor_id, event=event, resource=resource, metadata_json=meta))


def notify(db: Session, user_id: str, title: str, body: str, ref: str | None = None, kind: str = "info", expires_at=None):
    """Create an alert.

    kind="info": cleared once the action it asks for is taken (see clear_notifications / PATCH /notifications).
    kind="meet": a video-call link alert; it is never cleared by actions and disappears after its scheduled day (expires_at).
    """
    from .models import Notification

    db.add(Notification(user_id=user_id, title=title, body=body, ref=ref, kind=kind, expires_at=expires_at))


def clear_notifications(db: Session, ref: str, kind: str | None = "info") -> int:
    """Remove every alert about `ref` (default: only plain alerts, never video-link alerts)."""
    from .models import Notification

    q = db.query(Notification).filter(Notification.ref == ref)
    if kind:
        q = q.filter(Notification.kind == kind)
    return q.delete(synchronize_session=False)


def specialist_referral_filter(user: User):
    """A specialist only receives referrals the medical officer sent to them, or to their hospital without naming anyone."""
    from sqlalchemy import and_, or_

    own = Referral.specialist_id == user.id
    approved = Referral.approval_status.notin_(["PENDING", "REJECTED"])  # cross-district requests stay hidden until accepted
    if not user.hospital_id:
        return and_(own, approved)
    return and_(approved, or_(own, and_(Referral.specialist_id.is_(None), Referral.hospital_id == user.hospital_id)))


def referral_visible(user: User, r: Referral, db: Session | None = None) -> bool:
    if user.role == "navigator" and r.approval_status in {"PENDING", "REJECTED"}:
        return False
    if user.role != "specialist":
        return referral_in_area(db, user, r) if db is not None else True
    if r.approval_status in {"PENDING", "REJECTED"}:
        return False
    return r.specialist_id == user.id or (r.specialist_id is None and bool(user.hospital_id) and r.hospital_id == user.hospital_id)


def patient_visible(user: User, patient: Patient) -> bool:
    if user.role == "super_admin":
        return True
    if user.role == "admin":
        return False
    if user.role in {"health_worker", "medical_officer"}:
        return patient.facility == user.assigned_facility and same_area(user, patient.district, patient.mandal)
    if user.role in {"specialist", "navigator"}:
        return True
    return False


def get_patient_or_404(db: Session, patient_id: str) -> Patient:
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    return patient


def assert_patient_access(user: User, patient: Patient, db: Session | None = None):
    if user.role == "super_admin":
        return
    if user.role == "admin":
        raise HTTPException(403, "Administrators receive aggregate data only")
    if user.role in {"specialist", "navigator"}:
        q = db.query(Referral).filter(Referral.patient_id == patient.id) if db else None
        if q is not None and user.role == "specialist":
            q = q.filter(specialist_referral_filter(user))
        if q is not None and user.role == "navigator":
            rc = referral_area_clause(user)
            if rc is not None:
                q = q.filter(rc)
        if q is None or not q.first():
            raise HTTPException(403, "Access is limited to cases referred to you" if user.role == "specialist" else "Access is limited to referred cases")
        return
    if not patient_visible(user, patient):
        raise HTTPException(403, "Access denied for this patient record")


def assert_screening_access(db: Session, user: User, screening: Screening):
    patient = get_patient_or_404(db, screening.patient_id)
    assert_patient_access(user, patient, db)
    if user.role == "navigator":
        raise HTTPException(403, "Navigators cannot open screening photographs or exam notes")
