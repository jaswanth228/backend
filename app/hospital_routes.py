"""Hospitals, specialists for booking, and the cross-mandal approval flow.

Who does what
  District admin / super admin : adds, edits, hides and removes hospitals AND registers / updates the specialists of each hospital
  Medical officer              : assigns the hospital + specialist of the patient's own mandal; a request into another mandal / district goes to that district's medical officers, who accept or decline it
  Specialist                   : a doctor registered in one hospital; only receives patients promoted to them (or to their hospital)
  Navigator                    : receives patient + specialist + hospital details and guides the patient to the hospital
"""
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from .access import audit, clear_notifications, notify, district_officers
from .db import get_db
from .geo import clean, validate_geo
from .access import same_area, user_area
from .models import Appointment, Hospital, Notification, Patient, Referral, Screening, ClinicalReview, User
from .security import hash_password
from .routers import _activate_referral, appointment_out, fmt_slot, notify_appointment
from .security import current_user, require_roles

router = APIRouter()

MANAGERS = ("admin", "super_admin")  # maintain the hospital / specialist lists
DECIDERS = ("medical_officer", "super_admin")  # decide cross-mandal requests


class HospitalIn(BaseModel):
    name: str
    district: str
    mandal: str
    address: str | None = None
    contact_phone: str | None = None
    contact_email: str | None = None
    available: bool = True


class HospitalPatch(BaseModel):
    name: str | None = None
    district: str | None = None
    mandal: str | None = None
    address: str | None = None
    contact_phone: str | None = None
    contact_email: str | None = None
    available: bool | None = None


class DecisionIn(BaseModel):
    decision: str  # "approve" | "reject"
    note: str | None = None


def hospital_out(h: Hospital) -> dict:
    return {
        "id": h.id,
        "name": h.name,
        "district": h.district,
        "mandal": h.mandal,
        "address": h.address,
        "contact_phone": h.contact_phone,
        "contact_email": h.contact_email,
        "available": bool(h.available),
        "created_at": h.created_at.isoformat() if h.created_at else None,
    }


def _check_scope(user: User, district: str) -> None:
    """A district admin manages their own district only (if one is set); super admins manage everything."""
    if user.role == "admin" and user.district and user.district.lower() != district.lower():
        raise HTTPException(403, f"You can manage hospitals in {user.district} only")


def _contact_checks(phone: str | None, email: str | None) -> tuple[str | None, str | None]:
    phone, email = clean(phone), clean(email)
    if phone and not (6 <= len(phone) <= 40 and all(c.isdigit() or c in "+-() ,/" for c in phone)):
        raise HTTPException(422, "Enter a valid contact phone number")
    if email and ("@" not in email or "." not in email.split("@")[-1]):
        raise HTTPException(422, "Enter a valid contact email")
    return phone, email


def _duplicate(db: Session, name: str, district: str, mandal: str, skip_id: str | None = None) -> bool:
    q = db.query(Hospital.id).filter(func.lower(Hospital.name) == name.lower(), Hospital.district == district, Hospital.mandal == mandal)
    if skip_id:
        q = q.filter(Hospital.id != skip_id)
    return q.first() is not None


@router.get("/hospitals")
def list_hospitals(
    district: str | None = None,
    mandal: str | None = None,
    include_unavailable: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    """Hospitals in a mandal, for the booking form. Managers can also list the hidden (unavailable) ones."""
    q = db.query(Hospital)
    if clean(district):
        q = q.filter(Hospital.district == clean(district))
    if clean(mandal):
        q = q.filter(Hospital.mandal == clean(mandal))
    if not (include_unavailable and user.role in MANAGERS + ("medical_officer",)):
        q = q.filter(Hospital.available.is_(True))
    return [hospital_out(h) for h in q.order_by(Hospital.district, Hospital.mandal, Hospital.name).all()]


@router.post("/hospitals", status_code=201)
def add_hospital(body: HospitalIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*MANAGERS))):
    name = clean(body.name)
    if not name or len(name) < 3:
        raise HTTPException(422, "Enter the hospital name")
    district, mandal = validate_geo(db, body.district, body.mandal)
    if not district or not mandal:
        raise HTTPException(422, "Select the district and mandal this hospital belongs to")
    _check_scope(user, district)
    if _duplicate(db, name, district, mandal):
        raise HTTPException(409, "This hospital is already listed in that mandal")
    phone, email = _contact_checks(body.contact_phone, body.contact_email)
    h = Hospital(
        name=name, district=district, mandal=mandal, address=clean(body.address), contact_phone=phone,
        contact_email=email, available=body.available, created_by=user.id,
    )
    db.add(h)
    db.flush()
    audit(db, user.id, "hospital.add", f"hospital:{h.id}", json.dumps({"district": district, "mandal": mandal}))
    db.commit()
    return hospital_out(h)


@router.patch("/hospitals/{hospital_id}")
def update_hospital(hospital_id: str, body: HospitalPatch, db: Session = Depends(get_db), user: User = Depends(require_roles(*MANAGERS))):
    h = db.get(Hospital, hospital_id)
    if not h:
        raise HTTPException(404, "Hospital not found")
    _check_scope(user, h.district)
    data = body.model_dump(exclude_unset=True)
    name = clean(data["name"]) if "name" in data else h.name
    if not name or len(name) < 3:
        raise HTTPException(422, "Enter the hospital name")
    district, mandal = h.district, h.mandal
    if "district" in data or "mandal" in data:
        district, mandal = validate_geo(db, data.get("district", h.district), data.get("mandal", h.mandal))
        _check_scope(user, district)
    if _duplicate(db, name, district, mandal, skip_id=h.id):
        raise HTTPException(409, "This hospital is already listed in that mandal")
    phone, email = _contact_checks(data["contact_phone"] if "contact_phone" in data else h.contact_phone, data["contact_email"] if "contact_email" in data else h.contact_email)
    h.name, h.district, h.mandal = name, district, mandal
    if "address" in data:
        h.address = clean(data["address"])
    h.contact_phone, h.contact_email = phone, email
    if data.get("available") is not None:
        h.available = data["available"]
    h.updated_at = datetime.now(timezone.utc)
    audit(db, user.id, "hospital.update", f"hospital:{h.id}", json.dumps({k: v for k, v in data.items() if k != "address"}))
    db.commit()
    return hospital_out(h)


@router.delete("/hospitals/{hospital_id}")
def delete_hospital(hospital_id: str, db: Session = Depends(get_db), user: User = Depends(require_roles(*MANAGERS))):
    h = db.get(Hospital, hospital_id)
    if not h:
        raise HTTPException(404, "Hospital not found")
    _check_scope(user, h.district)
    if (
        db.query(Appointment.id).filter(Appointment.hospital_id == h.id).first()
        or db.query(Referral.id).filter(Referral.hospital_id == h.id).first()
        or db.query(User.id).filter(User.hospital_id == h.id).first()
    ):
        raise HTTPException(409, "This hospital has specialists, referrals or appointments on record. Mark it unavailable instead of deleting it")
    audit(db, user.id, "hospital.delete", f"hospital:{h.id}", json.dumps({"name": h.name}))
    db.delete(h)
    db.commit()
    return {"deleted": True, "id": hospital_id}


# ---------------- specialists (doctors registered in a hospital; maintained by the district admin) ----------------

class SpecialistIn(BaseModel):
    full_name: str
    email: str
    password: str
    hospital_id: str
    specialty: str | None = None
    phone: str | None = None


class SpecialistPatch(BaseModel):
    full_name: str | None = None
    hospital_id: str | None = None
    specialty: str | None = None
    phone: str | None = None
    password: str | None = None
    available: bool | None = None  # False = not taking new patients (account suspended)


def specialist_out(u: User, hospitals: dict[str, Hospital]) -> dict:
    h = hospitals.get(u.hospital_id or "")
    return {
        "id": u.id,
        "full_name": u.full_name,
        "email": u.email,
        "phone": u.phone,
        "specialty": u.specialty,
        "hospital_id": u.hospital_id,
        "hospital_name": h.name if h else None,
        "hospital_mandal": h.mandal if h else None,
        "hospital_district": h.district if h else None,
        "assigned_facility": u.assigned_facility,
        "district": u.district,
        "mandal": u.mandal,
        "available": u.status == "active",
        "status": u.status,
    }


def _hospital_map(db: Session) -> dict[str, Hospital]:
    return {h.id: h for h in db.query(Hospital).all()}


@router.get("/specialists")
def list_specialists(
    hospital_id: str | None = None,
    district: str | None = None,
    mandal: str | None = None,
    include_inactive: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("navigator", "specialist", "medical_officer", "admin", "super_admin")),
):
    """Specialists of a hospital (for the medical officer's choice) or the whole directory (admins)."""
    q = db.query(User).filter(User.role == "specialist")
    if not (include_inactive and user.role in MANAGERS):
        q = q.filter(User.status == "active")
    if hospital_id:
        q = q.filter(User.hospital_id == hospital_id)
    hospitals = _hospital_map(db)
    rows = [u for u in q.order_by(User.full_name).all()]
    # Medical officers / navigators get their own mandal's specialists by default. Naming a hospital, district or
    # mandal explicitly (cross-mandal / cross-district booking) still returns that place's specialists.
    if user.role in {"medical_officer", "navigator"} and not (hospital_id or clean(district) or clean(mandal)) and any(user_area(user)):
        rows = [u for u in rows if u.hospital_id in hospitals and same_area(user, hospitals[u.hospital_id].district, hospitals[u.hospital_id].mandal)]
    if clean(district) or clean(mandal):
        rows = [
            u for u in rows
            if u.hospital_id in hospitals
            and (not clean(district) or hospitals[u.hospital_id].district == clean(district))
            and (not clean(mandal) or hospitals[u.hospital_id].mandal == clean(mandal))
        ]
    return [specialist_out(u, hospitals) for u in rows]


def _hospital_for_specialist(db: Session, user: User, hospital_id: str) -> Hospital:
    h = db.get(Hospital, hospital_id)
    if not h:
        raise HTTPException(422, "Select the hospital this specialist is registered in")
    _check_scope(user, h.district)
    return h


@router.post("/specialists", status_code=201)
def add_specialist(body: SpecialistIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*MANAGERS))):
    name = clean(body.full_name)
    email = (body.email or "").strip().lower()
    if not name or len(name) < 2:
        raise HTTPException(422, "Enter the specialist's name")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(422, "Enter a valid email")
    if len(body.password or "") < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    if db.query(User.id).filter(User.email == email).first():
        raise HTTPException(409, "An account with this email already exists")
    h = _hospital_for_specialist(db, user, body.hospital_id)
    phone, _ = _contact_checks(body.phone, None)
    u = User(
        email=email, full_name=name, password_hash=hash_password(body.password), role="specialist", status="active",
        assigned_facility=h.name, district=h.district, mandal=h.mandal, hospital_id=h.id,
        specialty=clean(body.specialty), phone=phone,
    )
    db.add(u)
    db.flush()
    audit(db, user.id, "specialist.add", f"user:{u.id}", json.dumps({"hospital": h.id}))
    db.commit()
    return specialist_out(u, {h.id: h})


@router.patch("/specialists/{specialist_id}")
def update_specialist(specialist_id: str, body: SpecialistPatch, db: Session = Depends(get_db), user: User = Depends(require_roles(*MANAGERS))):
    u = db.get(User, specialist_id)
    if not u or u.role != "specialist":
        raise HTTPException(404, "Specialist not found")
    data = body.model_dump(exclude_unset=True)
    current = db.get(Hospital, u.hospital_id) if u.hospital_id else None
    if current:
        _check_scope(user, current.district)
    if "full_name" in data:
        name = clean(data["full_name"])
        if not name or len(name) < 2:
            raise HTTPException(422, "Enter the specialist's name")
        u.full_name = name
    if data.get("hospital_id"):
        h = _hospital_for_specialist(db, user, data["hospital_id"])
        u.hospital_id, u.assigned_facility, u.district, u.mandal = h.id, h.name, h.district, h.mandal
    if "specialty" in data:
        u.specialty = clean(data["specialty"])
    if "phone" in data:
        u.phone, _ = _contact_checks(data["phone"], None)
    if data.get("password"):
        if len(data["password"]) < 8:
            raise HTTPException(422, "Password must be at least 8 characters")
        u.password_hash = hash_password(data["password"])
    if data.get("available") is not None:
        u.status = "active" if data["available"] else "suspended"
        u.suspended_at = None if data["available"] else datetime.now(timezone.utc)
        u.suspension_reason = None if data["available"] else "Marked unavailable by the district admin"
    audit(db, user.id, "specialist.update", f"user:{u.id}", json.dumps({k: v for k, v in data.items() if k != "password"}))
    db.commit()
    return specialist_out(u, _hospital_map(db))


@router.delete("/specialists/{specialist_id}")
def delete_specialist(specialist_id: str, db: Session = Depends(get_db), user: User = Depends(require_roles(*MANAGERS))):
    u = db.get(User, specialist_id)
    if not u or u.role != "specialist":
        raise HTTPException(404, "Specialist not found")
    h = db.get(Hospital, u.hospital_id) if u.hospital_id else None
    if h:
        _check_scope(user, h.district)
    if db.query(Referral.id).filter(Referral.specialist_id == u.id).first() or db.query(Appointment.id).filter(Appointment.specialist_id == u.id).first():
        raise HTTPException(409, "This specialist has referrals or appointments on record. Mark them unavailable instead of deleting")
    db.query(Notification).filter(Notification.user_id == u.id).delete(synchronize_session=False)
    audit(db, user.id, "specialist.delete", f"user:{u.id}", json.dumps({"name": u.full_name}))
    db.delete(u)
    db.commit()
    return {"deleted": True, "id": specialist_id}


# ---------------- cross-mandal requests ----------------

def _request_out(db: Session, a: Appointment) -> dict:
    ref = db.get(Referral, a.referral_id)
    patient = db.get(Patient, ref.patient_id) if ref else None
    out = appointment_out(a, db)
    out.update(
        {
            "patient_district": patient.district if patient else None,
            "patient_mandal": patient.mandal if patient else None,
            "patient_village": patient.village if patient else None,
            "specialty": ref.specialty if ref else None,
            "priority": ref.priority if ref else None,
        }
    )
    return out  # no patient name or clinical details: officers decide on geography and reason only


def _in_officer_scope(user: User, a: Appointment, db: Session) -> bool:
    if user.role == "super_admin" or not (user.district or user.mandal):
        return True
    h = db.get(Hospital, a.hospital_id) if a.hospital_id else None
    ref = db.get(Referral, a.referral_id)
    p = db.get(Patient, ref.patient_id) if ref else None
    # a medical officer decides requests whose patient mandal or requested hospital mandal is their own
    return any(same_area(user, x.district, x.mandal) for x in (h, p) if x is not None)


def _referral_request_out(db: Session, r: Referral) -> dict:
    patient = db.get(Patient, r.patient_id)
    h = db.get(Hospital, r.hospital_id) if r.hospital_id else None
    sp = db.get(User, r.specialist_id) if r.specialist_id else None
    return {
        "kind": "referral", "id": r.id, "slot": None,
        "hospital_name": h.name if h else r.facility, "hospital_mandal": h.mandal if h else None, "hospital_district": h.district if h else None,
        "specialist_name": sp.full_name if sp else None,
        "patient_district": patient.district if patient else None, "patient_mandal": patient.mandal if patient else None,
        "patient_village": patient.village if patient else None,
        "specialty": r.specialty, "priority": r.priority,
        "cross_reason": r.cross_reason, "approval_status": r.approval_status, "approval_note": r.approval_note,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def _referral_in_scope(user: User, r: Referral, db: Session) -> bool:
    """The medical officers of the hospital's district decide (the officer who asked cannot decide their own request)."""
    if user.role == "super_admin":
        return True
    h = db.get(Hospital, r.hospital_id) if r.hospital_id else None
    return bool(h) and (h.district or "").strip().lower() == (user.district or "").strip().lower()


@router.get("/cross-mandal-requests")
def list_cross_mandal(status: str = "PENDING", db: Session = Depends(get_db), user: User = Depends(require_roles(*DECIDERS))):
    q = db.query(Appointment).filter(Appointment.cross_mandal.is_(True))
    if status != "all":
        q = q.filter(Appointment.approval_status == status)
    rows = [a for a in q.order_by(Appointment.created_at.desc()).all() if _in_officer_scope(user, a, db)]
    out = [{**_request_out(db, a), "kind": "appointment"} for a in rows]
    rq = db.query(Referral).filter(Referral.cross_mandal.is_(True))
    if status != "all":
        rq = rq.filter(Referral.approval_status == status)
    out += [_referral_request_out(db, r) for r in rq.order_by(Referral.created_at.desc()).all() if _referral_in_scope(user, r, db)]
    return out


@router.post("/referrals/{referral_id}/cross-mandal/decision")
def decide_referral_cross(referral_id: str, body: DecisionIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*DECIDERS))):
    r = db.get(Referral, referral_id)
    if not r or not r.cross_mandal:
        raise HTTPException(404, "Request not found")
    if not _referral_in_scope(user, r, db):
        raise HTTPException(403, "This request belongs to another district")
    if r.created_by and r.created_by == user.id:
        raise HTTPException(403, "You made this request yourself. A medical officer of the hospital's district must decide it")
    if r.approval_status != "PENDING":
        raise HTTPException(409, f"This request was already {r.approval_status.lower()}")
    if body.decision not in {"approve", "reject"}:
        raise HTTPException(422, "Decision must be approve or reject")
    note = clean(body.note)
    if body.decision == "reject" and not note:
        raise HTTPException(422, "Give a short reason so the requesting medical officer can choose another specialist")
    h = db.get(Hospital, r.hospital_id) if r.hospital_id else None
    if body.decision == "approve" and (not h or not h.available):
        raise HTTPException(409, "That hospital is no longer available. Decline the request or re-enable the hospital first")
    patient = db.get(Patient, r.patient_id)
    r.approval_note = note[:255] if note else None
    r.decided_by = user.id
    clear_notifications(db, f"xref:{r.id}")  # answered: removed from every officer's notifications
    where = h.name if h else "the chosen hospital"
    if body.decision == "approve":
        r.approval_status, r.status = "APPROVED", "OPEN"
        spec = db.get(User, r.specialist_id) if r.specialist_id else None
        _activate_referral(db, r, h, spec, patient)
        if r.created_by:
            notify(db, r.created_by, "Cross-mandal request accepted", f"{where} accepted your request. The specialist and navigator were told.")
    else:
        r.approval_status, r.status = "REJECTED", "CANCELLED"
        if r.created_by:
            notify(db, r.created_by, "Cross-mandal request declined", f"{where} declined your request. Reason: {note}. Assign a specialist of the patient's own mandal or try another hospital.")
    audit(db, user.id, f"referral.cross_mandal.{body.decision}", f"referral:{r.id}", note)
    db.commit()
    return _referral_request_out(db, r)


@router.post("/appointments/{appointment_id}/cross-mandal/decision")
def decide_cross_mandal(appointment_id: str, body: DecisionIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*DECIDERS))):
    a = db.get(Appointment, appointment_id)
    if not a or not a.cross_mandal:
        raise HTTPException(404, "Cross-mandal request not found")
    if not _in_officer_scope(user, a, db):
        raise HTTPException(403, "This request belongs to another mandal")
    if a.booked_by and a.booked_by == user.id:
        raise HTTPException(403, "You booked this request yourself. Another medical officer must decide it")
    if a.approval_status != "PENDING":
        raise HTTPException(409, f"This request was already {a.approval_status.lower()}")
    if body.decision not in {"approve", "reject"}:
        raise HTTPException(422, "Decision must be approve or reject")
    note = clean(body.note)
    if body.decision == "reject" and not note:
        raise HTTPException(422, "Give a short reason so the booking team can arrange another hospital")
    h = db.get(Hospital, a.hospital_id)
    if body.decision == "approve" and (not h or not h.available):
        raise HTTPException(409, "That hospital is no longer available. Decline the request or re-enable the hospital first")
    a.approval_note = note[:255] if note else None
    a.decided_by = user.id
    clear_notifications(db, f"xmandal:{a.id}")  # answered: remove the request from every officer's notifications
    where = f"{h.name}, {h.mandal}" if h else "the chosen hospital"
    booker = {a.booked_by} if a.booked_by else set()
    if body.decision == "approve":
        a.approval_status, a.status = "APPROVED", "CONFIRMED"
        ref = db.get(Referral, a.referral_id)
        s = db.query(Screening).join(ClinicalReview, ClinicalReview.screening_id == Screening.id).filter(ClinicalReview.id == ref.review_id).first() if ref else None
        if s:
            s.status = "APPOINTED"
        notify_appointment(db, a, user, "Hospital appointment confirmed", f"Cross-mandal visit to {where} on {fmt_slot(a.slot)} was accepted by the medical officer.", extra_ids=booker)
    else:
        a.approval_status, a.status = "REJECTED", "CANCELLED"
        notify_appointment(db, a, user, "Cross-mandal request declined", f"{where} on {fmt_slot(a.slot)} was declined. Reason: {note}. Please book a hospital in the patient's mandal.", extra_ids=booker)
    audit(db, user.id, f"appointment.cross_mandal.{body.decision}", f"appointment:{a.id}", note)
    db.commit()
    return _request_out(db, a)
