"""District / mandal master list helpers and dashboard scoping."""
from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import (
    AiAnalysis,
    Appointment,
    ClinicalReview,
    District,
    Investigation,
    Mandal,
    NavigationTask,
    Patient,
    Referral,
    Screening,
)


def clean(value: str | None) -> str | None:
    return (value or "").strip() or None


def geo_tree(db: Session) -> list[dict]:
    mandals: dict[str, list[dict]] = {}
    for m in db.query(Mandal).order_by(Mandal.name).all():
        mandals.setdefault(m.district_id, []).append({"id": m.id, "name": m.name})
    return [
        {"id": d.id, "name": d.name, "mandals": mandals.get(d.id, [])}
        for d in db.query(District).order_by(District.name).all()
    ]


def validate_geo(db: Session, district: str | None, mandal: str | None) -> tuple[str | None, str | None]:
    """Return the canonical (district, mandal) names from the master list.

    Once at least one district exists, every registration must pick a district and one of its mandals.
    With an empty master list nothing can be selected yet, so free values are accepted.
    """
    district, mandal = clean(district), clean(mandal)
    if db.query(District.id).first() is None:
        return district, mandal
    if not district or not mandal:
        raise HTTPException(422, "Select your district and mandal")
    d = db.query(District).filter(func.lower(District.name) == district.lower()).first()
    if not d:
        raise HTTPException(422, "Unknown district. Choose one from the list")
    m = db.query(Mandal).filter(Mandal.district_id == d.id, func.lower(Mandal.name) == mandal.lower()).first()
    if not m:
        raise HTTPException(422, f"'{mandal}' is not a mandal of {d.name}")
    return d.name, m.name


class GeoScope:
    """Restricts dashboard queries to one district and/or mandal (by the patient's location).

    `scope.q(Model, *columns)` behaves like `db.query(...)` but only counts rows that belong to patients in scope.
    With no district/mandal selected it is a plain, unfiltered query.
    """

    def __init__(self, db: Session, district: str | None = None, mandal: str | None = None):
        self.db = db
        self.district = clean(district)
        self.mandal = clean(mandal)
        self.active = bool(self.district or self.mandal)
        pids = select(Patient.id)
        if self.district:
            pids = pids.where(Patient.district == self.district)
        if self.mandal:
            pids = pids.where(Patient.mandal == self.mandal)
        self.pids = pids
        self.sids = select(Screening.id).where(Screening.patient_id.in_(pids))
        self.rids = select(Referral.id).where(Referral.patient_id.in_(pids))

    @property
    def level(self) -> str:
        return "mandal" if self.mandal else "district" if self.district else "all"

    def q(self, model, *cols):
        query = self.db.query(*(cols or (model,)))
        if not self.active:
            return query
        if model is Patient:
            return query.filter(Patient.id.in_(self.pids))
        if model in (Screening, Referral):
            return query.filter(model.patient_id.in_(self.pids))
        if model in (ClinicalReview, AiAnalysis):
            return query.filter(model.screening_id.in_(self.sids))
        if model in (Appointment, Investigation, NavigationTask):
            return query.filter(model.referral_id.in_(self.rids))
        return query
