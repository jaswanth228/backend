"""District / mandal master list (district admin + super admin) and the super admin console."""
import json
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, delete, func, or_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from . import supabase_storage
from .access import audit
from .db import Base, get_db
from .geo import clean, geo_tree, validate_geo
from .models import District, Hospital, Mandal, Notification, Patient, User
from .routers import user_out
from .security import ADMIN_ROLES, ROLES, hash_password, require_roles

router = APIRouter()

USER_STATUSES = {"active", "pending", "rejected", "suspended"}


def _names(raw: list[str] | None) -> list[str]:
    """Accept ['A', 'B, C'] or newline separated text; return de-duplicated clean names."""
    out: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        for part in re.split(r"[,\n;]+", item or ""):
            name = re.sub(r"\s+", " ", part).strip()
            if name and name.lower() not in seen:
                seen.add(name.lower())
                out.append(name[:120])
    return out


# ---------------------------------------------------------------------------------------------------------------
# Geography: districts and mandals
# ---------------------------------------------------------------------------------------------------------------
class DistrictsIn(BaseModel):
    names: list[str]


class MandalsIn(BaseModel):
    district_id: str
    names: list[str]


class RenameIn(BaseModel):
    name: str


@router.get("/geo")
def public_geo(db: Session = Depends(get_db)):
    """District -> mandal list used by every registration form (public: the sign-up page needs it)."""
    return {"districts": geo_tree(db)}


@router.post("/admin/geo/districts", status_code=201)
def add_districts(body: DistrictsIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES))):
    names = _names(body.names)
    if not names:
        raise HTTPException(422, "Enter at least one district name")
    existing = {n.lower() for (n,) in db.query(District.name).all()}
    added, skipped = [], []
    for n in names:
        if n.lower() in existing:
            skipped.append(n)
            continue
        db.add(District(name=n, created_by=user.id))
        added.append(n)
    if added:
        audit(db, user.id, "geo.district.add", "geo:districts", json.dumps({"added": added}))
    db.commit()
    return {"added": added, "skipped": skipped, "districts": geo_tree(db)}


@router.post("/admin/geo/mandals", status_code=201)
def add_mandals(body: MandalsIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES))):
    d = db.get(District, body.district_id)
    if not d:
        raise HTTPException(404, "District not found")
    names = _names(body.names)
    if not names:
        raise HTTPException(422, "Enter at least one mandal name")
    existing = {n.lower() for (n,) in db.query(Mandal.name).filter(Mandal.district_id == d.id).all()}
    added, skipped = [], []
    for n in names:
        if n.lower() in existing:
            skipped.append(n)
            continue
        db.add(Mandal(district_id=d.id, name=n, created_by=user.id))
        added.append(n)
    if added:
        audit(db, user.id, "geo.mandal.add", f"district:{d.id}", json.dumps({"district": d.name, "added": added}))
    db.commit()
    return {"added": added, "skipped": skipped, "districts": geo_tree(db)}


@router.patch("/admin/geo/districts/{district_id}")
def rename_district(district_id: str, body: RenameIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES))):
    d = db.get(District, district_id)
    if not d:
        raise HTTPException(404, "District not found")
    new = (_names([body.name]) or [None])[0]
    if not new:
        raise HTTPException(422, "Enter a district name")
    clash = db.query(District).filter(func.lower(District.name) == new.lower(), District.id != d.id).first()
    if clash:
        raise HTTPException(409, "A district with this name already exists")
    old = d.name
    d.name = new
    # Patients and staff keep the district as text, so carry the rename through to them.
    db.query(Patient).filter(Patient.district == old).update({Patient.district: new})
    db.query(User).filter(User.district == old).update({User.district: new})
    audit(db, user.id, "geo.district.rename", f"district:{d.id}", json.dumps({"from": old, "to": new}))
    db.commit()
    return {"districts": geo_tree(db)}


@router.patch("/admin/geo/mandals/{mandal_id}")
def rename_mandal(mandal_id: str, body: RenameIn, db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES))):
    m = db.get(Mandal, mandal_id)
    if not m:
        raise HTTPException(404, "Mandal not found")
    d = db.get(District, m.district_id)
    new = (_names([body.name]) or [None])[0]
    if not new:
        raise HTTPException(422, "Enter a mandal name")
    clash = db.query(Mandal).filter(Mandal.district_id == m.district_id, func.lower(Mandal.name) == new.lower(), Mandal.id != m.id).first()
    if clash:
        raise HTTPException(409, "This district already has a mandal with that name")
    old = m.name
    m.name = new
    db.query(Patient).filter(Patient.district == d.name, Patient.mandal == old).update({Patient.mandal: new})
    db.query(User).filter(User.district == d.name, User.mandal == old).update({User.mandal: new})
    audit(db, user.id, "geo.mandal.rename", f"mandal:{m.id}", json.dumps({"district": d.name, "from": old, "to": new}))
    db.commit()
    return {"districts": geo_tree(db)}


def _mandal_usage(db: Session, district: str, mandal: str) -> int:
    return (
        db.query(Patient.id).filter(Patient.district == district, Patient.mandal == mandal).count()
        + db.query(User.id).filter(User.district == district, User.mandal == mandal).count()
    )


@router.delete("/admin/geo/mandals/{mandal_id}")
def delete_mandal(mandal_id: str, force: bool = False, db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES))):
    m = db.get(Mandal, mandal_id)
    if not m:
        raise HTTPException(404, "Mandal not found")
    d = db.get(District, m.district_id)
    used = _mandal_usage(db, d.name, m.name) + db.query(Hospital.id).filter(Hospital.district == d.name, Hospital.mandal == m.name).count()
    if used and not (force and user.role == "super_admin"):
        raise HTTPException(409, f"{used} patient/staff/hospital record(s) use this mandal, so it cannot be removed from the list")
    db.delete(m)
    audit(db, user.id, "geo.mandal.delete", f"mandal:{m.id}", json.dumps({"district": d.name, "name": m.name, "forced": bool(used)}))
    db.commit()
    return {"districts": geo_tree(db)}


@router.delete("/admin/geo/districts/{district_id}")
def delete_district(district_id: str, force: bool = False, db: Session = Depends(get_db), user: User = Depends(require_roles(*ADMIN_ROLES))):
    d = db.get(District, district_id)
    if not d:
        raise HTTPException(404, "District not found")
    used = db.query(Patient.id).filter(Patient.district == d.name).count() + db.query(User.id).filter(User.district == d.name).count() + db.query(Hospital.id).filter(Hospital.district == d.name).count()
    if used and not (force and user.role == "super_admin"):
        raise HTTPException(409, f"{used} patient/staff/hospital record(s) use this district, so it cannot be removed from the list")
    db.query(Mandal).filter(Mandal.district_id == d.id).delete()
    db.delete(d)
    audit(db, user.id, "geo.district.delete", f"district:{d.id}", json.dumps({"name": d.name, "forced": bool(used)}))
    db.commit()
    return {"districts": geo_tree(db)}


# ---------------------------------------------------------------------------------------------------------------
# Super admin: accounts (add admins, edit any user, delete)
# ---------------------------------------------------------------------------------------------------------------
class SuperUserIn(BaseModel):
    full_name: str
    email: str
    password: str
    role: str
    assigned_facility: str | None = None
    district: str | None = None
    mandal: str | None = None


class SuperUserPatch(BaseModel):
    full_name: str | None = None
    email: str | None = None
    password: str | None = None
    role: str | None = None
    assigned_facility: str | None = None
    district: str | None = None
    mandal: str | None = None
    status: str | None = None


super_only = require_roles("super_admin")


def _other_active_super_admins(db: Session, user_id: str) -> int:
    return db.query(User).filter(User.role == "super_admin", User.status == "active", User.id != user_id).count()


@router.post("/super/users", status_code=201)
def super_create_user(body: SuperUserIn, db: Session = Depends(get_db), user: User = Depends(super_only)):
    """Create any account (district admin, super admin or staff) directly as active — no approval step."""
    email = body.email.strip().lower()
    name = body.full_name.strip()
    if body.role not in ROLES:
        raise HTTPException(422, "Choose a valid role")
    if len(name) < 2 or "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(422, "Enter a full name and a valid email")
    if len(body.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(409, "An account with this email already exists")
    district, mandal = clean(body.district), clean(body.mandal)
    if body.role not in ADMIN_ROLES:
        district, mandal = validate_geo(db, district, mandal)
    u = User(
        email=email,
        full_name=name,
        password_hash=hash_password(body.password),
        role=body.role,
        assigned_facility=clean(body.assigned_facility) or "PHC G. Madugula",
        district=district,
        mandal=mandal,
        status="active",
    )
    db.add(u)
    db.flush()
    audit(db, user.id, "superadmin.user.create", f"user:{u.id}", json.dumps({"role": u.role}))
    db.commit()
    return user_out(u)


@router.patch("/super/users/{user_id}")
def super_update_user(user_id: str, body: SuperUserPatch, db: Session = Depends(get_db), user: User = Depends(super_only)):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    data = body.model_dump(exclude_unset=True)
    if u.id == user.id and (data.get("role", u.role) != u.role or data.get("status", u.status) != u.status):
        raise HTTPException(400, "You cannot change your own role or status")
    if "role" in data:
        if data["role"] not in ROLES:
            raise HTTPException(422, "Choose a valid role")
        if u.role == "super_admin" and data["role"] != "super_admin" and not _other_active_super_admins(db, u.id):
            raise HTTPException(409, "At least one active super admin must remain")
        u.role = data["role"]
    if "status" in data:
        if data["status"] not in USER_STATUSES:
            raise HTTPException(422, "Choose a valid status")
        if u.role == "super_admin" and data["status"] != "active" and u.status == "active" and not _other_active_super_admins(db, u.id):
            raise HTTPException(409, "At least one active super admin must remain")
        if data["status"] == "suspended" and u.status != "suspended":
            u.suspended_at = datetime.now(timezone.utc)
        if data["status"] != "suspended":
            u.suspended_at = None
            u.suspension_reason = None
        u.status = data["status"]
    if "full_name" in data and clean(data["full_name"]):
        u.full_name = data["full_name"].strip()
    if "email" in data and clean(data["email"]):
        email = data["email"].strip().lower()
        if db.query(User).filter(User.email == email, User.id != u.id).first():
            raise HTTPException(409, "An account with this email already exists")
        u.email = email
    if "password" in data and data["password"]:
        if len(data["password"]) < 8:
            raise HTTPException(422, "Password must be at least 8 characters")
        u.password_hash = hash_password(data["password"])
    if "assigned_facility" in data and clean(data["assigned_facility"]):
        u.assigned_facility = data["assigned_facility"].strip()
    if "district" in data or "mandal" in data:
        district = clean(data["district"]) if "district" in data else u.district
        mandal = clean(data["mandal"]) if "mandal" in data else u.mandal
        if u.role not in ADMIN_ROLES:
            district, mandal = validate_geo(db, district, mandal)
        u.district, u.mandal = district, mandal
    meta = {k: ("***" if k == "password" else v) for k, v in data.items()}
    audit(db, user.id, "superadmin.user.update", f"user:{u.id}", json.dumps(meta))
    db.commit()
    return user_out(u)


@router.delete("/super/users/{user_id}")
def super_delete_user(user_id: str, cascade: bool = False, db: Session = Depends(get_db), user: User = Depends(super_only)):
    """Permanently remove an account. Use suspend instead to keep the person's history.

    Without `cascade` the delete is refused while patients, screenings or reviews still point at the account.
    """
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404, "User not found")
    if u.id == user.id:
        raise HTTPException(400, "You cannot delete your own account")
    if u.role == "super_admin" and not _other_active_super_admins(db, u.id):
        raise HTTPException(409, "At least one active super admin must remain")
    db.query(Notification).filter(Notification.user_id == u.id).delete()
    keys = _delete_row(db, User.__table__, u.id, cascade)
    audit(db, user.id, "superadmin.user.delete", f"user:{user_id}", json.dumps({"email": u.email, "role": u.role, "cascade": cascade}))
    db.commit()
    _purge_storage(keys)
    return {"ok": True}


# ---------------------------------------------------------------------------------------------------------------
# Super admin: generic data manager (add / alter / delete any row of any table)
# ---------------------------------------------------------------------------------------------------------------
HIDDEN_COLUMNS = {"password_hash"}


def _models() -> dict:
    return {m.class_.__tablename__: m.class_ for m in Base.registry.mappers}


def _model_or_404(table: str):
    model = _models().get(table)
    if not model:
        raise HTTPException(404, "Unknown table")
    return model


def _ctype(col) -> str:
    t = col.type
    if isinstance(t, Boolean):
        return "boolean"
    if isinstance(t, Integer):
        return "integer"
    if isinstance(t, Float):
        return "number"
    if isinstance(t, DateTime):
        return "datetime"
    if isinstance(t, Text):
        return "text"
    return "string"


def _columns(model) -> list[dict]:
    return [
        {"name": c.name, "type": _ctype(c), "nullable": bool(c.nullable), "pk": bool(c.primary_key)}
        for c in model.__table__.columns
        if c.name not in HIDDEN_COLUMNS
    ]


def _row_out(model, obj) -> dict:
    out = {}
    for c in model.__table__.columns:
        if c.name in HIDDEN_COLUMNS:
            continue
        v = getattr(obj, c.name)
        out[c.name] = v.isoformat() if isinstance(v, datetime) else v
    return out


def _coerce(col, value):
    if isinstance(value, str) and value.strip() == "" and (col.nullable or not isinstance(col.type, String)):
        value = None
    if value is None:
        return None
    try:
        if isinstance(col.type, Boolean):
            return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "y"}
        if isinstance(col.type, Integer):
            return int(value)
        if isinstance(col.type, Float):
            return float(value)
        if isinstance(col.type, DateTime):
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return str(value)
    except (ValueError, TypeError):
        raise HTTPException(422, f"'{col.name}' has an invalid value")


def _values(model, data: dict, *, creating: bool) -> dict:
    cols = {c.name: c for c in model.__table__.columns}
    values = {}
    for k, v in (data or {}).items():
        if k in HIDDEN_COLUMNS or k == "password":
            continue
        col = cols.get(k)
        if col is None:
            raise HTTPException(422, f"Unknown column '{k}'")
        if col.primary_key and not creating:
            continue
        if col.primary_key and (v is None or v == ""):
            continue
        coerced = _coerce(col, v)
        if coerced is None and not col.nullable:
            continue  # a required column left empty keeps its default / current value
        values[k] = coerced
    return values


def _purge_storage(keys: list[str]) -> None:
    for key in keys:
        try:
            supabase_storage.remove(key)
        except Exception:
            pass  # best effort: the database row is already gone


def _delete_row(db: Session, table, pk_value, cascade: bool, keys: list[str] | None = None) -> list[str]:
    """Delete one row. With cascade, rows that require it are deleted too and optional links are cleared."""
    keys = [] if keys is None else keys
    pkcol = next(iter(table.primary_key.columns))
    if cascade:
        for child in Base.metadata.tables.values():
            for fk in list(child.foreign_keys):
                if fk.column is not pkcol:
                    continue
                col = fk.parent
                if col.nullable:
                    db.execute(update(child).where(col == pk_value).values({col.name: None}))
                else:
                    cpk = next(iter(child.primary_key.columns))
                    for (cid,) in db.execute(select(cpk).where(col == pk_value)).all():
                        _delete_row(db, child, cid, True, keys)
    if table.name == "images":
        row = db.execute(select(table.c.storage_key).where(pkcol == pk_value)).first()
        if row and row[0]:
            keys.append(row[0])
    try:
        db.execute(delete(table).where(pkcol == pk_value))
        db.flush()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(409, "Other records still reference this row. Delete them first, or use cascade delete.")
    return keys


class RowIn(BaseModel):
    data: dict


@router.get("/super/data")
def data_tables(db: Session = Depends(get_db), user: User = Depends(super_only)):
    out = []
    for name, model in sorted(_models().items()):
        out.append({"table": name, "rows": db.query(func.count()).select_from(model).scalar() or 0, "columns": _columns(model)})
    return out


@router.get("/super/data/{table}")
def data_rows(
    table: str,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    user: User = Depends(super_only),
):
    model = _model_or_404(table)
    query = db.query(model)
    if clean(q):
        like = f"%{clean(q)}%"
        text_cols = [c for c in model.__table__.columns if isinstance(c.type, (String, Text)) and c.name not in HIDDEN_COLUMNS]
        if text_cols:
            query = query.filter(or_(*[c.ilike(like) for c in text_cols]))
    total = query.count()
    order = model.__table__.c.created_at.desc() if "created_at" in model.__table__.c else next(iter(model.__table__.primary_key.columns))
    rows = query.order_by(order).offset(max(0, offset)).limit(min(max(1, limit), 200)).all()
    return {"table": table, "columns": _columns(model), "total": total, "rows": [_row_out(model, r) for r in rows]}


@router.post("/super/data/{table}", status_code=201)
def data_create(table: str, body: RowIn, db: Session = Depends(get_db), user: User = Depends(super_only)):
    model = _model_or_404(table)
    values = _values(model, body.data, creating=True)
    if table == "users":
        if values.get("role") not in ROLES:
            raise HTTPException(422, "Choose a valid role")
        pw = body.data.get("password")
        if not pw or len(str(pw)) < 8:
            raise HTTPException(422, "Provide a 'password' of at least 8 characters for a new user")
        values["password_hash"] = hash_password(str(pw))
        values["email"] = str(values.get("email", "")).strip().lower()
    obj = model(**values)
    try:
        db.add(obj)
        db.flush()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(422, f"Could not add row: {str(getattr(exc, 'orig', exc)).splitlines()[0]}")
    pk = getattr(obj, next(iter(model.__table__.primary_key.columns)).name)
    audit(db, user.id, "superadmin.data.create", f"{table}:{pk}")
    db.commit()
    return _row_out(model, obj)


@router.patch("/super/data/{table}/{row_id}")
def data_update(table: str, row_id: str, body: RowIn, db: Session = Depends(get_db), user: User = Depends(super_only)):
    model = _model_or_404(table)
    obj = db.get(model, row_id)
    if not obj:
        raise HTTPException(404, "Row not found")
    values = _values(model, body.data, creating=False)
    if table == "users":
        if "role" in values and values["role"] not in ROLES:
            raise HTTPException(422, "Choose a valid role")
        if "email" in values:
            values["email"] = str(values["email"]).strip().lower()
        if body.data.get("password"):
            if len(str(body.data["password"])) < 8:
                raise HTTPException(422, "Password must be at least 8 characters")
            values["password_hash"] = hash_password(str(body.data["password"]))
        demoting = obj.role == "super_admin" and (values.get("role", obj.role) != "super_admin" or values.get("status", obj.status) != "active")
        if demoting and obj.status == "active" and not _other_active_super_admins(db, obj.id):
            raise HTTPException(409, "At least one active super admin must remain")
    for k, v in values.items():
        setattr(obj, k, v)
    try:
        db.flush()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(422, f"Could not save row: {str(getattr(exc, 'orig', exc)).splitlines()[0]}")
    audit(db, user.id, "superadmin.data.update", f"{table}:{row_id}", json.dumps({"columns": sorted(values)}))
    db.commit()
    return _row_out(model, obj)


@router.delete("/super/data/{table}/{row_id}")
def data_delete(table: str, row_id: str, cascade: bool = False, db: Session = Depends(get_db), user: User = Depends(super_only)):
    model = _model_or_404(table)
    obj = db.get(model, row_id)
    if not obj:
        raise HTTPException(404, "Row not found")
    if table == "users":
        if obj.id == user.id:
            raise HTTPException(400, "You cannot delete your own account")
        if obj.role == "super_admin" and not _other_active_super_admins(db, obj.id):
            raise HTTPException(409, "At least one active super admin must remain")
    keys = _delete_row(db, model.__table__, row_id, cascade)
    audit(db, user.id, "superadmin.data.delete", f"{table}:{row_id}", json.dumps({"cascade": cascade}))
    db.commit()
    _purge_storage(keys)
    return {"ok": True}
