import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import User

bearer = HTTPBearer(auto_error=False)

ROLES = {
    "health_worker",
    "medical_officer",
    "specialist",
    "navigator",
    "admin",  # district admin
    "super_admin",
}

# Roles that use the admin console. `admin` is the district admin; `super_admin` has every district-admin
# privilege and can additionally add / alter / delete any data in the application.
ADMIN_ROLES = ("admin", "super_admin")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 180_000)
    return f"{salt.hex()}:{dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, dk_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(dk_hex)
    except Exception:
        return False
    got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 180_000)
    return hmac.compare_digest(got, expected)


def create_token(user: User) -> str:
    payload = {
        "sub": user.id,
        "role": user.role,
        "facility": user.assigned_facility,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token")


def current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if creds is None:
        raise HTTPException(401, "Authentication required")
    data = decode_token(creds.credentials)
    user = db.get(User, data.get("sub"))
    if not user or user.status != "active":
        raise HTTPException(401, "Account not active")
    return user


def require_roles(*roles: str):
    def checker(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(403, "Not permitted for this role")
        return user

    return checker


def optional_idempotency(idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    return idempotency_key
