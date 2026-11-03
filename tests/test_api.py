import io
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use a SEPARATE PostgreSQL database for tests (tables are dropped/recreated).
# Point TEST_DATABASE_URL at a throwaway Postgres (tables are dropped!). NEVER your real Supabase project.
os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
os.environ["SEED_ON_START"] = "true"
os.environ["SEED_DEMO_DATA"] = "false"  # keep the journey tests deterministic; test_roles_load_straight_from_db seeds explicitly
os.environ["LOCAL_QUALITY_CHECK"] = "true"  # existing tests exercise the local gate; see test_ai_judges_quality_when_local_check_off
os.environ["JWT_SECRET"] = "test-secret-not-for-production-use-32b"
os.environ["STORAGE_DIR"] = str(ROOT / "storage-test")

from app import supabase_storage  # noqa: E402

_FAKE_BUCKET: dict[str, bytes] = {}
supabase_storage.ensure_bucket = lambda: None
supabase_storage.upload = lambda key, data, content_type="application/octet-stream": _FAKE_BUCKET.__setitem__(key, data)
supabase_storage.download = lambda key: _FAKE_BUCKET.get(key)

from app.db import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.seed import seed_if_empty  # noqa: E402

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
_db = SessionLocal()
seed_if_empty(_db)
_db.close()

client = TestClient(app)


def login(email: str) -> str:
    r = client.post("/auth/login", json={"email": email, "password": "Demo@123"})
    assert r.status_code == 200, r.text
    return r.json()["token"]


def auth(email: str):
    return {"Authorization": f"Bearer {login(email)}"}


def make_image(blur=False) -> bytes:
    img = Image.new("RGB", (400, 400), color=(180, 90, 90) if not blur else (40, 40, 40))
    if not blur:
        for x in range(0, 400, 4):
            for y in range(0, 400, 4):
                img.putpixel((x, y), (10, 200, 20) if (x + y) % 8 == 0 else (220, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["clinical_use"] is False


def test_login_and_role_isolation():
    token = login("hw@demo.local")
    assert token
    admin = auth("admin@demo.local")
    r = client.get("/patients", headers=admin)
    assert r.status_code == 403
    dash = client.get("/dashboard/summary", headers=admin)
    assert dash.status_code == 200


def test_consent_gate_and_journey():
    hw = auth("hw@demo.local")
    p = client.post(
        "/patients",
        headers={**hw, "Idempotency-Key": "p1"},
        json={
            "full_name": "Synthetic Test Patient",
            "age": 46,
            "sex": "female",
            "phone": "9111111111",
            "village": "G. Madugula",
            "tobacco_use": "gutkha",
            "symptoms": "white patch",
            "video_allowed": True,
        },
    )
    assert p.status_code == 200, p.text
    pid = p.json()["id"]
    again = client.post(
        "/patients",
        headers={**hw, "Idempotency-Key": "p1"},
        json={"full_name": "Synthetic Test Patient"},
    )
    assert again.json()["id"] == pid

    blocked = client.post("/screenings", headers=hw, json={"patient_id": pid, "exam_notes": "lesion"})
    assert blocked.status_code == 400

    for purpose in ("examination", "imaging"):
        c = client.post(f"/patients/{pid}/consents", headers=hw, json={"purpose": purpose, "language": "te"})
        assert c.status_code == 200

    s = client.post("/screenings", headers={**hw, "Idempotency-Key": "s1"}, json={"patient_id": pid, "exam_notes": "ulcer"})
    assert s.status_code == 200
    sid = s.json()["id"]

    img = client.post(
        f"/screenings/{sid}/images",
        headers=hw,
        files={"file": ("photo.jpg", make_image(), "image/jpeg")},
        data={"view": "cheek"},
    )
    assert img.status_code == 200, img.text
    assert img.json()["quality_status"] == "acceptable"
    image_id = img.json()["id"]

    blur = client.post(
        f"/screenings/{sid}/images",
        headers=hw,
        files={"file": ("blur.jpg", make_image(blur=True), "image/jpeg")},
        data={"view": "tongue"},
    )
    assert blur.status_code == 200
    insufficient = client.post(
        "/ai/analyze",
        headers=hw,
        json={"screening_id": sid, "image_id": blur.json()["id"]},
    )
    assert insufficient.status_code == 200
    assert insufficient.json()["status"] == "image_insufficient"

    ai = client.post("/ai/analyze", headers=hw, json={"screening_id": sid, "image_id": image_id})
    assert ai.status_code == 200
    assert ai.json()["clinical_use"] is False
    assert ai.json()["requires_clinician_review"] is True

    mo = auth("mo@demo.local")
    review = client.post(
        "/reviews",
        headers=mo,
        json={"screening_id": sid, "finding": "Clinical suspicion, independent of AI", "outcome": "refer", "ai_agreement": "disagree"},
    )
    assert review.status_code == 200
    replay = client.post(
        "/reviews",
        headers=mo,
        json={"screening_id": sid, "finding": "overwrite", "outcome": "refer"},
    )
    assert replay.status_code == 409

    spec = auth("spec@demo.local")
    denied = client.get(f"/patients/{pid}", headers=spec)
    assert denied.status_code == 403

    ref = client.post("/referrals", headers=mo, json={"review_id": review.json()["id"], "reason": "specialist exam", "hospital_id": client.get("/hospitals", headers=mo).json()[0]["id"]})
    assert ref.status_code == 200
    assert "te" in ref.json()["slip"]
    allowed = client.get(f"/patients/{pid}", headers=spec)
    assert allowed.status_code == 200

    hw_blocked = client.get("/referrals", headers=hw)
    assert hw_blocked.status_code == 403

    nav = auth("nav@demo.local")
    tasks = client.get("/navigation", headers=nav)
    assert tasks.status_code == 200
    assert tasks.json()

    appt = client.post(
        "/appointments",
        headers=spec,
        json={"referral_id": ref.json()["id"], "kind": "video", "slot": "2026-11-02T10:00:00+05:30", "status": "CONFIRMED"},
    )
    assert appt.status_code == 422  # a Google Meet link is required for video
    appt = client.post(
        "/appointments",
        headers=spec,
        json={"referral_id": ref.json()["id"], "kind": "video", "slot": "2026-11-02T10:00:00+05:30", "status": "CONFIRMED", "meet_url": "https://meet.google.com/abc-defg-hij"},
    )
    assert appt.status_code == 200, appt.text
    nav_start = client.post(
        "/appointments",
        headers=nav,
        json={"referral_id": ref.json()["id"], "kind": "video", "slot": "x", "meet_url": "https://meet.google.com/abc-defg-hij", "start_now": True},
    )
    assert nav_start.status_code == 403  # only specialists may start immediately
    now_call = client.post(
        "/appointments",
        headers=spec,
        json={"referral_id": ref.json()["id"], "kind": "video", "slot": "ignored", "meet_url": "https://meet.google.com/xyz-wxyz-abc", "start_now": True},
    )
    assert now_call.status_code == 200, now_call.text
    assert now_call.json()["status"] == "CONFIRMED"
    video = client.get(f"/appointments/{appt.json()['id']}/video", headers=spec)
    assert video.status_code == 200
    assert video.json()["url"] == "https://meet.google.com/abc-defg-hij"
    assert video.json()["recording_default"] is False

    inv = client.post("/investigations", headers=spec, json={"referral_id": ref.json()["id"], "kind": "biopsy"})
    assert inv.status_code == 200
    advanced = client.patch(
        f"/investigations/{inv.json()['id']}",
        headers=spec,
        json={"state": "CLINICIAN_REVIEWED", "clinician_result": "Synthetic clinician result"},
    )
    assert advanced.status_code == 200

    closed = client.patch(
        f"/navigation/{tasks.json()[0]['id']}",
        headers=nav,
        json={"status": "CLOSED", "contact_log": "Linked to care (synthetic)."},
    )
    assert closed.status_code == 200

    dash = client.get("/dashboard/summary", headers=auth("admin@demo.local"))
    assert dash.status_code == 200
    assert dash.json()["referrals"] >= 1


def test_wrong_password():
    r = client.post("/auth/login", json={"email": "hw@demo.local", "password": "wrong"})
    assert r.status_code == 401


def _journey_to_image(hw, name: str, phone: str):
    p = client.post(
        "/patients",
        headers={**hw, "Idempotency-Key": f"p-{name}"},
        json={"full_name": name, "age": 52, "sex": "male", "phone": phone, "village": "Test", "tobacco_use": "gutka"},
    )
    assert p.status_code == 200, p.text
    pid = p.json()["id"]
    for purpose in ("examination", "imaging"):
        assert client.post(f"/patients/{pid}/consents", headers=hw, json={"purpose": purpose}).status_code == 200
    s = client.post(
        "/screenings",
        headers={**hw, "Idempotency-Key": f"s-{name}"},
        json={"patient_id": pid, "symptoms": "sore for 3 weeks", "tobacco": "gutka daily"},
    )
    sid = s.json()["id"]
    img = client.post(
        f"/screenings/{sid}/images",
        headers=hw,
        files={"file": ("photo.jpg", make_image(), "image/jpeg")},
        data={"view": "cheek"},
    )
    assert img.status_code == 200, img.text
    return sid, img.json()["id"]


def test_ollama_feedback_is_stored_and_returned(monkeypatch):
    from app import routers
    from app.ai import ollama_adapter

    seen = {}

    def fake_ollama(raw, screening_id, context):
        seen["bytes"] = len(raw)
        seen["context"] = context
        return {
            "provider": "ollama",
            "model_version": "fake-vision:1",
            "status": "completed",
            "research_label": "suspicious_example",
            "score": None,
            "clinical_use": False,
            "requires_clinician_review": True,
            "warning": "NOT FOR CLINICAL USE",
            "screening_id": screening_id,
            "feedback": {
                "predicted_condition": {"disease_detected": True, "name": "Oral leukoplakia (most likely)", "likelihood": "medium"},
                "suggested_urgency": "soon",
                "clinician_summary": "Examine the left cheek.",
                "patient_message": {"en": "Please see a doctor soon.", "te": "త్వరగా డాక్టర్‌ని కలవండి."},
            },
        }

    monkeypatch.setattr(routers.settings, "AI_PROVIDER", "ollama")
    monkeypatch.setattr(routers, "ollama_analyze", fake_ollama)

    hw = auth("hw@demo.local")
    sid, image_id = _journey_to_image(hw, "Ollama Patient", "9222222222")
    r = client.post("/ai/analyze", headers=hw, json={"screening_id": sid, "image_id": image_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "ollama"
    assert body["clinical_use"] is False and body["requires_clinician_review"] is True
    assert body["feedback"]["suggested_urgency"] == "soon"
    # the backend, not the browser, read the stored photo and passed patient context along
    assert seen["bytes"] > 1000
    assert seen["context"]["view"] == "cheek" and seen["context"]["age"] == 52
    assert "full_name" not in seen["context"] and "phone" not in seen["context"]

    detail = client.get(f"/screenings/{sid}", headers=hw).json()
    assert detail["ai_analyses"][-1]["feedback"]["predicted_condition"]["name"].startswith("Oral leukoplakia")


def test_ollama_failure_preserves_screening(monkeypatch):
    from app import routers
    from app.ai.ollama_adapter import OllamaError

    def boom(*_a, **_k):
        raise OllamaError("Ollama Cloud timed out")

    monkeypatch.setattr(routers.settings, "AI_PROVIDER", "ollama")
    monkeypatch.setattr(routers, "ollama_analyze", boom)

    hw = auth("hw@demo.local")
    sid, image_id = _journey_to_image(hw, "Ollama Fail Patient", "9333333333")
    r = client.post("/ai/analyze", headers=hw, json={"screening_id": sid, "image_id": image_id})
    assert r.status_code == 503
    assert r.json()["detail"]["status"] == "failed"
    detail = client.get(f"/screenings/{sid}", headers=hw).json()
    assert detail["status"] == "AWAITING_CLINICIAN"
    assert detail["ai_analyses"][-1]["status"] == "failed"


def test_cannot_analyze_image_from_another_screening():
    hw = auth("hw@demo.local")
    sid_a, _ = _journey_to_image(hw, "Owner A", "9444444444")
    _sid_b, image_b = _journey_to_image(hw, "Owner B", "9555555555")
    r = client.post("/ai/analyze", headers=hw, json={"screening_id": sid_a, "image_id": image_b})
    assert r.status_code == 404


def test_ai_judges_quality_when_local_check_off(monkeypatch):
    from app import routers

    monkeypatch.setattr(routers.settings, "LOCAL_QUALITY_CHECK", False)
    monkeypatch.setattr(routers.settings, "AI_PROVIDER", "ollama")

    def fake_ollama(raw, screening_id, context):
        usable = context["view"] != "tongue"
        return {
            "provider": "ollama",
            "model_version": "fake-vision:1",
            "status": "completed" if usable else "image_insufficient",
            "research_label": "no_suspicious_finding_example" if usable else "inconclusive_example",
            "score": None,
            "clinical_use": False,
            "requires_clinician_review": True,
            "warning": "NOT FOR CLINICAL USE",
            "screening_id": screening_id,
            "feedback": {
                "image_assessment": {"is_oral_photo": usable, "usable": usable, "issues": [] if usable else ["too dark"]},
                "recapture_tips": [] if usable else ["Use more light"],
            },
        }

    monkeypatch.setattr(routers, "ollama_analyze", fake_ollama)

    hw = auth("hw@demo.local")
    sid, image_id = _journey_to_image(hw, "No Gate Patient", "9666666666")
    detail = client.get(f"/screenings/{sid}", headers=hw).json()
    # a blurry photo is no longer rejected locally
    blur = client.post(
        f"/screenings/{sid}/images",
        headers=hw,
        files={"file": ("blur.jpg", make_image(blur=True), "image/jpeg")},
        data={"view": "tongue"},
    )
    assert blur.status_code == 200 and blur.json()["quality_status"] == "pending_ai"

    ok = client.post("/ai/analyze", headers=hw, json={"screening_id": sid, "image_id": image_id})
    assert ok.status_code == 200 and ok.json()["status"] == "completed"

    bad = client.post("/ai/analyze", headers=hw, json={"screening_id": sid, "image_id": blur.json()["id"]})
    assert bad.status_code == 200
    assert bad.json()["status"] == "image_insufficient"
    assert bad.json()["feedback"]["recapture_tips"] == ["Use more light"]

    images = {i["id"]: i for i in client.get(f"/screenings/{sid}", headers=hw).json()["images"]}
    assert images[image_id]["quality_status"] == "acceptable"
    assert images[blur.json()["id"]]["quality_status"] == "recapture"
    assert images[blur.json()["id"]]["quality_reason"] == "too dark"


def test_registration_requires_admin_approval():
    body = {"full_name": "New Navigator", "email": "New.Nav@Example.com", "password": "Strong#123", "role": "navigator", "district": "Alluri Sitharama Raju", "mandal": "G. Madugula"}
    r = client.post("/auth/register", json=body)
    assert r.status_code == 201, r.text

    # duplicate email and bad designation are refused
    assert client.post("/auth/register", json=body).status_code == 409
    assert client.post("/auth/register", json={**body, "email": "x@y.com", "role": "admin"}).status_code == 422

    # pending users cannot sign in
    blocked = client.post("/auth/login", json={"email": "new.nav@example.com", "password": "Strong#123"})
    assert blocked.status_code == 403

    admin = auth("admin@demo.local")
    notes = client.get("/notifications", headers=admin).json()
    assert any("awaiting approval" in n["title"] for n in notes)

    # non-admins cannot approve
    assert client.get("/admin/users", headers=auth("hw@demo.local")).status_code == 403

    pending = client.get("/admin/users?status=pending", headers=admin).json()
    uid_ = next(u["id"] for u in pending if u["email"] == "new.nav@example.com")
    assert client.post(f"/admin/users/{uid_}/approve", headers=admin).status_code == 200
    assert client.post(f"/admin/users/{uid_}/approve", headers=admin).status_code == 409

    ok = client.post("/auth/login", json={"email": "new.nav@example.com", "password": "Strong#123"})
    assert ok.status_code == 200
    assert ok.json()["user"]["role"] == "navigator"

    detail = client.get(f"/admin/users/{uid_}", headers=admin).json()
    assert "tasks_assigned" in detail["work"]["metrics"]
    roles = {u["role"] for u in client.get("/admin/users", headers=admin).json()}
    assert "admin" not in roles


def test_rejected_registration_cannot_login():
    client.post("/auth/register", json={"full_name": "Rej User", "email": "rej@example.com", "password": "Strong#123", "role": "navigator", "district": "Alluri Sitharama Raju", "mandal": "G. Madugula"})
    admin = auth("admin@demo.local")
    pending = client.get("/admin/users?status=pending", headers=admin).json()
    uid_ = next(u["id"] for u in pending if u["email"] == "rej@example.com")
    assert client.post(f"/admin/users/{uid_}/reject", headers=admin).status_code == 200
    assert client.post("/auth/login", json={"email": "rej@example.com", "password": "Strong#123"}).status_code == 403


def test_slot_helpers():
    from app.routers import fmt_slot, parse_slot

    assert parse_slot("2026-11-02T05:00:00+00:00").hour == 5
    assert parse_slot("2026-11-02 10:00") is not None
    assert parse_slot("next tuesday") is None
    assert fmt_slot("2026-11-02T10:00:00+05:30") == "Mon, 02 Nov 2026, 10:00 (UTC+05:30)"
    assert fmt_slot("next tuesday") == "next tuesday"


def test_google_meet_link_validation():
    from fastapi import HTTPException
    from app.routers import clean_meet_url

    assert clean_meet_url("https://meet.google.com/abc-defg-hij") == "https://meet.google.com/abc-defg-hij"
    for bad in ["", "http://meet.google.com/abc-defg-hij", "https://meet.google.com.evil.com/abc-defg-hij", "https://example.com/abc-defg-hij"]:
        try:
            clean_meet_url(bad)
            raise AssertionError(bad)
        except HTTPException as e:
            assert e.status_code == 422


def test_roles_load_straight_from_db():
    """MO, specialist and navigator see database records without any health-worker action in this session."""
    from app.seed import seed_demo_pipeline

    db = SessionLocal()
    try:
        seed_demo_pipeline(db)
        seed_demo_pipeline(db)  # idempotent
    finally:
        db.close()
    mo, spec, nav = auth("mo@demo.local"), auth("spec@demo.local"), auth("nav@demo.local")
    queue = client.get("/screenings?awaiting_review=true", headers=mo)
    assert queue.status_code == 200
    names = [r["patient_name"] for r in queue.json()]
    assert any("Ramulu" in (n or "") for n in names) and any("Savitri" in (n or "") for n in names)
    assert all(r["status"] not in {"REVIEWED", "REFERRAL_PLANNED", "CARE_LINKED"} for r in queue.json())
    refs = client.get("/referrals", headers=spec)
    assert refs.status_code == 200 and any("Appa Rao" in (r["patient_name"] or "") for r in refs.json())
    assert client.get("/appointments", headers=spec).json()
    assert client.get("/investigations", headers=spec).json()
    assert client.get("/navigation", headers=nav).json()


def test_database_url_normalised_for_supabase():
    from app.db import normalize_database_url

    url = normalize_database_url("postgres://postgres:pw@db.abcd.supabase.co:5432/postgres")
    assert url.startswith("postgresql+psycopg2://") and "sslmode=require" in url
    local = normalize_database_url("postgresql+psycopg2://postgres:pw@localhost:5432/x")
    assert "sslmode" not in local


def test_ollama_returns_english_and_telugu_outputs(monkeypatch):
    import json as _json
    from types import SimpleNamespace

    from app.ai import ollama_adapter

    english = {
        "image_assessment": {"is_oral_photo": True, "usable": True, "issues": []},
        "predicted_condition": {"disease_detected": True, "name": "Oral leukoplakia (most likely)", "likelihood": "medium", "other_possibilities": ["Oral lichen planus"]},
        "analysis": "English analysis.", "research_label": "condition_possible_example", "confidence": "medium", "suggested_urgency": "soon",
        "health_suggestions": ["Stop tobacco"], "warning_signs": ["Ulcer"], "consult_advice": "See a dentist soon.",
        "health_worker_guidance": ["Arrange review"], "recapture_tips": [], "clinician_summary": "Examine cheek.",
        "patient_message": {"en": "See a doctor.", "te": "డాక్టర్‌ని కలవండి."}, "limitations": "Photo cannot confirm.",
    }
    telugu = {
        "condition_name": "ల్యూకోప్లాకియా", "other_possibilities": ["లైకెన్ ప్లానస్"], "photo_issues": [], "analysis": "తెలుగు విశ్లేషణ.",
        "health_suggestions": ["పొగాకు మానండి"], "warning_signs": ["పుండు"], "consult_advice": "త్వరలో డాక్టర్‌ను కలవండి.",
        "health_worker_guidance": ["సమీక్ష ఏర్పాటు చేయండి"], "recapture_tips": [], "clinician_summary": "చెంపను పరీక్షించండి.", "limitations": "ఫోటోతో నిర్ధారణ కాదు.",
    }
    replies = [english, telugu]
    calls = []

    class FakeClient:
        def chat(self, **kw):
            calls.append(kw)
            return SimpleNamespace(message=SimpleNamespace(content=_json.dumps(replies[len(calls) - 1], ensure_ascii=False)))

    monkeypatch.setattr(ollama_adapter, "_client", lambda: FakeClient())
    monkeypatch.setattr(ollama_adapter.settings, "OLLAMA_API_KEY", "test-key")
    out = ollama_adapter.analyze(make_image(), "scr-1", {"view": "cheek"})
    fb = out["feedback"]
    assert len(calls) == 2 and "images" not in calls[1]["messages"][-1]  # second call is text-only
    assert fb["analysis"] == "English analysis."
    te = fb["translations"]["te"]
    assert te["predicted_condition"]["name"] == "ల్యూకోప్లాకియా" and te["analysis"] == "తెలుగు విశ్లేషణ."
    assert te["health_suggestions"] == ["పొగాకు మానండి"] and te["patient_message"] == "డాక్టర్‌ని కలవండి."


def test_telugu_failure_keeps_english(monkeypatch):
    import json as _json
    from types import SimpleNamespace

    from app.ai import ollama_adapter

    n = {"i": 0}

    class FakeClient:
        def chat(self, **kw):
            n["i"] += 1
            if n["i"] == 2:
                raise RuntimeError("boom")
            return SimpleNamespace(message=SimpleNamespace(content=_json.dumps({"predicted_condition": {"name": "No disease detected", "disease_detected": False}, "analysis": "Fine."})))

    monkeypatch.setattr(ollama_adapter, "_client", lambda: FakeClient())
    monkeypatch.setattr(ollama_adapter.settings, "OLLAMA_API_KEY", "test-key")
    out = ollama_adapter.analyze(make_image(), "scr-2", {})
    assert out["feedback"]["analysis"] == "Fine." and "translations" not in out["feedback"]


def test_host_ends_or_deletes_call_and_alerts_roll_back():
    from app.models import Notification
    from app.seed import seed_demo_pipeline

    db = SessionLocal()
    try:
        seed_demo_pipeline(db)
    finally:
        db.close()
    spec, hw, nav = auth("spec@demo.local"), auth("hw@demo.local"), auth("nav@demo.local")
    ref = next(r for r in client.get("/referrals", headers=spec).json() if "Appa Rao" in (r["patient_name"] or ""))

    def alerts_about(headers, appt_id):
        return [n for n in client.get("/notifications", headers=headers).json() if f"[appt:{appt_id}]" in n["body"]]

    a = client.post("/appointments", headers=spec, json={"referral_id": ref["id"], "kind": "video", "start_now": True, "status": "CONFIRMED", "meet_url": "https://meet.google.com/abc-defg-hij"})
    assert a.status_code == 200, a.text
    aid = a.json()["id"]
    assert alerts_about(nav, aid)  # the live alert reached the navigator

    ended = client.post(f"/appointments/{aid}/end", headers=spec)
    assert ended.status_code == 200 and ended.json()["status"] == "CANCELLED" and ended.json()["has_meet_link"] is False
    assert alerts_about(nav, aid) == [] and alerts_about(hw, aid) == []
    assert client.get(f"/appointments/{aid}/video", headers=hw).status_code == 403  # link no longer active

    b = client.post("/appointments", headers=spec, json={"referral_id": ref["id"], "kind": "video", "slot": "2030-01-01T10:00:00+05:30", "status": "CONFIRMED", "meet_url": "https://meet.google.com/xyz-abcd-efg"})
    bid = b.json()["id"]
    assert alerts_about(nav, bid)
    assert client.delete(f"/appointments/{bid}", headers=hw).status_code == 403  # health worker cannot delete
    assert client.delete(f"/appointments/{bid}", headers=spec).status_code == 200
    assert alerts_about(nav, bid) == [] and client.get(f"/appointments/{bid}", headers=spec).status_code == 404


# ----------------------------------------------------------------------------------------------------------------
# Stage 11: super admin, district / mandal lists, scoped dashboards, suspension
# ----------------------------------------------------------------------------------------------------------------
def _register_staff(email: str, role: str = "navigator", district="Alluri Sitharama Raju", mandal="G. Madugula"):
    r = client.post("/auth/register", json={"full_name": "Staff " + email, "email": email, "password": "Strong#123", "role": role, "district": district, "mandal": mandal})
    assert r.status_code == 201, r.text
    admin = auth("admin@demo.local")
    pending = client.get("/admin/users?status=pending", headers=admin).json()
    uid_ = next(u["id"] for u in pending if u["email"] == email)
    assert client.post(f"/admin/users/{uid_}/approve", headers=admin).status_code == 200
    return uid_


def test_geography_lists_and_registration_selection():
    admin = auth("admin@demo.local")
    # public list used by the sign-up page
    geo = client.get("/geo").json()["districts"]
    assert any(d["name"] == "Alluri Sitharama Raju" and any(m["name"] == "G. Madugula" for m in d["mandals"]) for d in geo)

    # only admins may edit the list
    assert client.post("/admin/geo/districts", json={"names": ["Nope"]}, headers=auth("hw@demo.local")).status_code == 403

    r = client.post("/admin/geo/districts", json={"names": ["Test District A, Test District B\nTest District A"]}, headers=admin)
    assert r.status_code == 201 and r.json()["added"] == ["Test District A", "Test District B"]
    did = next(d["id"] for d in r.json()["districts"] if d["name"] == "Test District A")
    m = client.post("/admin/geo/mandals", json={"district_id": did, "names": ["Mandal One, Mandal Two"]}, headers=admin)
    assert m.status_code == 201 and len(m.json()["added"]) == 2
    again = client.post("/admin/geo/mandals", json={"district_id": did, "names": ["mandal one"]}, headers=admin).json()
    assert again["added"] == [] and again["skipped"] == ["mandal one"]

    base = {"full_name": "Geo User", "email": "geo.user@example.com", "password": "Strong#123", "role": "navigator"}
    assert client.post("/auth/register", json=base).status_code == 422  # district + mandal are required
    assert client.post("/auth/register", json={**base, "district": "Test District A", "mandal": "G. Madugula"}).status_code == 422  # wrong district
    assert client.post("/auth/register", json={**base, "district": "Test District A", "mandal": "Mandal Two"}).status_code == 201

    # a mandal in use cannot be removed by a district admin; the unused one can
    uid_ = next(u["id"] for u in client.get("/admin/users?status=pending", headers=admin).json() if u["email"] == "geo.user@example.com")
    mandals = {x["name"]: x["id"] for d in client.get("/geo").json()["districts"] if d["id"] == did for x in d["mandals"]}
    assert client.delete(f"/admin/geo/mandals/{mandals['Mandal Two']}", headers=admin).status_code == 409
    assert client.delete(f"/admin/geo/mandals/{mandals['Mandal One']}", headers=admin).status_code == 200
    assert client.post(f"/admin/users/{uid_}/reject", headers=admin).status_code == 200


def test_patient_registration_carries_district_and_mandal():
    hw = auth("hw@demo.local")
    r = client.post("/patients", json={"full_name": "Geo Patient", "village": "G. Madugula", "district": "Alluri Sitharama Raju", "mandal": "G. Madugula"}, headers=hw)
    assert r.status_code == 200 and r.json()["district"] == "Alluri Sitharama Raju" and r.json()["mandal"] == "G. Madugula"
    # falls back to the health worker's own district / mandal when none is sent
    r2 = client.post("/patients", json={"full_name": "Default Geo Patient"}, headers=hw)
    assert r2.status_code == 200 and r2.json()["district"] == "Alluri Sitharama Raju"
    bad = client.post("/patients", json={"full_name": "Bad Geo", "district": "Alluri Sitharama Raju", "mandal": "Nowhere"}, headers=hw)
    assert bad.status_code == 422


def test_district_and_mandal_dashboards_are_scoped():
    admin = auth("superadmin@demo.local")
    hw = auth("hw@demo.local")
    client.post("/admin/geo/districts", json={"names": ["Scope District"]}, headers=admin)
    did = next(d["id"] for d in client.get("/geo").json()["districts"] if d["name"] == "Scope District")
    client.post("/admin/geo/mandals", json={"district_id": did, "names": ["Scope Mandal"]}, headers=admin)
    # a patient registered in another district by someone who belongs there (super admin creates the staff member)
    sup = auth("superadmin@demo.local")
    created = client.post("/super/users", json={"full_name": "Scope HW", "email": "scope.hw@example.com", "password": "Strong#123", "role": "health_worker", "district": "Scope District", "mandal": "Scope Mandal"}, headers=sup)
    assert created.status_code == 201, created.text
    other = {"Authorization": "Bearer " + client.post("/auth/login", json={"email": "scope.hw@example.com", "password": "Strong#123"}).json()["token"]}
    assert client.post("/patients", json={"full_name": "Scoped Patient"}, headers=other).status_code == 200

    everything = client.get("/admin/monitoring", headers=admin).json()
    scoped = client.get("/admin/monitoring?district=Scope%20District", headers=admin).json()
    mandal = client.get("/admin/monitoring?district=Scope%20District&mandal=Scope%20Mandal", headers=admin).json()
    nowhere = client.get("/admin/monitoring?district=Alluri%20Sitharama%20Raju&mandal=Nope", headers=admin).json()
    assert scoped["funnel"][0]["value"] == 1 and mandal["funnel"][0]["value"] == 1 and nowhere["funnel"][0]["value"] == 0
    assert everything["funnel"][0]["value"] > scoped["funnel"][0]["value"]
    assert scoped["scope"]["level"] == "district" and mandal["scope"]["level"] == "mandal"
    assert {r["area"] for r in everything["by_district"]} >= {"Alluri Sitharama Raju", "Scope District"}
    assert client.get("/dashboard/summary?district=Scope%20District", headers=admin).json()["screenings"] == 0
    assert client.get("/admin/monitoring", headers=hw).status_code == 403


def test_district_admin_can_suspend_and_reactivate_without_losing_data():
    admin = auth("admin@demo.local")
    uid_ = _register_staff("suspend.me@example.com", "health_worker")
    login = lambda: client.post("/auth/login", json={"email": "suspend.me@example.com", "password": "Strong#123"})
    token = login().json()["token"]
    mine = {"Authorization": f"Bearer {token}"}
    p = client.post("/patients", json={"full_name": "Kept Patient"}, headers=mine)
    assert p.status_code == 200

    s = client.post(f"/admin/users/{uid_}/suspend", json={"reason": "left the programme"}, headers=admin)
    assert s.status_code == 200 and s.json()["status"] == "suspended" and s.json()["suspension_reason"] == "left the programme"
    assert login().status_code == 403                                   # cannot sign in
    assert client.get("/auth/me", headers=mine).status_code == 401       # existing session is cut off at once
    assert client.post(f"/admin/users/{uid_}/suspend", headers=admin).status_code == 409  # already suspended

    detail = client.get(f"/admin/users/{uid_}", headers=admin).json()    # the account and its history are still there
    assert detail["user"]["status"] == "suspended" and detail["work"]["metrics"]["patients_registered"] == 1
    assert uid_ in {u["id"] for u in client.get("/admin/users?status=suspended", headers=admin).json()}

    assert client.post(f"/admin/users/{uid_}/reactivate", headers=admin).status_code == 200
    assert login().status_code == 200
    assert client.get("/auth/me", headers=mine).status_code == 200

    # guard rails: not yourself, not other admins, not outside the admin roles
    me = client.get("/auth/me", headers=admin).json()["id"]
    assert client.post(f"/admin/users/{me}/suspend", headers=admin).status_code == 404  # admins are not manageable by a district admin
    assert client.post(f"/admin/users/{uid_}/suspend", headers=auth("hw@demo.local")).status_code == 403


def test_super_admin_privileges():
    sup = auth("superadmin@demo.local")
    admin = auth("admin@demo.local")

    # sees everything a district admin sees ...
    assert client.get("/dashboard/summary", headers=sup).status_code == 200
    assert client.get("/admin/monitoring", headers=sup).status_code == 200
    assert client.get("/audit", headers=sup).status_code == 200
    assert client.get("/patients", headers=sup).status_code == 200      # ... plus identifiable records
    # ... but a district admin cannot use the super admin console
    assert client.get("/super/data", headers=admin).status_code == 403
    assert client.post("/super/users", json={"full_name": "X Y", "email": "x@y.com", "password": "Strong#123", "role": "admin"}, headers=admin).status_code == 403

    # adds a district admin manually; the new admin can sign in immediately
    r = client.post("/super/users", json={"full_name": "New District Admin", "email": "new.admin@example.com", "password": "Strong#123", "role": "admin"}, headers=sup)
    assert r.status_code == 201 and r.json()["status"] == "active" and r.json()["role"] == "admin"
    assert client.post("/auth/login", json={"email": "new.admin@example.com", "password": "Strong#123"}).status_code == 200
    assert client.post("/super/users", json={"full_name": "Dup", "email": "NEW.admin@example.com", "password": "Strong#123", "role": "admin"}, headers=sup).status_code == 409
    new_id = r.json()["id"]
    listed = {u["id"] for u in client.get("/admin/users", headers=sup).json()}
    assert new_id in listed and new_id not in {u["id"] for u in client.get("/admin/users", headers=admin).json()}

    # alters any account, resets a password, suspends an admin
    assert client.patch(f"/super/users/{new_id}", json={"full_name": "Renamed Admin", "password": "Another#123"}, headers=sup).json()["full_name"] == "Renamed Admin"
    assert client.post("/auth/login", json={"email": "new.admin@example.com", "password": "Another#123"}).status_code == 200
    assert client.post(f"/admin/users/{new_id}/suspend", headers=sup).status_code == 200
    assert client.post("/auth/login", json={"email": "new.admin@example.com", "password": "Another#123"}).status_code == 403

    # cannot lock the application out of super admins
    me = client.get("/auth/me", headers=sup).json()["id"]
    assert client.delete(f"/super/users/{me}", headers=sup).status_code == 400
    assert client.patch(f"/super/users/{me}", json={"role": "admin"}, headers=sup).status_code == 400

    # deletes an account for good
    assert client.delete(f"/super/users/{new_id}", headers=sup).status_code == 200
    assert new_id not in {u["id"] for u in client.get("/admin/users", headers=sup).json()}


def test_super_admin_data_manager_can_add_alter_and_delete_any_row():
    sup = auth("superadmin@demo.local")
    tables = {t["table"]: t for t in client.get("/super/data", headers=sup).json()}
    assert {"users", "patients", "screenings", "referrals", "districts", "mandals"} <= set(tables)
    assert all(c["name"] != "password_hash" for c in tables["users"]["columns"])  # hashes are never exposed

    created = client.post("/super/data/patients", json={"data": {"full_name": "Manual Patient", "district": "Alluri Sitharama Raju", "mandal": "G. Madugula", "age": "52"}}, headers=sup)
    assert created.status_code == 201 and created.json()["age"] == 52
    pid = created.json()["id"]
    upd = client.patch(f"/super/data/patients/{pid}", json={"data": {"full_name": "Edited Patient", "phone": "9111111111"}}, headers=sup)
    assert upd.status_code == 200 and upd.json()["full_name"] == "Edited Patient"
    found = client.get("/super/data/patients?q=Edited", headers=sup).json()
    assert found["total"] >= 1 and any(r["id"] == pid for r in found["rows"])

    # a row with dependants is refused until cascade is requested
    hw = auth("hw@demo.local")
    victim = client.post("/patients", json={"full_name": "Cascade Victim"}, headers=hw).json()["id"]
    assert client.post("/patients/%s/consents" % victim, json={"purpose": "examination", "language": "en", "granted": True}, headers=hw).status_code == 200
    assert client.delete(f"/super/data/patients/{victim}", headers=sup).status_code == 409
    assert client.delete(f"/super/data/patients/{victim}?cascade=true", headers=sup).status_code == 200
    assert client.get(f"/super/data/patients?q=Cascade%20Victim", headers=sup).json()["total"] == 0

    assert client.delete(f"/super/data/patients/{pid}", headers=sup).status_code == 200
    assert client.delete(f"/super/data/patients/{pid}", headers=sup).status_code == 404
    assert client.get("/super/data/not_a_table", headers=sup).status_code == 404
    assert client.get("/super/data/audit_events?q=superadmin.data", headers=sup).json()["total"] >= 3  # every change is audited


# ----------------------------------------------------------------------------------------------------------------
# Stage 12: hospitals, patient's hospital choice, cross-mandal approval, self-clearing notifications
# ----------------------------------------------------------------------------------------------------------------
def test_hospitals_booking_cross_mandal_and_notification_clearing():
    from app.seed import seed_demo_pipeline

    db = SessionLocal()
    try:
        seed_demo_pipeline(db)
    finally:
        db.close()
    ho, nav, spec = auth("mo@demo.local"), auth("nav@demo.local"), auth("spec@demo.local")
    asr = "Alluri Sitharama Raju"

    # only a district admin (or super admin) adds hospitals; the medical officer cannot
    adm = auth("admin@demo.local")
    assert client.post("/hospitals", json={"name": "MO Not Allowed", "district": asr, "mandal": "G. Madugula"}, headers=ho).status_code == 403
    # (original comment) only an admin adds hospitals; duplicates are refused
    body = {"name": "Test Mandal Hospital", "district": asr, "mandal": "G. Madugula", "contact_phone": "08935-111111"}
    assert client.post("/hospitals", json=body, headers=nav).status_code == 403
    made = client.post("/hospitals", json=body, headers=adm)
    assert made.status_code == 201, made.text
    assert client.post("/hospitals", json=body, headers=adm).status_code == 409
    hid = made.json()["id"]

    # booking form: hospitals of a mandal
    local = client.get("/hospitals", params={"district": asr, "mandal": "G. Madugula"}, headers=nav).json()
    assert any(h["id"] == hid for h in local)
    paderu = client.get("/hospitals", params={"district": asr, "mandal": "Paderu"}, headers=nav).json()
    assert paderu, "demo seed has a hospital in Paderu"

    ref = next(r for r in client.get("/referrals", headers=spec).json() if "Appa Rao" in (r["patient_name"] or ""))
    slot = "2030-02-01T10:00:00+05:30"

    # a hospital visit needs a hospital
    assert client.post("/appointments", headers=nav, json={"referral_id": ref["id"], "kind": "physical", "slot": slot}).status_code == 422

    # same mandal: booked straight away
    ok = client.post("/appointments", headers=nav, json={"referral_id": ref["id"], "kind": "physical", "slot": slot, "status": "CONFIRMED", "hospital_id": hid})
    assert ok.status_code == 200, ok.text
    assert ok.json()["approval_status"] == "NOT_REQUIRED" and ok.json()["hospital_name"] == "Test Mandal Hospital"

    # a plain alert is cleared as soon as it is acted on
    notes = client.get("/notifications", headers=spec).json()
    plain = next(n for n in notes if "Hospital appointment scheduled" in n["title"] and f"[appt:{ok.json()['id']}]" in n["body"])
    assert plain["kind"] == "info"
    assert client.patch(f"/notifications/{plain['id']}", headers=spec).json()["cleared"] is True
    assert all(n["id"] != plain["id"] for n in client.get("/notifications", headers=spec).json())

    # other mandal: needs a reason and a medical officer's acceptance
    cross = {"referral_id": ref["id"], "kind": "physical", "slot": slot, "status": "CONFIRMED", "hospital_id": paderu[0]["id"]}
    assert client.post("/appointments", headers=nav, json=cross).status_code == 422
    req = client.post("/appointments", headers=nav, json={**cross, "cross_reason": "Only oral surgery unit within reach"})
    assert req.status_code == 200, req.text
    rid = req.json()["id"]
    assert req.json()["status"] == "PENDING_APPROVAL" and req.json()["approval_status"] == "PENDING"
    assert client.patch(f"/appointments/{rid}", headers=nav, json={"status": "CONFIRMED"}).status_code == 409  # cannot skip the officer

    pending = client.get("/cross-mandal-requests", headers=ho).json()
    assert any(r["id"] == rid and "patient_name" not in r for r in pending)
    assert client.get("/cross-mandal-requests", headers=nav).status_code == 403
    officer_note = next(n for n in client.get("/notifications", headers=ho).json() if f"[xmandal:{rid}]" in n["body"])
    assert officer_note["action_pending"] is True
    client.patch(f"/notifications/{officer_note['id']}", headers=ho)  # opening it does not clear it: no decision yet
    assert any(n["id"] == officer_note["id"] for n in client.get("/notifications", headers=ho).json())

    assert client.post(f"/appointments/{rid}/cross-mandal/decision", headers=nav, json={"decision": "approve"}).status_code == 403
    done = client.post(f"/appointments/{rid}/cross-mandal/decision", headers=ho, json={"decision": "approve", "note": "Accepted"})
    assert done.status_code == 200 and done.json()["approval_status"] == "APPROVED" and done.json()["status"] == "CONFIRMED"
    assert all(f"[xmandal:{rid}]" not in n["body"] for n in client.get("/notifications", headers=ho).json())  # answered -> cleared
    assert client.post(f"/appointments/{rid}/cross-mandal/decision", headers=ho, json={"decision": "approve"}).status_code == 409

    # declining needs a reason and cancels the booking
    req2 = client.post("/appointments", headers=nav, json={**cross, "cross_reason": "Patient prefers this town"}).json()
    assert client.post(f"/appointments/{req2['id']}/cross-mandal/decision", headers=ho, json={"decision": "reject"}).status_code == 422
    no = client.post(f"/appointments/{req2['id']}/cross-mandal/decision", headers=ho, json={"decision": "reject", "note": "Use the local hospital"})
    assert no.json()["approval_status"] == "REJECTED" and no.json()["status"] == "CANCELLED"

    # a medical officer cannot approve a request they booked themselves
    own = client.post("/appointments", headers=ho, json={**cross, "cross_reason": "Own request for test"}).json()
    assert client.post(f"/appointments/{own['id']}/cross-mandal/decision", headers=ho, json={"decision": "approve"}).status_code == 403
    assert client.post(f"/appointments/{own['id']}/cross-mandal/decision", headers=auth("superadmin@demo.local"), json={"decision": "approve"}).status_code == 200

    # hospitals with appointments cannot be deleted, only hidden
    assert client.delete(f"/hospitals/{hid}", headers=adm).status_code == 409
    assert client.patch(f"/hospitals/{hid}", headers=adm, json={"available": False}).status_code == 200
    assert all(h["id"] != hid for h in client.get("/hospitals", params={"district": asr, "mandal": "G. Madugula"}, headers=nav).json())
    assert client.post("/appointments", headers=nav, json={"referral_id": ref["id"], "kind": "physical", "slot": slot, "hospital_id": hid}).status_code == 422


def test_meet_alerts_stay_until_their_day_is_over():
    from app.seed import seed_demo_pipeline

    db = SessionLocal()
    try:
        seed_demo_pipeline(db)
    finally:
        db.close()
    spec, nav = auth("spec@demo.local"), auth("nav@demo.local")
    ref = next(r for r in client.get("/referrals", headers=spec).json() if "Appa Rao" in (r["patient_name"] or ""))
    meet = {"referral_id": ref["id"], "kind": "video", "status": "CONFIRMED", "meet_url": "https://meet.google.com/abc-defg-hij"}

    future = client.post("/appointments", headers=spec, json={**meet, "slot": "2030-03-01T10:00:00+05:30"}).json()
    note = next(n for n in client.get("/notifications", headers=nav).json() if f"[appt:{future['id']}]" in n["body"])
    assert note["kind"] == "meet" and note["expires_at"].startswith("2030-03-01")
    assert client.patch(f"/notifications/{note['id']}", headers=nav).json()["cleared"] is False  # opening it keeps the link
    assert any(n["id"] == note["id"] for n in client.get("/notifications", headers=nav).json())

    past = client.post("/appointments", headers=spec, json={**meet, "slot": "2020-03-01T10:00:00+05:30"}).json()
    assert all(f"[appt:{past['id']}]" not in n["body"] for n in client.get("/notifications", headers=nav).json())  # day is over -> gone



# ----------------------------------------------------------------------------------------------------------------
# Corrected roles: district-locked admin, video privilege, MO assignment + cross-district requests
# ----------------------------------------------------------------------------------------------------------------
def test_district_admin_is_locked_to_own_district():
    admin, sup = auth("admin@demo.local"), auth("superadmin@demo.local")
    mine = client.get("/admin/monitoring", headers=admin).json()
    other = client.get("/admin/monitoring", params={"district": "Somewhere Else"}, headers=admin).json()
    assert mine["scope"]["district"] == "Alluri Sitharama Raju" and other["scope"]["district"] == "Alluri Sitharama Raju"  # request cannot widen it
    assert client.get("/dashboard/summary", params={"district": "Somewhere Else"}, headers=admin).json()["scope"]["district"] == "Alluri Sitharama Raju"
    assert client.get("/admin/monitoring", headers=sup).json()["scope"]["level"] == "all"
    assert client.post("/admin/reset-demo", headers=admin).status_code == 403  # wiping all data is super admin only


def test_video_needs_specialist_and_patient_privilege_and_mo_assigns_cross_district():
    from app.seed import seed_demo_pipeline

    db = SessionLocal()
    try:
        seed_demo_pipeline(db)
    finally:
        db.close()
    mo, nav, spec, hw = auth("mo@demo.local"), auth("nav@demo.local"), auth("spec@demo.local"), auth("hw@demo.local")
    ref = next(r for r in client.get("/referrals", headers=spec).json() if "Appa Rao" in (r["patient_name"] or ""))
    meet = {"referral_id": ref["id"], "kind": "video", "slot": "2030-05-01T10:00:00+05:30", "status": "CONFIRMED", "meet_url": "https://meet.google.com/abc-defg-hij"}
    assert client.post("/appointments", headers=mo, json=meet).status_code == 403   # video moved from the MO to the specialist
    assert client.post("/appointments", headers=nav, json=meet).status_code == 403
    assert client.post("/appointments", headers=spec, json=meet).status_code == 200

    # a patient registered without the video privilege cannot get a video consultation
    p = client.post("/patients", headers=hw, json={"full_name": "No Video Patient", "village": "G. Madugula"}).json()
    assert p["video_allowed"] is False
    opts = client.get(f"/patients/{p['id']}/assignment-options", headers=mo)
    assert opts.status_code == 200 and opts.json()["hospitals"], "same-mandal hospitals with specialists are offered"
    assert all(sp["id"] for h in opts.json()["hospitals"] for sp in h["specialists"])
    assert client.get(f"/patients/{p['id']}/assignment-options", headers=spec).status_code == 403
