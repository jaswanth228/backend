"""One-off helper. Safe to re-run.

1) Uploads old local ./storage/<id>.bin files (if any) to Supabase.
2) Renames objects already in the bucket that end in .bin to .jpg/.png/... and updates images.storage_key.

Run after setting backend/.env:  python migrate_local_images.py
"""
from pathlib import Path

from app import supabase_storage
from app.config import settings
from app.db import SessionLocal
from app.models import ImageRecord


def main() -> None:
    supabase_storage.ensure_bucket()
    folder = Path(settings.STORAGE_DIR)
    db = SessionLocal()
    moved = renamed = missing = 0
    try:
        for rec in db.query(ImageRecord).all():
            key = rec.storage_key
            if "/" not in key:  # old local file
                f = folder / key
                if not f.exists():
                    missing += 1
                    print("missing local file:", f)
                    continue
                data, old_remote = f.read_bytes(), None
            elif key.endswith(".bin"):  # already in bucket with the generic name
                data, old_remote = supabase_storage.download(key), key
                if data is None:
                    missing += 1
                    print("missing in bucket:", key)
                    continue
            else:
                continue
            ext, mime = supabase_storage.detect_type(data)
            new_key = f"{rec.screening_id}/{rec.id}{ext}"
            supabase_storage.upload(new_key, data, mime)
            rec.storage_key = new_key
            db.commit()
            if old_remote and old_remote != new_key:
                supabase_storage.remove(old_remote)
                renamed += 1
            else:
                moved += 1
    finally:
        db.close()
    print(f"uploaded from local: {moved}, renamed in bucket: {renamed}, missing: {missing}")


if __name__ == "__main__":
    main()
