import json
from pathlib import Path
from typing import Optional

from recommendation_module.user_profile import User

DATA_FILE = Path("data/user_profiles.json")


def _ensure_parent():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)


def _read_all() -> dict:
    _ensure_parent()
    if not DATA_FILE.exists():
        return {}
    try:
        with DATA_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_all(data: dict) -> None:
    _ensure_parent()
    tmp = DATA_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(DATA_FILE)


def load_user(user_id: str) -> Optional[User]:
    all_data = _read_all()
    payload = all_data.get(user_id)
    if not payload:
        return None
    try:
        return User.from_dict(payload)
    except Exception:
        return None


def save_user(user: User) -> None:
    if user is None:
        return
    all_data = _read_all()
    all_data[user.user_id] = user.to_dict()
    _write_all(all_data)


def list_users() -> list:
    return list(_read_all().keys())
