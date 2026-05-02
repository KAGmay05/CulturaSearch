import json
from pathlib import Path
from typing import Optional, Tuple

USERS_FILE = Path("data/users.json")


def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    try:
        with USERS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def authenticate(user_id: str, password: str) -> Tuple[bool, Optional[str]]:
    """
    Verifica credenciales.
    Retorna (success: bool, user_name: str|None)
    """
    users_data = _load_users()
    users_list = users_data.get("users", [])
    
    for user in users_list:
        if user.get("user_id") == user_id and user.get("password") == password:
            return True, user.get("name", user_id)
    
    return False, None


def user_exists(user_id: str) -> bool:
    """Verifica si un usuario existe."""
    users_data = _load_users()
    users_list = users_data.get("users", [])
    return any(u.get("user_id") == user_id for u in users_list)


def list_all_users() -> list:
    """Devuelve lista de user_ids disponibles."""
    users_data = _load_users()
    users_list = users_data.get("users", [])
    return [u.get("user_id") for u in users_list]
