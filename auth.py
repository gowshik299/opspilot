# auth.py
import os
from datetime import datetime, timedelta
from typing import Optional
import json

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import text

from memory import engine
from cache import r, REDIS_AVAILABLE

# Config
SECRET_KEY = os.getenv("JWT_SECRET", "opspilot-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


# ── Password utils ────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT utils ─────────────────────────────────────

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── Database utils ────────────────────────────────

def get_user(username: str) -> Optional[dict]:
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT * FROM users WHERE username = :u AND is_active = TRUE"),
            {"u": username}
        ).fetchone()
        if result:
            return dict(result._mapping)
    return None

def create_user(username: str, email: str, password: str, role: str = "viewer") -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO users (username, email, hashed_password, role)
                VALUES (:u, :e, :p, :r)
            """), {
                "u": username,
                "e": email,
                "p": hash_password(password),
                "r": role
            })
            conn.commit()
        return True
    except Exception as e:
        print(f"Create user error: {e}")
        return False


# ── Token blacklist (Redis) ───────────────────────

def blacklist_token(token: str):
    if REDIS_AVAILABLE:
        r.setex(f"blacklist:{token}", 
                int(timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS).total_seconds()), 
                "1")

def is_blacklisted(token: str) -> bool:
    if REDIS_AVAILABLE:
        return r.exists(f"blacklist:{token}") > 0
    return False


# ── Auth dependency ───────────────────────────────

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    token = credentials.credentials
    
    # Check blacklist
    if is_blacklisted(token):
        raise HTTPException(status_code=401, detail="Token has been revoked")
    
    # Verify token
    payload = decode_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Get user from DB
    user = get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

def require_admin(user: dict = Security(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user