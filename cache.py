# cache.py
import redis
import json
import os
import hashlib

# Connect to Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

try:
    r = redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
    print("✅ Redis connected!")
    REDIS_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Redis not available: {e}")
    REDIS_AVAILABLE = False

def make_key(query: str) -> str:
    """Create cache key from query"""
    return f"opspilot:{hashlib.md5(query.encode()).hexdigest()}"

def get_cached(query: str):
    """Get cached response"""
    if not REDIS_AVAILABLE:
        return None
    try:
        key = make_key(query)
        value = r.get(key)
        if value:
            print(f"✅ Cache hit: {query[:50]}")
            return json.loads(value)
        return None
    except Exception:
        return None

def set_cached(query: str, response: str, ttl: int = 300):
    """Cache response for TTL seconds (default 5 mins)"""
    if not REDIS_AVAILABLE:
        return
    try:
        key = make_key(query)
        r.setex(key, ttl, json.dumps(response))
        print(f"✅ Cached: {query[:50]}")
    except Exception:
        pass

def invalidate_cache():
    """Clear all cached responses"""
    if not REDIS_AVAILABLE:
        return
    try:
        keys = r.keys("opspilot:*")
        if keys:
            r.delete(*keys)
        print("✅ Cache cleared")
    except Exception:
        pass
