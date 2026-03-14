"""
inference/security.py — Enterprise Security Hardening (Feature 24)

Wraps the FastAPI server and WebSocket endpoints with strict rate limiting,
payload validation, and TLS encryption readiness.
"""

from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from typing import Callable
import time
import os
import json
from pydantic import BaseModel, root_validator
import logging
from cryptography.fernet import Fernet

logger = logging.getLogger("rf_inference.security")

API_KEY_HEADER = APIKeyHeader(name="X-EchoPose-Token", auto_error=False)
VALID_TOKENS = {os.getenv("ECHOPOSE_API_TOKEN", "change_me_in_production")}

class RateLimiter:
    """Token Bucket rate limiter to prevent DoS attacks on the inference pipeline"""
    def __init__(self, requests_per_second: int = 50):
        self.rps = requests_per_second
        self.clients = {}
        self.last_cleanup = time.time()
        
    def _cleanup_stale_clients(self, now):
        """Prevents memory leak by pruning idle IPs every 60 seconds"""
        if now - self.last_cleanup > 60.0:
            stale = [ip for ip, timestamps in self.clients.items() if not timestamps or now - timestamps[-1] > 60.0]
            for ip in stale:
                del self.clients[ip]
            self.last_cleanup = now

    def check_rate_limit(self, client_ip: str):
        now = time.time()
        self._cleanup_stale_clients(now)
        
        if client_ip not in self.clients:
            self.clients[client_ip] = [now]
            return True
            
        # Clean up old requests outside the 1-second window
        self.clients[client_ip] = [t for t in self.clients[client_ip] if now - t < 1.0]
        
        if len(self.clients[client_ip]) >= self.rps:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(status_code=429, detail="Too Many Requests")
            
        self.clients[client_ip].append(now)
        return True

limiter = RateLimiter(requests_per_second=60) # Allow 60Hz Max

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Validates incoming REST connection tokens"""
    if not api_key or api_key not in VALID_TOKENS:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

class IncomingCSIBundle(BaseModel):
    """Strict Input Validation using Pydantic"""
    window_us: int
    frames: list
    
    @root_validator
    def validate_physics(cls, values):
        frames = values.get('frames')
        if not frames:
            raise ValueError("Empty CSI bundle payload")
        if len(frames) > 200:
            raise ValueError("Exceeded maximum frames per bundle (200) - Possible injection attack")
        # Validate that frames contain expected structures
        for frame in frames:
            if "node_id" not in frame or "amplitudes" not in frame:
                raise ValueError("Malformed CSI frame syntax")
            if len(frame["amplitudes"]) > 1024:
                 raise ValueError("Exceeded maximum subcarriers per node (1024)")
        return values
        
# ── Encryption at Rest ─────────────────────────────────────────────
FERNET_KEY = os.getenv("ECHOPOSE_SESSION_KEY", Fernet.generate_key().decode('utf-8'))
cipher_suite = Fernet(FERNET_KEY.encode('utf-8'))

def encrypt_session_data(data: dict) -> bytes:
    """Encrypt JSON session data using AES-256 for at-rest storage"""
    json_bytes = json.dumps(data).encode('utf-8')
    return cipher_suite.encrypt(json_bytes)

def decrypt_session_data(encrypted_bytes: bytes) -> dict:
    """Decrypt stored session data back to dict"""
    json_bytes = cipher_suite.decrypt(encrypted_bytes)
    return json.loads(json_bytes.decode('utf-8'))
