"""
Middleware Module
Handles authentication, rate limiting, and CORS.
"""

import time
import asyncio
from collections import defaultdict
from typing import Optional, Callable, Dict
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

from .database import get_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sliding window rate limiter.
    
    Tracks request timestamps per user and enforces rate limits.
    Owner (is_owner=True) bypasses all rate limits.
    """
    
    def __init__(self):
        # Dict of user_id -> list of request timestamps
        self._requests: Dict[int, list] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, user_id: int, rate_limit: float, is_owner: bool) -> tuple:
        """
        Check if a request is allowed under rate limiting.
        
        Args:
            user_id: The user making the request
            rate_limit: Requests per second allowed (0 = unlimited)
            is_owner: If True, bypass rate limiting
        
        Returns:
            Tuple of (allowed: bool, retry_after: float or None)
        """
        # Owner bypasses all rate limits
        if is_owner or rate_limit <= 0:
            return True, None
        
        current_time = time.time()
        window_size = 1.0  # 1 second window
        
        async with self._lock:
            # Get requests in the current window
            user_requests = self._requests[user_id]
            
            # Remove old requests outside the window
            cutoff = current_time - window_size
            user_requests = [t for t in user_requests if t > cutoff]
            self._requests[user_id] = user_requests
            
            # Check if under limit
            if len(user_requests) < rate_limit:
                # Allow and record this request
                user_requests.append(current_time)
                return True, None
            else:
                # Rate limited - calculate retry time
                oldest_in_window = min(user_requests) if user_requests else current_time
                retry_after = oldest_in_window + window_size - current_time
                return False, max(0.1, retry_after)
    
    async def cleanup(self):
        """Remove old entries to prevent memory growth."""
        current_time = time.time()
        cutoff = current_time - 60  # Keep last 60 seconds
        
        async with self._lock:
            for user_id in list(self._requests.keys()):
                self._requests[user_id] = [
                    t for t in self._requests[user_id] if t > cutoff
                ]
                # Remove empty entries
                if not self._requests[user_id]:
                    del self._requests[user_id]


# Global rate limiter instance
rate_limiter = RateLimiter()


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication and rate limiting middleware.
    
    Validates API keys from the 'key' header and enforces rate limits.
    """
    
    # Paths that don't require authentication
    PUBLIC_PATHS = {
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process the request through auth and rate limiting."""
        
        # Skip auth for public paths
        path = request.url.path
        if path in self.PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)
        
        # Get API key from header
        api_key = request.headers.get("key") or request.headers.get("authorization")
        
        # Also check query parameter for convenience
        if not api_key:
            api_key = request.query_params.get("key")
        
        # Strip "Bearer " prefix if present
        if api_key and api_key.startswith("Bearer "):
            api_key = api_key[7:]
        
        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing API key. Provide 'key' header."}
            )
        
        # Validate the key (could be master token or API key)
        db = await get_database()
        
        # Try as API key first
        user = await db.validate_api_key(api_key)
        
        # If not found, try as master token
        if not user:
            user = await db.validate_master_token(api_key)
        
        if not user:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid API key"}
            )
        
        # Check rate limit
        allowed, retry_after = await rate_limiter.is_allowed(
            user_id=user["id"],
            rate_limit=user["rate_limit"],
            is_owner=user["is_owner"]
        )
        
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded. Max {user['rate_limit']} requests/second.",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(int(retry_after) + 1)}
            )
        
        # Attach user to request state for use in endpoints
        request.state.user = user
        
        # Process the request
        response = await call_next(request)
        
        return response


def get_current_user(request: Request) -> dict:
    """
    Get the current authenticated user from request state.
    
    Use as a dependency in FastAPI endpoints:
        user: dict = Depends(get_current_user)
    """
    if not hasattr(request.state, "user"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return request.state.user


async def require_owner(request: Request) -> dict:
    """
    Dependency that requires the user to be the owner.
    
    Use for owner-only endpoints.
    """
    user = get_current_user(request)
    if not user.get("is_owner"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Owner access required"
        )
    return user


async def start_rate_limiter_cleanup():
    """Background task to periodically clean up rate limiter."""
    while True:
        await asyncio.sleep(60)
        await rate_limiter.cleanup()
