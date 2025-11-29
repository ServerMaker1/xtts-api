"""
Authentication and Authorization Module
Handles master users, API keys, and tiered access control.
"""

import os
import secrets
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import aiosqlite
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path (persistent storage on HF Spaces)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "auth.db"

# Ensure directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)


def hash_token(token: str) -> str:
    """Hash a token using SHA-256."""
    return hashlib.sha256(token.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return f"tts_{secrets.token_urlsafe(32)}"


class AuthManager:
    """
    Manages master users and API keys with tiered access.
    
    Access Tiers:
    - Owner (you): Unlimited rate, unlimited API keys
    - Friends (5): 3 req/s rate limit, max 5 API keys each
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self):
        """Initialize the database and seed master users."""
        if self._initialized:
            return
        
        logger.info("Initializing authentication database...")
        
        async with aiosqlite.connect(DB_PATH) as db:
            # Create master_users table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS master_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    token_hash TEXT NOT NULL UNIQUE,
                    is_owner INTEGER DEFAULT 0,
                    rate_limit REAL DEFAULT 3.0,
                    max_keys INTEGER DEFAULT 5,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create api_keys table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT NOT NULL UNIQUE,
                    key_prefix TEXT NOT NULL,
                    master_id INTEGER NOT NULL,
                    name TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_used TEXT,
                    FOREIGN KEY (master_id) REFERENCES master_users(id)
                )
            """)
            
            await db.commit()
        
        # Seed master users from environment variables
        await self._seed_master_users()
        
        self._initialized = True
        logger.info("Authentication database initialized.")
    
    async def _seed_master_users(self):
        """
        Seed master users from environment variables.
        
        Expected env vars:
        - OWNER_TOKEN: Your master token (unlimited access)
        - OWNER_NAME: Your display name
        - FRIEND_1_TOKEN, FRIEND_1_NAME: Friend 1's token and name
        - FRIEND_2_TOKEN, FRIEND_2_NAME: Friend 2's token and name
        - ... up to FRIEND_5_TOKEN, FRIEND_5_NAME
        """
        async with aiosqlite.connect(DB_PATH) as db:
            # Seed owner
            owner_token = os.environ.get("OWNER_TOKEN")
            owner_name = os.environ.get("OWNER_NAME", "owner")
            
            if owner_token:
                token_hash = hash_token(owner_token)
                try:
                    await db.execute("""
                        INSERT OR IGNORE INTO master_users 
                        (name, token_hash, is_owner, rate_limit, max_keys)
                        VALUES (?, ?, 1, 0, 0)
                    """, (owner_name, token_hash))
                    logger.info(f"Owner '{owner_name}' seeded.")
                except Exception as e:
                    logger.warning(f"Owner already exists or error: {e}")
            else:
                logger.warning("OWNER_TOKEN not set! Owner access disabled.")
            
            # Seed friends (1-5)
            for i in range(1, 6):
                friend_token = os.environ.get(f"FRIEND_{i}_TOKEN")
                friend_name = os.environ.get(f"FRIEND_{i}_NAME", f"friend_{i}")
                
                if friend_token:
                    token_hash = hash_token(friend_token)
                    try:
                        await db.execute("""
                            INSERT OR IGNORE INTO master_users 
                            (name, token_hash, is_owner, rate_limit, max_keys)
                            VALUES (?, ?, 0, 3.0, 5)
                        """, (friend_name, token_hash))
                        logger.info(f"Friend '{friend_name}' seeded.")
                    except Exception as e:
                        logger.warning(f"Friend {i} already exists or error: {e}")
            
            await db.commit()
    
    async def validate_master_token(self, token: str) -> Optional[dict]:
        """
        Validate a master token and return user info.
        
        Returns:
            User dict if valid, None if invalid
        """
        token_hash = hash_token(token)
        
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, name, is_owner, rate_limit, max_keys
                FROM master_users
                WHERE token_hash = ?
            """, (token_hash,))
            row = await cursor.fetchone()
            
            if row:
                return {
                    "id": row["id"],
                    "name": row["name"],
                    "is_owner": bool(row["is_owner"]),
                    "rate_limit": row["rate_limit"],
                    "max_keys": row["max_keys"],
                    "type": "master"
                }
            
            return None
    
    async def validate_api_key(self, api_key: str) -> Optional[dict]:
        """
        Validate an API key and return associated user info.
        
        Returns:
            User dict if valid, None if invalid
        """
        key_hash = hash_token(api_key)
        
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            
            # Get API key and associated master user
            cursor = await db.execute("""
                SELECT k.id as key_id, k.name as key_name, k.master_id,
                       m.name as master_name, m.is_owner, m.rate_limit, m.max_keys
                FROM api_keys k
                JOIN master_users m ON k.master_id = m.id
                WHERE k.key_hash = ?
            """, (key_hash,))
            row = await cursor.fetchone()
            
            if row:
                # Update last_used timestamp
                await db.execute("""
                    UPDATE api_keys SET last_used = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (row["key_id"],))
                await db.commit()
                
                return {
                    "id": row["master_id"],
                    "name": row["master_name"],
                    "key_id": row["key_id"],
                    "key_name": row["key_name"],
                    "is_owner": bool(row["is_owner"]),
                    "rate_limit": row["rate_limit"],
                    "max_keys": row["max_keys"],
                    "type": "api_key"
                }
            
            return None
    
    async def create_api_key(
        self, 
        master_token: str, 
        key_name: str = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Create a new API key for a master user.
        
        Args:
            master_token: Master user's token
            key_name: Optional name for the key
        
        Returns:
            Tuple of (api_key, error_message)
        """
        # Validate master token
        user = await self.validate_master_token(master_token)
        if not user:
            return None, "Invalid master token"
        
        # Check key limit (0 = unlimited for owner)
        if user["max_keys"] > 0:
            async with aiosqlite.connect(DB_PATH) as db:
                cursor = await db.execute("""
                    SELECT COUNT(*) FROM api_keys WHERE master_id = ?
                """, (user["id"],))
                count = (await cursor.fetchone())[0]
                
                if count >= user["max_keys"]:
                    return None, f"API key limit reached ({user['max_keys']} max)"
        
        # Generate new API key
        api_key = generate_api_key()
        key_hash = hash_token(api_key)
        key_prefix = api_key[:12]  # Store prefix for identification
        
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                INSERT INTO api_keys (key_hash, key_prefix, master_id, name)
                VALUES (?, ?, ?, ?)
            """, (key_hash, key_prefix, user["id"], key_name))
            await db.commit()
        
        logger.info(f"API key created for user '{user['name']}': {key_prefix}...")
        return api_key, None
    
    async def list_api_keys(self, master_token: str) -> Tuple[Optional[List[dict]], Optional[str]]:
        """
        List all API keys for a master user.
        
        Args:
            master_token: Master user's token
        
        Returns:
            Tuple of (list of keys, error_message)
        """
        user = await self.validate_master_token(master_token)
        if not user:
            return None, "Invalid master token"
        
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, key_prefix, name, created_at, last_used
                FROM api_keys
                WHERE master_id = ?
                ORDER BY created_at DESC
            """, (user["id"],))
            rows = await cursor.fetchall()
            
            keys = [
                {
                    "id": row["id"],
                    "key_prefix": row["key_prefix"],
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "last_used": row["last_used"]
                }
                for row in rows
            ]
            
            return keys, None
    
    async def revoke_api_key(self, master_token: str, key_id: int) -> Tuple[bool, Optional[str]]:
        """
        Revoke (delete) an API key.
        
        Args:
            master_token: Master user's token
            key_id: ID of the key to revoke
        
        Returns:
            Tuple of (success, error_message)
        """
        user = await self.validate_master_token(master_token)
        if not user:
            return False, "Invalid master token"
        
        async with aiosqlite.connect(DB_PATH) as db:
            # Verify key belongs to this user
            cursor = await db.execute("""
                SELECT id FROM api_keys WHERE id = ? AND master_id = ?
            """, (key_id, user["id"]))
            row = await cursor.fetchone()
            
            if not row:
                return False, "API key not found or not owned by you"
            
            await db.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
            await db.commit()
        
        logger.info(f"API key {key_id} revoked by user '{user['name']}'")
        return True, None
    
    async def get_master_users(self) -> List[dict]:
        """Get list of all master users (for admin purposes)."""
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, name, is_owner, rate_limit, max_keys, created_at
                FROM master_users
            """)
            rows = await cursor.fetchall()
            
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "is_owner": bool(row["is_owner"]),
                    "rate_limit": row["rate_limit"],
                    "max_keys": row["max_keys"],
                    "created_at": row["created_at"]
                }
                for row in rows
            ]


# Global instance
_auth_manager: Optional[AuthManager] = None


async def get_auth_manager() -> AuthManager:
    """Get the global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
        await _auth_manager.initialize()
    return _auth_manager
