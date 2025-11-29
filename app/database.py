"""
Database Module - SQLite with usage tracking
Simple, reliable, and free!
"""

import os
import hashlib
import secrets
from typing import Optional, List, Tuple
import logging
import aiosqlite
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "tts_api.db"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def hash_token(token: str) -> str:
    """Hash a token using SHA-256."""
    return hashlib.sha256(token.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return f"tts_{secrets.token_urlsafe(32)}"


class DatabaseManager:
    """SQLite database manager with usage tracking."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def initialize(self):
        """Initialize database schema."""
        if self._initialized:
            return
        
        logger.info(f"Initializing database at {DB_PATH}...")
        
        async with aiosqlite.connect(DB_PATH) as db:
            # Master users table
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
            
            # API keys table
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
            
            # Usage tracking table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    master_id INTEGER NOT NULL,
                    api_key_id INTEGER,
                    text_length INTEGER NOT NULL,
                    audio_duration_seconds REAL,
                    audio_size_bytes INTEGER,
                    voice_id TEXT DEFAULT 'default',
                    language TEXT DEFAULT 'en',
                    output_format TEXT DEFAULT 'mp3',
                    cached INTEGER DEFAULT 0,
                    processing_time_seconds REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (master_id) REFERENCES master_users(id),
                    FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
                )
            """)
            
            await db.commit()
        
        # Seed users from environment
        await self._seed_master_users()
        
        self._initialized = True
        logger.info("Database initialized successfully")
    
    async def _seed_master_users(self):
        """Seed master users from environment variables."""
        async with aiosqlite.connect(DB_PATH) as db:
            # Owner
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
                    logger.info(f"Owner '{owner_name}' seeded")
                except Exception as e:
                    logger.warning(f"Owner seed error: {e}")
            
            # Friends (1-5)
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
                        logger.info(f"Friend '{friend_name}' seeded")
                    except Exception as e:
                        logger.warning(f"Friend {i} seed error: {e}")
            
            await db.commit()
    
    async def validate_master_token(self, token: str) -> Optional[dict]:
        """Validate a master token and return user info."""
        token_hash = hash_token(token)
        
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, name, is_owner, rate_limit, max_keys
                FROM master_users WHERE token_hash = ?
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
        """Validate an API key and return associated user info."""
        key_hash = hash_token(api_key)
        
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT k.id as key_id, k.name as key_name, k.master_id,
                       m.name as master_name, m.is_owner, m.rate_limit, m.max_keys
                FROM api_keys k
                JOIN master_users m ON k.master_id = m.id
                WHERE k.key_hash = ?
            """, (key_hash,))
            row = await cursor.fetchone()
            
            if row:
                # Update last_used
                await db.execute("""
                    UPDATE api_keys SET last_used = CURRENT_TIMESTAMP WHERE id = ?
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
    
    async def create_api_key(self, master_token: str, key_name: str = None) -> Tuple[Optional[str], Optional[str]]:
        """Create a new API key for a master user."""
        user = await self.validate_master_token(master_token)
        if not user:
            return None, "Invalid master token"
        
        # Check key limit (0 = unlimited)
        if user["max_keys"] > 0:
            async with aiosqlite.connect(DB_PATH) as db:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM api_keys WHERE master_id = ?",
                    (user["id"],)
                )
                count = (await cursor.fetchone())[0]
                
                if count >= user["max_keys"]:
                    return None, f"API key limit reached ({user['max_keys']} max)"
        
        # Generate key
        api_key = generate_api_key()
        key_hash = hash_token(api_key)
        key_prefix = api_key[:12]
        
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                INSERT INTO api_keys (key_hash, key_prefix, master_id, name)
                VALUES (?, ?, ?, ?)
            """, (key_hash, key_prefix, user["id"], key_name))
            await db.commit()
        
        logger.info(f"API key created for '{user['name']}': {key_prefix}...")
        return api_key, None
    
    async def list_api_keys(self, master_token: str) -> Tuple[Optional[List[dict]], Optional[str]]:
        """List all API keys for a master user."""
        user = await self.validate_master_token(master_token)
        if not user:
            return None, "Invalid master token"
        
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, key_prefix, name, created_at, last_used
                FROM api_keys WHERE master_id = ? ORDER BY created_at DESC
            """, (user["id"],))
            rows = await cursor.fetchall()
            
            return [{
                "id": r["id"],
                "key_prefix": r["key_prefix"],
                "name": r["name"],
                "created_at": r["created_at"],
                "last_used": r["last_used"]
            } for r in rows], None
    
    async def revoke_api_key(self, master_token: str, key_id: int) -> Tuple[bool, Optional[str]]:
        """Revoke an API key."""
        user = await self.validate_master_token(master_token)
        if not user:
            return False, "Invalid master token"
        
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute(
                "SELECT id FROM api_keys WHERE id = ? AND master_id = ?",
                (key_id, user["id"])
            )
            if not await cursor.fetchone():
                return False, "API key not found"
            
            await db.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
            await db.commit()
        
        logger.info(f"API key {key_id} revoked by '{user['name']}'")
        return True, None
    
    async def log_usage(
        self,
        master_id: int,
        api_key_id: Optional[int],
        text_length: int,
        audio_duration_seconds: float,
        audio_size_bytes: int,
        voice_id: str,
        language: str,
        output_format: str,
        cached: bool,
        processing_time_seconds: float
    ):
        """Log a TTS request for analytics."""
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                INSERT INTO usage_logs 
                (master_id, api_key_id, text_length, audio_duration_seconds, audio_size_bytes,
                 voice_id, language, output_format, cached, processing_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (master_id, api_key_id, text_length, audio_duration_seconds, audio_size_bytes,
                  voice_id, language, output_format, int(cached), processing_time_seconds))
            await db.commit()
    
    async def get_user_stats(self, master_id: int) -> dict:
        """Get usage statistics for a user."""
        async with aiosqlite.connect(DB_PATH) as db:
            # Total requests
            cursor = await db.execute(
                "SELECT COUNT(*) FROM usage_logs WHERE master_id = ?", (master_id,))
            total_requests = (await cursor.fetchone())[0]
            
            # Total audio duration
            cursor = await db.execute(
                "SELECT COALESCE(SUM(audio_duration_seconds), 0) FROM usage_logs WHERE master_id = ?",
                (master_id,))
            total_audio_seconds = (await cursor.fetchone())[0]
            
            # Total characters
            cursor = await db.execute(
                "SELECT COALESCE(SUM(text_length), 0) FROM usage_logs WHERE master_id = ?",
                (master_id,))
            total_characters = (await cursor.fetchone())[0]
            
            # Language usage
            cursor = await db.execute("""
                SELECT language, COUNT(*) as count FROM usage_logs 
                WHERE master_id = ? GROUP BY language ORDER BY count DESC
            """, (master_id,))
            language_usage = {r[0]: r[1] for r in await cursor.fetchall()}
            
            # Voice usage
            cursor = await db.execute("""
                SELECT voice_id, COUNT(*) as count FROM usage_logs 
                WHERE master_id = ? GROUP BY voice_id ORDER BY count DESC
            """, (master_id,))
            voice_usage = {r[0]: r[1] for r in await cursor.fetchall()}
        
        return {
            "total_requests": total_requests,
            "total_audio_seconds": total_audio_seconds or 0,
            "total_characters": total_characters or 0,
            "language_usage": language_usage or {"en": 0},
            "voice_usage": voice_usage or {"default": 0}
        }
    
    async def get_recent_usage(self, master_id: int, limit: int = 20) -> List[dict]:
        """Get recent usage logs for a user."""
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute("""
                SELECT text_length, audio_duration_seconds, voice_id, language,
                       output_format, cached, timestamp
                FROM usage_logs WHERE master_id = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (master_id, limit))
            rows = await cursor.fetchall()
            
            return [{
                "characters": r[0],
                "duration": r[1],
                "voice": r[2],
                "language": r[3],
                "format": r[4],
                "cached": bool(r[5]),
                "timestamp": r[6]
            } for r in rows]
    
    async def get_master_users(self) -> List[dict]:
        """Get list of all master users."""
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT id, name, is_owner, rate_limit, max_keys, created_at
                FROM master_users
            """)
            rows = await cursor.fetchall()
            
            return [{
                "id": r["id"],
                "name": r["name"],
                "is_owner": bool(r["is_owner"]),
                "rate_limit": r["rate_limit"],
                "max_keys": r["max_keys"],
                "created_at": r["created_at"]
            } for r in rows]


# Global instance
_db_manager: Optional[DatabaseManager] = None


async def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    return _db_manager
