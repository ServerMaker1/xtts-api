"""
Audio Cache Module
LRU cache for generated TTS audio to avoid redundant processing.
"""

import os
import hashlib
import json
import asyncio
import aiofiles
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
from collections import OrderedDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache settings
MAX_CACHE_SIZE_MB = int(os.environ.get("CACHE_SIZE_MB", "500"))  # 500MB default
MAX_CACHE_AGE_HOURS = int(os.environ.get("CACHE_AGE_HOURS", "24"))  # 24 hours default
MAX_CACHE_ENTRIES = int(os.environ.get("CACHE_ENTRIES", "1000"))  # Max entries


def generate_cache_key(text: str, voice_id: str, language: str, output_format: str, speed: float) -> str:
    """
    Generate a unique cache key for TTS parameters.
    
    Uses SHA-256 hash of normalized parameters.
    """
    # Normalize text (strip whitespace, lowercase for key generation)
    normalized_text = text.strip()
    
    # Create a deterministic string representation
    key_data = f"{normalized_text}|{voice_id}|{language}|{output_format}|{speed:.2f}"
    
    # Hash it
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


class AudioCache:
    """
    LRU cache for TTS audio files.
    
    Features:
    - Disk-based storage for persistence
    - LRU eviction when size/count limits reached
    - Automatic cleanup of old entries
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._index: OrderedDict = OrderedDict()  # key -> metadata
            self._lock = asyncio.Lock()
            self._total_size = 0
    
    async def initialize(self):
        """Load existing cache index from disk."""
        index_path = CACHE_DIR / "index.json"
        
        if index_path.exists():
            try:
                async with aiofiles.open(index_path, "r") as f:
                    content = await f.read()
                    data = json.loads(content)
                    self._index = OrderedDict(data.get("entries", {}))
                    self._total_size = data.get("total_size", 0)
                    logger.info(f"Loaded cache index: {len(self._index)} entries, {self._total_size / 1024 / 1024:.2f}MB")
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = OrderedDict()
                self._total_size = 0
        
        # Verify cache files exist
        await self._verify_cache()
        
        # Clean old entries
        await self._cleanup_old_entries()
    
    async def _save_index(self):
        """Save cache index to disk."""
        index_path = CACHE_DIR / "index.json"
        
        try:
            data = {
                "entries": dict(self._index),
                "total_size": self._total_size,
                "updated_at": datetime.utcnow().isoformat()
            }
            async with aiofiles.open(index_path, "w") as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    async def _verify_cache(self):
        """Verify all cached files exist, remove orphaned entries."""
        async with self._lock:
            keys_to_remove = []
            
            for key, meta in self._index.items():
                cache_path = CACHE_DIR / meta["filename"]
                if not cache_path.exists():
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                meta = self._index.pop(key)
                self._total_size -= meta.get("size", 0)
            
            if keys_to_remove:
                logger.info(f"Removed {len(keys_to_remove)} orphaned cache entries")
                await self._save_index()
    
    async def _cleanup_old_entries(self):
        """Remove entries older than MAX_CACHE_AGE_HOURS."""
        cutoff = datetime.utcnow() - timedelta(hours=MAX_CACHE_AGE_HOURS)
        
        async with self._lock:
            keys_to_remove = []
            
            for key, meta in self._index.items():
                created_at = datetime.fromisoformat(meta["created_at"])
                if created_at < cutoff:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                await self._remove_entry(key)
            
            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} old cache entries")
    
    async def _remove_entry(self, key: str):
        """Remove a single cache entry (assumes lock is held)."""
        if key not in self._index:
            return
        
        meta = self._index.pop(key)
        self._total_size -= meta.get("size", 0)
        
        # Delete file
        cache_path = CACHE_DIR / meta["filename"]
        try:
            cache_path.unlink(missing_ok=True)
        except Exception:
            pass
    
    async def _evict_if_needed(self):
        """Evict oldest entries if cache limits are exceeded (assumes lock is held)."""
        max_size_bytes = MAX_CACHE_SIZE_MB * 1024 * 1024
        
        while (self._total_size > max_size_bytes or 
               len(self._index) > MAX_CACHE_ENTRIES) and self._index:
            # Remove oldest (first) entry
            key = next(iter(self._index))
            await self._remove_entry(key)
            logger.debug(f"Evicted cache entry: {key}")
    
    async def get(
        self,
        text: str,
        voice_id: str,
        language: str,
        output_format: str,
        speed: float
    ) -> Optional[Tuple[bytes, str]]:
        """
        Get cached audio if available.
        
        Returns:
            Tuple of (audio_bytes, content_type) or None
        """
        key = generate_cache_key(text, voice_id, language, output_format, speed)
        
        async with self._lock:
            if key not in self._index:
                return None
            
            meta = self._index[key]
            cache_path = CACHE_DIR / meta["filename"]
            
            if not cache_path.exists():
                # File missing, remove entry
                await self._remove_entry(key)
                await self._save_index()
                return None
            
            # Move to end (most recently used)
            self._index.move_to_end(key)
            
            # Update access time
            meta["last_accessed"] = datetime.utcnow().isoformat()
            meta["access_count"] = meta.get("access_count", 0) + 1
        
        # Read file
        try:
            async with aiofiles.open(cache_path, "rb") as f:
                audio_bytes = await f.read()
            
            content_type = "audio/mpeg" if output_format == "mp3" else "audio/wav"
            
            logger.debug(f"Cache hit: {key}")
            return audio_bytes, content_type
            
        except Exception as e:
            logger.error(f"Failed to read cached audio: {e}")
            return None
    
    async def set(
        self,
        text: str,
        voice_id: str,
        language: str,
        output_format: str,
        speed: float,
        audio_bytes: bytes
    ):
        """
        Cache generated audio.
        
        Args:
            text, voice_id, language, output_format, speed: TTS parameters
            audio_bytes: Generated audio data
        """
        key = generate_cache_key(text, voice_id, language, output_format, speed)
        ext = "mp3" if output_format == "mp3" else "wav"
        filename = f"{key}.{ext}"
        cache_path = CACHE_DIR / filename
        
        async with self._lock:
            # Evict if needed before adding
            await self._evict_if_needed()
            
            # Remove existing entry if present
            if key in self._index:
                await self._remove_entry(key)
            
            # Save file
            try:
                async with aiofiles.open(cache_path, "wb") as f:
                    await f.write(audio_bytes)
            except Exception as e:
                logger.error(f"Failed to write cache file: {e}")
                return
            
            # Add to index
            size = len(audio_bytes)
            self._index[key] = {
                "filename": filename,
                "size": size,
                "text_preview": text[:100],
                "voice_id": voice_id,
                "language": language,
                "output_format": output_format,
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat(),
                "access_count": 0
            }
            self._total_size += size
            
            # Move to end (most recently used)
            self._index.move_to_end(key)
            
            await self._save_index()
            
            logger.debug(f"Cached audio: {key} ({size / 1024:.1f}KB)")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "entries": len(self._index),
            "total_size_mb": round(self._total_size / 1024 / 1024, 2),
            "max_size_mb": MAX_CACHE_SIZE_MB,
            "max_entries": MAX_CACHE_ENTRIES,
            "max_age_hours": MAX_CACHE_AGE_HOURS
        }
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            for key in list(self._index.keys()):
                await self._remove_entry(key)
            
            await self._save_index()
            logger.info("Cache cleared")


# Global instance
_cache: Optional[AudioCache] = None


async def get_cache() -> AudioCache:
    """Get the global audio cache instance."""
    global _cache
    if _cache is None:
        _cache = AudioCache()
        await _cache.initialize()
    return _cache
