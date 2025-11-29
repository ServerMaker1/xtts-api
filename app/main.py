"""
XTTS-v2 TTS API - Main FastAPI Application
"""

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.responses import Response, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

from .tts_engine import get_engine, preload_model
from .database import get_database
from .queue import get_job_queue, JobStatus
from .cache import get_cache
from .middleware import AuthMiddleware, get_current_user, start_rate_limiter_cleanup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class TTSRequest(BaseModel):
    """Request model for TTS synthesis."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice: str = Field(default="default", description="Voice ID (default or cloned voice name)")
    language: str = Field(default="en", description="Language code (en, hi, es, fr, etc.)")
    format: str = Field(default="mp3", pattern="^(mp3|wav)$", description="Output format")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed (0.5-2.0)")
    use_cache: bool = Field(default=True, description="Use cached audio if available")
    async_mode: bool = Field(default=False, description="Return job_id for async processing")


class TTSResponse(BaseModel):
    """Response model for async TTS job submission."""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    text_preview: Optional[str] = None
    voice_id: Optional[str] = None
    language: Optional[str] = None
    output_format: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    audio_size_bytes: Optional[int] = None
    processing_time_seconds: Optional[float] = None


class VoiceInfo(BaseModel):
    """Voice information model."""
    voice_id: str
    description: str
    type: str = "built-in"


class APIKeyRequest(BaseModel):
    """Request model for API key generation."""
    name: Optional[str] = Field(None, max_length=100, description="Optional name for the key")


class APIKeyResponse(BaseModel):
    """Response model for API key generation."""
    api_key: str
    message: str


class APIKeyInfo(BaseModel):
    """API key information (without the actual key)."""
    id: int
    key_prefix: str
    name: Optional[str]
    created_at: str
    last_used: Optional[str]


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle handler."""
    logger.info("Starting TTS API server...")
    
    # Initialize database
    db = await get_database()
    logger.info("Database initialized")
    
    # Initialize cache
    cache = await get_cache()
    logger.info("Audio cache initialized")
    
    # Initialize TTS engine (lazy load, will load model on first request)
    engine = get_engine()
    logger.info("TTS engine initialized (model will load on first request)")
    
    # Initialize and start job queue
    job_queue = get_job_queue()
    await job_queue.start(engine)
    logger.info("Job queue started")
    
    # Start rate limiter cleanup task
    asyncio.create_task(start_rate_limiter_cleanup())
    
    logger.info("TTS API server ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down TTS API server...")
    await job_queue.stop()
    logger.info("Server stopped")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="XTTS-v2 TTS API",
    description="""
    Private Text-to-Speech API using XTTS-v2 model.
    
    ## Features
    - High-quality multilingual TTS
    - Voice cloning from reference audio
    - Async job processing
    - Audio caching for repeated requests
    - API key management with rate limiting
    - Usage analytics dashboard
    
    ## Authentication
    All endpoints (except /health) require an API key.
    Pass your key in the `key` header.
    
    ## Rate Limits
    - Owner: Unlimited
    - Friends: 3 requests/second, max 5 API keys
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add auth middleware
app.add_middleware(AuthMiddleware)

# Mount static files for frontend
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/", tags=["Info"])
async def root(request: Request):
    """Serve the dashboard if no key, otherwise return API info."""
    # Check if this is a browser request
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        # Serve the dashboard
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
    
    return {
        "name": "XTTS-v2 TTS API",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/static/index.html"
    }


@app.get("/health", tags=["Info"])
async def health():
    """Health check endpoint."""
    engine = get_engine()
    job_queue = get_job_queue()
    
    return {
        "status": "healthy",
        "model_loaded": engine.is_model_loaded(),
        "device": engine.get_device(),
        "queue": job_queue.get_queue_status()
    }


# ============================================================================
# TTS Endpoints
# ============================================================================

@app.post("/tts", tags=["TTS"])
async def synthesize_speech(
    request: Request,
    tts_request: TTSRequest,
    user: dict = Depends(get_current_user)
):
    """
    Synthesize speech from text.
    
    **Sync mode (default):** Returns audio directly.
    
    **Async mode (async_mode=true):** Returns job_id for long texts.
    Poll /status/{job_id} to check progress and get audio.
    """
    start_time = time.time()
    engine = get_engine()
    cache = await get_cache()
    db = await get_database()
    
    # Validate voice exists
    available_voices = engine.get_available_voices()
    if tts_request.voice not in available_voices:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{tts_request.voice}' not found. Available: {list(available_voices.keys())}"
        )
    
    # Validate language
    if tts_request.language not in engine.get_supported_languages():
        raise HTTPException(
            status_code=400,
            detail=f"Language '{tts_request.language}' not supported. Supported: {engine.get_supported_languages()}"
        )
    
    # Check cache first
    if tts_request.use_cache:
        cached = await cache.get(
            text=tts_request.text,
            voice_id=tts_request.voice,
            language=tts_request.language,
            output_format=tts_request.format,
            speed=tts_request.speed
        )
        
        if cached:
            audio_bytes, content_type = cached
            processing_time = time.time() - start_time
            
            # Log usage
            await db.log_usage(
                master_id=user["id"],
                api_key_id=user.get("key_id"),
                text_length=len(tts_request.text),
                audio_duration_seconds=len(audio_bytes) / 32000,  # Estimate
                audio_size_bytes=len(audio_bytes),
                voice_id=tts_request.voice,
                language=tts_request.language,
                output_format=tts_request.format,
                cached=True,
                processing_time_seconds=processing_time
            )
            
            logger.info(f"Serving cached audio for: '{tts_request.text[:50]}...'")
            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={
                    "X-Cache": "HIT",
                    "Content-Disposition": f'attachment; filename="tts_output.{tts_request.format}"'
                }
            )
    
    # Async mode - submit job and return job_id
    if tts_request.async_mode:
        job_queue = get_job_queue()
        job = await job_queue.submit_job(
            text=tts_request.text,
            voice_id=tts_request.voice,
            language=tts_request.language,
            output_format=tts_request.format,
            speed=tts_request.speed,
            user_id=user["id"]
        )
        
        return TTSResponse(
            job_id=job.id,
            status=job.status.value,
            message="Job submitted. Poll /status/{job_id} for progress."
        )
    
    # Sync mode - generate audio directly
    try:
        audio_bytes, content_type = engine.synthesize(
            text=tts_request.text,
            voice_id=tts_request.voice,
            language=tts_request.language,
            output_format=tts_request.format,
            speed=tts_request.speed
        )
        
        processing_time = time.time() - start_time
        
        # Estimate audio duration (rough estimate based on text length)
        audio_duration = len(tts_request.text) * 0.06  # ~60ms per character
        
        # Log usage
        await db.log_usage(
            master_id=user["id"],
            api_key_id=user.get("key_id"),
            text_length=len(tts_request.text),
            audio_duration_seconds=audio_duration,
            audio_size_bytes=len(audio_bytes),
            voice_id=tts_request.voice,
            language=tts_request.language,
            output_format=tts_request.format,
            cached=False,
            processing_time_seconds=processing_time
        )
        
        # Cache the result
        if tts_request.use_cache:
            await cache.set(
                text=tts_request.text,
                voice_id=tts_request.voice,
                language=tts_request.language,
                output_format=tts_request.format,
                speed=tts_request.speed,
                audio_bytes=audio_bytes
            )
        
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "X-Cache": "MISS",
                "X-Processing-Time": f"{processing_time:.2f}s",
                "Content-Disposition": f'attachment; filename="tts_output.{tts_request.format}"'
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["TTS"])
async def get_job_status(
    job_id: str,
    user: dict = Depends(get_current_user)
):
    """
    Get the status of a TTS job.
    
    When status is 'completed', use /download/{job_id} to get the audio.
    """
    job_queue = get_job_queue()
    job = job_queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(**job.to_dict())


@app.get("/download/{job_id}", tags=["TTS"])
async def download_job_audio(
    job_id: str,
    user: dict = Depends(get_current_user)
):
    """
    Download the audio for a completed job.
    """
    job_queue = get_job_queue()
    job = job_queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job.status.value}"
        )
    
    result = job_queue.get_job_audio(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    audio_bytes, content_type = result
    
    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="tts_{job_id}.{job.output_format}"'
        }
    )


# ============================================================================
# Voice Endpoints
# ============================================================================

@app.get("/voices", tags=["Voices"])
async def list_voices(user: dict = Depends(get_current_user)):
    """
    List all available voices (built-in and cloned).
    """
    engine = get_engine()
    voices = engine.get_available_voices()
    
    voice_list = []
    for voice_id, description in voices.items():
        voice_info = engine.get_voice_info(voice_id)
        voice_list.append({
            "voice_id": voice_id,
            "description": description,
            "type": voice_info.get("type", "built-in") if voice_info else "built-in"
        })
    
    return {
        "voices": voice_list,
        "total": len(voice_list)
    }


@app.post("/clone", tags=["Voices"])
async def clone_voice(
    audio: UploadFile = File(..., description="Reference audio (WAV/MP3, 6-30 seconds)"),
    name: str = Form(..., min_length=1, max_length=50, description="Voice name"),
    description: str = Form(default="", max_length=200, description="Voice description"),
    user: dict = Depends(get_current_user)
):
    """
    Clone a voice from reference audio.
    
    Upload a clear audio sample (6-30 seconds) of the voice you want to clone.
    The audio should have minimal background noise.
    """
    engine = get_engine()
    
    # Validate file type
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an audio file (WAV or MP3)."
        )
    
    try:
        result = engine.clone_voice(
            audio_file=audio.file,
            voice_name=name,
            description=description
        )
        
        return {
            "success": True,
            "voice": result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/voices/{voice_name}", tags=["Voices"])
async def delete_voice(
    voice_name: str,
    user: dict = Depends(get_current_user)
):
    """
    Delete a cloned voice.
    """
    engine = get_engine()
    
    try:
        deleted = engine.delete_voice(voice_name)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Voice not found")
        
        return {"success": True, "message": f"Voice '{voice_name}' deleted"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# API Key Management Endpoints
# ============================================================================

@app.post("/keys", response_model=APIKeyResponse, tags=["API Keys"])
async def create_api_key(
    request: Request,
    key_request: APIKeyRequest = None,
    user: dict = Depends(get_current_user)
):
    """
    Generate a new API key.
    
    **Requires master token in header.**
    
    - Owner: Unlimited keys
    - Friends: Max 5 keys each
    """
    master_token = request.headers.get("key") or request.headers.get("authorization")
    if master_token and master_token.startswith("Bearer "):
        master_token = master_token[7:]
    
    if user.get("type") != "master":
        raise HTTPException(
            status_code=403,
            detail="Only master tokens can create API keys. Use your master token, not an API key."
        )
    
    db = await get_database()
    
    key_name = key_request.name if key_request else None
    api_key, error = await db.create_api_key(master_token, key_name)
    
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    return APIKeyResponse(
        api_key=api_key,
        message="API key created successfully. Store it securely - it won't be shown again."
    )


@app.get("/keys", tags=["API Keys"])
async def list_api_keys(
    request: Request,
    user: dict = Depends(get_current_user)
):
    """
    List all your API keys.
    
    **Requires master token in header.**
    """
    master_token = request.headers.get("key") or request.headers.get("authorization")
    if master_token and master_token.startswith("Bearer "):
        master_token = master_token[7:]
    
    if user.get("type") != "master":
        raise HTTPException(
            status_code=403,
            detail="Only master tokens can list API keys."
        )
    
    db = await get_database()
    keys, error = await db.list_api_keys(master_token)
    
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    return {
        "keys": keys,
        "total": len(keys),
        "max_keys": user["max_keys"] if user["max_keys"] > 0 else "unlimited"
    }


@app.delete("/keys/{key_id}", tags=["API Keys"])
async def revoke_api_key(
    key_id: int,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """
    Revoke (delete) an API key.
    
    **Requires master token in header.**
    """
    master_token = request.headers.get("key") or request.headers.get("authorization")
    if master_token and master_token.startswith("Bearer "):
        master_token = master_token[7:]
    
    if user.get("type") != "master":
        raise HTTPException(
            status_code=403,
            detail="Only master tokens can revoke API keys."
        )
    
    db = await get_database()
    success, error = await db.revoke_api_key(master_token, key_id)
    
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    return {"success": True, "message": f"API key {key_id} revoked"}


# ============================================================================
# Stats & Usage Endpoints
# ============================================================================

@app.get("/stats", tags=["Stats"])
async def get_user_stats(
    request: Request,
    user: dict = Depends(get_current_user)
):
    """
    Get your usage statistics.
    """
    db = await get_database()
    stats = await db.get_user_stats(user["id"])
    return stats


@app.get("/usage", tags=["Stats"])
async def get_usage_history(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    user: dict = Depends(get_current_user)
):
    """
    Get your recent usage history.
    """
    db = await get_database()
    usage = await db.get_recent_usage(user["id"], limit)
    return {"usage": usage}


# ============================================================================
# Admin Endpoints (Owner Only)
# ============================================================================

@app.get("/admin/cache", tags=["Admin"])
async def get_cache_stats(user: dict = Depends(get_current_user)):
    """
    Get cache statistics. (Owner only)
    """
    if not user.get("is_owner"):
        raise HTTPException(status_code=403, detail="Owner access required")
    
    cache = await get_cache()
    return cache.get_stats()


@app.delete("/admin/cache", tags=["Admin"])
async def clear_cache(user: dict = Depends(get_current_user)):
    """
    Clear the audio cache. (Owner only)
    """
    if not user.get("is_owner"):
        raise HTTPException(status_code=403, detail="Owner access required")
    
    cache = await get_cache()
    await cache.clear()
    
    return {"success": True, "message": "Cache cleared"}


@app.get("/admin/queue", tags=["Admin"])
async def get_queue_stats(user: dict = Depends(get_current_user)):
    """
    Get job queue statistics. (Owner only)
    """
    if not user.get("is_owner"):
        raise HTTPException(status_code=403, detail="Owner access required")
    
    job_queue = get_job_queue()
    return job_queue.get_queue_status()


@app.get("/admin/users", tags=["Admin"])
async def list_master_users(user: dict = Depends(get_current_user)):
    """
    List all master users. (Owner only)
    """
    if not user.get("is_owner"):
        raise HTTPException(status_code=403, detail="Owner access required")
    
    db = await get_database()
    users = await db.get_master_users()
    
    return {
        "users": users,
        "total": len(users)
    }


@app.get("/languages", tags=["Info"])
async def list_languages(user: dict = Depends(get_current_user)):
    """
    List all supported languages.
    """
    engine = get_engine()
    languages = engine.get_supported_languages()
    
    language_names = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "pl": "Polish",
        "tr": "Turkish",
        "ru": "Russian",
        "nl": "Dutch",
        "cs": "Czech",
        "ar": "Arabic",
        "zh-cn": "Chinese (Simplified)",
        "ja": "Japanese",
        "hu": "Hungarian",
        "ko": "Korean",
        "hi": "Hindi"
    }
    
    return {
        "languages": [
            {"code": code, "name": language_names.get(code, code)}
            for code in languages
        ],
        "total": len(languages)
    }
