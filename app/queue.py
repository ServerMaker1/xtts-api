"""
Async Job Queue Module
Handles TTS job queuing, processing, and status tracking.
"""

import asyncio
import uuid
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Storage paths
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


class JobStatus(str, Enum):
    """Job status enum."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TTSJob:
    """Represents a TTS synthesis job."""
    id: str
    text: str
    voice_id: str
    language: str
    output_format: str
    speed: float
    user_id: int
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    audio_path: Optional[str] = None
    audio_size: Optional[int] = None
    processing_time: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert job to dictionary for API response."""
        return {
            "job_id": self.id,
            "status": self.status.value,
            "text_preview": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "voice_id": self.voice_id,
            "language": self.language,
            "output_format": self.output_format,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "audio_size_bytes": self.audio_size,
            "processing_time_seconds": self.processing_time
        }


class JobQueue:
    """
    Async job queue for TTS processing.
    
    Features:
    - Async job submission and status tracking
    - Background worker for processing
    - Auto-cleanup of old jobs
    - Audio file management
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
            self._jobs: Dict[str, TTSJob] = {}
            self._queue: asyncio.Queue = None
            self._worker_task: Optional[asyncio.Task] = None
            self._running = False
            self._tts_engine = None
    
    async def start(self, tts_engine):
        """Start the job queue worker."""
        if self._running:
            return
        
        self._tts_engine = tts_engine
        self._queue = asyncio.Queue()
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_worker())
        
        logger.info("Job queue started")
    
    async def stop(self):
        """Stop the job queue worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Job queue stopped")
    
    async def _worker(self):
        """Background worker that processes jobs from the queue."""
        logger.info("Job worker started")
        
        while self._running:
            try:
                # Wait for a job with timeout
                try:
                    job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                job = self._jobs.get(job_id)
                if not job or job.status != JobStatus.PENDING:
                    continue
                
                # Process the job
                await self._process_job(job)
                self._queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)
        
        logger.info("Job worker stopped")
    
    async def _process_job(self, job: TTSJob):
        """Process a single TTS job."""
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        
        logger.info(f"Processing job {job.id}: '{job.text[:50]}...'")
        
        try:
            # Run synthesis in thread pool to not block async loop
            start_time = asyncio.get_event_loop().time()
            
            audio_bytes, content_type = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._tts_engine.synthesize(
                    text=job.text,
                    voice_id=job.voice_id,
                    language=job.language,
                    output_format=job.output_format,
                    speed=job.speed
                )
            )
            
            end_time = asyncio.get_event_loop().time()
            
            # Save audio to file
            ext = "mp3" if job.output_format == "mp3" else "wav"
            audio_path = JOBS_DIR / f"{job.id}.{ext}"
            
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            
            # Update job status
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.audio_path = str(audio_path)
            job.audio_size = len(audio_bytes)
            job.processing_time = round(end_time - start_time, 2)
            
            logger.info(f"Job {job.id} completed in {job.processing_time}s")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            
            logger.error(f"Job {job.id} failed: {e}")
    
    async def _cleanup_worker(self):
        """Background task to clean up old jobs and audio files."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_old_jobs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_old_jobs(self):
        """Remove jobs older than 1 hour."""
        cutoff = datetime.utcnow() - timedelta(hours=1)
        jobs_to_remove = []
        
        for job_id, job in self._jobs.items():
            if job.created_at < cutoff:
                jobs_to_remove.append(job_id)
                
                # Delete audio file if exists
                if job.audio_path:
                    try:
                        Path(job.audio_path).unlink(missing_ok=True)
                    except Exception:
                        pass
        
        for job_id in jobs_to_remove:
            del self._jobs[job_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    async def submit_job(
        self,
        text: str,
        voice_id: str,
        language: str,
        output_format: str,
        speed: float,
        user_id: int
    ) -> TTSJob:
        """
        Submit a new TTS job to the queue.
        
        Returns:
            The created job object
        """
        job_id = str(uuid.uuid4())
        
        job = TTSJob(
            id=job_id,
            text=text,
            voice_id=voice_id,
            language=language,
            output_format=output_format,
            speed=speed,
            user_id=user_id
        )
        
        self._jobs[job_id] = job
        await self._queue.put(job_id)
        
        logger.info(f"Job {job_id} submitted (queue size: {self._queue.qsize()})")
        
        return job
    
    def get_job(self, job_id: str) -> Optional[TTSJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)
    
    def get_job_audio(self, job_id: str) -> Optional[tuple]:
        """
        Get the audio file for a completed job.
        
        Returns:
            Tuple of (audio_bytes, content_type) or None
        """
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.COMPLETED or not job.audio_path:
            return None
        
        audio_path = Path(job.audio_path)
        if not audio_path.exists():
            return None
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        content_type = "audio/mpeg" if job.output_format == "mp3" else "audio/wav"
        return audio_bytes, content_type
    
    def get_user_jobs(self, user_id: int, limit: int = 20) -> list:
        """Get recent jobs for a user."""
        user_jobs = [
            job for job in self._jobs.values()
            if job.user_id == user_id
        ]
        
        # Sort by created_at descending
        user_jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return user_jobs[:limit]
    
    def get_queue_status(self) -> dict:
        """Get overall queue status."""
        pending = sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)
        processing = sum(1 for j in self._jobs.values() if j.status == JobStatus.PROCESSING)
        completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)
        
        return {
            "queue_size": self._queue.qsize() if self._queue else 0,
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "total_jobs": len(self._jobs)
        }


# Global instance
_job_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get the global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue
