"""
XTTS-v2 TTS Engine Module
Handles model loading, text-to-speech synthesis, voice cloning, and audio export.
"""

import os

# Accept Coqui TTS license automatically (required for Docker/headless)
os.environ["COQUI_TOS_AGREED"] = "1"
# Disable torchcodec to avoid the requirement error
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"

import io
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Optional, List, BinaryIO, Tuple
import torch

# Fix for PyTorch 2.6+ weights_only default change
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import numpy as np
import soundfile as sf
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths for persistent storage (HF Spaces uses /data)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
VOICES_DIR = DATA_DIR / "voices"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"

# Ensure directories exist
VOICES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Supported languages for XTTS-v2
SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
]

# Default speaker reference for built-in voices
DEFAULT_VOICES = {
    "default": "Built-in default voice",
    "female_1": "Built-in female voice 1",
    "male_1": "Built-in male voice 1",
}


class TTSEngine:
    """
    XTTS-v2 Text-to-Speech Engine with voice cloning support.
    """
    
    _instance = None
    _model = None
    _model_loaded = False
    
    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the TTS engine (lazy model loading)."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"TTS Engine initialized. Device: {self._device}")
    
    def _load_model(self):
        """Lazy load the XTTS-v2 model on first use."""
        if self._model_loaded:
            return
        
        logger.info("Loading XTTS-v2 model (this may take a few minutes on first run)...")
        
        try:
            # Set torchaudio backend to avoid torchcodec issues
            try:
                import torchaudio
                # Try to set a backend that doesn't require torchcodec
                if hasattr(torchaudio, 'set_audio_backend'):
                    try:
                        torchaudio.set_audio_backend("soundfile")
                    except Exception:
                        try:
                            torchaudio.set_audio_backend("sox_io")
                        except Exception:
                            pass
            except ImportError:
                pass
            
            from TTS.api import TTS
            
            # Load XTTS-v2 model
            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            
            # Move to appropriate device
            if self._device == "cuda":
                self._model.to(self._device)
            
            self._model_loaded = True
            logger.info("XTTS-v2 model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load XTTS-v2 model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def get_device(self) -> str:
        """Return the current device (cuda/cpu)."""
        return self._device
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        return SUPPORTED_LANGUAGES.copy()
    
    def get_available_voices(self) -> dict:
        """
        Get all available voices (built-in + cloned).
        Returns dict with voice_id -> description.
        """
        voices = DEFAULT_VOICES.copy()
        
        # Add cloned voices from storage
        if VOICES_DIR.exists():
            for voice_file in VOICES_DIR.glob("*.wav"):
                voice_name = voice_file.stem
                voices[voice_name] = f"Cloned voice: {voice_name}"
        
        return voices
    
    def _get_speaker_wav(self, voice_id: str) -> Optional[str]:
        """
        Get the speaker WAV file path for a voice.
        XTTS-v2 requires a reference speaker audio for all synthesis.
        """
        # Check for cloned voice first
        voice_path = VOICES_DIR / f"{voice_id}.wav"
        if voice_path.exists():
            return str(voice_path)
        
        # For built-in voices, use the default sample speaker from TTS
        if voice_id in ["default", "female_1", "male_1"]:
            # Try to get sample from TTS package
            default_speaker_path = VOICES_DIR / "default_speaker.wav"
            
            if not default_speaker_path.exists():
                # Download/create a default speaker sample
                try:
                    import urllib.request
                    # Use a sample from Coqui's samples
                    sample_url = "https://github.com/coqui-ai/TTS/raw/main/tests/data/ljspeech/wavs/LJ001-0001.wav"
                    urllib.request.urlretrieve(sample_url, str(default_speaker_path))
                    logger.info("Downloaded default speaker sample")
                except Exception as e:
                    logger.warning(f"Could not download sample: {e}")
                    # Create a simple tone as fallback (won't sound great but will work)
                    import numpy as np
                    import soundfile as sf
                    sr = 22050
                    duration = 3
                    t = np.linspace(0, duration, int(sr * duration))
                    # Generate simple speech-like sound
                    audio = np.sin(2 * np.pi * 200 * t) * 0.3
                    sf.write(str(default_speaker_path), audio, sr)
                    logger.info("Created fallback speaker sample")
            
            return str(default_speaker_path)
        
        raise ValueError(f"Voice '{voice_id}' not found")
    
    def synthesize(
        self,
        text: str,
        voice_id: str = "default",
        language: str = "en",
        output_format: str = "mp3",
        speed: float = 1.0
    ) -> Tuple[bytes, str]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice_id: Voice identifier (default or cloned voice name)
            language: Language code (e.g., 'en', 'hi', 'es')
            output_format: Output format ('mp3' or 'wav')
            speed: Speech speed multiplier (0.5-2.0)
        
        Returns:
            Tuple of (audio_bytes, content_type)
        """
        # Ensure model is loaded
        self._load_model()
        
        # Validate inputs
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")
        
        if not 0.5 <= speed <= 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0")
        
        if not text or len(text.strip()) == 0:
            raise ValueError("Text cannot be empty")
        
        if len(text) > 5000:
            raise ValueError("Text too long. Maximum 5000 characters.")
        
        logger.info(f"Synthesizing: '{text[:50]}...' with voice={voice_id}, lang={language}")
        
        try:
            # Get speaker reference (XTTS-v2 always requires speaker audio)
            speaker_wav = self._get_speaker_wav(voice_id)
            
            # Generate speech
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # XTTS-v2 always requires speaker_wav
            self._model.tts_to_file(
                text=text,
                file_path=tmp_path,
                speaker_wav=speaker_wav,
                language=language,
                split_sentences=True
            )
            
            # Apply speed adjustment if needed
            audio = AudioSegment.from_wav(tmp_path)
            
            if speed != 1.0:
                # Adjust speed using pydub
                audio = audio._spawn(audio.raw_data, overrides={
                    "frame_rate": int(audio.frame_rate * speed)
                }).set_frame_rate(audio.frame_rate)
            
            # Export to desired format
            output_buffer = io.BytesIO()
            
            if output_format.lower() == "mp3":
                audio.export(output_buffer, format="mp3", bitrate="192k")
                content_type = "audio/mpeg"
            else:
                audio.export(output_buffer, format="wav")
                content_type = "audio/wav"
            
            # Cleanup temp file
            os.unlink(tmp_path)
            
            output_buffer.seek(0)
            return output_buffer.read(), content_type
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise RuntimeError(f"Speech synthesis failed: {e}")
    
    def clone_voice(
        self,
        audio_file: BinaryIO,
        voice_name: str,
        description: str = ""
    ) -> dict:
        """
        Clone a voice from reference audio.
        
        Args:
            audio_file: Audio file (WAV/MP3, 6-30 seconds recommended)
            voice_name: Name for the cloned voice
            description: Optional description
        
        Returns:
            Dict with voice info
        """
        # Validate voice name
        if not voice_name or not voice_name.replace("_", "").isalnum():
            raise ValueError("Voice name must be alphanumeric (underscores allowed)")
        
        if voice_name in DEFAULT_VOICES:
            raise ValueError(f"Cannot use reserved voice name: {voice_name}")
        
        voice_path = VOICES_DIR / f"{voice_name}.wav"
        
        logger.info(f"Cloning voice: {voice_name}")
        
        try:
            # Read uploaded audio
            audio_data = audio_file.read()
            
            # Convert to WAV format (XTTS requires WAV)
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
            
            # Validate duration (6-30 seconds ideal)
            duration_sec = len(audio) / 1000
            if duration_sec < 3:
                raise ValueError("Audio too short. Minimum 3 seconds required.")
            if duration_sec > 60:
                raise ValueError("Audio too long. Maximum 60 seconds allowed.")
            
            # Convert to mono, 22050Hz (XTTS requirement)
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(22050)
            
            # Normalize audio
            audio = audio.normalize()
            
            # Save as WAV
            audio.export(str(voice_path), format="wav")
            
            # Save metadata
            meta_path = VOICES_DIR / f"{voice_name}.meta"
            with open(meta_path, "w") as f:
                f.write(f"description={description}\n")
                f.write(f"duration={duration_sec:.2f}\n")
            
            logger.info(f"Voice cloned successfully: {voice_name}")
            
            return {
                "voice_id": voice_name,
                "description": description,
                "duration_seconds": round(duration_sec, 2),
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            # Cleanup on failure
            if voice_path.exists():
                voice_path.unlink()
            raise RuntimeError(f"Voice cloning failed: {e}")
    
    def delete_voice(self, voice_name: str) -> bool:
        """
        Delete a cloned voice.
        
        Args:
            voice_name: Name of the voice to delete
        
        Returns:
            True if deleted, False if not found
        """
        if voice_name in DEFAULT_VOICES:
            raise ValueError("Cannot delete built-in voices")
        
        voice_path = VOICES_DIR / f"{voice_name}.wav"
        meta_path = VOICES_DIR / f"{voice_name}.meta"
        
        if not voice_path.exists():
            return False
        
        voice_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        
        logger.info(f"Voice deleted: {voice_name}")
        return True
    
    def get_voice_info(self, voice_name: str) -> Optional[dict]:
        """Get information about a specific voice."""
        if voice_name in DEFAULT_VOICES:
            return {
                "voice_id": voice_name,
                "description": DEFAULT_VOICES[voice_name],
                "type": "built-in"
            }
        
        voice_path = VOICES_DIR / f"{voice_name}.wav"
        meta_path = VOICES_DIR / f"{voice_name}.meta"
        
        if not voice_path.exists():
            return None
        
        info = {
            "voice_id": voice_name,
            "type": "cloned"
        }
        
        if meta_path.exists():
            with open(meta_path, "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        info[key] = value
        
        return info


# Global engine instance
_engine: Optional[TTSEngine] = None


def get_engine() -> TTSEngine:
    """Get the global TTS engine instance."""
    global _engine
    if _engine is None:
        _engine = TTSEngine()
    return _engine


def preload_model():
    """Pre-load the model (call during startup)."""
    engine = get_engine()
    engine._load_model()
