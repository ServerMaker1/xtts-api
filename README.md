# 🎙️ XTTS-v2 TTS API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![HuggingFace Spaces](https://img.shields.io/badge/🤗-HuggingFace%20Spaces-orange.svg)](https://huggingface.co/spaces)

A production-ready, self-hosted Text-to-Speech API powered by [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) with voice cloning, multi-language support, and a beautiful admin dashboard. Deploy for **free** on HuggingFace Spaces!

**Author:** [Anubhav N. Mishra](https://github.com/anubhav-n-mishra)

---

## ✨ Features

- 🗣️ **High-Quality TTS** - XTTS-v2 model for natural-sounding speech
- 🌍 **17 Languages** - English, Hindi, Spanish, French, German, Japanese, Chinese, and more
- 🎭 **Voice Cloning** - Clone any voice from 6-30 second audio samples
- 📊 **Admin Dashboard** - Beautiful UI to manage API keys and view usage analytics
- 🔐 **Multi-Tier Auth** - Owner + Friends access system with rate limiting
- ⚡ **Async Processing** - Queue long texts for background processing
- 💾 **Audio Caching** - Automatic caching for repeated requests
- 📈 **Usage Analytics** - Track requests, characters, audio minutes, languages, and voices
- 🆓 **100% Free** - Deploy on HuggingFace Spaces at no cost

---

## 🚀 Quick Deploy to HuggingFace Spaces

### Step 1: Create a HuggingFace Space

1. Go to [HuggingFace Spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure:
   - **Space name:** `tts` (or any name you prefer)
   - **SDK:** `Docker`
   - **Visibility:** `Public` (recommended) or `Private`
4. Click **"Create Space"**

### Step 2: Generate Your Secure Tokens

Run this Python script locally to generate secure tokens:

```python
import secrets

print("=" * 60)
print("🔐 YOUR TTS API TOKENS - SAVE THESE SECURELY!")
print("=" * 60)
print(f"\nOWNER_TOKEN={secrets.token_urlsafe(32)}")
print(f"OWNER_NAME=YourName")
print()
for i in range(1, 6):
    print(f"FRIEND_{i}_TOKEN={secrets.token_urlsafe(32)}")
    print(f"FRIEND_{i}_NAME=Friend{i}")
print("\n" + "=" * 60)
```

**⚠️ Important:** Save these tokens securely! You'll need them to access your API.

### Step 3: Configure Space Secrets

In your HuggingFace Space:
1. Go to **Settings** → **Repository secrets**
2. Add these secrets:

| Secret Name | Description |
|-------------|-------------|
| `OWNER_TOKEN` | Your master access token (unlimited access) |
| `OWNER_NAME` | Your display name |
| `FRIEND_1_TOKEN` | Friend 1's access token |
| `FRIEND_1_NAME` | Friend 1's display name |
| `FRIEND_2_TOKEN` | Friend 2's access token (optional) |
| ... | Add up to 5 friends |

### Step 4: Clone and Push Code

```bash
# Clone this repository
git clone https://github.com/anubhav-n-mishra/xtts-api.git
cd xtts-api

# Clone your HuggingFace Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME hf-space
cd hf-space

# Copy all files
cp -r ../xtts-api/* .

# Push to HuggingFace
git add -A
git commit -m "Initial deployment"
git push origin main
```

### Step 5: Wait & Access

- **Build time:** 5-10 minutes
- **First TTS request:** Model downloads (~2GB), takes 2-3 minutes
- **Dashboard:** `https://YOUR_USERNAME-YOUR_SPACE.hf.space/static/index.html`

---

## 🔧 Local Development

### Prerequisites

- Python 3.10 or higher
- FFmpeg installed on your system
- CUDA-capable GPU (optional, for faster inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/anubhav-n-mishra/xtts-api.git
cd xtts-api

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
# Linux/Mac:
export OWNER_TOKEN="your_secure_token_here"
export OWNER_NAME="YourName"
export DATA_DIR="./data"

# Windows PowerShell:
$env:OWNER_TOKEN="your_secure_token_here"
$env:OWNER_NAME="YourName"
$env:DATA_DIR="./data"

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
# Build the image
docker build -t xtts-api .

# Run the container
docker run -p 7860:7860 \
  -e OWNER_TOKEN="your_token_here" \
  -e OWNER_NAME="YourName" \
  -v $(pwd)/data:/data \
  xtts-api
```

---

## 📖 API Reference

### Authentication

All endpoints (except `/health`) require authentication. Pass your token in the `key` header:

```bash
curl -H "key: YOUR_API_KEY" https://your-space.hf.space/voices
```

### Core Endpoints

#### `POST /tts` - Generate Speech

Convert text to speech audio.

```bash
curl -X POST "https://your-space.hf.space/tts" \
  -H "key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test!",
    "voice": "default",
    "language": "en",
    "format": "mp3",
    "speed": 1.0
  }' \
  --output speech.mp3
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to synthesize (max 5000 chars) |
| `voice` | string | "default" | Voice ID |
| `language` | string | "en" | Language code |
| `format` | string | "mp3" | Output format: "mp3" or "wav" |
| `speed` | float | 1.0 | Speech speed (0.5-2.0) |
| `async_mode` | bool | false | Return job_id for async processing |

#### `GET /voices` - List Available Voices

```bash
curl -H "key: YOUR_API_KEY" "https://your-space.hf.space/voices"
```

**Response:**
```json
{
  "voices": [
    {"voice_id": "default", "description": "Built-in default voice", "type": "built-in"},
    {"voice_id": "female_1", "description": "Built-in female voice 1", "type": "built-in"},
    {"voice_id": "my_clone", "description": "Cloned voice: my_clone", "type": "cloned"}
  ],
  "total": 3
}
```

#### `POST /clone` - Clone a Voice

Upload audio to create a custom voice.

```bash
curl -X POST "https://your-space.hf.space/clone" \
  -H "key: YOUR_API_KEY" \
  -F "audio=@sample.wav" \
  -F "name=my_voice" \
  -F "description=My custom cloned voice"
```

**Requirements:**
- Audio: WAV or MP3 format
- Duration: 6-30 seconds (ideal), 3-60 seconds (allowed)
- Quality: Clear speech, minimal background noise

#### `GET /languages` - List Supported Languages

```bash
curl -H "key: YOUR_API_KEY" "https://your-space.hf.space/languages"
```

### API Key Management

#### `POST /keys` - Create API Key

Create a new API key (requires master token).

```bash
curl -X POST "https://your-space.hf.space/keys" \
  -H "key: YOUR_MASTER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "My App"}'
```

#### `GET /keys` - List Your API Keys

```bash
curl -H "key: YOUR_MASTER_TOKEN" "https://your-space.hf.space/keys"
```

#### `DELETE /keys/{key_id}` - Revoke API Key

```bash
curl -X DELETE "https://your-space.hf.space/keys/123" \
  -H "key: YOUR_MASTER_TOKEN"
```

### Analytics

#### `GET /stats` - Usage Statistics

```bash
curl -H "key: YOUR_API_KEY" "https://your-space.hf.space/stats"
```

**Response:**
```json
{
  "total_requests": 150,
  "total_audio_seconds": 3600.5,
  "total_characters": 50000,
  "language_usage": {"en": 100, "hi": 30, "es": 20},
  "voice_usage": {"default": 120, "my_clone": 30}
}
```

#### `GET /usage` - Recent Usage History

```bash
curl -H "key: YOUR_API_KEY" "https://your-space.hf.space/usage?limit=10"
```

### Full API Documentation

Interactive Swagger docs available at: `https://your-space.hf.space/docs`

---

## 🌍 Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `ko` | Korean |
| `es` | Spanish | `ja` | Japanese |
| `fr` | French | `zh-cn` | Chinese (Simplified) |
| `de` | German | `ar` | Arabic |
| `it` | Italian | `hi` | Hindi |
| `pt` | Portuguese | `pl` | Polish |
| `ru` | Russian | `tr` | Turkish |
| `nl` | Dutch | `cs` | Czech |
| `hu` | Hungarian | | |

---

## 🏗️ Project Structure

```
xtts-api/
├── app/
│   ├── __init__.py          # Package init
│   ├── main.py              # FastAPI application & routes
│   ├── tts_engine.py        # XTTS-v2 model wrapper
│   ├── database.py          # SQLite database & usage tracking
│   ├── auth.py              # Authentication module
│   ├── middleware.py        # Auth & rate limiting middleware
│   ├── queue.py             # Async job processing queue
│   ├── cache.py             # Audio caching system
│   └── static/
│       └── index.html       # Admin dashboard
├── Dockerfile               # HuggingFace Spaces compatible
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── LICENSE                  # MIT License
```

---

## 🔒 Access Control

### User Tiers

| Tier | Rate Limit | Max API Keys | Description |
|------|------------|--------------|-------------|
| **Owner** | Unlimited | Unlimited | Full admin access, all features |
| **Friend** | 3 req/sec | 5 keys | Shared access for trusted friends |

### Authentication Flow

1. **Master Token** → Used to log into dashboard and create API keys
2. **API Key** → Used by applications to call TTS endpoints

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OWNER_TOKEN` | ✅ Yes | - | Owner's master authentication token |
| `OWNER_NAME` | No | "owner" | Owner's display name in dashboard |
| `FRIEND_1_TOKEN` | No | - | Friend 1's master token |
| `FRIEND_1_NAME` | No | "friend_1" | Friend 1's display name |
| `FRIEND_2_TOKEN` | No | - | Friend 2's master token |
| ... | ... | ... | Up to `FRIEND_5_TOKEN` / `FRIEND_5_NAME` |
| `DATA_DIR` | No | "/data" | Directory for persistent storage |

---

## 💻 Code Examples

### Python

```python
import requests

API_URL = "https://your-space.hf.space"
API_KEY = "tts_your_api_key_here"

def text_to_speech(text, language="en", voice="default"):
    response = requests.post(
        f"{API_URL}/tts",
        headers={"key": API_KEY},
        json={
            "text": text,
            "language": language,
            "voice": voice,
            "format": "mp3"
        }
    )
    response.raise_for_status()
    return response.content

# Generate and save audio
audio = text_to_speech("Hello from Python!", language="en")
with open("output.mp3", "wb") as f:
    f.write(audio)
```

### JavaScript / Node.js

```javascript
const fetch = require('node-fetch');
const fs = require('fs');

const API_URL = 'https://your-space.hf.space';
const API_KEY = 'tts_your_api_key_here';

async function textToSpeech(text, language = 'en') {
    const response = await fetch(`${API_URL}/tts`, {
        method: 'POST',
        headers: {
            'key': API_KEY,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: text,
            language: language,
            format: 'mp3'
        })
    });
    
    const buffer = await response.buffer();
    fs.writeFileSync('output.mp3', buffer);
}

textToSpeech('Hello from JavaScript!');
```

### Browser JavaScript

```javascript
async function generateSpeech(text) {
    const response = await fetch('https://your-space.hf.space/tts', {
        method: 'POST',
        headers: {
            'key': 'your_api_key',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: text,
            language: 'en',
            format: 'mp3'
        })
    });
    
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    
    const audio = new Audio(audioUrl);
    audio.play();
}
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** your feature branch: `git checkout -b feature/AmazingFeature`
3. **Commit** your changes: `git commit -m 'Add AmazingFeature'`
4. **Push** to the branch: `git push origin feature/AmazingFeature`
5. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/xtts-api.git
cd xtts-api

# Create branch
git checkout -b feature/my-feature

# Make changes, test locally, then submit PR
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Anubhav N. Mishra

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) - The amazing XTTS-v2 model
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [HuggingFace](https://huggingface.co/) - Free model hosting and Spaces

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/anubhav-n-mishra/xtts-api/issues)
- **Author:** [Anubhav N. Mishra](https://github.com/anubhav-n-mishra)

---

## ⭐ Star This Project

If you find this project useful, please give it a star! It helps others discover it and motivates continued development.

---

**Made with ❤️ by [Anubhav N. Mishra](https://github.com/anubhav-n-mishra)**
