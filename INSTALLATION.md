# 🚀 Installation Guide

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed (for frontend)
- [ ] Git installed
- [ ] Azure OpenAI subscription with deployed models
- [ ] Pinecone account
- [ ] Tavily account
- [ ] ~2GB free disk space (for TTS models)
- [ ] ~2GB free RAM (for TTS model loading)

## Step-by-Step Installation

### 1. Environment Setup

```bash
# Clone repository
git clone <your-repository-url>
cd crag-assistant

# Create Python virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Backend Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify critical installations
python -c "import TTS; print(f'Coqui TTS: {TTS.__version__}')"
python -c "import fastapi; print('FastAPI: OK')"
python -c "import langchain; print('LangChain: OK')"
```

### 3. Configure API Keys

```bash
# Copy environment template
cp .env.template .env

# Edit .env file with your actual API keys
nano .env  # or use your preferred editor
```

**Required API Keys:**
- **Azure OpenAI**: Get from Azure Portal > Azure OpenAI > Keys
- **Pinecone**: Get from https://app.pinecone.io/
- **Tavily**: Get from https://tavily.com/

### 4. Azure OpenAI Model Deployment

Before running the application, deploy these models in Azure OpenAI Studio:

1. **text-embedding-3-small**
   - Model: `text-embedding-3-small`
   - Deployment name: `text-embedding-3-small`

2. **gpt-4o-mini**
   - Model: `gpt-4o-mini`
   - Deployment name: `gpt-4o-mini`

### 5. Test Backend Installation

```bash
# Test Coqui TTS installation
python -c "
from TTS.api import TTS
print('✅ Coqui TTS installation verified')
print('Available models:')
models = TTS.list_models()
print(f'Found {len(models)} models')
"

# Start backend server
python crag_api.py
```

**Expected output:**
```
🚀 Initializing CRAG components with Azure OpenAI...
📍 Embedding Endpoint: https://your-embedding-resource.openai.azure.com/
📍 LLM Endpoint: https://your-llm-resource.openai.azure.com/
✅ Vector store initialized successfully
✅ LLM components initialized successfully
✅ CRAG workflow initialized successfully
🎉 CRAG initialization complete!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6. Test TTS Functionality

```bash
# Test TTS model loading (will download on first use)
curl -X POST "http://localhost:8000/ask_with_tts" \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello, this is a test of the TTS system."}'
```

**Note**: First TTS request will take 5-10 minutes to download the model (~500MB).

### 7. Install Frontend (Optional)

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start development server
ng serve
```

Frontend will be available at `http://localhost:4200`

### 8. Verify Installation

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test basic query
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello, how are you?"}'

# Test TTS models endpoint
curl http://localhost:8000/tts_models

# Test model switching
curl -X POST "http://localhost:8000/set_tts_model" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "tts_models/en/ljspeech/glow-tts"}'
```

## Troubleshooting Installation

### Python Dependencies

**Issue**: `pip install TTS` fails
```bash
# Update pip and setuptools
python -m pip install --upgrade pip setuptools

# Install with verbose output
pip install TTS -v

# Alternative: Install specific version
pip install TTS==0.22.0
```

**Issue**: System dependencies missing (Linux)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install espeak espeak-data libespeak1 libespeak-dev
sudo apt-get install ffmpeg

# CentOS/RHEL
sudo yum install espeak espeak-devel ffmpeg
```

**Issue**: System dependencies missing (macOS)
```bash
# Install with Homebrew
brew install espeak ffmpeg
```

### Coqui TTS Specific Issues

**Issue**: Model download fails
```bash
# Check internet connection and try again
# Models are downloaded to ~/.local/share/tts/

# Clear cache if corrupted
rm -rf ~/.local/share/tts/
```

**Issue**: "No module named 'torch'"
```bash
# TTS should install PyTorch automatically, but if it fails:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Issue**: GPU-related errors
```bash
# Force CPU-only usage (add to your .env or environment)
export CUDA_VISIBLE_DEVICES=""

# Or install CPU-only PyTorch explicitly
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Azure Configuration

**Issue**: Authentication errors
- Verify API keys are correct
- Check endpoint URLs format
- Ensure models are deployed
- Verify API version compatibility

**Issue**: Content policy violations
- Test with simple questions first
- Avoid trigger words in initial testing
- Check Azure content filter settings

### TTS Model Loading

**Issue**: First TTS request timeout
- First request downloads model (~500MB-1GB)
- Can take 5-10 minutes on slow connections
- Subsequent requests are much faster (2-5 seconds)
- Increase client timeout if needed

**Issue**: Out of memory errors
```bash
# TTS models require significant RAM
# Close other applications
# Use lighter models:
curl -X POST "http://localhost:8000/set_tts_model" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "tts_models/en/ljspeech/glow-tts"}'
```

**Issue**: Audio file not found
```bash
# Check audio_files directory exists and has permissions
mkdir -p audio_files
chmod 755 audio_files

# Check static file serving
curl http://localhost:8000/audio/test.wav
```

### Network Issues

**Issue**: CORS errors (frontend)
- Backend includes CORS middleware
- Verify frontend runs on port 4200
- Check browser console for exact errors

**Issue**: Connection refused
- Verify backend is running on port 8000
- Check firewall settings
- Ensure no port conflicts

### Model Selection Issues

**Issue**: Specific model not working
```bash
# List all available models
python -c "
from TTS.api import TTS
models = TTS.list_models()
for model in models:
    if 'en' in model:
        print(model)
"

# Try alternative models
curl -X POST "http://localhost:8000/set_tts_model" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "tts_models/en/ljspeech/tacotron2-DDC"}'
```

## Production Deployment

### Environment Variables

For production, set these additional variables:
```env
# Production settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# TTS Configuration
TTS_CACHE_PATH=/app/tts_cache
TTS_DEFAULT_MODEL=tts_models/en/ljspeech/glow-tts

# Security (optional)
API_KEY_REQUIRED=true
RATE_LIMIT_ENABLED=true
```

### Performance Optimization

```bash
# Pre-download TTS models to avoid first-request delay
python -c "
from TTS.api import TTS
TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC')
TTS(model_name='tts_models/en/ljspeech/glow-tts')
print('Models pre-loaded successfully')
"
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

# Install system dependencies for TTS
RUN apt-get update && apt-get install -y \
    espeak espeak-data libespeak1 libespeak-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download TTS models (optional, increases image size)
# RUN python -c "from TTS.api import TTS; TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC')"

COPY . .
EXPOSE 8000

CMD ["python", "crag_api.py"]
```

## Maintenance

### Updating Dependencies

```bash
# Update Python packages
pip install --upgrade TTS
pip install --upgrade -r requirements.txt

# Update Node.js packages
cd frontend && npm update
```

### TTS Model Management

```bash
# Clear TTS model cache
rm -rf ~/.local/share/tts/

# List models currently cached
python -c "
import os
cache_dir = os.path.expanduser('~/.local/share/tts/')
if os.path.exists(cache_dir):
    print('Cached models:', os.listdir(cache_dir))
else:
    print('No cached models found')
"

# Pre-load specific models
python -c "
from TTS.api import TTS
TTS(model_name='tts_models/en/vctk/vits')  # High quality
TTS(model_name='tts_models/en/ljspeech/glow-tts')  # Fast
"
```

### Log Management

```bash
# View API logs
tail -f api.log

# Clear audio files (optional)
find audio_files/ -name "*.wav" -mtime +1 -delete

# Monitor disk usage
du -sh ~/.local/share/tts/
du -sh audio_files/
```

## TTS Model Recommendations

### For Development
- **Model**: `tts_models/en/ljspeech/tacotron2-DDC`
- **Pros**: Balanced quality/speed, reliable
- **Cons**: Medium file size

### For Production (Speed Priority)
- **Model**: `tts_models/en/ljspeech/glow-tts`
- **Pros**: Fast generation, good quality
- **Cons**: Slightly robotic on complex sentences

### For Production (Quality Priority)
- **Model**: `tts_models/en/vctk/vits`
- **Pros**: Excellent quality, natural sounding
- **Cons**: Slower generation, larger model

---

**Need help?** Check the main README troubleshooting section or create an issue.