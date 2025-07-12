# 🚀 Installation Guide

## Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed (for frontend)
- [ ] Git installed
- [ ] Azure OpenAI subscription with deployed models
- [ ] Pinecone account
- [ ] Tavily account

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
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print('Transformers: OK')"
python -c "import fastapi; print('FastAPI: OK')"
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
# Test model loading
python -c "
from transformers import SpeechT5Processor
processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
print('✅ SpeechT5 installation verified')
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

### 6. Install Frontend (Optional)

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start development server
ng serve
```

Frontend will be available at `http://localhost:4200`

### 7. Verify Installation

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test basic query
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello, how are you?"}'

# Test TTS query
curl -X POST "http://localhost:8000/ask_with_tts" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are AI agents?"}'
```

## Troubleshooting Installation

### Python Dependencies

**Issue**: `pip install` fails
```bash
# Update pip
python -m pip install --upgrade pip

# Clear cache
pip cache purge

# Install with verbose output
pip install -r requirements.txt -v
```

**Issue**: PyTorch CUDA errors
```bash
# Force CPU-only installation
pip uninstall torch torchvision torchaudio
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
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

### Model Loading

**Issue**: First request timeout
- First TTS request takes 10-15 seconds
- Models download automatically (~300MB)
- Ensure stable internet connection
- Increase request timeout if needed

**Issue**: Memory errors
- TTS models require ~1-2GB RAM
- Close other applications if needed
- Consider using smaller model variants

### Network Issues

**Issue**: CORS errors (frontend)
- Backend includes CORS middleware
- Verify frontend runs on port 4200
- Check browser console for exact errors

**Issue**: Connection refused
- Verify backend is running on port 8000
- Check firewall settings
- Ensure no port conflicts

## Production Deployment

### Environment Variables

For production, set these additional variables:
```env
# Production settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# Security (optional)
API_KEY_REQUIRED=true
RATE_LIMIT_ENABLED=true
```

### Performance Optimization

```bash
# Install optional performance packages
pip install accelerate  # Faster model loading
pip install optimum     # Model optimization
```

### Docker Deployment (Advanced)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "crag_api.py"]
```

## Maintenance

### Updating Dependencies

```bash
# Update Python packages
pip install --upgrade -r requirements.txt

# Update Node.js packages
cd frontend && npm update
```

### Model Updates

```bash
# Clear model cache if needed
rm -rf ~/.cache/huggingface/transformers/

# Models will re-download automatically
```

### Log Management

```bash
# View API logs
tail -f api.log

# Clear audio files (optional)
rm -rf audio_files/*.wav
```

---

**Need help?** Check the main README troubleshooting section or create an issue.