# 🤖 CRAG Assistant with Text-to-Speech

A sophisticated AI chatbot system using **Corrective RAG (Retrieval-Augmented Generation)** with **Coqui TTS** capabilities. The system intelligently retrieves information from documents, searches the web when needed, and provides high-quality audio responses.

## 🌟 Features

- **🔍 Corrective RAG**: Smart document retrieval with web search fallback
- **🗣️ Text-to-Speech**: High-quality audio responses using Coqui TTS
- **🌐 Web Search Integration**: Tavily search when documents are insufficient
- **📚 Knowledge Base Management**: Dynamic URL management for document sources
- **🎨 Modern UI**: Angular frontend with real-time audio controls
- **☁️ Azure OpenAI**: Separate keys for embedding and LLM models
- **🛡️ Content Safety**: Azure content policy handling

## 🏗️ Architecture

```
User Question → Document Retrieval → Relevance Grading → Decision
                     ↓                      ↓             ↓
                 Found Relevant?        Need Web Search?  Generate Answer
                     ↓                      ↓             ↓
                 Generate Answer    →   Web Search   →   Coqui TTS Audio
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- Azure OpenAI API keys
- Pinecone API key
- Tavily API key

### Backend Setup

1. **Clone repository**
```bash
git clone <your-repo>
cd <project-directory>
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.template .env
# Edit .env with your API keys
```

4. **Start backend server**
```bash
python crag_api.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install Angular dependencies**
```bash
cd frontend
npm install
```

2. **Start development server**
```bash
ng serve
```

The frontend will be available at `http://localhost:4200`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Azure OpenAI Configuration - Separate Keys
AZURE_OPENAI_API_KEY_EMBEDDING=your_embedding_api_key
AZURE_OPENAI_API_KEY_LLM=your_llm_api_key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_ENDPOINT_EMBEDDING=https://your-embedding-resource.openai.azure.com/
AZURE_OPENAI_ENDPOINT_LLM=https://your-llm-resource.openai.azure.com/

# Model Deployments
AZURE_EMBEDDING_MODEL=text-embedding-3-small
AZURE_GPT4_MODEL=gpt-4o-mini

# Vector Database
PINECONE_API_KEY=your_pinecone_api_key

# Web Search
TAVILY_API_KEY=your_tavily_api_key

# Optional
USER_AGENT=CRAG-Assistant/1.0 (Educational Purpose)
```

### Azure OpenAI Models Required

Deploy these models in your Azure OpenAI resource:
- **text-embedding-3-small**: For document embeddings
- **gpt-4o-mini**: For LLM operations (grading, generation, rewriting)

## 📖 API Endpoints

### Core Endpoints

- `POST /ask` - Text-only question answering
- `POST /ask_with_tts` - Question answering with TTS audio
- `POST /update_urls` - Update knowledge base URLs
- `GET /health` - Health check
- `GET /audio/{filename}` - Serve TTS audio files

### TTS Management Endpoints

- `POST /set_tts_model` - Change TTS model
- `GET /tts_models` - List available TTS models

### Example Usage

```bash
# Text-only query
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are AI agents?"}'

# Query with TTS
curl -X POST "http://localhost:8000/ask_with_tts" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain prompt engineering"}'

# Update knowledge base
curl -X POST "http://localhost:8000/update_urls" \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://example.com/doc1", "https://example.com/doc2"]}'

# Change TTS model
curl -X POST "http://localhost:8000/set_tts_model" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "tts_models/en/vctk/vits"}'

# List available TTS models
curl http://localhost:8000/tts_models
```

## 🎵 Text-to-Speech Features

- **High-quality synthesis** using Coqui TTS
- **Multiple voice models** available:
  - `tts_models/en/ljspeech/tacotron2-DDC` (default)
  - `tts_models/en/ljspeech/glow-tts`
  - `tts_models/en/vctk/vits`
  - `tts_models/en/sam/tacotron-DDC`
- **Runtime model switching** without restart
- **Audio controls** (play/pause/stop)
- **Fallback support** to Web Speech API
- **Auto-play option** for new responses
- **File-based audio** serving via static endpoints

### Recommended TTS Models

| Model | Quality | Speed | Description |
|-------|---------|-------|-------------|
| `tts_models/en/ljspeech/tacotron2-DDC` | High | Medium | Default, balanced quality/speed |
| `tts_models/en/ljspeech/glow-tts` | High | Fast | Flow-based, natural sounding |
| `tts_models/en/vctk/vits` | Very High | Medium | Multi-speaker, excellent quality |
| `tts_models/en/sam/tacotron-DDC` | High | Medium | Alternative voice character |

## 🎯 CRAG Workflow

1. **Retrieve**: Search knowledge base for relevant documents
2. **Grade**: Assess document relevance using LLM
3. **Decision**: Use documents if relevant, otherwise search web
4. **Generate**: Create accurate, source-based response
5. **Synthesize**: Convert text to speech using Coqui TTS (optional)

## 🔧 Troubleshooting

### Common Issues and Solutions

#### **CORS Errors**
```
Access to fetch blocked by CORS policy
```
**Fix**: CORS middleware already configured for common dev ports. Restart API server.

#### **Azure OpenAI Authentication (401)**
```
Incorrect API key provided
```
**Fixes**:
- Verify API keys in `.env` file
- Check endpoint URLs are correct
- Ensure models are deployed in Azure
- Verify API version compatibility

#### **Pinecone Conflicts**
```
Pinecone warnings about duplicate vectors
```
**Fix**: Automatic deduplication implemented. Clear existing index if needed.

#### **Coqui TTS Installation Issues**
```
ModuleNotFoundError: No module named 'TTS'
```
**Fix**:
```bash
pip install TTS
```

#### **TTS Model Download Issues**
```
Failed to download model
```
**Fix**:
- Ensure stable internet connection
- First model download can take 5-10 minutes
- Check available disk space (models ~100-500MB each)

#### **TTS Generation Timeout**
```
TTS generation failed: timeout
```
**Fixes**:
- First request takes longer for model loading
- Check available memory (models need ~500MB-1GB RAM)
- Try a different, lighter model

#### **Audio File Access Issues**
```
Cannot access audio file
```
**Fix**: 
- Check `audio_files/` directory permissions
- Verify static file serving is working
- Try different browser if CORS issues

#### **Azure Content Policy Violations**
```
Response was filtered due to content management policy
```
**Fix**: 
- Rephrase questions to avoid trigger words
- Use broader, less specific terms
- Error handling implemented for graceful degradation

### Performance Optimization

#### **Memory Usage**
- Coqui TTS models: ~500MB-1GB RAM when loaded
- Lazy loading: Models load only when needed
- Audio files: Auto-cleanup can be implemented

#### **Response Times**
- First TTS request: 5-10 seconds (model loading)
- Subsequent requests: 2-5 seconds
- Text-only queries: 1-3 seconds
- Model switching: 3-8 seconds

#### **Model Selection for Performance**
- **Fastest**: `tts_models/en/ljspeech/glow-tts`
- **Best Quality**: `tts_models/en/vctk/vits`
- **Balanced**: `tts_models/en/ljspeech/tacotron2-DDC` (default)

## 📁 Project Structure

```
project/
├── crag_api.py              # Main API server with Coqui TTS
├── requirements.txt         # Python dependencies (updated for Coqui)
├── .env.template           # Environment template
├── audio_files/            # Generated TTS audio
├── frontend/               # Angular application
│   ├── src/app/
│   │   ├── app.component.ts
│   │   ├── app.component.html
│   │   └── app.component.scss
│   └── package.json
└── README.md
```

## 🛠️ Development

### Backend Development
```bash
# Install in development mode
pip install -e .

# Run with auto-reload
uvicorn crag_api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend
ng serve --host 0.0.0.0 --port 4200
```

### Adding New TTS Models

```python
# Change model at runtime via API
import requests

response = requests.post("http://localhost:8000/set_tts_model", 
                        json={"model_name": "tts_models/en/vctk/vits"})

# Or modify default in TTSService class
self.model_name = "tts_models/en/your_preferred_model"
```

## 📊 Monitoring

### Health Checks
```bash
curl http://localhost:8000/health
```

### TTS Model Status
```bash
curl http://localhost:8000/tts_models
```

### Logs
- API startup logs show TTS model loading status
- TTS generation logs show audio file creation
- Error logs provide debugging information

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with appropriate tests
4. Submit pull request with clear description

## 📄 License

[Your License Here]

## 🆘 Support

For issues and questions:
1. Check troubleshooting section above
2. Review API logs for error details
3. Verify environment configuration
4. Test with different TTS models
5. Create issue with reproduction steps

---

**🎉 Enjoy your CRAG Assistant with Coqui TTS capabilities!**