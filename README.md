# 🤖 CRAG Assistant with Text-to-Speech

A sophisticated AI chatbot system using **Corrective RAG (Retrieval-Augmented Generation)** with **SpeechT5 Text-to-Speech** capabilities. The system intelligently retrieves information from documents, searches the web when needed, and provides audio responses.

## 🌟 Features

- **🔍 Corrective RAG**: Smart document retrieval with web search fallback
- **🗣️ Text-to-Speech**: High-quality audio responses using SpeechT5
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
                 Generate Answer    →   Web Search   →   TTS Audio
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
```

## 🎵 Text-to-Speech Features

- **High-quality synthesis** using SpeechT5
- **Natural female voice** with optimized characteristics
- **Audio controls** (play/pause/stop)
- **Fallback support** to Web Speech API
- **Auto-play option** for new responses
- **File-based audio** serving via static endpoints

## 🎯 CRAG Workflow

1. **Retrieve**: Search knowledge base for relevant documents
2. **Grade**: Assess document relevance using LLM
3. **Decision**: Use documents if relevant, otherwise search web
4. **Generate**: Create accurate, source-based response
5. **Synthesize**: Convert text to speech (optional)

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
**Fix**: Automatic deduplication implemented. Clear existing index if needed:
```python
# In API, index is auto-managed with deduplication
```

#### **TTS Dependencies Missing**
```
ModuleNotFoundError: No module named 'sentencepiece'
```
**Fix**:
```bash
pip install sentencepiece soundfile librosa
```

#### **PyTorch Version Issues**
```
PyTorch >= 2.1 is required but found 2.0.1
```
**Fix**:
```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

#### **CUDA Library Errors**
```
libcufft.so.10: cannot open shared object file
```
**Fix**: Use CPU-only PyTorch (already configured in requirements)

#### **Dataset Loading Deprecated**
```
Dataset scripts are no longer supported
```
**Fix**: Custom speaker embeddings implemented (automatically handled)

#### **Azure Content Policy Violations**
```
Response was filtered due to content management policy
```
**Fix**: 
- Rephrase questions to avoid trigger words
- Use broader, less specific terms
- Error handling implemented for graceful degradation

#### **Model Loading Timeout**
```
TTS generation failed: timeout
```
**Fix**: First request takes 10-15 seconds for model download. Wait for completion.

### Performance Optimization

#### **Memory Usage**
- TTS models: ~1-2GB RAM when loaded
- Lazy loading: Models load only when needed
- Audio files: Auto-cleanup can be implemented

#### **Response Times**
- First TTS request: 10-15 seconds (model loading)
- Subsequent requests: 3-8 seconds
- Text-only queries: 1-3 seconds

## 📁 Project Structure

```
project/
├── crag_api.py              # Main API server
├── requirements.txt         # Python dependencies
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

### Adding New Features

1. **New TTS Languages**: Extend TTSService with additional models
2. **Custom Voices**: Modify speaker embedding generation
3. **Additional RAG Sources**: Update URL management system
4. **Enhanced UI**: Modify Angular components

## 📊 Monitoring

### Health Checks
```bash
curl http://localhost:8000/health
```

### Logs
- API startup logs show model loading status
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
4. Create issue with reproduction steps

---

**🎉 Enjoy your CRAG Assistant with TTS capabilities!**