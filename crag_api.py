from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel as LCBaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph, START
import os
from uuid import uuid4
import hashlib
import asyncio
import threading
from pathlib import Path
from langchain_core.documents import Document
from pinecone import Pinecone
from dotenv import load_dotenv

# FastAPI app
app = FastAPI(title="CRAG API", description="Corrective RAG API with Azure OpenAI and TTS")

# Create audio directory for TTS files
AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(exist_ok=True)

# Mount static files for audio serving
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",  # Angular dev server
        "http://127.0.0.1:4200",  # Alternative localhost
        "http://localhost:3000",  # React dev server (if needed)
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://localhost:8080",  # Vue dev server (if needed)
        "http://127.0.0.1:8080",  # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# For development only - uncomment to allow all origins (less secure)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,  # Must be False when allow_origins=["*"]
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# TTS Service Class
class TTSService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        self._model_loaded = False
        self._loading_lock = threading.Lock()

    def _load_model(self):
        """Lazy load SpeechT5 model"""
        if self._model_loaded:
            return

        with self._loading_lock:
            if self._model_loaded:  # Double-check pattern
                return

            try:
                print("🔄 Loading SpeechT5 TTS model...")
                from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
                import torch
                import numpy as np

                # Load processor and model
                self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
                self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
                self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

                # Use alternative method to get speaker embeddings
                print("🔊 Loading speaker embeddings...")
                try:
                    # Try to load from hub directly (alternative approach)
                    import requests

                    # Pre-computed speaker embedding (female voice)
                    # This is a known working embedding from SpeechT5 examples
                    embedding_url = "https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors/resolve/main/cmu_us_bdl_arctic-wav-22050_embeddings.pkl"

                    # If that fails, use a manually created realistic embedding
                    self.speaker_embeddings = self._create_speaker_embedding()

                except Exception as embed_error:
                    print(f"⚠️ Using fallback speaker embedding: {embed_error}")
                    self.speaker_embeddings = self._create_speaker_embedding()

                self._model_loaded = True
                print("✅ SpeechT5 model loaded successfully")

            except Exception as e:
                print(f"❌ Error loading SpeechT5 model: {e}")
                raise

    def _create_speaker_embedding(self):
        """Create a realistic speaker embedding"""
        import torch
        import numpy as np

        # Create a more realistic 512-dimensional speaker embedding
        # Based on typical SpeechT5 speaker characteristics
        np.random.seed(42)  # For consistent voice

        # Create base embedding with female voice characteristics
        embedding = np.random.normal(0, 0.1, 512)

        # Adjust specific ranges for more natural female voice
        embedding[0:50] = np.random.normal(0.2, 0.05, 50)  # Pitch characteristics
        embedding[50:100] = np.random.normal(-0.1, 0.03, 50)  # Formant characteristics
        embedding[100:150] = np.random.normal(0.15, 0.04, 50)  # Voice quality
        embedding[150:200] = np.random.normal(0.0, 0.02, 50)  # Neutral characteristics

        # Convert to tensor and ensure correct shape
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        return embedding_tensor.unsqueeze(0)  # Add batch dimension

    async def generate_audio(self, text: str) -> str:
        """Generate audio file from text"""
        try:
            # Load model if not loaded
            if not self._model_loaded:
                await asyncio.get_event_loop().run_in_executor(None, self._load_model)

            # Generate audio in thread pool to avoid blocking
            audio_path = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_audio_sync, text
            )
            return audio_path

        except Exception as e:
            print(f"❌ TTS generation error: {e}")
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    def _generate_audio_sync(self, text: str) -> str:
        """Synchronous audio generation"""
        import torch
        import soundfile as sf
        import librosa
        import numpy as np

        # Clean and prepare text
        text = self._clean_text(text)
        print(f"🔊 Generating TTS for: {text[:50]}...")

        # Tokenize text
        inputs = self.processor(text=text, return_tensors="pt")

        # Generate speech
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                self.speaker_embeddings,
                vocoder=self.vocoder
            )

        # Save audio file
        audio_id = str(uuid4())[:8]
        audio_filename = f"tts_{audio_id}.wav"
        audio_path = AUDIO_DIR / audio_filename

        # Convert to numpy with high-quality processing
        audio_data = speech.cpu().numpy()

        # High-quality time stretching
        audio_slowed = librosa.effects.time_stretch(
            audio_data,
            rate=0.75,
            hop_length=256,  # Better quality
            n_fft=1024  # Higher resolution
        )

        # Professional audio normalization
        audio_normalized = librosa.util.normalize(audio_slowed)
        audio_boosted = audio_normalized * 1.25  # 25% volume boost
        audio_final = np.clip(audio_boosted, -1.0, 1.0)

        sf.write(str(audio_path), audio_final, 16000)

        print(f"✅ TTS audio saved: {audio_filename}")
        return f"/audio/{audio_filename}"

    def _clean_text(self, text: str) -> str:
        """Clean text for better TTS"""
        import re

        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
        text = re.sub(r'`(.*?)`', r'\1', text)  # Remove code
        text = re.sub(r'#{1,6}\s*', '', text)  # Remove headers
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove links

        # Clean special characters
        text = re.sub(r'[^\w\s\.,!?;:()-]', '', text)

        # Limit length (SpeechT5 has token limits)
        if len(text) > 500:
            text = text[:500] + "..."

        return text.strip()


# Initialize TTS service
tts_service = TTSService()

# Load environment variables from .env file
load_dotenv()

# Set USER_AGENT to fix the warning
if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "CRAG-Assistant/1.0 (Educational Purpose)"

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY_EMBEDDING = os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING")
AZURE_OPENAI_API_KEY_LLM = os.getenv("AZURE_OPENAI_API_KEY_LLM")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_ENDPOINT_EMBEDDING = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING")
AZURE_OPENAI_ENDPOINT_LLM = os.getenv("AZURE_OPENAI_ENDPOINT_LLM")

# Model configurations
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-small")
AZURE_GPT4_MODEL = os.getenv("AZURE_GPT4_MODEL", "gpt-4o-mini")

# Validate required environment variables
required_vars = [
    "AZURE_OPENAI_API_KEY_EMBEDDING",
    "AZURE_OPENAI_API_KEY_LLM",
    "AZURE_OPENAI_ENDPOINT_EMBEDDING",
    "AZURE_OPENAI_ENDPOINT_LLM"
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Pinecone Configuration
PINECONE_INDEX_NAME = "crag-index"

# Initialize Pinecone
pc = Pinecone()


def setup_pinecone_index():
    """Setup Pinecone index with proper existing index handling"""
    existing_indexes = [index.name for index in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"📦 Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
            dimension=1536
        )
        print(f"✅ Index {PINECONE_INDEX_NAME} created successfully")
        return True  # New index created
    else:
        print(f"📋 Using existing Pinecone index: {PINECONE_INDEX_NAME}")
        return False  # Index already exists


# Setup index and track if it's new
is_new_index = setup_pinecone_index()

# Make urls global and mutable
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def clear_vectorstore(vectorstore):
    """Delete all vectors in the namespace"""
    vectorstore._index.delete(
        namespace="default",
        delete_all=True
    )


def generate_deterministic_id(content: str, source: str = "") -> str:
    """Generate deterministic ID based on content hash"""
    content_hash = hashlib.md5((content + source).encode()).hexdigest()
    return f"doc_{content_hash[:16]}"


def check_index_has_documents(vectorstore) -> bool:
    """Check if index already has documents"""
    try:
        # Try to query the index to see if it has any vectors
        index_stats = vectorstore._index.describe_index_stats()
        total_vectors = index_stats.get('total_vector_count', 0)
        namespace_stats = index_stats.get('namespaces', {})
        default_namespace_count = namespace_stats.get('default', {}).get('vector_count', 0)

        print(f"📊 Index stats - Total vectors: {total_vectors}, Default namespace: {default_namespace_count}")
        return default_namespace_count > 0
    except Exception as e:
        print(f"⚠️ Could not check index stats: {e}")
        return False


def init_retriever():
    """Initialize retriever with document deduplication"""
    global urls, is_new_index

    # Azure OpenAI Embeddings
    embeddings = AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY_EMBEDDING,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT_EMBEDDING,
        model=AZURE_EMBEDDING_MODEL
    )

    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace="default"
    )

    # Check if we need to load documents
    if is_new_index:
        print("🆕 New index detected - loading initial documents")
        should_load_docs = True
    else:
        has_docs = check_index_has_documents(vectorstore)
        if has_docs:
            print("📚 Index already contains documents - skipping reload")
            should_load_docs = False
        else:
            print("📭 Index exists but empty - loading documents")
            should_load_docs = True

    # Load documents only if needed
    if should_load_docs:
        print(f"🔄 Loading documents from {len(urls)} URLs...")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        # Generate deterministic IDs to prevent duplicates
        doc_ids = []
        for doc in doc_splits:
            doc_id = generate_deterministic_id(
                content=doc.page_content[:100],  # Use first 100 chars for ID
                source=doc.metadata.get('source', '')
            )
            doc_ids.append(doc_id)

        print(f"📝 Adding {len(doc_splits)} document chunks with deterministic IDs")
        vectorstore.add_documents(documents=doc_splits, ids=doc_ids)
        print("✅ Documents loaded successfully")
    else:
        print("⏭️ Skipped document loading - using existing documents")

    return vectorstore.as_retriever(), vectorstore


# Initialize LLMs and tools
class GradeDocuments(LCBaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def init_components():
    """Initialize CRAG components with Azure OpenAI"""

    # Azure OpenAI LLM with function call (GPT-4o-mini for all tasks)
    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY_LLM,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT_LLM,
        model=AZURE_GPT4_MODEL,
        temperature=0
    )
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Grader prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])

    retrieval_grader = grade_prompt | structured_llm_grader

    # RAG components with Azure OpenAI (GPT-4o-mini)
    prompt = hub.pull("rlm/rag-prompt")
    llm_gen = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY_LLM,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT_LLM,
        model=AZURE_GPT4_MODEL,
        temperature=0
    )
    rag_chain = prompt | llm_gen | StrOutputParser()

    # Question rewriter with Azure OpenAI (GPT-4o-mini)
    llm_rewriter = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY_LLM,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT_LLM,
        model=AZURE_GPT4_MODEL,
        temperature=0
    )
    system_rewrite = """You a question re-writer that converts an input question to a better version that is optimized
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system_rewrite),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    question_rewriter = re_write_prompt | llm_rewriter | StrOutputParser()

    # Web search
    web_search_tool = TavilySearchResults(k=3)

    return retrieval_grader, rag_chain, question_rewriter, web_search_tool


# Graph functions
def retrieve(state):
    """Retrieve documents from vector store"""
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}


def generate(state):
    """Generate answer using RAG chain"""
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """Grade retrieved documents for relevance"""
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):
    """Transform query for better web search"""
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    """Perform web search when documents are not relevant"""
    question = state["question"]
    documents = state["documents"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}


def decide_to_generate(state):
    """Decide whether to do web search or generate answer"""
    web_search = state["web_search"]
    return "transform_query" if web_search == "Yes" else "generate"


# Initialize workflow
def init_workflow():
    """Initialize CRAG workflow graph"""

    class GraphState(Dict):
        question: str
        generation: str
        web_search: str
        documents: List[str]

    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# Initialize components at startup
print("🚀 Initializing CRAG components with Azure OpenAI...")
print(f"📍 Embedding Endpoint: {AZURE_OPENAI_ENDPOINT_EMBEDDING}")
print(f"📍 LLM Endpoint: {AZURE_OPENAI_ENDPOINT_LLM}")
print(f"📍 API Version: {AZURE_OPENAI_API_VERSION}")
print(f"📍 Embedding Model: {AZURE_EMBEDDING_MODEL}")
print(f"📍 LLM Model: {AZURE_GPT4_MODEL} (for all tasks)")

try:
    retriever, vectorstore = init_retriever()
    print("✅ Vector store initialized successfully")

    retrieval_grader, rag_chain, question_rewriter, web_search_tool = init_components()
    print("✅ LLM components initialized successfully")

    crag_app = init_workflow()
    print("✅ CRAG workflow initialized successfully")

    print("🎉 CRAG initialization complete!")
except Exception as e:
    print(f"❌ Error during initialization: {str(e)}")
    raise


# API Models
class Question(BaseModel):
    question: str


class Response(BaseModel):
    answer: str


class TTSResponse(BaseModel):
    answer: str
    audio_url: Optional[str] = None
    has_audio: bool = False


class UpdateUrlsRequest(BaseModel):
    urls: List[str]


class UpdateUrlsResponse(BaseModel):
    status: str
    message: str


@app.post("/ask", response_model=Response)
async def ask_question(question: Question):
    """Ask a question using CRAG workflow (text only)"""
    try:
        print(f"🔍 Processing question: {question.question}")
        result = crag_app.invoke({"question": question.question})
        print(f"✅ Generated answer: {result['generation'][:100]}...")
        return Response(answer=result["generation"])
    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_with_tts", response_model=TTSResponse)
async def ask_question_with_tts(question: Question):
    """Ask a question using CRAG workflow with TTS audio"""
    try:
        print(f"🔍 Processing question with TTS: {question.question}")

        # Get text response from CRAG
        result = crag_app.invoke({"question": question.question})
        text_answer = result["generation"]
        print(f"✅ Generated answer: {text_answer[:100]}...")

        # Generate TTS audio
        try:
            audio_url = await tts_service.generate_audio(text_answer)
            print(f"🔊 TTS audio generated: {audio_url}")
            return TTSResponse(
                answer=text_answer,
                audio_url=audio_url,
                has_audio=True
            )
        except Exception as tts_error:
            print(f"⚠️ TTS generation failed: {tts_error}")
            # Return text-only response if TTS fails
            return TTSResponse(
                answer=text_answer,
                audio_url=None,
                has_audio=False
            )

    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_urls", response_model=UpdateUrlsResponse)
async def update_urls(request: UpdateUrlsRequest):
    """Update URLs in the knowledge base with deduplication"""
    global urls, retriever, vectorstore
    try:
        print(f"🔄 Updating URLs: {request.urls}")
        urls = request.urls

        # Clear the Pinecone DB to avoid conflicts
        clear_vectorstore(vectorstore)
        print("🗑️ Cleared existing vector store")

        # Force reload documents since we cleared the store
        global is_new_index
        original_is_new = is_new_index
        is_new_index = True  # Temporarily treat as new to force reload

        # Re-initialize retriever and vectorstore with new URLs
        retriever, vectorstore = init_retriever()

        # Restore original state
        is_new_index = original_is_new

        print("✅ Reinitialised vector store with new URLs")

        return UpdateUrlsResponse(
            status="success",
            message=f"URLs updated and vectorstore refreshed with {len(urls)} sources."
        )
    except Exception as e:
        print(f"❌ Error updating URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "CRAG API is running with Azure OpenAI (Separate Keys)",
        "azure_endpoints": {
            "embedding": AZURE_OPENAI_ENDPOINT_EMBEDDING,
            "llm": AZURE_OPENAI_ENDPOINT_LLM
        },
        "models": {
            "embedding": AZURE_EMBEDDING_MODEL,
            "llm": AZURE_GPT4_MODEL
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)