# Import necessary libraries for FastAPI, AI, and utility functions
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

# Initialize the FastAPI app for your AI backend
app = FastAPI(title="CRAG API", description="Corrective RAG API with Azure OpenAI and Coqui TTS")

# Create a directory to store generated audio files for TTS responses
AUDIO_DIR = Path("audio_files")
AUDIO_DIR.mkdir(exist_ok=True)

# Serve audio files via HTTP for easy access by the frontend
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# Add CORS middleware so your frontend web app can communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",  # Angular
        "http://127.0.0.1:4200",  # Alternative Angular
        "http://localhost:3000",  # React
        "http://127.0.0.1:3000",  # Alternative React
        "http://localhost:8080",  # Vue
        "http://127.0.0.1:8080",  # Alternative Vue
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# TTSService provides text-to-speech functionality using Coqui TTS
class TTSService:
    def __init__(self):
        # Model components are loaded only when needed to save memory and startup time
        self.tts = None
        self._model_loaded = False
        self._loading_lock = threading.Lock()  # Prevent race conditions on model load

        # Available Coqui TTS models - you can change this to other models
        self.model_name = "tts_models/en/ljspeech/tacotron2-DDC"

        # Alternative models you can try:
        # self.model_name = "tts_models/en/ljspeech/glow-tts"
        # self.model_name = "tts_models/en/vctk/vits"
        # self.model_name = "tts_models/en/sam/tacotron-DDC"

    def _load_model(self):
        """Load the Coqui TTS model."""
        if self._model_loaded:
            return

        with self._loading_lock:
            if self._model_loaded:
                return

            try:
                print("🔄 Loading Coqui TTS model...")
                from TTS.api import TTS

                # Initialize TTS with the selected model
                self.tts = TTS(model_name=self.model_name, progress_bar=False)

                self._model_loaded = True
                print(f"✅ Coqui TTS model loaded successfully: {self.model_name}")

            except Exception as e:
                print(f"❌ Error loading Coqui TTS model: {e}")
                print("💡 Make sure you have installed TTS: pip install TTS")
                raise

    async def generate_audio(self, text: str) -> str:
        """Generate a speech audio file from text, using Coqui TTS."""
        try:
            # Load the TTS model if it's not already loaded
            if not self._model_loaded:
                await asyncio.get_event_loop().run_in_executor(None, self._load_model)

            # Generate audio in a background thread to avoid blocking the main async loop
            audio_path = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_audio_sync, text
            )
            return audio_path

        except Exception as e:
            print(f"❌ TTS generation error: {e}")
            raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    def _generate_audio_sync(self, text: str) -> str:
        """Synchronously generate and save speech audio from text using Coqui TTS."""
        try:
            # Clean up text for speech
            text = self._clean_text(text)
            print(f"🔊 Generating TTS for: {text[:50]}...")

            # Generate unique filename for this audio
            audio_id = str(uuid4())[:8]
            audio_filename = f"tts_{audio_id}.wav"
            audio_path = AUDIO_DIR / audio_filename

            # Generate speech using Coqui TTS
            self.tts.tts_to_file(
                text=text,
                file_path=str(audio_path)
            )

            print(f"✅ TTS audio saved: {audio_filename}")
            return f"/audio/{audio_filename}"

        except Exception as e:
            print(f"❌ Error in TTS generation: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """Remove markdown and special characters for clearer speech."""
        import re

        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # *italic*
        text = re.sub(r'`(.*?)`', r'\1', text)  # `code`
        text = re.sub(r'#{1,6}\s*', '', text)  # Headers
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # [text](link)

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:()-]', '', text)

        # Limit length for TTS
        if len(text) > 500:
            text = text[:500] + "..."

        return text.strip()

    def set_model(self, model_name: str):
        """Change the TTS model (requires reloading)."""
        if model_name != self.model_name:
            self.model_name = model_name
            self._model_loaded = False
            self.tts = None
            print(f"🔄 TTS model changed to: {model_name}")

    def get_available_models(self):
        """Get list of available Coqui TTS models."""
        try:
            from TTS.api import TTS
            return TTS.list_models()
        except Exception as e:
            print(f"❌ Error getting available models: {e}")
            return []


# Instantiate the TTS service at startup
tts_service = TTSService()

# Load environment variables for API keys, endpoints, and configs
load_dotenv()
if not os.getenv("USER_AGENT"):
    os.environ["USER_AGENT"] = "CRAG-Assistant/1.0 (Educational Purpose)"

# Validate environment variables for Azure OpenAI
AZURE_OPENAI_API_KEY_EMBEDDING = os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING")
AZURE_OPENAI_API_KEY_LLM = os.getenv("AZURE_OPENAI_API_KEY_LLM")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_ENDPOINT_EMBEDDING = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING")
AZURE_OPENAI_ENDPOINT_LLM = os.getenv("AZURE_OPENAI_ENDPOINT_LLM")
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-small")
AZURE_GPT4_MODEL = os.getenv("AZURE_GPT4_MODEL", "gpt-4o-mini")

# Ensure all required Azure keys are present, or fail early
required_vars = [
    "AZURE_OPENAI_API_KEY_EMBEDDING",
    "AZURE_OPENAI_API_KEY_LLM",
    "AZURE_OPENAI_ENDPOINT_EMBEDDING",
    "AZURE_OPENAI_ENDPOINT_LLM"
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Pinecone vector DB configuration for retrieval-augmented generation (RAG)
PINECONE_INDEX_NAME = "crag-index"
pc = Pinecone()


def setup_pinecone_index():
    """Check if the Pinecone index exists; create it if missing."""
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


# Track whether the index was created fresh or already exists
is_new_index = setup_pinecone_index()

# Default URLs for initial knowledge base
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def clear_vectorstore(vectorstore):
    """Remove all vectors from the index (used when updating URLs)."""
    vectorstore._index.delete(
        namespace="default",
        delete_all=True
    )


def generate_deterministic_id(content: str, source: str = "") -> str:
    """Generate a unique, repeatable ID for each document chunk."""
    content_hash = hashlib.md5((content + source).encode()).hexdigest()
    return f"doc_{content_hash[:16]}"


def check_index_has_documents(vectorstore) -> bool:
    """Check if the Pinecone index has any documents loaded."""
    try:
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
    """Set up the document retriever and load documents if necessary."""
    global urls, is_new_index
    # Set up AzureOpenAI embeddings for semantic document search
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

    # Decide whether to reload documents based on index state
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

    # Load documents from URLs if needed
    if should_load_docs:
        print(f"🔄 Loading documents from {len(urls)} URLs...")
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)
        doc_ids = []
        for doc in doc_splits:
            doc_id = generate_deterministic_id(
                content=doc.page_content[:100],
                source=doc.metadata.get('source', '')
            )
            doc_ids.append(doc_id)
        print(f"📝 Adding {len(doc_splits)} document chunks with deterministic IDs")
        vectorstore.add_documents(documents=doc_splits, ids=doc_ids)
        print("✅ Documents loaded successfully")
    else:
        print("⏭️ Skipped document loading - using existing documents")
    return vectorstore.as_retriever(), vectorstore


# Define a Pydantic model for grading document relevance
class GradeDocuments(LCBaseModel):
    """Used by the LLM to return a binary score for document relevance."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def init_components():
    """Set up the main CRAG components: LLMs, prompts, retrievers, graders."""
    # Set up the main Azure OpenAI LLM for all tasks
    llm = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY_LLM,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT_LLM,
        model=AZURE_GPT4_MODEL,
        temperature=0
    )
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    # Prompt for document grading
    system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    retrieval_grader = grade_prompt | structured_llm_grader

    # RAG prompt chain for generating answers based on context
    prompt = hub.pull("rlm/rag-prompt")
    llm_gen = AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY_LLM,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT_LLM,
        model=AZURE_GPT4_MODEL,
        temperature=0
    )
    rag_chain = prompt | llm_gen | StrOutputParser()

    # LLM for rewriting queries for better web search (improves retrieval)
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

    # Web search tool for fallback information retrieval
    web_search_tool = TavilySearchResults(k=3)

    return retrieval_grader, rag_chain, question_rewriter, web_search_tool


# Main logic functions for each workflow step
def retrieve(state):
    """Retrieve documents from the vector store based on the user question."""
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}


def generate(state):
    """Generate an answer to the question using the RAG chain and retrieved documents."""
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """Grade the relevance of each document and decide if web search is needed."""
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
    """Rewrite the question to improve its suitability for web search."""
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    """Perform a web search if the knowledge base is insufficient."""
    question = state["question"]
    documents = state["documents"]
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}


def decide_to_generate(state):
    """Decide whether to generate an answer or perform a web search next."""
    web_search = state["web_search"]
    return "transform_query" if web_search == "Yes" else "generate"


def init_workflow():
    """Define the CRAG workflow as a directed graph of steps."""

    class GraphState(Dict):
        question: str
        generation: str
        web_search: str
        documents: List[str]

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)
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


# Initialization block: sets up the retriever, LLMs, workflow, etc.
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


# API Models: define data formats for requests and responses
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


class TTSModelRequest(BaseModel):
    model_name: str


class TTSModelResponse(BaseModel):
    status: str
    message: str
    current_model: str


# Main API endpoints
@app.post("/ask", response_model=Response)
async def ask_question(question: Question):
    """Handle text-only question requests (no TTS)."""
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
    """Handle requests for both text and TTS (audio) answers."""
    try:
        print(f"🔍 Processing question with TTS: {question.question}")
        result = crag_app.invoke({"question": question.question})
        text_answer = result["generation"]
        print(f"✅ Generated answer: {text_answer[:100]}...")
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
    """Update the knowledge base URLs and reload the vector store."""
    global urls, retriever, vectorstore
    try:
        print(f"🔄 Updating URLs: {request.urls}")
        urls = request.urls
        clear_vectorstore(vectorstore)
        print("🗑️ Cleared existing vector store")
        global is_new_index
        original_is_new = is_new_index
        is_new_index = True
        retriever, vectorstore = init_retriever()
        is_new_index = original_is_new
        print("✅ Reinitialised vector store with new URLs")
        return UpdateUrlsResponse(
            status="success",
            message=f"URLs updated and vectorstore refreshed with {len(urls)} sources."
        )
    except Exception as e:
        print(f"❌ Error updating URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_tts_model", response_model=TTSModelResponse)
async def set_tts_model(request: TTSModelRequest):
    """Change the TTS model."""
    try:
        tts_service.set_model(request.model_name)
        return TTSModelResponse(
            status="success",
            message=f"TTS model changed to: {request.model_name}",
            current_model=request.model_name
        )
    except Exception as e:
        print(f"❌ Error setting TTS model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts_models")
async def get_tts_models():
    """Get list of available TTS models."""
    try:
        models = tts_service.get_available_models()
        return {
            "status": "success",
            "current_model": tts_service.model_name,
            "available_models": models
        }
    except Exception as e:
        print(f"❌ Error getting TTS models: {str(e)}")
        return {
            "status": "error",
            "current_model": tts_service.model_name,
            "available_models": [],
            "error": str(e)
        }


@app.get("/health")
async def health_check():
    """Health check endpoint to confirm API status and model configs."""
    return {
        "status": "healthy",
        "message": "CRAG API is running with Azure OpenAI and Coqui TTS",
        "azure_endpoints": {
            "embedding": AZURE_OPENAI_ENDPOINT_EMBEDDING,
            "llm": AZURE_OPENAI_ENDPOINT_LLM
        },
        "models": {
            "embedding": AZURE_EMBEDDING_MODEL,
            "llm": AZURE_GPT4_MODEL,
            "tts": tts_service.model_name
        }
    }


# Local run entrypoint for development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)