import streamlit as st
import requests
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import time

# Configuration
API_BASE_URL = "http://localhost:8000"
DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

# Page config
st.set_page_config(
    page_title="CRAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CRAG Assistant"
    }
)

# Force dark theme với JavaScript
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Force dark mode
    function forceDarkMode() {
        const stApp = document.querySelector('.stApp');
        const sidebar = document.querySelector('[data-testid="stSidebar"]');

        if (stApp) {
            stApp.setAttribute('data-theme', 'dark');
            stApp.style.setProperty('--bg-color', '#0f172a', 'important');
            stApp.style.setProperty('--text-primary', '#f1f5f9', 'important');
            stApp.style.setProperty('--card-bg', '#1e293b', 'important');
            stApp.style.setProperty('--input-bg', '#334155', 'important');
        }

        if (sidebar) {
            sidebar.style.backgroundColor = '#1e293b';
        }
    }

    // Apply immediately and on mutations
    forceDarkMode();

    const observer = new MutationObserver(forceDarkMode);
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
""", unsafe_allow_html=True)

# Custom CSS to match Angular design
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');

    /* CSS Variables for Light Theme */
    :root {
        --primary: #667eea;
        --secondary: #14b8a6;
        --accent: #ec4899;
        --bg-color: #ffffff;
        --sidebar-bg: #f8fafc;
        --border-color: #e2e8f0;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --card-bg: #ffffff;
        --input-bg: #ffffff;
    }
    /* Force dark mode CSS */
    .stApp {
        background-color: #0f172a !important;
        color: #f1f5f9 !important;
    }
    
    .stApp [data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }
    /* Dark Theme Variables */
    [data-theme="dark"], .stApp[data-theme="dark"] {
        --bg-color: #0f172a;
        --sidebar-bg: #1e293b;
        --border-color: #334155;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --card-bg: #1e293b;
        --input-bg: #334155;
    }

    /* Detect Streamlit dark mode */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-color: #0f172a;
            --sidebar-bg: #1e293b;
            --border-color: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #cbd5e1;
            --card-bg: #1e293b;
            --input-bg: #334155;
        }
    }

    /* Main app styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: none;
    }

    /* Sidebar styling - Increase width for better text display */
    .css-1d391kg {
        width: 320px !important;
        min-width: 320px !important;
    }

    .css-1cypcdb {
        width: 320px !important;
        min-width: 320px !important;
    }

    /* Sidebar content spacing */
    .css-1lcbmhc, .css-1v0mbdj {
        width: 320px !important;
        min-width: 320px !important;
    }

    /* Chat messages - Responsive to theme */
    .user-message {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 4px solid var(--primary);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        margin-left: 20%;
        color: #1a237e !important; /* Dark blue for good contrast */
    }

    .assistant-message {
        background-color: var(--card-bg);
        border-left: 4px solid var(--secondary);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        margin-right: 20%;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color);
    }

    .system-message {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        border-left: 4px solid #20c997;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
    }

    /* Input styling - Match button height */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid var(--border-color);
        padding: 0.75rem 1rem;
        font-size: 15px;
        height: 48px;
        background-color: var(--input-bg) !important;
        color: var(--text-primary) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }

    /* Form input specific styling */
    .stTextInput input {
        background-color: var(--input-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }

    /* Button styling - Match input height */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
        height: 48px;
        font-size: 15px;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    /* Sidebar elements */
    .sidebar-header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 600;
    }

    .conversation-item {
        background-color: var(--card-bg);
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid var(--border-color);
        cursor: pointer;
        transition: all 0.2s ease;
        color: var(--text-primary);
    }

    .conversation-item:hover {
        background-color: var(--sidebar-bg);
        border-color: var(--primary);
        transform: translateY(-1px);
    }

    .conversation-item.active {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border-color: var(--primary);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }

    /* URL management */
    .url-section {
        background-color: var(--card-bg);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin-top: 1rem;
    }

    /* Text area styling */
    .stTextArea > div > div > textarea {
        background-color: var(--input-bg);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 0.25rem;
    }

    .status-active {
        background-color: rgba(20, 184, 166, 0.1);
        color: var(--secondary);
    }

    .status-inactive {
        background-color: rgba(107, 114, 128, 0.1);
        color: var(--text-secondary);
    }

    /* Welcome screen */
    .welcome-screen {
        text-align: center;
        padding: 2rem;
        background: var(--card-bg);
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }

    .welcome-screen h1 {
        color: var(--text-primary);
    }

    .welcome-screen p {
        color: var(--text-secondary);
    }

    .feature-card {
        background-color: var(--card-bg);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--secondary);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
        transition: transform 0.2s;
        color: var(--text-primary);
    }

    .feature-card:hover {
        transform: translateY(-1px);
    }

    .workflow-step {
        background-color: var(--card-bg);
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 6px;
        border-left: 3px solid var(--primary);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        color: var(--text-primary);
    }

    /* Audio controls */
    .audio-controls {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }

    /* Loading animation */
    .loading-dots {
        display: inline-block;
    }

    .loading-dots span {
        animation: blink 1.4s infinite both;
    }

    .loading-dots span:nth-child(2) {
        animation-delay: 0.2s;
    }

    .loading-dots span:nth-child(3) {
        animation-delay: 0.4s;
    }

    @keyframes blink {
        0%, 80%, 100% { opacity: 0; }
        40% { opacity: 1; }
    }

    /* Error styling */
    .error-message {
        background-color: rgba(248, 215, 218, 0.9);
        color: #721c24;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 4px solid #dc3545;
    }

    /* Success styling */
    .success-message {
        background-color: rgba(212, 237, 218, 0.9);
        color: #155724;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }

    /* Streamlit specific overrides for dark mode */
    .stMarkdown, .stMarkdown p {
        color: var(--text-primary) !important;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-primary) !important;
    }

    /* Sidebar text colors */
    .css-1d391kg .stMarkdown {
        color: white !important;
    }

    .css-1d391kg .stMarkdown h1, 
    .css-1d391kg .stMarkdown h2, 
    .css-1d391kg .stMarkdown h3 {
        color: white !important;
    }

    /* Sidebar subheader styling */
    .stSidebar .stMarkdown h3 {
        color: var(--text-primary) !important;
        background: none !important;
    }

    /* Override Streamlit default styles */
    .stSidebar [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }

    .stSidebar .stTextInput label {
        color: var(--text-primary) !important;
    }

    .stSidebar .stTextArea label {
        color: var(--text-primary) !important;
    }

    .stSidebar .stToggle label {
        color: var(--text-primary) !important;
    }

    /* Caption text */
    .stCaption {
        color: var(--text-secondary) !important;
    }

    /* Force text colors for all elements */
    .stApp {
        color: var(--text-primary);
    }

    .stApp .stMarkdown {
        color: var(--text-primary) !important;
    }

    /* Input placeholder text */
    .stTextInput input::placeholder {
        color: var(--text-secondary) !important;
        opacity: 0.7;
    }

    /* Message content text override */
    .user-message * {
        color: #1a237e !important;
    }

    .assistant-message * {
        color: var(--text-primary) !important;
    }

    .system-message * {
        color: white !important;
    }

    /* Welcome screen text */
    .welcome-screen * {
        color: var(--text-primary) !important;
    }

    .welcome-screen h1 {
        color: var(--primary) !important;
    }

    /* Session info styling */
    .session-info {
        text-align: center;
        background-color: var(--card-bg);
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
    }

    /* Message timestamp */
    .message-timestamp {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-align: right;
        margin-top: 0.5rem;
        opacity: 0.8;
    }

    /* Loading message styling */
    .loading-message {
        text-align: center;
        font-style: italic;
        color: var(--text-secondary);
        padding: 1rem;
        background-color: var(--card-bg);
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }

    /* Strong CSS overrides for Streamlit */
    .stApp, .stApp * {
        color: var(--text-primary) !important;
    }

    .stApp .stButton button {
        color: white !important;
    }

    .stApp .sidebar-header {
        color: white !important;
    }

    .stApp .user-message, .stApp .user-message * {
        color: #1a237e !important;
    }

    .stApp .system-message, .stApp .system-message * {
        color: white !important;
    }

    /* Dark mode detection for Streamlit */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: var(--bg-color);
            color: var(--text-primary) !important;
        }

        .stApp .stTextInput input {
            background-color: var(--input-bg) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-color) !important;
        }
    }

    /* Light mode overrides */
    @media (prefers-color-scheme: light) {
        .stApp {
            background-color: var(--bg-color);
            color: var(--text-primary) !important;
        }

        .stApp .stTextInput input {
            background-color: var(--input-bg) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-color) !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# Data classes
class Message:
    def __init__(self, role: str, content: str, timestamp: datetime = None, audio_url: str = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.audio_url = audio_url
        self.is_playing = False


class ConversationInfo:
    def __init__(self, chat_id: str, last_modified: str, message_count: int, preview: str):
        self.chat_id = chat_id
        self.last_modified = last_modified
        self.message_count = message_count
        self.preview = preview


# Initialize session state
def init_session_state():
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = f"chat-{str(uuid.uuid4())[:9]}"
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversations' not in st.session_state:
        st.session_state.conversations = []
    if 'tts_enabled' not in st.session_state:
        st.session_state.tts_enabled = True
    if 'loading' not in st.session_state:
        st.session_state.loading = False
    if 'error' not in st.session_state:
        st.session_state.error = ""
    if 'urls_input' not in st.session_state:
        st.session_state.urls_input = "\n".join(DEFAULT_URLS)
    if 'updating_urls' not in st.session_state:
        st.session_state.updating_urls = False
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False


# Utility functions
def generate_new_chat_id():
    return f"chat-{str(uuid.uuid4())[:9]}"


def get_short_uuid(uuid_str: str) -> str:
    return uuid_str[:8]


def format_date(date_str: str) -> str:
    try:
        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        now = datetime.now()
        diff = (now - date).days
        if diff == 0:
            return "Today"
        elif diff == 1:
            return "Yesterday"
        elif diff <= 7:
            return f"{diff} days ago"
        else:
            return date.strftime("%b %d")
    except:
        return "Unknown"


def is_task_message(content: str) -> bool:
    return ('Knowledge Base Updated' in content or
            '✅' in content or
            '🔍' in content or
            '**' in content or
            'sources:' in content)


# API functions
def send_question(question: str, use_tts: bool = False) -> Dict:
    try:
        endpoint = "/ask_with_tts" if use_tts else "/ask"
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json={"question": question},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to CRAG API. Please ensure the backend is running on http://localhost:8000")
        return None
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None


def update_urls(urls: List[str]) -> Dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/update_urls",
            json={"urls": urls},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"❌ Error updating URLs: {str(e)}")
        return None


def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


# UI Components
def render_sidebar():
    with st.sidebar:
        # Sidebar header
        st.markdown("""
        <div class="sidebar-header">
            🤖 CRAG Sessions
        </div>
        """, unsafe_allow_html=True)

        # New conversation button
        if st.button("➕ New Session", use_container_width=True):
            create_new_conversation()

        # Conversations list
        st.markdown('<h3 style="color: var(--text-primary) !important;">📋 Recent Sessions</h3>', unsafe_allow_html=True)

        if not st.session_state.conversations:
            st.markdown("""
            <div style="text-align: center; color: var(--text-secondary); padding: 1rem;">
                <div style="color: var(--text-primary);">No sessions yet</div>
                <small style="color: var(--text-secondary);">Start asking questions to begin</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            for conv in st.session_state.conversations:
                is_active = conv.chat_id == st.session_state.chat_id

                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(
                            f"🤖 {get_short_uuid(conv.chat_id)}",
                            key=f"conv_{conv.chat_id}",
                            help=f"Session: {conv.chat_id}",
                            use_container_width=True
                    ):
                        select_conversation(conv)

                with col2:
                    st.markdown(f'<small style="color: var(--text-secondary);">{conv.message_count} msgs</small>',
                                unsafe_allow_html=True)

        st.divider()

        # URL Management Section
        st.markdown('<h3 style="color: var(--text-primary) !important;">📚 Knowledge Base</h3>', unsafe_allow_html=True)

        urls_text = st.text_area(
            "URLs (one per line)",
            value=st.session_state.urls_input,
            height=100,
            disabled=st.session_state.updating_urls,
            key="urls_textarea"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                    "🔄 Update" if not st.session_state.updating_urls else "⏳ Updating...",
                    disabled=st.session_state.updating_urls,
                    use_container_width=True
            ):
                update_knowledge_base(urls_text)

        with col2:
            if st.button("↺ Reset", disabled=st.session_state.updating_urls, use_container_width=True):
                st.session_state.urls_input = "\n".join(DEFAULT_URLS)
                st.rerun()

        st.divider()

        # Settings
        st.markdown('<h3 style="color: var(--text-primary) !important;">⚙️ Settings</h3>', unsafe_allow_html=True)

        # TTS Toggle
        tts_enabled = st.toggle(
            "🔊 Text-to-Speech",
            value=st.session_state.tts_enabled,
            key="tts_toggle"
        )
        if tts_enabled != st.session_state.tts_enabled:
            st.session_state.tts_enabled = tts_enabled

        # Clear conversation
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_conversation()

        # API Status
        st.divider()
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")


def render_welcome_screen():
    st.markdown("""
    <div class="welcome-screen">
        <h1 style="color: var(--primary) !important;">🔍 Corrective RAG Assistant</h1>
        <p style="color: var(--text-secondary) !important;"><strong>Advanced AI Assistant with Retrieval-Augmented Generation and Text-to-Speech</strong></p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 style="color: var(--text-primary) !important;">🚀 Features</h3>', unsafe_allow_html=True)
        features = [
            "📚 Smart document retrieval from knowledge base",
            "🌐 Web search when information is not found locally",
            "🔊 Text-to-speech for all responses",
            "🛡️ Corrective mechanisms to prevent hallucination",
            "⚡ Real-time source grading and query rewriting"
        ]
        for feature in features:
            st.markdown(f'<div class="feature-card" style="color: var(--text-primary) !important;">{feature}</div>',
                        unsafe_allow_html=True)

    with col2:
        st.markdown('<h3 style="color: var(--text-primary) !important;">🔄 How CRAG Works</h3>', unsafe_allow_html=True)
        steps = [
            "1️⃣ **Retrieve:** Search knowledge base for relevant documents",
            "2️⃣ **Grade:** Assess relevance of retrieved documents",
            "3️⃣ **Decision:** Use documents or search web if irrelevant",
            "4️⃣ **Generate:** Provide accurate, source-based answer",
            "5️⃣ **Speak:** Convert answer to speech (optional)"
        ]
        for step in steps:
            st.markdown(f'<div class="workflow-step" style="color: var(--text-primary) !important;">{step}</div>',
                        unsafe_allow_html=True)

    st.markdown('<h3 style="color: var(--text-primary) !important;">💬 Try asking:</h3>', unsafe_allow_html=True)
    examples = [
        '"What are AI agents and how do they work?"',
        '"Explain different prompt engineering techniques"',
        '"How do adversarial attacks target large language models?"',
        '"What is the difference between fine-tuning and RAG?"'
    ]

    for example in examples:
        st.markdown(f'<div style="color: var(--text-primary) !important;">- {example}</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start New Session", use_container_width=True, key="start_session"):
            create_new_conversation()


def render_chat_messages():
    if not st.session_state.messages:
        render_welcome_screen()
        return

    # Session info
    st.markdown(f"""
    <div class="session-info">
        <b style="color: var(--text-primary);">Session:</b> {get_short_uuid(st.session_state.chat_id)} | 
        <span class="status-indicator status-active">🔍 RAG Active</span>
        <span class="status-indicator {'status-active' if st.session_state.tts_enabled else 'status-inactive'}">
            {'🔊 TTS On' if st.session_state.tts_enabled else '🔇 TTS Off'}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Messages container
    for i, msg in enumerate(st.session_state.messages):
        if msg.role == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong style="color: #1a237e;">👤 You</strong>
                <div style="margin-top: 0.5rem; color: #1a237e;">{msg.content}</div>
                <div class="message-timestamp" style="color: #64748b;">
                    {msg.timestamp.strftime('%H:%M:%S')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif msg.role == "assistant":
            message_class = "system-message" if is_task_message(msg.content) else "assistant-message"
            text_color = "white" if is_task_message(msg.content) else "var(--text-primary)"

            with st.container():
                st.markdown(f"""
                <div class="{message_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <strong style="color: {text_color};">🤖 CRAG Assistant</strong>
                    </div>
                    <div style="color: {text_color};">
                """, unsafe_allow_html=True)

                # Message content
                st.markdown(msg.content)

                st.markdown(f"""
                    </div>
                """, unsafe_allow_html=True)

                # Audio controls and timestamp
                col1, col2 = st.columns([3, 1])
                with col1:
                    if msg.audio_url and st.session_state.tts_enabled:
                        if st.button(f"🔊 Play Audio", key=f"audio_{i}"):
                            st.audio(f"{API_BASE_URL}{msg.audio_url}")

                with col2:
                    timestamp_color = "rgba(255,255,255,0.8)" if is_task_message(
                        msg.content) else "var(--text-secondary)"
                    st.markdown(
                        f'<div class="message-timestamp" style="color: {timestamp_color};">{msg.timestamp.strftime("%H:%M:%S")}</div>',
                        unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

    # Loading indicator
    if st.session_state.loading:
        st.markdown("""
        <div class="loading-message">
            <div style="color: var(--text-primary);">🔍 CRAG is processing your question</div>
            <div class="loading-dots">
                <span>.</span><span>.</span><span>.</span>
            </div>
            <div style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8; color: var(--text-secondary);">
                Searching documents and web if needed...
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_chat_input():
    if st.session_state.error:
        st.markdown(f"""
        <div class="error-message">
            <span style="color: #721c24;">⚠️ {st.session_state.error}</span>
        </div>
        """, unsafe_allow_html=True)

    # Chat input with proper form handling for Enter key
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Ask me anything about AI, ML, or the documents in the knowledge base... (Press Enter to send)",
                disabled=st.session_state.loading,
                key="user_input_form",
                label_visibility="collapsed"
            )

        with col2:
            send_button = st.form_submit_button(
                "🔍 Ask CRAG" if not st.session_state.loading else "⏳ Processing...",
                disabled=st.session_state.loading,
                use_container_width=True
            )

        # Handle form submission (Enter key or button click)
        if send_button and user_input and user_input.strip():
            send_message(user_input.strip())


# Core functions
def create_new_conversation():
    st.session_state.chat_id = generate_new_chat_id()
    st.session_state.messages = []
    st.session_state.error = ""
    st.session_state.loading = False

    # Add welcome message
    welcome_msg = Message(
        role="assistant",
        content="""🤖 **Welcome to CRAG (Corrective RAG) Assistant!**

I can help you find information from documents and the web using advanced retrieval techniques.

**How it works:**
• I first search through the knowledge base
• If information isn't found, I search the web  
• I provide accurate, source-based answers
• Responses can be played as audio using SpeechT5 TTS

**Try asking:**
• "What are AI agents?"
• "Explain prompt engineering techniques"  
• "How do adversarial attacks work on LLMs?"

Feel free to ask any question!"""
    )
    st.session_state.messages.append(welcome_msg)
    update_conversations_list()
    st.rerun()


def select_conversation(conversation: ConversationInfo):
    st.session_state.chat_id = conversation.chat_id
    # Note: In a real app, you'd load the conversation history here
    # Since the API is stateless, we just switch the ID
    st.rerun()


def clear_conversation():
    st.session_state.messages = []
    st.session_state.chat_id = generate_new_chat_id()
    st.session_state.error = ""
    st.session_state.loading = False
    update_conversations_list()
    st.rerun()


def send_message(message: str):
    # Add user message
    user_msg = Message(role="user", content=message)
    st.session_state.messages.append(user_msg)

    # Set loading state
    st.session_state.loading = True
    st.session_state.error = ""

    # Immediate rerun to show user message and loading state
    st.rerun()


def update_knowledge_base(urls_text: str):
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]

    if not urls:
        st.session_state.error = "Please enter at least one URL"
        return

    st.session_state.updating_urls = True
    st.session_state.error = ""
    st.rerun()

    response = update_urls(urls)

    if response and response.get("status") == "success":
        # Add success message to chat
        system_msg = Message(
            role="assistant",
            content=f"""✅ **Knowledge Base Updated**

{response["message"]}

**Updated URLs:**
{chr(10).join([f"• {url}" for url in urls])}

You can now ask questions about the content from these sources."""
        )
        st.session_state.messages.append(system_msg)
        st.session_state.urls_input = urls_text

        st.success("✅ Knowledge base updated successfully!")
    else:
        st.session_state.error = "Failed to update knowledge base. Please try again."

    st.session_state.updating_urls = False
    st.rerun()


def update_conversations_list():
    # Update the current conversation in the list
    current_conv = ConversationInfo(
        chat_id=st.session_state.chat_id,
        last_modified=datetime.now().isoformat(),
        message_count=len(st.session_state.messages),
        preview="Current RAG session"
    )

    # Update or add to conversations list
    found = False
    for i, conv in enumerate(st.session_state.conversations):
        if conv.chat_id == st.session_state.chat_id:
            st.session_state.conversations[i] = current_conv
            found = True
            break

    if not found:
        st.session_state.conversations.insert(0, current_conv)

    # Keep only last 10 conversations
    st.session_state.conversations = st.session_state.conversations[:10]


# Main app
def main():
    init_session_state()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white !important;">🤖 CRAG Assistant</h1>
        <p style="color: white !important;">Corrective RAG with Text-to-Speech</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize with welcome conversation if needed
    if not st.session_state.initialized:
        if not st.session_state.messages:
            create_new_conversation()
        st.session_state.initialized = True

    # Handle pending message sending
    if st.session_state.loading and st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg.role == "user":
            # Continue with API call
            response = send_question(last_msg.content, st.session_state.tts_enabled)

            if response:
                assistant_msg = Message(
                    role="assistant",
                    content=response["answer"]
                )

                if "has_audio" in response and response["has_audio"] and response.get("audio_url"):
                    assistant_msg.audio_url = response["audio_url"]

                st.session_state.messages.append(assistant_msg)
                update_conversations_list()

                # Show success feedback
                st.success("✅ Response received!")
                time.sleep(0.5)  # Brief pause to show success
            else:
                st.session_state.messages.pop()
                st.session_state.error = "Failed to get response from CRAG API. Please try again."

            st.session_state.loading = False
            st.rerun()

    # Layout
    render_sidebar()

    # Main content
    render_chat_messages()
    render_chat_input()


if __name__ == "__main__":
    main()