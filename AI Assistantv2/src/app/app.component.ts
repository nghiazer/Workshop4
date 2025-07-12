import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { RouterOutlet } from '@angular/router';
import { MarkdownComponent } from 'ngx-markdown';

// Interfaces for structured data in the chat application
interface Message {
  role: 'user' | 'assistant'; // Indicates who sent the message
  content: string;
  timestamp?: Date;
  audioUrl?: string;
  isPlaying?: boolean;
}

interface AskQuestionRequest {
  question: string;
}

interface AskQuestionResponse {
  answer: string;
}

interface TTSResponse {
  answer: string;
  audio_url?: string;
  has_audio: boolean;
}

interface UpdateUrlsRequest {
  urls: string[];
}

interface UpdateUrlsResponse {
  status: string;
  message: string;
}

interface ConversationInfo {
  chatId: string;
  lastModified: string;
  messageCount: number;
  preview: string;
}

// Main Angular component for the CRAG Assistant UI
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule, HttpClientModule, FormsModule, ReactiveFormsModule, MarkdownComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent implements OnInit {
  // Current conversation/chat ID
  chatId = '';
  // All messages in the current chat
  messages: Message[] = [];
  // User's input to send as a question
  userInput = '';
  // Flags for loading state and errors
  loading = false;
  loaded = false;
  error = '';

  // Sidebar: List of conversations (mocked for now)
  conversations: ConversationInfo[] = [];
  sidebarOpen = true;
  loadingConversations = false;

  // URL management for the knowledge base
  urlsInput = '';
  defaultUrls = [
    'https://lilianweng.github.io/posts/2023-06-23-agent/',
    'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/',
    'https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/'
  ];
  updatingUrls = false;

  // TTS: Enable/disable and track playback
  ttsEnabled = true;
  currentAudio: HTMLAudioElement | null = null;
  speechSynthesis = window.speechSynthesis;

  // API base URL (change as needed for your backend)
  private readonly API_BASE_URL = 'http://localhost:8000';

  constructor(private http: HttpClient) {
    // Generate a unique chat ID at startup
    this.generateNewChatId();
    // Pre-fill URLs for initial knowledge base
    this.urlsInput = this.defaultUrls.join('\n');
  }

  ngOnInit() {
    // Load sidebar conversations (mocked, as backend doesn't support this)
    this.loadConversationsList();
    this.loaded = true; // Enable chat UI immediately
  }

  /**
   * Generate a new random chat/conversation ID
   */
  generateNewChatId() {
    this.chatId = 'chat-' + Math.random().toString(36).substr(2, 9);
  }

  /**
   * Load all conversations for sidebar.
   * Since the CRAG API is stateless, this is just mock data.
   */
  loadConversationsList() {
    this.loadingConversations = true;
    setTimeout(() => {
      // Mock a single "current session" conversation
      this.conversations = [
        {
          chatId: this.chatId,
          lastModified: new Date().toISOString(),
          messageCount: this.messages.length,
          preview: 'Current RAG session'
        }
      ];
      this.loadingConversations = false;
    }, 500);
  }

  /**
   * Select a conversation (sidebar)
   * In CRAG, chat history isn't loaded, just switches ID.
   */
  selectConversation(conversation: ConversationInfo) {
    this.chatId = conversation.chatId;
    // No history to load, as API is stateless
  }

  /**
   * Show/hide sidebar
   */
  toggleSidebar() {
    this.sidebarOpen = !this.sidebarOpen;
  }

  /**
   * Format date for display in UI (today, yesterday, n days ago, etc.)
   */
  formatDate(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    if (diffDays === 1) {
      return 'Today';
    } else if (diffDays === 2) {
      return 'Yesterday';
    } else if (diffDays <= 7) {
      return `${diffDays - 1} days ago`;
    } else {
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
      });
    }
  }

  /**
   * Return a shortened version of a UUID for UI display
   */
  getShortUuid(uuid: string): string {
    return uuid.substring(0, 8);
  }

  /**
   * Send user's question to the backend API.
   * Handles TTS or text-only endpoints depending on ttsEnabled.
   */
  sendPrompt() {
    if (!this.userInput.trim()) {
      this.error = 'Please enter a question';
      return;
    }
    // Optimistically add user message to chat
    const userMessage: Message = {
      role: 'user',
      content: this.userInput.trim(),
      timestamp: new Date()
    };
    this.messages.push(userMessage);

    // Prepare request and endpoint
    const currentPrompt = this.userInput.trim();
    this.userInput = '';
    this.loading = true;
    this.error = '';
    const request: AskQuestionRequest = { question: currentPrompt };
    const endpoint = this.ttsEnabled ? '/ask_with_tts' : '/ask';

    // Send request to CRAG API
    this.http.post<TTSResponse | AskQuestionResponse>(`${this.API_BASE_URL}${endpoint}`, request).subscribe({
      next: (response) => {
        // Add assistant reply to chat
        const assistantMessage: Message = {
          role: 'assistant',
          content: response.answer,
          timestamp: new Date()
        };
        // If TTS audio is available, add URL for playback
        if ('has_audio' in response && response.has_audio && response.audio_url) {
          assistantMessage.audioUrl = `${this.API_BASE_URL}${response.audio_url}`;
          console.log('🔊 TTS audio available:', assistantMessage.audioUrl);
        }
        this.messages.push(assistantMessage);
        this.loading = false;
        this.scrollToBottom();
        // If TTS is enabled and audio is available, auto-play it
        if (this.ttsEnabled && assistantMessage.audioUrl) {
          this.playServerTTS(assistantMessage);
        }
        // Refresh sidebar conversation list
        this.loadConversationsList();
      },
      error: (err) => {
        console.error('Error sending message:', err);
        this.error = 'Error getting response from CRAG API. Please try again.';
        this.loading = false;
        // Remove optimistic user message
        this.messages.pop();
        // Restore input for retry
        this.userInput = currentPrompt;
      }
    });
  }

  /**
   * Update the knowledge base URLs.
   * Calls backend to reload document store with these URLs.
   */
  updateUrls() {
    const urls = this.urlsInput.split('\n').filter(url => url.trim()).map(url => url.trim());
    if (urls.length === 0) {
      this.error = 'Please enter at least one URL';
      return;
    }
    this.updatingUrls = true;
    this.error = '';
    const request: UpdateUrlsRequest = { urls };
    this.http.post<UpdateUrlsResponse>(`${this.API_BASE_URL}/update_urls`, request).subscribe({
      next: (response) => {
        this.updatingUrls = false;
        // Add a system message to chat to confirm update
        const systemMessage: Message = {
          role: 'assistant',
          content: `✅ **Knowledge Base Updated**\n\n${response.message}\n\nUpdated URLs:\n${urls.map(url => `• ${url}`).join('\n')}\n\nYou can now ask questions about the content from these sources.`,
          timestamp: new Date()
        };
        this.messages.push(systemMessage);
        this.scrollToBottom();
      },
      error: (err) => {
        console.error('Error updating URLs:', err);
        this.error = 'Error updating knowledge base URLs. Please try again.';
        this.updatingUrls = false;
      }
    });
  }

  /**
   * Reset URLs input to default values
   */
  resetUrls() {
    this.urlsInput = this.defaultUrls.join('\n');
  }

  /**
   * Start a new conversation (clears messages, generates new chatId, adds welcome message)
   */
  createNewConversation() {
    this.generateNewChatId();
    this.messages = [];
    this.error = '';
    this.userInput = '';
    // Add a welcome message explaining CRAG's features
    const welcomeMessage: Message = {
      role: 'assistant',
      content: `🤖 **Welcome to CRAG (Corrective RAG) Assistant!**\n\nI can help you find information from documents and the web using advanced retrieval techniques.\n\n**How it works:**\n• I first search through the knowledge base\n• If information isn't found, I search the web\n• I provide accurate, source-based answers\n• Responses can be played as audio using SpeechT5 TTS\n\n**Try asking:**\n• "What are AI agents?"\n• "Explain prompt engineering techniques"\n• "How do adversarial attacks work on LLMs?"\n\nFeel free to ask any question!`,
      timestamp: new Date()
    };
    this.messages.push(welcomeMessage);
    this.loadConversationsList();
  }

  /**
   * Clear current chat and stop audio
   */
  clearConversation() {
    this.messages = [];
    this.generateNewChatId();
    this.error = '';
    this.userInput = '';
    this.stopCurrentAudio();
  }

  /**
   * Copy chat ID to clipboard for sharing/reference
   */
  copyChatId() {
    if (this.chatId) {
      navigator.clipboard.writeText(this.chatId).then(() => {
        console.log('Chat ID copied to clipboard');
      });
    }
  }

  /**
   * Enable/disable TTS playback
   */
  toggleTTS() {
    this.ttsEnabled = !this.ttsEnabled;
    if (!this.ttsEnabled) {
      this.stopCurrentAudio();
    }
  }

  /**
   * Play server-generated TTS audio for a specific assistant message
   */
  playServerTTS(message: Message) {
    if (!message.audioUrl) {
      console.warn('No audio URL available for message');
      return;
    }
    this.stopCurrentAudio();
    try {
      this.currentAudio = new Audio(message.audioUrl);
      this.currentAudio.onloadstart = () => {
        message.isPlaying = true;
        console.log('🔊 Loading TTS audio...');
      };
      this.currentAudio.oncanplay = () => {
        console.log('🎵 TTS audio ready to play');
      };
      this.currentAudio.onplay = () => {
        message.isPlaying = true;
        console.log('▶️ TTS audio started');
      };
      this.currentAudio.onpause = () => {
        message.isPlaying = false;
        console.log('⏸️ TTS audio paused');
      };
      this.currentAudio.onended = () => {
        message.isPlaying = false;
        this.currentAudio = null;
        console.log('⏹️ TTS audio ended');
      };
      this.currentAudio.onerror = (error) => {
        message.isPlaying = false;
        this.currentAudio = null;
        console.error('❌ TTS audio error:', error);
      };
      this.currentAudio.play().catch(error => {
        console.error('❌ Failed to play TTS audio:', error);
        message.isPlaying = false;
      });
    } catch (error) {
      console.error('❌ Error creating audio element:', error);
      message.isPlaying = false;
    }
  }

  /**
   * Use browser's Web Speech API as fallback TTS (if server TTS is not available)
   */
  playWebSpeechTTS(text: string, message: Message) {
    this.stopCurrentAudio();
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 0.8;
      utterance.onstart = () => {
        message.isPlaying = true;
      };
      utterance.onend = () => {
        message.isPlaying = false;
      };
      utterance.onerror = () => {
        message.isPlaying = false;
        console.error('TTS error occurred');
      };
      this.speechSynthesis.speak(utterance);
    } else {
      console.warn('Speech synthesis not supported');
    }
  }

  /**
   * Stop all audio playback (server-side and browser TTS)
   */
  stopCurrentAudio() {
    // Stop HTML audio playback
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.currentAudio = null;
    }
    // Stop browser speech synthesis
    if (this.speechSynthesis.speaking) {
      this.speechSynthesis.cancel();
    }
    // Reset playing state for all messages
    this.messages.forEach(msg => {
      msg.isPlaying = false;
    });
  }

  /**
   * Toggle audio playback for a message (play/pause)
   */
  toggleAudio(message: Message) {
    if (message.isPlaying) {
      this.stopCurrentAudio();
    } else {
      // Use server TTS if available, fallback to browser TTS
      if (message.audioUrl) {
        this.playServerTTS(message);
      } else {
        this.playWebSpeechTTS(message.content, message);
      }
    }
  }

  /**
   * Generate dummy audio URL (not used in production)
   */
  private generateDummyAudioUrl(): string {
    return `data:audio/wav;base64,dummy-${Date.now()}`;
  }

  /**
   * Scroll chat view to bottom after new message
   */
  private scrollToBottom() {
    setTimeout(() => {
      const chatContainer = document.querySelector('.chat-messages-container');
      if (chatContainer) {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      }
    }, 100);
  }

  /**
   * Check if a message is a system/task message (for styling)
   */
  isTaskMessage(content: string): boolean {
    return content.includes('Knowledge Base Updated') ||
           content.includes('✅') ||
           content.includes('🔍') ||
           content.includes('**') ||
           content.includes('sources:');
  }

  /**
   * Angular ngFor track function for efficient rendering of messages
   */
  trackByMessage(index: number, message: Message): string {
    return `${message.role}-${index}-${message.content.substring(0, 50)}`;
  }

  /**
   * Angular ngFor track function for efficient rendering of conversations
   */
  trackByConversation(index: number, conversation: ConversationInfo): string {
    return conversation.chatId;
  }
}