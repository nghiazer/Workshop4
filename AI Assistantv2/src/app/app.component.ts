import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { RouterOutlet } from '@angular/router';
import { MarkdownComponent } from 'ngx-markdown';

interface Message {
  role: 'user' | 'assistant';
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

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule, HttpClientModule, FormsModule, ReactiveFormsModule, MarkdownComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent implements OnInit {
  // Chat properties
  chatId = '';
  messages: Message[] = [];
  userInput = '';
  loading = false;
  loaded = false;
  error = '';

  // Sidebar related properties
  conversations: ConversationInfo[] = [];
  sidebarOpen = true;
  loadingConversations = false;

  // URL management
  urlsInput = '';
  defaultUrls = [
    'https://lilianweng.github.io/posts/2023-06-23-agent/',
    'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/',
    'https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/'
  ];
  updatingUrls = false;

  // TTS properties
  ttsEnabled = true;
  currentAudio: HTMLAudioElement | null = null;
  speechSynthesis = window.speechSynthesis;

  // API base URL - adjust this to your CRAG API server
  private readonly API_BASE_URL = 'http://localhost:8000';

  constructor(private http: HttpClient) {
    this.generateNewChatId();
    this.urlsInput = this.defaultUrls.join('\n');
  }

  ngOnInit() {
    // Load conversations list when component initializes
    this.loadConversationsList();
    this.loaded = true; // Enable chat immediately for CRAG
  }

  /**
   * Generate new chat ID
   */
  generateNewChatId() {
    this.chatId = 'chat-' + Math.random().toString(36).substr(2, 9);
  }

  /**
   * Load list of all conversations (mock data since CRAG API doesn't have this)
   */
  loadConversationsList() {
    this.loadingConversations = true;
    
    // Mock conversations for now since CRAG API doesn't have conversation management
    setTimeout(() => {
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
   * Select conversation from sidebar
   */
  selectConversation(conversation: ConversationInfo) {
    this.chatId = conversation.chatId;
    // For CRAG, we don't need to load chat history as API is stateless
  }

  /**
   * Toggle sidebar visibility
   */
  toggleSidebar() {
    this.sidebarOpen = !this.sidebarOpen;
  }

  /**
   * Format date for display
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
   * Get short UUID for display
   */
  getShortUuid(uuid: string): string {
    return uuid.substring(0, 8);
  }

  /**
   * Send message to CRAG API
   */
  sendPrompt() {
    if (!this.userInput.trim()) {
      this.error = 'Please enter a question';
      return;
    }

    // Add user message to UI immediately
    const userMessage: Message = {
      role: 'user',
      content: this.userInput.trim(),
      timestamp: new Date()
    };
    this.messages.push(userMessage);

    const currentPrompt = this.userInput.trim();
    this.userInput = '';
    this.loading = true;
    this.error = '';

    // Send to CRAG API
    const request: AskQuestionRequest = { question: currentPrompt };

    this.http.post<AskQuestionResponse>(`${this.API_BASE_URL}/ask`, request).subscribe({
      next: (response) => {
        const assistantMessage: Message = {
          role: 'assistant',
          content: response.answer,
          timestamp: new Date(),
          audioUrl: this.generateDummyAudioUrl() // Dummy audio URL
        };
        this.messages.push(assistantMessage);
        this.loading = false;
        this.scrollToBottom();

        // Auto-play TTS if enabled
        if (this.ttsEnabled) {
          this.playTTS(assistantMessage.content, assistantMessage);
        }

        // Update conversations list
        this.loadConversationsList();
      },
      error: (err) => {
        console.error('Error sending message:', err);
        this.error = 'Error getting response from CRAG API. Please try again.';
        this.loading = false;

        // Remove the user message that was added optimistically
        this.messages.pop();

        // Restore the user input
        this.userInput = currentPrompt;
      }
    });
  }

  /**
   * Update URLs in the vector store
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
        
        // Add system message about URL update
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
   * Reset URLs to default
   */
  resetUrls() {
    this.urlsInput = this.defaultUrls.join('\n');
  }

  /**
   * Create new conversation
   */
  createNewConversation() {
    this.generateNewChatId();
    this.messages = [];
    this.error = '';
    this.userInput = '';

    // Add welcome message
    const welcomeMessage: Message = {
      role: 'assistant',
      content: `🤖 **Welcome to CRAG (Corrective RAG) Assistant!**\n\nI can help you find information from documents and the web using advanced retrieval techniques.\n\n**How it works:**\n• I first search through the knowledge base\n• If information isn't found, I search the web\n• I provide accurate, source-based answers\n• Responses can be played as audio\n\n**Try asking:**\n• "What are AI agents?"\n• "Explain prompt engineering techniques"\n• "How do adversarial attacks work on LLMs?"\n\nFeel free to ask any question!`,
      timestamp: new Date()
    };
    this.messages.push(welcomeMessage);

    this.loadConversationsList();
  }

  /**
   * Clear current conversation
   */
  clearConversation() {
    this.messages = [];
    this.generateNewChatId();
    this.error = '';
    this.userInput = '';
    this.stopCurrentAudio();
  }

  /**
   * Copy chat ID to clipboard
   */
  copyChatId() {
    if (this.chatId) {
      navigator.clipboard.writeText(this.chatId).then(() => {
        console.log('Chat ID copied to clipboard');
      });
    }
  }

  /**
   * Toggle TTS on/off
   */
  toggleTTS() {
    this.ttsEnabled = !this.ttsEnabled;
    if (!this.ttsEnabled) {
      this.stopCurrentAudio();
    }
  }

  /**
   * Play text-to-speech for message (dummy implementation)
   */
  playTTS(text: string, message: Message) {
    this.stopCurrentAudio();

    // Dummy implementation using Web Speech API
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
      this.currentAudio = utterance as any; // Type casting for consistency
    } else {
      // Fallback: simulate audio playback
      message.isPlaying = true;
      setTimeout(() => {
        message.isPlaying = false;
      }, text.length * 50); // Simulate reading time
    }
  }

  /**
   * Stop current audio playback
   */
  stopCurrentAudio() {
    if (this.speechSynthesis.speaking) {
      this.speechSynthesis.cancel();
    }
    
    this.messages.forEach(msg => {
      msg.isPlaying = false;
    });
  }

  /**
   * Toggle audio playback for specific message
   */
  toggleAudio(message: Message) {
    if (message.isPlaying) {
      this.stopCurrentAudio();
    } else {
      this.playTTS(message.content, message);
    }
  }

  /**
   * Generate dummy audio URL (for future real TTS integration)
   */
  private generateDummyAudioUrl(): string {
    return `data:audio/wav;base64,dummy-${Date.now()}`;
  }

  /**
   * Scroll to bottom of chat
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
   * Check if message contains RAG/search results
   */
  isTaskMessage(content: string): boolean {
    return content.includes('Knowledge Base Updated') ||
           content.includes('✅') ||
           content.includes('🔍') ||
           content.includes('**') ||
           content.includes('sources:');
  }

  /**
   * Track function for ngFor to improve performance
   */
  trackByMessage(index: number, message: Message): string {
    return `${message.role}-${index}-${message.content.substring(0, 50)}`;
  }

  /**
   * Track function for conversations list
   */
  trackByConversation(index: number, conversation: ConversationInfo): string {
    return conversation.chatId;
  }
}