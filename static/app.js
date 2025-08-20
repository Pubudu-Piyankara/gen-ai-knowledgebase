// RAG Knowledge Base Frontend JavaScript

class RAGInterface {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.checkSystemHealth();
    }

    initializeElements() {
        // Upload elements
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.fileSelectBtn = document.getElementById('file-select-btn');
        this.progressContainer = document.getElementById('progress-container');
        this.progressBar = document.getElementById('progress-bar');
        this.progressText = document.getElementById('progress-text');
        this.progressPercent = document.getElementById('progress-percent');
        this.uploadStatus = document.getElementById('upload-status');
        this.fileList = document.getElementById('file-list');

        // Query elements
        this.queryInput = document.getElementById('query-input');
        this.queryBtn = document.getElementById('query-btn');
        this.queryResults = document.getElementById('query-results');
        this.advancedToggle = document.getElementById('advanced-toggle');
        this.advancedOptions = document.getElementById('advanced-options');
        this.topKSelect = document.getElementById('top-k');
        this.includeMetadataCheckbox = document.getElementById('include-metadata');

        // Chat elements
        this.chatContainer = document.getElementById('chat-container');
        this.chatInput = document.getElementById('chat-input');
        this.chatSendBtn = document.getElementById('chat-send-btn');

        // Document management
        this.documentList = document.getElementById('document-list');
        this.refreshDocsBtn = document.getElementById('refresh-docs-btn');

        // Other elements
        this.statusIndicator = document.getElementById('status-indicator');
        this.loadingOverlay = document.getElementById('loading-overlay');
    }

    attachEventListeners() {
        // Upload functionality
        this.fileSelectBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // Query functionality
        this.queryBtn.addEventListener('click', () => this.performQuery());
        this.queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.performQuery();
        });
        this.advancedToggle.addEventListener('click', () => this.toggleAdvancedOptions());

        // Chat functionality
        this.chatSendBtn.addEventListener('click', () => this.sendChatMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendChatMessage();
        });

        // Document management
        this.refreshDocsBtn.addEventListener('click', () => this.loadDocuments());
    }

    // Drag and Drop Handlers
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        this.uploadFiles(files);
    }

    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.uploadFiles(files);
    }

    // File Upload
    async uploadFiles(files) {
        if (files.length === 0) return;

        this.showProgress();
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            await this.uploadSingleFile(file, i + 1, files.length);
        }

        this.hideProgress();
        this.loadDocuments(); // Refresh document list
    }

    async uploadSingleFile(file, current, total) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            this.updateProgress(`Uploading ${file.name}...`, (current - 1) / total * 100);

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.showUploadSuccess(file.name, result);
            this.updateProgress(`Uploaded ${file.name}`, current / total * 100);

        } catch (error) {
            console.error('Upload error:', error);
            this.showUploadError(file.name, error.message);
        }
    }

    // Query Functionality
    async performQuery() {
        const query = this.queryInput.value.trim();
        if (!query) return;

        this.showLoading();
        this.queryBtn.disabled = true;
        this.queryBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Searching...';

        try {
            const requestData = {
                query: query,
                top_k: parseInt(this.topKSelect.value),
                include_metadata: this.includeMetadataCheckbox.checked
            };

            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`Query failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayQueryResults(result);

        } catch (error) {
            console.error('Query error:', error);
            this.showError('Query failed: ' + error.message);
        } finally {
            this.hideLoading();
            this.queryBtn.disabled = false;
            this.queryBtn.innerHTML = '<i class="fas fa-search mr-2"></i>Search';
        }
    }

    // Chat Functionality
    async sendChatMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        this.addChatMessage(message, 'user');
        this.chatInput.value = '';

        // Show typing indicator
        const typingId = this.addTypingIndicator();

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: message,
                    top_k: 3,
                    include_metadata: false
                })
            });

            if (!response.ok) {
                throw new Error(`Chat failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.removeChatMessage(typingId);
            
            // Display AI response
            const aiResponse = result.response || this.generateResponseFromResults(result.results);
            this.addChatMessage(aiResponse, 'bot');

        } catch (error) {
            console.error('Chat error:', error);
            this.removeChatMessage(typingId);
            this.addChatMessage('Sorry, I encountered an error while processing your message.', 'bot');
        }
    }

    // Document Management
    async loadDocuments() {
        try {
            const response = await fetch('/api/documents');
            if (!response.ok) {
                throw new Error(`Failed to load documents: ${response.statusText}`);
            }

            const documents = await response.json();
            this.displayDocuments(documents);

        } catch (error) {
            console.error('Error loading documents:', error);
            this.documentList.innerHTML = `
                <div class="text-center text-red-500 py-4">
                    <i class="fas fa-exclamation-triangle text-2xl mb-2"></i>
                    <p>Error loading documents: ${error.message}</p>
                </div>
            `;
        }
    }

    // System Health Check
    async checkSystemHealth() {
        try {
            const response = await fetch('/health');
            if (response.ok) {
                const health = await response.json();
                this.updateSystemStatus('connected', health);
            } else {
                this.updateSystemStatus('error', null);
            }
        } catch (error) {
            this.updateSystemStatus('error', null);
        }
    }

    // UI Helper Methods
    showProgress() {
        this.progressContainer.classList.remove('hidden');
    }

    hideProgress() {
        this.progressContainer.classList.add('hidden');
        this.progressBar.style.width = '0%';
    }

    updateProgress(text, percent) {
        this.progressText.textContent = text;
        this.progressPercent.textContent = Math.round(percent) + '%';
        this.progressBar.style.width = percent + '%';
    }

    showLoading() {
        this.loadingOverlay.classList.remove('hidden');
    }

    hideLoading() {
        this.loadingOverlay.classList.add('hidden');
    }

    showUploadSuccess(filename, result) {
        const successDiv = document.createElement('div');
        successDiv.className = 'bg-green-50 border border-green-200 rounded-md p-3 mb-2';
        successDiv.innerHTML = `
            <div class="flex items-center">
                <i class="fas fa-check-circle text-green-500 mr-3"></i>
                <div class="flex-1">
                    <p class="text-sm font-medium text-green-800">${filename}</p>
                    <p class="text-xs text-green-600">
                        ${result.chunks_created} chunks created • Document ID: ${result.document_id.substring(0, 8)}...
                    </p>
                </div>
            </div>
        `;
        this.fileList.appendChild(successDiv);
    }

    showUploadError(filename, error) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'bg-red-50 border border-red-200 rounded-md p-3 mb-2';
        errorDiv.innerHTML = `
            <div class="flex items-center">
                <i class="fas fa-exclamation-circle text-red-500 mr-3"></i>
                <div class="flex-1">
                    <p class="text-sm font-medium text-red-800">${filename}</p>
                    <p class="text-xs text-red-600">${error}</p>
                </div>
            </div>
        `;
        this.fileList.appendChild(errorDiv);
    }

    displayQueryResults(result) {
        this.queryResults.innerHTML = '';

        if (result.results.length === 0) {
            this.queryResults.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <i class="fas fa-search text-2xl mb-2"></i>
                    <p>No relevant documents found for your query.</p>
                </div>
            `;
            return;
        }

        // Show AI response if available
        if (result.response) {
            const responseDiv = document.createElement('div');
            responseDiv.className = 'bg-blue-50 border border-blue-200 rounded-md p-4 mb-4';
            responseDiv.innerHTML = `
                <div class="flex items-start">
                    <i class="fas fa-robot text-blue-600 mr-3 mt-1"></i>
                    <div>
                        <h4 class="font-medium text-blue-800 mb-2">AI Response</h4>
                        <p class="text-blue-700">${result.response}</p>
                    </div>
                </div>
            `;
            this.queryResults.appendChild(responseDiv);
        }

        // Show search results
        const resultsHeader = document.createElement('div');
        resultsHeader.className = 'flex justify-between items-center mb-4';
        resultsHeader.innerHTML = `
            <h4 class="font-medium text-gray-800">Search Results (${result.total_results})</h4>
            <span class="text-sm text-gray-500">Processing time: ${result.processing_time.toFixed(3)}s</span>
        `;
        this.queryResults.appendChild(resultsHeader);

        result.results.forEach((item, index) => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'border border-gray-200 rounded-md p-4 mb-3 hover:bg-gray-50';
            resultDiv.innerHTML = `
                <div class="flex justify-between items-start mb-2">
                    <h5 class="font-medium text-gray-800">${item.metadata?.filename || 'Unknown Document'}</h5>
                    <span class="text-xs text-gray-500">Score: ${(item.score || 0).toFixed(3)}</span>
                </div>
                <p class="text-gray-700 text-sm mb-2">${item.content}</p>
                ${item.metadata && this.includeMetadataCheckbox.checked ? `
                    <div class="text-xs text-gray-500">
                        <span class="mr-4">Type: ${item.metadata.document_type || 'Unknown'}</span>
                        <span>Chunk: ${item.metadata.chunk_index || 'N/A'}</span>
                    </div>
                ` : ''}
            `;
            this.queryResults.appendChild(resultDiv);
        });
    }

    displayDocuments(documents) {
        if (!documents || (documents.total_entities === 0)) {
            this.documentList.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <i class="fas fa-file-alt text-4xl mb-4"></i>
                    <p>No documents uploaded yet. Start by uploading some files above!</p>
                </div>
            `;
            return;
        }

        this.documentList.innerHTML = `
            <div class="bg-blue-50 border border-blue-200 rounded-md p-4">
                <div class="flex items-center justify-between">
                    <div>
                        <h4 class="font-medium text-blue-800">Collection Statistics</h4>
                        <p class="text-blue-600 text-sm">Total documents: ${documents.total_entities || 0}</p>
                    </div>
                    <i class="fas fa-database text-blue-600 text-2xl"></i>
                </div>
            </div>
        `;
    }

    addChatMessage(message, sender) {
        const messageDiv = document.createElement('div');
        const isUser = sender === 'user';
        messageDiv.className = `chat-message ${isUser ? 'user-message ml-auto' : 'bot-message'} text-white p-4 rounded-lg`;
        
        messageDiv.innerHTML = `
            <div class="flex items-start">
                <i class="fas fa-${isUser ? 'user' : 'robot'} mr-3 mt-1"></i>
                <div>
                    <p>${message}</p>
                </div>
            </div>
        `;
        
        this.chatContainer.appendChild(messageDiv);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        return messageDiv;
    }

    addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message bot-message text-white p-4 rounded-lg';
        typingDiv.innerHTML = `
            <div class="flex items-start">
                <i class="fas fa-robot mr-3 mt-1"></i>
                <div class="flex items-center">
                    <div class="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        `;
        
        this.chatContainer.appendChild(typingDiv);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        return typingDiv;
    }

    removeChatMessage(messageElement) {
        if (messageElement && messageElement.parentNode) {
            messageElement.parentNode.removeChild(messageElement);
        }
    }

    generateResponseFromResults(results) {
        if (results.length === 0) {
            return "I couldn't find any relevant information in your documents for that question.";
        }

        const topResult = results[0];
        return `Based on your documents, here's what I found: ${topResult.content}`;
    }

    toggleAdvancedOptions() {
        const isHidden = this.advancedOptions.classList.contains('hidden');
        this.advancedOptions.classList.toggle('hidden');
        
        const chevron = this.advancedToggle.querySelector('.fa-chevron-down, .fa-chevron-up');
        chevron.className = isHidden ? 'fas fa-chevron-up ml-1' : 'fas fa-chevron-down ml-1';
    }

    updateSystemStatus(status, health) {
        const indicator = this.statusIndicator;
        const dot = indicator.querySelector('div');
        const text = indicator.querySelector('span').lastChild;

        if (status === 'connected') {
            dot.className = 'w-2 h-2 bg-green-500 rounded-full mr-2';
            text.textContent = 'Connected';
        } else {
            dot.className = 'w-2 h-2 bg-red-500 rounded-full mr-2';
            text.textContent = 'Disconnected';
        }
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'bg-red-50 border border-red-200 rounded-md p-4 mb-4';
        errorDiv.innerHTML = `
            <div class="flex items-center">
                <i class="fas fa-exclamation-triangle text-red-500 mr-3"></i>
                <p class="text-red-700">${message}</p>
            </div>
        `;
        this.queryResults.appendChild(errorDiv);
    }
}

// CSS for typing indicator
const style = document.createElement('style');
style.textContent = `
    .typing-indicator {
        display: flex;
        align-items: center;
    }
    .typing-indicator span {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        background-color: rgba(255, 255, 255, 0.7);
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    @keyframes typing {
        0%, 60%, 100% {
            transform: translateY(0);
        }
        30% {
            transform: translateY(-10px);
        }
    }
`;
document.head.appendChild(style);

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RAGInterface();
});
