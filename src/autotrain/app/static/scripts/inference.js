document.addEventListener('DOMContentLoaded', () => {
    const modelListContainer = document.getElementById('model-list-container');
    const refreshBtn = document.getElementById('refresh-models');
    const modelHeader = document.getElementById('model-header');
    const selectedModelName = document.getElementById('selected-model-name');
    const selectedModelType = document.getElementById('selected-model-type');
    const inferenceInterface = document.getElementById('inference-interface');
    const rightPanel = document.getElementById('right-panel');
    const modelDescription = document.getElementById('model-description');

    let currentModel = null;

    // Helper function to get HF token from session or env
    function getHFToken() {
        return sessionStorage.getItem('hf_token') || window.HF_TOKEN;
    }

    // Helper function to add auth headers
    function addAuthHeaders(headers = {}) {
        const token = getHFToken();
        if (token && token !== "None") {
            headers['Authorization'] = `Bearer ${token}`;
        }
        return headers;
    }

    // Templates
    const templates = {
        modelItem: document.getElementById('model-item-template'),
        llm: document.getElementById('llm-interface-template'),
        textClassification: document.getElementById('text-classification-interface-template'),
        imageUpload: document.getElementById('image-upload-template')
    };

    // Inference Settings State
    const inferenceSettings = {
        temperature: 0.7,
        maxTokens: 256,  // Reasonable default - early stopping handles short responses
        topP: 0.95,
        topK: 50,
        doSample: true,
        systemPrompt: ''
    };

    // Initialize settings controls
    function initializeSettingsControls() {
        const tempSlider = document.getElementById('temperature');
        const tokensSlider = document.getElementById('max-tokens');
        const topPSlider = document.getElementById('top-p');
        const topKSlider = document.getElementById('top-k');
        const doSampleToggle = document.getElementById('do-sample');
        const systemPromptInput = document.getElementById('system-prompt');
        const resetBtn = document.getElementById('reset-settings');

        // Update displayed values
        function updateDisplayValue(id, value) {
            const display = document.getElementById(id);
            if (display) display.textContent = value;
        }

        // Temperature slider
        if (tempSlider) {
            tempSlider.addEventListener('input', (e) => {
                inferenceSettings.temperature = parseFloat(e.target.value);
                updateDisplayValue('temp-value', inferenceSettings.temperature);
            });
        }

        // Max tokens slider
        if (tokensSlider) {
            tokensSlider.addEventListener('input', (e) => {
                inferenceSettings.maxTokens = parseInt(e.target.value);
                updateDisplayValue('tokens-value', inferenceSettings.maxTokens);
            });
        }

        // Top P slider
        if (topPSlider) {
            topPSlider.addEventListener('input', (e) => {
                inferenceSettings.topP = parseFloat(e.target.value);
                updateDisplayValue('topp-value', inferenceSettings.topP);
            });
        }

        // Top K slider
        if (topKSlider) {
            topKSlider.addEventListener('input', (e) => {
                inferenceSettings.topK = parseInt(e.target.value);
                updateDisplayValue('topk-value', inferenceSettings.topK);
            });
        }

        // Do Sample toggle
        if (doSampleToggle) {
            doSampleToggle.addEventListener('change', (e) => {
                inferenceSettings.doSample = e.target.checked;
                updateDisplayValue('sampling-value', e.target.checked ? 'On' : 'Off');
            });
        }

        // System prompt
        if (systemPromptInput) {
            systemPromptInput.addEventListener('input', (e) => {
                inferenceSettings.systemPrompt = e.target.value;
            });
        }

        // Reset button
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                inferenceSettings.temperature = 0.7;
                inferenceSettings.maxTokens = 512;
                inferenceSettings.topP = 0.95;
                inferenceSettings.topK = 50;
                inferenceSettings.doSample = true;
                inferenceSettings.systemPrompt = '';

                if (tempSlider) tempSlider.value = 0.7;
                if (tokensSlider) tokensSlider.value = 512;
                if (topPSlider) topPSlider.value = 0.95;
                if (topKSlider) topKSlider.value = 50;
                if (doSampleToggle) doSampleToggle.checked = true;
                if (systemPromptInput) systemPromptInput.value = '';

                updateDisplayValue('temp-value', 0.7);
                updateDisplayValue('tokens-value', 512);
                updateDisplayValue('topp-value', 0.95);
                updateDisplayValue('topk-value', 50);
                updateDisplayValue('sampling-value', 'On');
            });
        }
    }

    // Call this after DOM is ready
    initializeSettingsControls();

    // Fetch and display models
        async function fetchModels() {
        modelListContainer.innerHTML = '<div class="text-center text-gray-500 text-sm mt-4">Loading models...</div>';
        try {
            const headers = addAuthHeaders();
            const response = await fetch('/api/models/list', { headers });
            const models = await response.json();
            
            modelListContainer.innerHTML = '';
            if (models.length === 0) {
                modelListContainer.innerHTML = '<div class="text-center text-gray-500 text-sm mt-4">No models found</div>';
                return;
            }

            // Group models by type
            const groupedModels = {};
            models.forEach(model => {
                if (!groupedModels[model.type]) {
                    groupedModels[model.type] = [];
                }
                groupedModels[model.type].push(model);
            });

            // Render groups
            for (const [type, typeModels] of Object.entries(groupedModels)) {
                const groupHeader = document.createElement('div');
                groupHeader.className = 'px-2 py-1 text-xs font-semibold text-gray-500 uppercase tracking-wider mt-2 border-b dark:border-gray-700 mb-1';
                groupHeader.textContent = type;
                modelListContainer.appendChild(groupHeader);

                typeModels.forEach(model => {
                    const clone = templates.modelItem.content.cloneNode(true);
                    const item = clone.querySelector('.model-item');
                    clone.querySelector('.model-id').textContent = model.id;
                    clone.querySelector('.model-type').textContent = new Date(model.metadata.created_at * 1000).toLocaleString();
                    
                    item.addEventListener('click', () => selectModel(model));
                    modelListContainer.appendChild(clone);
                });
            }
        } catch (error) {
            console.error('Error fetching models:', error);
            modelListContainer.innerHTML = '<div class="text-center text-red-500 text-sm mt-4">Error loading models</div>';
        }
    }

    // Select a model
    function selectModel(model) {
        currentModel = model;
        
        // Update sidebar selection style
        document.querySelectorAll('.model-item').forEach(el => {
            el.classList.remove('bg-blue-100', 'dark:bg-blue-900');
        });
        
        // Update Header
        modelHeader.classList.remove('hidden');
        selectedModelName.textContent = model.id;
        selectedModelType.textContent = model.type;
        
        // Show Right Panel
        rightPanel.classList.remove('hidden');
        modelDescription.textContent = `Type: ${model.type}\nPath: ${model.model_path}`;

        // Show/hide inference settings based on model type
        const inferenceSettingsDiv = document.getElementById('inference-settings');
        console.log('Model type:', model.type, 'Settings div found:', !!inferenceSettingsDiv);
        // Show settings for generative models (llm and seq2seq)
        if (model.type === 'llm' || model.type === 'seq2seq') {
            if (inferenceSettingsDiv) {
                inferenceSettingsDiv.classList.remove('hidden');
                console.log('Removed hidden class from inference settings');
            }
        } else {
            if (inferenceSettingsDiv) inferenceSettingsDiv.classList.add('hidden');
        }

        // Clear Interface
        inferenceInterface.innerHTML = '';

        // Load appropriate interface
        if (model.type === 'llm') {
            loadLLMInterface();
        } else if (model.type === 'text-classification' || model.type === 'token-classification' || model.type === 'text-regression' || model.type === 'seq2seq') {
            loadTextClassificationInterface();
        } else if (model.type === 'extractive-question-answering') {
            loadQuestionAnsweringInterface();
        } else if (model.type === 'sentence-transformers') {
            loadSentenceTransformersInterface();
        } else if (model.type === 'image-classification' || model.type === 'image-object-detection' || model.type === 'image-regression' || model.type === 'vlm') {
            loadImageInterface();
        } else if (model.type === 'tabular') {
            loadTabularInterface();
        } else {
            inferenceInterface.innerHTML = `<div class="text-center text-gray-500">
                <i class="fas fa-exclamation-triangle text-4xl mb-4"></i>
                <p>Interface for ${model.type} not implemented yet.</p>
            </div>`;
        }
    }

    // --- LLM Interface ---
    function loadLLMInterface() {
        const clone = templates.llm.content.cloneNode(true);
        inferenceInterface.innerHTML = ''; // Clear previous interface content properly before appending new one.
        // Wait, selectModel clears it. But loadDefaultInterface calls this too.
        // I should be careful.
        
        // Note: cloneNode returns DocumentFragment.
        // If inferenceInterface already has content (e.g. default interface), we should clear it.
        // selectModel clears it.
        
        if (inferenceInterface.innerHTML !== '') {
             inferenceInterface.innerHTML = '';
        }
        
        inferenceInterface.appendChild(clone);
        
        const chatHistory = document.getElementById('chat-history');
        const input = document.getElementById('llm-input');
        const sendBtn = document.getElementById('llm-send-btn');
        
        let currentConversation = {
            timestamp: Date.now(),
            messages: []
        };

        // Load previous conversations if model is selected
        if (currentModel) {
            loadConversations();
        } else {
            // Default state
             appendMessage('assistant', 'Please select a model to start chatting.');
             input.disabled = true;
             sendBtn.disabled = true;
        }

        async function loadConversations() {
            if (!currentModel) return;
            try {
                const headers = addAuthHeaders();
                // Use query parameter for model_id to avoid path routing issues
                const params = new URLSearchParams({ model_id: currentModel.id });
                const response = await fetch(`/api/conversations?${params}`, { headers });

                if (!response.ok) {
                    console.warn('No conversations found for this model');
                    return;
                }

                const conversations = await response.json();

                // Ensure conversations is an array
                if (!Array.isArray(conversations)) {
                    console.warn('Conversations response is not an array:', conversations);
                    return;
                }

                const listContainer = document.getElementById('example-inputs');
                if(listContainer) {
                    listContainer.innerHTML = ''; // Reuse example-inputs container in right panel for history

                    if (conversations.length > 0) {
                        const title = document.createElement('h3');
                        title.className = 'font-bold mb-2 mt-4';
                        title.textContent = 'History';
                        listContainer.appendChild(title);
                    }

                    conversations.forEach(conv => {
                        const div = document.createElement('div');
                        div.className = 'text-xs p-2 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer rounded truncate';
                        // Timestamp is already in milliseconds
                        const date = new Date(conv.timestamp).toLocaleString('en-US', {
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                        });
                        const preview = conv.messages.length > 0 ? conv.messages[0].content.substring(0, 30) : 'Empty';
                        div.textContent = `${date} - ${preview}...`;
                        div.onclick = () => loadConversation(conv);
                        listContainer.appendChild(div);
                    });
                }

            } catch (e) {
                console.error('Failed to load conversations', e);
            }
        }

        function loadConversation(conv) {
            currentConversation = conv;
            chatHistory.innerHTML = '';
            conv.messages.forEach(msg => {
                appendMessage(msg.role, msg.content, false, msg.params);
            });
        }

        async function saveConversation() {
            if (!currentModel) return;
            try {
                const headers = addAuthHeaders({'Content-Type': 'application/json'});
                // Use query parameter for model_id to avoid path routing issues
                const params = new URLSearchParams({ model_id: currentModel.id });
                const response = await fetch(`/api/conversations/save?${params}`, {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify(currentConversation)
                });

                if (!response.ok) {
                    throw new Error(`Failed to save: ${response.status}`);
                }
            } catch (e) {
                console.error('Failed to save conversation', e);
            }
        }

        async function sendMessage() {
            const text = input.value.trim();
            if (!text) return;
            
            if (!currentModel) {
                alert("Please select a model first.");
                return;
            }

            appendMessage('user', text);
            currentConversation.messages.push({
                role: 'user',
                content: text,
                timestamp: Date.now()
            });
            input.value = '';

            const loadingId = appendMessage('assistant', 'Thinking...', true);

            try {
                const headers = addAuthHeaders({'Content-Type': 'application/json'});
                // Build request with inference settings
                const requestBody = {
                    model_id: currentModel.id,
                    inputs: { text: text },
                    parameters: {
                        temperature: inferenceSettings.temperature,
                        max_new_tokens: inferenceSettings.maxTokens,
                        top_p: inferenceSettings.topP,
                        top_k: inferenceSettings.topK,
                        do_sample: inferenceSettings.doSample
                    }
                };

                // Add system prompt if provided
                if (inferenceSettings.systemPrompt && inferenceSettings.systemPrompt.trim()) {
                    requestBody.inputs.system_prompt = inferenceSettings.systemPrompt;
                }

                const response = await fetch('/api/inference/universal', {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) throw new Error('Inference failed');

                const data = await response.json();
                const output = data.outputs[0];

                // Prepare params for display and storage
                const messageParams = {
                    temperature: inferenceSettings.temperature,
                    max_tokens: inferenceSettings.maxTokens,
                    top_p: inferenceSettings.topP,
                    top_k: inferenceSettings.topK,
                    do_sample: inferenceSettings.doSample,
                    system_prompt: inferenceSettings.systemPrompt
                };

                updateMessage(loadingId, output, messageParams);

                // Save message with generation parameters for future reference
                currentConversation.messages.push({
                    role: 'assistant',
                    content: output,
                    timestamp: Date.now(),
                    params: messageParams
                });

                // Save conversation
                saveConversation();

            } catch (error) {
                updateMessage(loadingId, 'Error: ' + error.message);
            }
        }

        sendBtn.addEventListener('click', sendMessage);
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    function appendMessage(role, text, isLoading = false, params = null) {
        const chatHistory = document.getElementById('chat-history');
        const msgDiv = document.createElement('div');
        msgDiv.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;

        const bubble = document.createElement('div');
        bubble.className = `max-w-[80%] p-3 rounded-lg ${
            role === 'user'
            ? 'bg-orange-600 text-white rounded-br-none'
            : 'bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-bl-none'
        }`;

        // Add tooltip with generation params for assistant messages
        if (params && role === 'assistant') {
            const tooltip = `Temperature: ${params.temperature}\nMax Tokens: ${params.max_tokens}\nTop P: ${params.top_p}\nTop K: ${params.top_k}\nSampling: ${params.do_sample ? 'On' : 'Off'}${params.system_prompt ? '\nSystem: ' + params.system_prompt : ''}`;
            bubble.title = tooltip;
            bubble.style.cursor = 'help';
        }

        if (isLoading) {
            bubble.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Thinking...';
            bubble.id = 'loading-' + Date.now();
        } else {
            // Add info icon for messages with params
            if (params && role === 'assistant') {
                bubble.innerHTML = `
                    <div style="display: flex; align-items: start; gap: 8px;">
                        <span style="flex: 1;">${text}</span>
                        <i class="fas fa-info-circle" style="color: #9ca3af; font-size: 14px; flex-shrink: 0; margin-top: 2px;" title="Hover for generation details"></i>
                    </div>
                `;
            } else {
                bubble.textContent = text;
            }
        }

        msgDiv.appendChild(bubble);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return bubble.id;
    }

    function updateMessage(id, text, params = null) {
        const bubble = document.getElementById(id);
        if (bubble) {
            // Add info icon for messages with params
            if (params) {
                bubble.innerHTML = `
                    <div style="display: flex; align-items: start; gap: 8px;">
                        <span style="flex: 1;">${text}</span>
                        <i class="fas fa-info-circle" style="color: #9ca3af; font-size: 14px; flex-shrink: 0; margin-top: 2px;" title="Hover for generation details"></i>
                    </div>
                `;

                const tooltip = `Temperature: ${params.temperature}\nMax Tokens: ${params.max_tokens}\nTop P: ${params.top_p}\nTop K: ${params.top_k}\nSampling: ${params.do_sample ? 'On' : 'Off'}${params.system_prompt ? '\nSystem: ' + params.system_prompt : ''}`;
                bubble.title = tooltip;
                bubble.style.cursor = 'help';
            } else {
                bubble.textContent = text;
            }

            bubble.removeAttribute('id');
        }
    }

    // --- Text Classification Interface ---
    function loadTextClassificationInterface() {
        const clone = templates.textClassification.content.cloneNode(true);
        inferenceInterface.innerHTML = '';
        inferenceInterface.appendChild(clone);
        
        const input = document.getElementById('text-input');
        const classifyBtn = document.getElementById('classify-btn');
        const resultsDiv = document.getElementById('classification-results');
        const probsDiv = document.getElementById('class-probabilities');

        if (currentModel.type === 'seq2seq') classifyBtn.textContent = "Generate";

        classifyBtn.addEventListener('click', async () => {
            const text = input.value.trim();
            if (!text) return;

            classifyBtn.disabled = true;
            classifyBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

            try {
                const headers = addAuthHeaders({'Content-Type': 'application/json'});
                // Build request with inference settings
                const requestBody = {
                    model_id: currentModel.id,
                    inputs: { text: text },
                    parameters: {
                        temperature: inferenceSettings.temperature,
                        max_new_tokens: inferenceSettings.maxTokens,
                        top_p: inferenceSettings.topP,
                        top_k: inferenceSettings.topK,
                        do_sample: inferenceSettings.doSample
                    }
                };

                // Add system prompt if provided
                if (inferenceSettings.systemPrompt && inferenceSettings.systemPrompt.trim()) {
                    requestBody.inputs.system_prompt = inferenceSettings.systemPrompt;
                }

                const response = await fetch('/api/inference/universal', {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify(requestBody)
                });
                
                const data = await response.json();
                resultsDiv.classList.remove('hidden');
                
                probsDiv.innerHTML = '';
                
                if (currentModel.type === 'text-classification' && Array.isArray(data.outputs) && data.outputs.length > 0 && Array.isArray(data.outputs[0])) {
                    // Text Classification - Render bars
                    data.outputs[0].forEach(item => {
                        const percentage = (item.score * 100).toFixed(1);
                        const bar = document.createElement('div');
                        bar.className = 'flex items-center gap-2 text-sm';
                        bar.innerHTML = `
                            <div class="w-20 truncate text-right font-mono">${item.label}</div>
                            <div class="flex-1 bg-gray-200 dark:bg-gray-700 rounded h-4 overflow-hidden relative">
                                <div class="bg-blue-500 h-full" style="width: ${percentage}%"></div>
                            </div>
                            <div class="w-12 text-right">${percentage}%</div>
                        `;
                        probsDiv.appendChild(bar);
                    });
                } else if (currentModel.type === 'token-classification') {
                    // Basic rendering for tokens
                    probsDiv.innerHTML = `<pre class="text-xs overflow-auto">${JSON.stringify(data.outputs, null, 2)}</pre>`;
                } else {
                    // Default / Seq2Seq / Regression
                    const output = Array.isArray(data.outputs) && data.outputs[0].generated_text ? data.outputs[0].generated_text : JSON.stringify(data.outputs);
                    probsDiv.textContent = output;
                }

            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                classifyBtn.disabled = false;
                classifyBtn.textContent = currentModel.type === 'seq2seq' ? 'Generate' : 'Classify';
            }
        });
    }

    // --- Image Interface ---
    function loadImageInterface() {
        const clone = templates.imageUpload.content.cloneNode(true);
        inferenceInterface.innerHTML = '';
        inferenceInterface.appendChild(clone);

        const fileInput = document.getElementById('dropzone-file');
        const preview = document.getElementById('image-preview');
        const previewImg = preview.querySelector('img');
        const analyzeBtn = document.getElementById('analyze-btn');
        const resultsDiv = document.getElementById('vision-results');

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                if (file.size > 10 * 1024 * 1024) {
                    alert("File too large (max 10MB)");
                    return;
                }
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImg.src = e.target.result;
                    preview.classList.remove('hidden');
                    analyzeBtn.classList.remove('hidden');
                };
                reader.readAsDataURL(file);
            }
        });
        
        if (currentModel.type === 'vlm') {
            const container = document.getElementById('vision-input-container');
            container.classList.remove('hidden');
            container.innerHTML = `
                <label class="block mb-2 text-sm font-medium">Prompt</label>
                <textarea id="vlm-prompt" class="w-full p-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-700" rows="2" placeholder="Describe this image..."></textarea>
            `;
        }

        analyzeBtn.addEventListener('click', async () => {
            if (!previewImg.src) return;

            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            
            const payload = {
                model_id: currentModel.id,
                inputs: { image: previewImg.src }
            };

            if (currentModel.type === 'vlm') {
                const prompt = document.getElementById('vlm-prompt').value.trim();
                if (prompt) payload.inputs.text = prompt;
            }

            try {
                const headers = addAuthHeaders({'Content-Type': 'application/json'});
                const response = await fetch('/api/inference/universal', {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                resultsDiv.classList.remove('hidden');
                
                resultsDiv.innerHTML = '';
                if (currentModel.type === 'image-classification' && Array.isArray(data.outputs)) {
                    // Render bars
                    data.outputs.forEach(item => {
                        const percentage = (item.score * 100).toFixed(1);
                        const bar = document.createElement('div');
                        bar.className = 'flex items-center gap-2 text-sm mb-1';
                        bar.innerHTML = `
                            <div class="w-24 truncate text-right font-mono">${item.label}</div>
                            <div class="flex-1 bg-gray-200 dark:bg-gray-700 rounded h-4 overflow-hidden relative">
                                <div class="bg-green-500 h-full" style="width: ${percentage}%"></div>
                            </div>
                            <div class="w-12 text-right">${percentage}%</div>
                        `;
                        resultsDiv.appendChild(bar);
                    });
                } else if (currentModel.type === 'image-regression' && Array.isArray(data.outputs)) {
                    // Show regression value
                    resultsDiv.innerHTML = `
                        <div class="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded">
                            <div class="text-sm text-gray-600 dark:text-gray-400 mb-1">Predicted Value:</div>
                            <div class="text-2xl font-bold text-blue-600 dark:text-blue-400">${data.outputs[0].toFixed(4)}</div>
                        </div>
                    `;
                } else if (currentModel.type === 'image-object-detection' && Array.isArray(data.outputs)) {
                    // List objects
                    data.outputs.forEach(item => {
                        const percentage = (item.score * 100).toFixed(1);
                        const box = document.createElement('div');
                        box.className = 'flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-700 rounded mb-1 border border-gray-200 dark:border-gray-600';
                        box.innerHTML = `
                            <span class="font-bold text-sm">${item.label}</span>
                            <span class="text-xs bg-blue-100 text-blue-800 px-2 py-0.5 rounded dark:bg-blue-900 dark:text-blue-300">${percentage}%</span>
                        `;
                        resultsDiv.appendChild(box);
                    });
                } else {
                    // VLM or raw
                    resultsDiv.innerHTML = `<div class="p-2">${Array.isArray(data.outputs) ? data.outputs[0] : JSON.stringify(data.outputs)}</div>`;
                }

            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze';
            }
        });
    }

    // --- Question Answering Interface ---
    function loadQuestionAnsweringInterface() {
        inferenceInterface.innerHTML = `
            <div class="w-full max-w-3xl">
                <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <h2 class="text-lg font-bold mb-4">Question Answering</h2>
                    <div class="mb-4">
                        <label class="block mb-2 text-sm font-medium">Context</label>
                        <textarea id="qa-context" class="w-full p-3 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-700" rows="6" placeholder="Enter the context text here..."></textarea>
                    </div>
                    <div class="mb-4">
                        <label class="block mb-2 text-sm font-medium">Question</label>
                        <input type="text" id="qa-question" class="w-full p-3 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-700" placeholder="Ask a question about the context..." />
                    </div>
                    <button id="qa-submit-btn" class="bg-orange-500 hover:bg-orange-600 text-white px-6 py-2 rounded font-medium">
                        Find Answer
                    </button>
                    <div id="qa-result" class="mt-4 hidden">
                        <h3 class="font-bold mb-2">Answer:</h3>
                        <div id="qa-answer" class="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-700 rounded"></div>
                    </div>
                </div>
            </div>
        `;

        const contextInput = document.getElementById('qa-context');
        const questionInput = document.getElementById('qa-question');
        const submitBtn = document.getElementById('qa-submit-btn');
        const resultDiv = document.getElementById('qa-result');
        const answerDiv = document.getElementById('qa-answer');

        submitBtn.addEventListener('click', async () => {
            const context = contextInput.value.trim();
            const question = questionInput.value.trim();

            if (!context || !question) {
                alert('Please provide both context and question');
                return;
            }

            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Finding...';

            try {
                const headers = addAuthHeaders({'Content-Type': 'application/json'});
                const response = await fetch('/api/inference/universal', {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify({
                        model_id: currentModel.id,
                        inputs: {
                            text: context,
                            question: question
                        }
                    })
                });

                const data = await response.json();
                if (response.ok) {
                    answerDiv.textContent = data.outputs[0];
                    resultDiv.classList.remove('hidden');
                } else {
                    alert('Error: ' + data.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Find Answer';
            }
        });
    }

    // --- Sentence Transformers Interface ---
    function loadSentenceTransformersInterface() {
        inferenceInterface.innerHTML = `
            <div class="w-full max-w-3xl">
                <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <h2 class="text-lg font-bold mb-4">Sentence Embeddings</h2>
                    <div class="mb-4">
                        <label class="block mb-2 text-sm font-medium">Text (one per line for batch)</label>
                        <textarea id="st-input" class="w-full p-3 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-700" rows="6" placeholder="Enter text to generate embeddings...&#10;You can enter multiple lines for batch processing"></textarea>
                    </div>
                    <button id="st-submit-btn" class="bg-orange-500 hover:bg-orange-600 text-white px-6 py-2 rounded font-medium">
                        Generate Embeddings
                    </button>
                    <div id="st-result" class="mt-4 hidden">
                        <h3 class="font-bold mb-2">Results:</h3>
                        <div id="st-output" class="p-3 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded font-mono text-sm overflow-auto max-h-96"></div>
                    </div>
                </div>
            </div>
        `;

        const inputArea = document.getElementById('st-input');
        const submitBtn = document.getElementById('st-submit-btn');
        const resultDiv = document.getElementById('st-result');
        const outputDiv = document.getElementById('st-output');

        submitBtn.addEventListener('click', async () => {
            const text = inputArea.value.trim();
            if (!text) {
                alert('Please enter some text');
                return;
            }

            // Split by newlines for batch processing
            const texts = text.split('\n').filter(line => line.trim());

            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

            try {
                const headers = addAuthHeaders({'Content-Type': 'application/json'});
                const response = await fetch('/api/inference/universal', {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify({
                        model_id: currentModel.id,
                        inputs: {
                            texts: texts.length === 1 ? texts[0] : texts
                        }
                    })
                });

                const data = await response.json();
                if (response.ok) {
                    const embeddings = data.outputs;
                    let output = `Generated ${embeddings.length} embedding(s)\n`;
                    output += `Dimension: ${embeddings[0].length}\n\n`;
                    embeddings.forEach((emb, idx) => {
                        output += `Text ${idx + 1}: ${texts[idx].substring(0, 50)}...\n`;
                        output += `Embedding (first 10): [${emb.slice(0, 10).map(v => v.toFixed(4)).join(', ')}...]\n\n`;
                    });
                    outputDiv.textContent = output;
                    resultDiv.classList.remove('hidden');
                } else {
                    alert('Error: ' + data.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Generate Embeddings';
            }
        });
    }

    // --- Tabular Interface ---
    function loadTabularInterface() {
        inferenceInterface.innerHTML = `
            <div class="w-full max-w-3xl">
                <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <h2 class="text-lg font-bold mb-4">Tabular Prediction</h2>
                    <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
                        Enter feature values as JSON
                    </p>
                    <div class="mb-4">
                        <label class="block mb-2 text-sm font-medium">Features (JSON)</label>
                        <textarea id="tabular-input" class="w-full p-3 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-700 font-mono text-sm" rows="8" placeholder='{"feature1": 1.0, "feature2": 2.5, ...}'></textarea>
                    </div>
                    <button id="tabular-submit-btn" class="bg-orange-500 hover:bg-orange-600 text-white px-6 py-2 rounded font-medium">
                        Predict
                    </button>
                    <div id="tabular-result" class="mt-4 hidden">
                        <h3 class="font-bold mb-2">Prediction:</h3>
                        <div id="tabular-output" class="p-3 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded"></div>
                    </div>
                </div>
            </div>
        `;

        const inputArea = document.getElementById('tabular-input');
        const submitBtn = document.getElementById('tabular-submit-btn');
        const resultDiv = document.getElementById('tabular-result');
        const outputDiv = document.getElementById('tabular-output');

        submitBtn.addEventListener('click', async () => {
            const text = inputArea.value.trim();
            if (!text) {
                alert('Please enter feature values');
                return;
            }

            let features;
            try {
                features = JSON.parse(text);
            } catch (e) {
                alert('Invalid JSON format');
                return;
            }

            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Predicting...';

            try {
                const headers = addAuthHeaders({'Content-Type': 'application/json'});
                const response = await fetch('/api/inference/universal', {
                    method: 'POST',
                    headers: headers,
                    body: JSON.stringify({
                        model_id: currentModel.id,
                        inputs: {
                            features: features
                        }
                    })
                });

                const data = await response.json();
                if (response.ok) {
                    outputDiv.textContent = JSON.stringify(data.outputs, null, 2);
                    resultDiv.classList.remove('hidden');
                } else {
                    alert('Error: ' + data.detail);
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Predict';
            }
        });
    }

    // HuggingFace Model Loader
    const hfModelInput = document.getElementById('hf-model-input');
    const hfTokenInput = document.getElementById('hf-token-input');
    const loadHfModelBtn = document.getElementById('load-hf-model-btn');

    // Load/save HF token from sessionStorage (cleared when browser closes)
    const savedToken = sessionStorage.getItem('hf_token');
    if (savedToken && hfTokenInput) {
        hfTokenInput.value = savedToken;
    }

    // Save token to sessionStorage when user enters it
    if (hfTokenInput) {
        hfTokenInput.addEventListener('change', (e) => {
            const token = e.target.value.trim();
            if (token) {
                sessionStorage.setItem('hf_token', token);
                console.log('HF token saved to session');
            } else {
                sessionStorage.removeItem('hf_token');
            }
        });
    }

    loadHfModelBtn.addEventListener('click', async () => {
        const modelId = hfModelInput.value.trim();
        if (!modelId) {
            alert('Please enter a model ID');
            return;
        }

        loadHfModelBtn.disabled = true;
        loadHfModelBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

        try {
            // Get token from input or sessionStorage
            const token = hfTokenInput?.value.trim() || sessionStorage.getItem('hf_token') || window.HF_TOKEN;

            // Try to detect model type by fetching from HF Hub
            const headers = {'Content-Type': 'application/json'};
            if (token && token !== "None") {
                headers['Authorization'] = `Bearer ${token}`;
            }

            // Make a test request to see if model exists and what type it is
            // We'll use the inference endpoint which will auto-detect the type
            const testRequest = {
                model_id: modelId,
                inputs: {text: "test"}  // Minimal input for type detection
            };

            const response = await fetch('/api/inference/universal', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(testRequest)
            });

            if (response.ok || response.status === 400) {
                // Either it worked (200) or failed with validation error (400)
                // Both mean the model exists and was detected
                const data = await response.json();
                const modelType = data.model_type || 'unknown';

                // Create a pseudo-model object and select it
                const hfModel = {
                    id: modelId,
                    type: modelType,
                    model_path: modelId,
                    metadata: {
                        created_at: Date.now() / 1000,
                        size: null,
                        files: []
                    },
                    isHfHub: true
                };

                // Add to model list
                const groupHeader = document.createElement('div');
                groupHeader.className = 'px-2 py-1 text-xs font-semibold text-gray-500 uppercase tracking-wider mt-2 border-b dark:border-gray-700 mb-1';
                groupHeader.textContent = 'HuggingFace Hub';

                const modelItem = document.createElement('div');
                modelItem.className = 'model-item p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded cursor-pointer mb-1';
                modelItem.innerHTML = `
                    <div class="font-medium text-sm model-id">${modelId}</div>
                    <div class="text-xs text-gray-500 model-type">${modelType}</div>
                `;
                modelItem.addEventListener('click', () => selectModel(hfModel));

                // Add to top of model list
                const container = modelListContainer;
                container.insertBefore(modelItem, container.firstChild);
                container.insertBefore(groupHeader, container.firstChild);

                // Auto-select the model
                selectModel(hfModel);

                // Clear input
                hfModelInput.value = '';

                console.log(`Loaded HF model: ${modelId} (type: ${modelType})`);
            } else {
                const error = await response.json();
                alert(`Failed to load model: ${error.detail || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error loading HF model:', error);
            alert('Error loading model: ' + error.message);
        } finally {
            loadHfModelBtn.disabled = false;
            loadHfModelBtn.innerHTML = 'Load';
        }
    });

    // Allow Enter key to load model
    hfModelInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            loadHfModelBtn.click();
        }
    });

    // Initial load
    fetchModels();
    refreshBtn.addEventListener('click', fetchModels);

    // Load default interface (LLM) even if no model, so UI is visible
    // But we need to wait for fetchModels to finish or just load it initially
    // fetchModels updates the list.
    // Let's just load the LLM interface by default (in disconnected state)
    loadLLMInterface();
});
