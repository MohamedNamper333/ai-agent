const API_BASE = 'http://127.0.0.1:8080';

let currentConvId = null;
let isStreaming = false;
let ragActive = false;

// DOM references
const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome');
const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const convList = document.getElementById('conv-list');
const newChatBtn = document.getElementById('new-chat-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const toolsBadge = document.getElementById('tools-badge');
const fileInput = document.getElementById('file-input');
const ragBtn = document.getElementById('rag-btn');
const councilBtn = document.getElementById('council-btn');
const ragIndicator = document.getElementById('rag-indicator');
const ragClose = document.getElementById('rag-close');

// ─── API ───
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(`${API_BASE}${path}`, opts);
  if (!r.ok) throw new Error(`API error: ${r.status}`);
  return r;
}

// ─── Status ───
async function loadStatus() {
  try {
    const r = await api('GET', '/status');
    const data = await r.json();
    statusDot.className = 'status-dot ' + (data.model_loaded ? 'loaded' : 'loading');
    statusText.textContent = data.model_loaded ? (data.model_name || 'ready').slice(0, 20) : 'loading...';
    if (toolsBadge) {
      const tc = await api('GET', '/stats');
      const stats = await tc.json().catch(() => ({}));
      toolsBadge.textContent = (stats.tool_count || '46') + ' tools';
    }
  } catch {
    statusDot.className = 'status-dot';
    statusText.textContent = 'disconnected';
  }
}

// ─── Conversations ───
async function loadConversations() {
  try {
    const r = await api('GET', '/conversations');
    const data = await r.json();
    convList.innerHTML = '';
    if (data.conversations && data.conversations.length) {
      data.conversations.forEach(cid => {
        const item = document.createElement('div');
        item.className = 'conv-item' + (cid === data.current ? ' active' : '');
        item.textContent = cid.replace('conv_', '').replace(/_/g, ' ').slice(0, 24);
        item.dataset.id = cid;
        item.onclick = () => switchConversation(cid);

        const delBtn = document.createElement('button');
        delBtn.className = 'delete-conv';
        delBtn.textContent = '×';
        delBtn.onclick = async (e) => {
          e.stopPropagation();
          await api('DELETE', `/conversations/${cid}`).catch(() => {});
          if (currentConvId === cid) {
            messagesEl.innerHTML = '';
            welcomeEl.classList.remove('hidden');
            currentConvId = null;
          }
          loadConversations();
        };
        item.appendChild(delBtn);
        convList.appendChild(item);
      });
      currentConvId = data.current;
    } else {
      convList.innerHTML = '<div style="padding:16px;text-align:center;color:var(--text-muted);font-size:13px">No conversations yet</div>';
    }
  } catch {}
}

// ─── Messages ───
async function switchConversation(cid) {
  if (isStreaming) return;
  currentConvId = cid;
  await loadConversations();
  await loadMessages();
}

async function loadMessages() {
  if (!currentConvId) return;
  try {
    const r = await api('GET', `/conversations/${currentConvId}`);
    const data = await r.json();
    messagesEl.innerHTML = '';
    welcomeEl.classList.add('hidden');
    if (data.messages) {
      data.messages.forEach(m => {
        if (m.role === 'system') return;
        addMessage(m.role, m.content, null, false);
      });
    }
    scrollToBottom();
  } catch {}
}

// ─── Markdown ───
function renderMarkdown(text) {
  if (typeof marked === 'undefined') {
    return text.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
               .replace(/`([^`]+)`/g, '<code>$1</code>')
               .replace(/\n/g, '<br>');
  }
  return marked.parse(text, { breaks: true, gfm: true });
}

function scrollToBottom() {
  const container = messagesEl;
  container.scrollTop = container.scrollHeight;
}

function addMessage(role, content, msgId, animate = true) {
  welcomeEl.classList.add('hidden');

  const div = document.createElement('div');

  if (content === '_streaming_') {
    div.className = 'message assistant streaming';
    div.id = `msg-${msgId || 'stream'}`;
    div.innerHTML = `
      <div class="assistant-avatar-row">
        <div class="assistant-avatar">⟁</div>
        <span class="assistant-name">AI Agent</span>
      </div>
      <div class="message-content"></div>
    `;
    messagesEl.appendChild(div);
    scrollToBottom();
    return div;
  }

  if (role === 'user') {
    div.className = 'message user';
    div.innerHTML = `
      <div class="user-label">
        <div class="user-avatar">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
          <circle cx="12" cy="7" r="4"/>
        </svg>
      </div>
        <span class="user-name">You</span>
      </div>
      <div class="message-content">${escHtml(content)}</div>
    `;
  } else if (role === 'assistant') {
    div.className = 'message assistant';
    const html = renderMarkdown(content);
    div.innerHTML = `
      <div class="assistant-avatar-row">
        <div class="assistant-avatar">⟁</div>
        <span class="assistant-name">AI Agent</span>
      </div>
      <div class="message-content">${html}</div>
      <div class="message-timestamp">${new Date().toLocaleTimeString()}</div>
    `;

    div.querySelectorAll('pre').forEach(pre => {
      const btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.textContent = 'Copy';
      btn.onclick = () => {
        const code = pre.querySelector('code');
        const text = code ? code.textContent : pre.textContent;
        navigator.clipboard.writeText(text).then(() => {
          showToast('Copied!');
          btn.textContent = 'Done';
          setTimeout(() => btn.textContent = 'Copy', 1500);
        });
      };
      pre.style.position = 'relative';
      pre.appendChild(btn);
    });
  } else {
    div.className = `message ${role}`;
    div.textContent = content;
  }

  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

function escHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

// ─── Send message ───
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isStreaming || !currentConvId) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';
  addMessage('user', text);
  isStreaming = true;
  sendBtn.disabled = true;

  const msgEl = addMessage('assistant', '_streaming_', 'stream');

  try {
    const body = {
      conversation_id: currentConvId,
      message: text,
      stream: true,
      use_rag: ragActive,
    };

    // Check for council trigger
    if (text.toLowerCase().includes('/council') || text.toLowerCase().includes('council this')) {
      body.stream = false;
      const r = await api('POST', '/chat', body);
      const data = await r.json();
      msgEl.className = 'message assistant';
      msgEl.innerHTML = `
        <div class="assistant-avatar-row">
          <div class="assistant-avatar">🧠</div>
          <span class="assistant-name">Council</span>
        </div>
        <div class="message-content">${renderMarkdown(data.text || data.response || '')}</div>
        <div class="message-timestamp">${new Date().toLocaleTimeString()}</div>
      `;
    } else {
      const r = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!r.ok) throw new Error(`HTTP ${r.status}`);

      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let full = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim();
            if (data === '[DONE]') continue;
            try {
              const parsed = JSON.parse(data);
              if (parsed.text) {
                full += parsed.text;
                msgEl.querySelector('.message-content').innerHTML = renderMarkdown(full);
                scrollToBottom();
              }
            } catch {}
          }
        }
      }

      msgEl.className = 'message assistant';
      const ts = document.createElement('div');
      ts.className = 'message-timestamp';
      ts.textContent = new Date().toLocaleTimeString();
      msgEl.appendChild(ts);
    }

    scrollToBottom();
  } catch (err) {
    msgEl.className = 'message assistant';
    msgEl.innerHTML = `
      <div class="assistant-avatar-row">
        <div class="assistant-avatar">!</div>
        <span class="assistant-name">Error</span>
      </div>
      <div class="message-content" style="color:#ff6b6b">${escHtml(err.message)}</div>
    `;
  }

  isStreaming = false;
  sendBtn.disabled = false;
  inputEl.focus();
}

// ─── Image Upload Warning ───
function showImageWarning() {
  const warningEl = document.createElement('div');
  warningEl.className = 'message assistant';
  warningEl.innerHTML = `
    <div class="assistant-avatar-row">
      <div class="assistant-avatar">🖼</div>
      <span class="assistant-name">System</span>
    </div>
    <div class="message-content" style="border-color: #ff9800;">
      ⚠️ This model doesn't support image analysis. To use image features:
      <br><br>
      • <b>Option 1:</b> Install a vision model in Ollama:<br>
      <code>ollama pull llava</code> or <code>ollama pull bakllava</code>
      <br><br>
      • <b>Option 2:</b> OCR text from images is available via<br>
      <code>pip install pytesseract Pillow</code>
      <br><br>
      • <b>Option 3:</b> The file is uploaded to RAG for text search.
    </div>
  `;
  messagesEl.appendChild(warningEl);
  scrollToBottom();
}

// ─── Toast ───
let toastTimer;

function showToast(msg, duration = 2000) {
  const toast = document.getElementById('toast');
  if (!toast) return;
  clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.classList.remove('hidden');
  toast.classList.add('show');
  toastTimer = setTimeout(() => {
    toast.classList.remove('show');
    toast.classList.add('hidden');
  }, duration);
}

// ─── Event listeners ───
sendBtn.onclick = sendMessage;

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
});

newChatBtn.onclick = async () => {
  if (isStreaming) return;
  try {
    const r = await api('POST', '/conversations/new');
    const data = await r.json();
    currentConvId = data.conversation_id;
    messagesEl.innerHTML = '';
    welcomeEl.classList.remove('hidden');
    await loadConversations();
  } catch {}
};

// Suggestion chips
document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.onclick = () => {
    const prompt = chip.dataset.prompt;
    if (prompt && currentConvId) {
      inputEl.value = prompt;
      inputEl.style.height = 'auto';
      sendMessage();
    }
  };
});

// RAG toggle
ragBtn.onclick = () => {
  ragActive = !ragActive;
  ragBtn.classList.toggle('active', ragActive);
  ragIndicator.classList.toggle('hidden', !ragActive);
};

ragClose.onclick = () => {
  ragActive = false;
  ragBtn.classList.remove('active');
  ragIndicator.classList.add('hidden');
};

// Upload
document.getElementById('upload-label').onclick = function(e) {
  if (e.target.tagName !== 'INPUT') fileInput.click();
};

fileInput.onchange = async () => {
  const files = fileInput.files;
  if (!files.length) return;

  const imageFiles = [];
  const docFiles = [];

  for (const file of files) {
    if (file.type.startsWith('image/')) {
      imageFiles.push(file.name);
    } else {
      docFiles.push(file);
    }
  }

  if (imageFiles.length > 0) {
    showImageWarning();
    showToast(`Images uploaded to RAG. Use a vision model (llava) for AI analysis.`, 4000);
  }

  if (docFiles.length > 0 || imageFiles.length > 0) {
    showToast(`Uploading ${files.length} file(s)...`);
    for (const file of files) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
      } catch {}
    }
    showToast('Upload complete! Use RAG mode to query your files.', 3000);
  }

  fileInput.value = '';
};

// Load conversations on scroll (lazy)
convList.addEventListener('scroll', () => {});

// ─── Init ───
(async () => {
  await loadStatus();
  await loadConversations();

  if (currentConvId) {
    await loadMessages();
  } else {
    try {
      const r = await api('POST', '/conversations/new');
      const data = await r.json();
      currentConvId = data.conversation_id;
      await loadConversations();
    } catch {}
  }

  setInterval(loadStatus, 10000);
})();
