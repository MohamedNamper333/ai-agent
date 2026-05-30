const API_BASE = 'http://127.0.0.1:8080';

let currentConvId = null;
let isStreaming = false;

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const convList = document.getElementById('conv-list');
const newChatBtn = document.getElementById('new-chat-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const ragToggle = document.getElementById('rag-toggle');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');

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

async function loadStatus() {
  try {
    const r = await api('GET', '/status');
    const data = await r.json();
    statusDot.className = 'status-dot ' + (data.model_loaded ? 'loaded' : 'loading');
    statusText.textContent = data.model_loaded ? data.model_name || 'ready' : 'loading model...';
  } catch {
    statusDot.className = 'status-dot';
    statusText.textContent = 'disconnected';
  }
}

async function loadConversations() {
  try {
    const r = await api('GET', '/conversations');
    const data = await r.json();
    convList.innerHTML = '';
    if (data.conversations) {
      data.conversations.forEach(cid => {
        const item = document.createElement('div');
        item.className = 'conv-item' + (cid === data.current ? ' active' : '');
        item.textContent = cid.replace('conv_', '').replace(/_/g, ' ');
        item.dataset.id = cid;
        item.onclick = () => switchConversation(cid);
        convList.appendChild(item);
      });
      currentConvId = data.current;
    }
  } catch {}
}

async function switchConversation(cid) {
  if (isStreaming) return;
  currentConvId = cid;
  await loadConversations();
  await loadMessages();
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addMessage(role, content, msgId) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  if (content === '_streaming_') {
    div.className += ' streaming';
    div.id = `msg-${msgId || 'stream'}`;
    div.textContent = '';
  } else {
    div.textContent = content;
    const ts = document.createElement('div');
    ts.className = 'timestamp';
    ts.textContent = new Date().toLocaleTimeString();
    div.appendChild(ts);
  }
  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

async function loadMessages() {
  if (!currentConvId) return;
  try {
    const r = await api('GET', `/conversations/${currentConvId}`);
    const data = await r.json();
    messagesEl.innerHTML = '';
    if (data.messages) {
      data.messages.forEach(m => {
        if (m.role !== 'system') addMessage(m.role, m.content);
      });
    }
    scrollToBottom();
  } catch {}
}

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
    const useRag = ragToggle.checked;
    const r = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conversation_id: currentConvId,
        message: text,
        stream: true,
        use_rag: useRag,
      }),
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
              msgEl.textContent = full;
              scrollToBottom();
            }
          } catch {}
        }
      }
    }

    msgEl.className = 'message assistant';
    const ts = document.createElement('div');
    ts.className = 'timestamp';
    ts.textContent = new Date().toLocaleTimeString();
    msgEl.appendChild(ts);
    scrollToBottom();
  } catch (err) {
    msgEl.textContent = `Error: ${err.message}`;
    msgEl.className = 'message assistant';
  }

  isStreaming = false;
  sendBtn.disabled = false;
  inputEl.focus();
}

// Event listeners
sendBtn.onclick = sendMessage;
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + 'px';
});

newChatBtn.onclick = async () => {
  if (isStreaming) return;
  try {
    const r = await api('POST', '/conversations/new');
    const data = await r.json();
    currentConvId = data.conversation_id;
    messagesEl.innerHTML = '';
    await loadConversations();
  } catch {}
};

uploadBtn.onclick = () => fileInput.click();
fileInput.onchange = async () => {
  const files = fileInput.files;
  if (!files.length) return;
  for (const file of files) {
    const formData = new FormData();
    formData.append('file', file);
    try {
      await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
    } catch {}
  }
  fileInput.value = '';
};

// Init
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
  setInterval(loadStatus, 5000);
})();
