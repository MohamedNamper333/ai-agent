/* ═══════════════════════════════════════════
   AI Agent — app.js  v2 (premium + settings)
═══════════════════════════════════════════ */

'use strict';

const API_BASE   = window.location.origin;
const MAX_CHARS  = 4000;

const ACTION_PROMPTS = {
  review:  'Please review my code for bugs, performance issues, and best practices.',
  analyze: 'Analyze the structure and architecture of this project and summarize key findings.',
  write:   'Write a clean, well-documented Python script to help me automate a task.',
  search:  'Search for the latest developments in AI and summarize the most important news.'
};

const CMD_PALETTE_ITEMS = [
  { icon: '✨', name: 'New conversation',    desc: 'Start a fresh chat',          kbd: 'Ctrl+N', action: () => newConversation() },
  { icon: '🔍', name: 'Search conversations', desc: 'Find past chats',             kbd: '/',      action: () => focusSearch() },
  { icon: '🌙', name: 'Toggle dark mode',     desc: 'Switch theme',               kbd: 'Ctrl+D', action: () => toggleTheme() },
  { icon: '💬', name: 'Review my code',       desc: 'Ask AI to review code',      kbd: '',       action: () => quickAction('review') },
  { icon: '📊', name: 'Analyze project',      desc: 'Get a project overview',     kbd: '',       action: () => quickAction('analyze') },
  { icon: '✏️',  name: 'Write a script',       desc: 'Generate a script',          kbd: '',       action: () => quickAction('write') },
  { icon: '🌐', name: 'Search docs',          desc: 'Search the knowledge base',  kbd: '',       action: () => quickAction('search') },
];

/* ── State ── */
let currentConvId = null;
let isStreaming   = false;
let currentView   = 'dashboard';
let convTitles    = {};
let recognition   = null;
let isListening   = false;
let cmdOpen       = false;
let toolsData     = null;

/* ── Persist ── */
try { convTitles = JSON.parse(localStorage.getItem('ai_conv_titles') || '{}'); } catch (_) {}
function saveTitles() { try { localStorage.setItem('ai_conv_titles', JSON.stringify(convTitles)); } catch(_) {} }
function getTitle(cid) { return convTitles[cid] || 'Conversation'; }
function setTitle(cid, title) { convTitles[cid] = title.slice(0, 80); saveTitles(); }

/* ── DOM ── */
const $  = id => document.getElementById(id);
const qs = sel => document.querySelector(sel);

const appEl       = $('app');
const messagesEl  = $('messages');
const inputEl     = $('user-input');
const sendBtn     = $('send-btn');
const dashInput   = $('dash-input');
const dashSendBtn = $('dash-send-btn');
const dashView    = $('dashboard-view');
const chatView    = $('chat-view');
const convList    = $('conv-list');
const dashConvList= $('dashboard-conv-list');
const charCounter = $('chat-char-counter');
const dashCharCnt = $('dash-char-counter');
const topbarTitle = $('topbar-title');
const breadcrumb  = $('breadcrumb-current');
const statusText  = qs('.status-text');

/* ══════════════════════════════
   INIT
══════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  restoreTheme();
  initInputs();
  initTree();
  initActionPills();
  initChatPills();
  initSidebarToggle();
  initDarkMode();
  initMic();
  initKeyboardShortcuts();
  initGlobalSearch();
  initCmdPalette();
  initSettings();
  initSwipeGesture();
  loadConversations();
  switchView('dashboard');
});

/* ══════════════════════════════
   VIEW SWITCHING
══════════════════════════════ */
function switchView(name) {
  dashView.classList.remove('view--active');
  chatView.classList.remove('view--active');
  chatView.style.display = 'none';
  dashView.style.display = 'none';

  if (name === 'chat') {
    chatView.style.display = '';
    requestAnimationFrame(() => chatView.classList.add('view--active'));
    currentView = 'chat';
    breadcrumb && (breadcrumb.textContent = getTitle(currentConvId) || 'Chat');
    topbarTitle && (topbarTitle.textContent = getTitle(currentConvId) || '');
    setStatus('Ready');
    setTimeout(() => inputEl?.focus(), 100);
  } else {
    dashView.style.display = '';
    requestAnimationFrame(() => dashView.classList.add('view--active'));
    currentView = 'dashboard';
    breadcrumb && (breadcrumb.textContent = 'Overview');
    topbarTitle && (topbarTitle.textContent = '');
    dashInput?.focus();
  }
}

/* ══════════════════════════════
   INPUTS
══════════════════════════════ */
function autoGrow(el, maxH) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, maxH) + 'px';
}

function updateCounter(el, displayEl) {
  if (!displayEl) return;
  const n = el.value.length;
  if (n === 0) { displayEl.textContent = ''; displayEl.className = displayEl.className.replace(/ ?(warn|limit)/g, ''); return; }
  displayEl.textContent = `${n} / ${MAX_CHARS}`;
  displayEl.className = displayEl.className.replace(/ ?(warn|limit)/g, '') +
    (n > MAX_CHARS * 0.9 ? ' warn' : '') + (n >= MAX_CHARS ? ' limit' : '');
}

function initInputs() {
  sendBtn?.addEventListener('click', sendMessage);
  inputEl?.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
  inputEl?.addEventListener('input', () => { autoGrow(inputEl, 180); updateCounter(inputEl, charCounter); });
  dashSendBtn?.addEventListener('click', handleDashSend);
  dashInput?.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleDashSend(); } });
  dashInput?.addEventListener('input', () => { autoGrow(dashInput, 140); updateCounter(dashInput, dashCharCnt); });
}

async function handleDashSend() {
  const text = dashInput?.value.trim();
  if (!text) return;
  inputEl.value = text;
  autoGrow(inputEl, 180);
  dashInput.value = '';
  autoGrow(dashInput, 140);
  switchView('chat');
  if (!currentConvId) await createNewConversation();
  sendMessage();
}

/* ══════════════════════════════
   NEW CHAT
══════════════════════════════ */
$('new-chat-btn')?.addEventListener('click', newConversation);

async function newConversation() {
  await createNewConversation();
  switchView('chat');
}

/* ══════════════════════════════
   TREE
══════════════════════════════ */
function initTree() {
  document.querySelectorAll('.tree-item[data-toggle]').forEach(item => {
    const open = () => {
      const children = $(item.dataset.toggle);
      if (!children) return;
      const isOpen = children.classList.toggle('open');
      item.querySelector('.tree-arrow')?.classList.toggle('open', isOpen);
      item.setAttribute('aria-expanded', String(isOpen));
      document.querySelectorAll('.tree-item').forEach(el => el.classList.remove('active'));
      item.classList.add('active');
    };
    item.addEventListener('click', open);
    item.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  });

  document.querySelectorAll('.tree-leaf').forEach(leaf => {
    leaf.addEventListener('click', async () => {
      const prompt = leaf.dataset.prompt;
      if (!prompt) return;
      inputEl.value = prompt;
      autoGrow(inputEl, 180);
      switchView('chat');
      if (!currentConvId) await createNewConversation();
      sendMessage();
    });
  });
}

/* ══════════════════════════════
   ACTION PILLS
══════════════════════════════ */
function initActionPills() {
  document.querySelectorAll('.action-pill[data-action]').forEach(btn => {
    btn.addEventListener('click', () => quickAction(btn.dataset.action));
  });
}

function initChatPills() {
  document.querySelectorAll('.chat-pill[data-action]').forEach(btn => {
    btn.addEventListener('click', () => {
      inputEl.value = ACTION_PROMPTS[btn.dataset.action] || '';
      inputEl.focus();
      autoGrow(inputEl, 180);
    });
  });
}

async function quickAction(action) {
  inputEl.value = ACTION_PROMPTS[action] || '';
  autoGrow(inputEl, 180);
  switchView('chat');
  if (!currentConvId) await createNewConversation();
  sendMessage();
}

/* ══════════════════════════════
   SIDEBAR TOGGLE
══════════════════════════════ */
function initSidebarToggle() {
  $('sidebar-toggle')?.addEventListener('click', () => {
    appEl.classList.toggle('sidebar-collapsed');
  });
  document.addEventListener('click', e => {
    if (window.innerWidth > 720) return;
    const sidebar  = $('sidebar');
    const toggle   = $('sidebar-toggle');
    if (!appEl.classList.contains('sidebar-collapsed') &&
        sidebar && !sidebar.contains(e.target) &&
        e.target !== toggle && !toggle?.contains(e.target)) {
      appEl.classList.add('sidebar-collapsed');
    }
  });
}

/* ══════════════════════════════
   DARK MODE
══════════════════════════════ */
function restoreTheme() {
  const stored = localStorage.getItem('ai_theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (stored === 'dark' || (!stored && prefersDark)) applyTheme('dark');
}

function applyTheme(mode) {
  const isDark = mode === 'dark';
  document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
  localStorage.setItem('ai_theme', isDark ? 'dark' : 'light');
  const moonIcon = $('theme-icon-moon');
  const sunIcon  = $('theme-icon-sun');
  const label    = qs('.theme-label');
  if (moonIcon) moonIcon.style.display = isDark ? 'none' : '';
  if (sunIcon)  sunIcon.style.display  = isDark ? '' : 'none';
  if (label)    label.textContent      = isDark ? 'Light mode' : 'Dark mode';
}

function toggleTheme() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  applyTheme(isDark ? 'light' : 'dark');
}

function initDarkMode() {
  $('dark-mode-toggle')?.addEventListener('click', toggleTheme);
}

/* ══════════════════════════════
   GLOBAL SEARCH
══════════════════════════════ */
function initGlobalSearch() {
  const el = $('global-search');
  if (!el) return;
  let timer;
  el.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => filterConvs(el.value.trim().toLowerCase()), 220);
  });
  el.addEventListener('keydown', e => {
    if (e.key === 'Escape') { el.value = ''; filterConvs(''); }
  });
}

function filterConvs(q) {
  document.querySelectorAll('.conv-item').forEach(item => {
    const title = item.querySelector('.conv-item-title')?.textContent.toLowerCase() || '';
    item.style.display = !q || title.includes(q) ? '' : 'none';
  });
}

function focusSearch() { $('global-search')?.focus(); }

/* ══════════════════════════════
   MIC / SPEECH
══════════════════════════════ */
function initMic() {
  const micBtn = $('mic-btn');
  if (!micBtn) return;
  const SpeechRecog = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecog) { micBtn.disabled = true; micBtn.title = 'Voice input not supported'; return; }
  recognition = new SpeechRecog();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';
  micBtn.addEventListener('click', () => {
    if (isListening) { recognition.stop(); return; }
    try { recognition.start(); isListening = true; micBtn.classList.add('listening'); inputEl.placeholder = 'Listening…'; } catch (e) { showToast('Could not start microphone'); }
  });
  recognition.onresult = e => {
    let transcript = '';
    for (let i = e.resultIndex; i < e.results.length; i++) transcript += e.results[i][0].transcript;
    inputEl.value = transcript;
    autoGrow(inputEl, 180);
  };
  recognition.onend = () => { isListening = false; micBtn.classList.remove('listening'); inputEl.placeholder = 'Ask me anything…'; if (inputEl.value.trim()) sendMessage(); };
  recognition.onerror = ev => { isListening = false; micBtn.classList.remove('listening'); inputEl.placeholder = 'Ask me anything…'; if (ev.error === 'not-allowed') showToast('Microphone access denied'); else showToast('Voice input error: ' + ev.error); };
}

/* ══════════════════════════════
   KEYBOARD SHORTCUTS
══════════════════════════════ */
function initKeyboardShortcuts() {
  document.addEventListener('keydown', e => {
    const tag = document.activeElement?.tagName;
    const typing = tag === 'TEXTAREA' || tag === 'INPUT';
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') { e.preventDefault(); newConversation(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') { e.preventDefault(); toggleCmdPalette(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key === 'd') { e.preventDefault(); toggleTheme(); return; }
    if (e.key === '/' && !typing) { e.preventDefault(); focusSearch(); return; }
    if (e.key === 'Escape') {
      if (cmdOpen) { closeCmdPalette(); return; }
      if (currentView === 'chat' && !inputEl.value.trim()) switchView('dashboard');
    }
  });
}

/* ══════════════════════════════
   COMMAND PALETTE
══════════════════════════════ */
function initCmdPalette() {
  const palette = $('cmd-palette');
  const overlay = palette?.querySelector('.cmd-overlay');
  const cmdInput= $('cmd-input');
  const cmdList = $('cmd-list');
  if (!palette) return;
  overlay?.addEventListener('click', closeCmdPalette);
  cmdInput?.addEventListener('input', () => renderCmdItems(cmdInput.value.trim().toLowerCase()));
  cmdInput?.addEventListener('keydown', e => {
    const items = cmdList.querySelectorAll('.cmd-item');
    const active = cmdList.querySelector('.cmd-item.active');
    if (e.key === 'ArrowDown') { e.preventDefault(); const next = active?.nextElementSibling || items[0]; active?.classList.remove('active'); next?.classList.add('active'); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); const prev = active?.previousElementSibling || items[items.length - 1]; active?.classList.remove('active'); prev?.classList.add('active'); }
    else if (e.key === 'Enter') { e.preventDefault(); const focused = cmdList.querySelector('.cmd-item.active') || cmdList.querySelector('.cmd-item'); focused?.click(); }
  });
  renderCmdItems('');
}

function renderCmdItems(q) {
  const cmdList = $('cmd-list');
  if (!cmdList) return;
  cmdList.innerHTML = '';
  const filtered = q ? CMD_PALETTE_ITEMS.filter(i => i.name.toLowerCase().includes(q) || i.desc.toLowerCase().includes(q)) : CMD_PALETTE_ITEMS;
  if (!filtered.length) { cmdList.innerHTML = '<div style="padding:20px;text-align:center;color:var(--tx-3);font-size:13px">No commands found</div>'; return; }
  filtered.forEach((item, idx) => {
    const el = document.createElement('div');
    el.className = 'cmd-item' + (idx === 0 ? ' active' : '');
    el.setAttribute('role', 'option');
    el.innerHTML = `<span class="cmd-item-icon">${item.icon}</span><div class="cmd-item-text"><div class="cmd-item-name">${item.name}</div><div class="cmd-item-desc">${item.desc}</div></div>${item.kbd ? `<kbd class="cmd-item-kbd">${item.kbd}</kbd>` : ''}`;
    el.addEventListener('click', () => { closeCmdPalette(); item.action(); });
    cmdList.appendChild(el);
  });
}

function toggleCmdPalette() { cmdOpen ? closeCmdPalette() : openCmdPalette(); }
function openCmdPalette() { const p = $('cmd-palette'); if (!p) return; p.style.display = ''; cmdOpen = true; setTimeout(() => { $('cmd-input')?.focus(); renderCmdItems(''); }, 30); }
function closeCmdPalette() { const p = $('cmd-palette'); if (!p) return; p.style.display = 'none'; cmdOpen = false; }

/* ══════════════════════════════
   SETTINGS (Fast Mode, RAG, Tools)
══════════════════════════════ */
async function initSettings() {
  try {
    const r = await apiCall('GET', '/settings');
    const s = await r.json();
    const fastToggle = $('toggle-fast');
    const ragToggle = $('toggle-rag');
    if (fastToggle) fastToggle.checked = s.fast_mode === 'on';
    if (ragToggle) ragToggle.checked = s.rag_enabled;
  } catch (e) { console.error('Failed to load settings:', e); }

  $('toggle-fast')?.addEventListener('change', async (e) => {
    try {
      const r = await apiCall('POST', '/settings/fast-mode');
      const data = await r.json();
      e.target.checked = data.fast_mode === 'on';
      showToast('Fast Mode: ' + data.fast_mode.toUpperCase());
    } catch (err) { showToast('Error toggling Fast Mode'); }
  });

  $('toggle-rag')?.addEventListener('change', async (e) => {
    try {
      const r = await apiCall('POST', '/settings/rag');
      const data = await r.json();
      e.target.checked = data.rag_enabled;
      showToast('RAG: ' + (data.rag_enabled ? 'ON' : 'OFF'));
    } catch (err) { showToast('Error toggling RAG'); }
  });

  $('settings-toggle-btn')?.addEventListener('click', () => {
    const panel = $('tools-settings');
    panel?.classList.toggle('hidden');
    if (!panel?.classList.contains('hidden') && !toolsData) loadToolsSettings();
  });
}

async function loadToolsSettings() {
  try {
    const r = await apiCall('GET', '/tools');
    toolsData = await r.json();
    renderToolsSettings(toolsData);
  } catch (e) { showToast('Failed to load tools'); }
}

function initSwipeGesture() {
  const app = document.getElementById('app');
  const sidebar = document.getElementById('sidebar');
  if (!app || !sidebar) return;
  if (app.dataset.swipeBound === '1') return;
  app.dataset.swipeBound = '1';

  const EDGE_PX = 40;
  const MIN_DX = 60;
  const MAX_DY = 30;
  const MAX_DT = 300;

  let startX = 0, startY = 0, startT = 0, tracking = false;

  sidebar.addEventListener('touchstart', (e) => {
    if (e.touches.length !== 1) return;
    const t = e.touches[0];
    startX = t.clientX;
    startY = t.clientY;
    startT = Date.now();
    tracking = t.clientX <= EDGE_PX;
  }, { passive: true });

  sidebar.addEventListener('touchmove', (e) => {
    if (!tracking || e.touches.length !== 1) return;
    const t = e.touches[0];
    if (Math.abs(t.clientY - startY) > MAX_DY) tracking = false;
  }, { passive: true });

  sidebar.addEventListener('touchend', (e) => {
    if (!tracking) return;
    tracking = false;
    const dt = Date.now() - startT;
    if (dt > MAX_DT) return;
    const t = e.changedTouches[0];
    if (!t) return;
    const dx = t.clientX - startX;
    const dy = t.clientY - startY;
    if (Math.abs(dy) > MAX_DY) return;
    if (dx >= MIN_DX) {
      app.classList.remove('sidebar-collapsed');
    } else if (dx <= -MIN_DX) {
      app.classList.add('sidebar-collapsed');
    }
  }, { passive: true });
}

function renderToolsSettings(data) {
  const panel = $('tools-settings');
  if (!panel) return;
  panel.innerHTML = '';

  const totalEl = document.createElement('div');
  totalEl.style.cssText = 'padding:4px;font-size:10px;color:var(--tx-3);text-align:center';
  totalEl.textContent = data.enabled + '/' + data.total + ' tools enabled';
  panel.appendChild(totalEl);

  Object.keys(data.tools).sort().forEach(cat => {
    const tools = data.tools[cat];
    const allOn = tools.every(t => t.enabled);

    const catRow = document.createElement('div');
    catRow.className = 'cat-toggle-row';
    catRow.innerHTML = '<span>' + cat + ' (' + tools.filter(t => t.enabled).length + '/' + tools.length + ')</span>' +
      '<button class="cat-toggle-btn" data-cat="' + cat + '" data-action="' + (allOn ? 'disable' : 'enable') + '">' +
      (allOn ? 'All OFF' : 'All ON') + '</button>';
    catRow.querySelector('.cat-toggle-btn').addEventListener('click', async (e) => {
      const action = e.target.dataset.action;
      const catName = e.target.dataset.cat;
      try {
        await apiCall('POST', '/tools/category/' + catName + '/' + (action === 'enable' ? 'enable' : 'disable'));
        showToast((action === 'enable' ? 'Enabled' : 'Disabled') + ' all ' + catName);
        loadToolsSettings();
      } catch (err) { showToast('Error'); }
    });
    panel.appendChild(catRow);

    tools.forEach(t => {
      const row = document.createElement('div');
      row.className = 'tool-setting-row';
      row.innerHTML =
        '<span class="tool-setting-name">' + t.name + '</span>' +
        '<span class="tool-setting-cat">' + cat + '</span>' +
        '<button class="tool-toggle-btn ' + (t.enabled ? 'on' : 'off') + '" data-tool="' + t.name + '">' +
        (t.enabled ? 'ON' : 'OFF') + '</button>';
      row.querySelector('.tool-toggle-btn').addEventListener('click', async (e) => {
        const name = e.target.dataset.tool;
        const isOn = e.target.classList.contains('on');
        try {
          await apiCall('POST', '/tools/' + name + '/' + (isOn ? 'disable' : 'enable'));
          showToast((isOn ? 'Disabled' : 'Enabled') + ' ' + name);
          loadToolsSettings();
        } catch (err) { showToast('Error'); }
      });
      panel.appendChild(row);
    });
  });

  const btn = $('settings-toggle-btn');
  if (btn) {
    const span = btn.querySelector('span:last-child');
    if (span) span.textContent = 'Tools (' + data.enabled + '/' + data.total + ')';
  }
}

/* ══════════════════════════════
   API
══════════════════════════════ */
async function apiCall(method, path, body) {
  const opts = { method, headers: {} };
  if (body) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  const r = await fetch(API_BASE + path, opts);
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r;
}

/* ══════════════════════════════
   CONVERSATIONS
══════════════════════════════ */
async function createNewConversation() {
  messagesEl.innerHTML = '';
  try {
    const r = await apiCall('POST', '/conversations/new');
    const d = await r.json();
    currentConvId = d.conversation_id;
  } catch (_) { currentConvId = 'local_' + Date.now(); }
  await loadConversations();
  return currentConvId;
}

async function loadConversations() {
  try {
    const r = await apiCall('GET', '/conversations');
    const d = await r.json();
    renderSidebarConvs(d.conversations || [], d.current);
    renderDashConvs(d.conversations || []);
  } catch (_) { renderSidebarConvs([], null); renderDashConvs([]); }
}

function renderSidebarConvs(convs, currentId) {
  if (!convList) return;
  convList.innerHTML = '';
  if (!convs.length) { convList.innerHTML = '<div class="conv-empty">No conversations yet</div>'; return; }
  convs.forEach(cid => {
    const title = getTitle(cid);
    const item = document.createElement('div');
    item.className = 'conv-item' + (cid === currentId ? ' active' : '');
    item.setAttribute('role', 'listitem');
    item.setAttribute('tabindex', '0');
    item.innerHTML = `<div class="conv-item-icon" aria-hidden="true">💬</div><span class="conv-item-title">${escHtml(title)}</span><button class="conv-del-btn" title="Delete" aria-label="Delete conversation">×</button>`;
    item.querySelector('.conv-del-btn').addEventListener('click', e => { e.stopPropagation(); deleteConv(cid); });
    item.addEventListener('click', () => openConv(cid));
    item.addEventListener('keydown', e => { if (e.key === 'Enter') openConv(cid); });
    convList.appendChild(item);
  });
}

function renderDashConvs(convs) {
  if (!dashConvList) return;
  dashConvList.innerHTML = '';
  if (!convs.length) { dashConvList.innerHTML = '<div class="recent-empty">Start your first conversation above ↑</div>'; return; }
  convs.slice(0, 6).forEach((cid, i) => {
    const card = document.createElement('div');
    card.className = 'recent-card';
    card.setAttribute('role', 'listitem');
    card.setAttribute('tabindex', '0');
    const timeAgo = getRelativeTime(Date.now() - i * 7200000 - 3600000);
    card.innerHTML = `<div class="recent-card-icon" aria-hidden="true">💬</div><div class="recent-card-body"><div class="recent-card-title">${escHtml(getTitle(cid))}</div><div class="recent-card-time">${timeAgo}</div></div>`;
    card.addEventListener('click', () => openConv(cid));
    card.addEventListener('keydown', e => { if (e.key === 'Enter') openConv(cid); });
    dashConvList.appendChild(card);
  });
}

function getRelativeTime(ts) {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return mins + 'm ago';
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + 'h ago';
  const days = Math.floor(hrs / 24);
  if (days < 30) return days + 'd ago';
  return 'long ago';
}

async function openConv(cid) {
  if (isStreaming) return;
  currentConvId = cid;
  switchView('chat');
  breadcrumb && (breadcrumb.textContent = getTitle(cid));
  topbarTitle && (topbarTitle.textContent = getTitle(cid));
  document.querySelectorAll('.conv-item').forEach(el => {
    const t = el.querySelector('.conv-item-title')?.textContent;
    el.classList.toggle('active', t === getTitle(cid));
  });
  await loadMessages(cid);
}

async function loadMessages(cid) {
  messagesEl.innerHTML = '';
  try {
    const r = await apiCall('GET', '/conversations/' + cid);
    const d = await r.json();
    if (d.messages?.length) { d.messages.filter(m => m.role !== 'system').forEach(m => appendMessage(m.role, m.content)); }
    else { appendMessage('assistant', 'Hello! How can I help you today?'); }
  } catch (_) { appendMessage('assistant', 'Hello! How can I help you today?'); }
  scrollBottom(true);
}

function showConfirm(msg) {
  return new Promise(resolve => {
    const modal = $('confirm-modal');
    const msgEl = $('confirm-msg');
    const okBtn = $('modal-ok');
    const cancelBtn = $('modal-cancel');
    if (!modal || !msgEl || !okBtn || !cancelBtn) { resolve(confirm(msg)); return; }
    msgEl.textContent = msg;
    modal.classList.remove('hidden');
    function cleanup(result) { modal.classList.add('hidden'); okBtn.removeEventListener('click', onOk); cancelBtn.removeEventListener('click', onCancel); resolve(result); }
    function onOk() { cleanup(true); }
    function onCancel() { cleanup(false); }
    okBtn.addEventListener('click', onOk);
    cancelBtn.addEventListener('click', onCancel);
  });
}

async function deleteConv(cid) {
  if (!await showConfirm('Delete this conversation?')) return;
  try { await apiCall('DELETE', '/conversations/' + cid); } catch (_) {}
  delete convTitles[cid]; saveTitles();
  if (currentConvId === cid) { currentConvId = null; messagesEl.innerHTML = ''; switchView('dashboard'); }
  await loadConversations();
  showToast('Conversation deleted');
}

/* ══════════════════════════════
   MESSAGES
══════════════════════════════ */
function appendMessage(role, content) {
  const group = document.createElement('div');
  group.className = `msg-group msg-group--${role}`;
  const sender = document.createElement('div');
  sender.className = 'msg-sender';
  const avatarText = role === 'user' ? 'U' : '⬟';
  sender.innerHTML = `<div class="msg-sender-avatar" aria-hidden="true">${avatarText}</div><span class="msg-sender-name">${role === 'user' ? 'You' : 'AI Agent'}</span><span class="msg-sender-time">${formatTime(new Date())}</span>`;
  group.appendChild(sender);
  const bubble = document.createElement('div');
  bubble.className = `message message--${role}`;
  const content_el = document.createElement('div');
  content_el.className = 'msg-content';
  content_el.innerHTML = role === 'user' ? escHtml(content).replace(/\n/g, '<br>') : renderMd(content);
  bubble.appendChild(content_el);
  if (role === 'assistant') attachCopyBtns(bubble);
  group.appendChild(bubble);
  if (role === 'assistant') {
    const actions = document.createElement('div');
    actions.className = 'msg-actions';
    actions.innerHTML = `<button class="msg-action-btn" title="Copy response"><svg width="11" height="11" viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="9" height="9" rx="1.5" stroke="currentColor" stroke-width="1.4"/><path d="M3 11V3a1 1 0 0 1 1-1h8" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg> Copy</button><button class="msg-action-btn" title="Regenerate"><svg width="11" height="11" viewBox="0 0 16 16" fill="none"><path d="M13.5 8A5.5 5.5 0 1 1 8 2.5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/><path d="M11 2.5h2.5V5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg> Retry</button>`;
    actions.querySelectorAll('.msg-action-btn')[0]?.addEventListener('click', () => {
      navigator.clipboard.writeText(content_el.textContent).then(() => showToast('Copied to clipboard')).catch(() => showToast('Copy failed'));
    });
    group.appendChild(actions);
  }
  messagesEl.appendChild(group);
  scrollBottom();
  return { group, bubble, content_el };
}

function attachCopyBtns(parent) {
  parent.querySelectorAll('pre').forEach(pre => {
    if (pre.querySelector('.copy-code-btn')) return;
    pre.style.position = 'relative';
    const btn = document.createElement('button');
    btn.className = 'copy-code-btn';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code');
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code')?.textContent || pre.textContent;
      navigator.clipboard.writeText(code).then(() => { btn.textContent = 'Copied!'; setTimeout(() => btn.textContent = 'Copy', 1800); }).catch(() => showToast('Copy failed'));
    });
    pre.appendChild(btn);
  });
}

/* ══════════════════════════════
   SEND MESSAGE
══════════════════════════════ */
async function sendMessage() {
  const text = inputEl?.value.trim();
  if (!text || isStreaming) return;
  if (text.length > MAX_CHARS) { showToast(`Message too long (${text.length}/${MAX_CHARS})`); return; }
  const isNew = !currentConvId;
  if (!currentConvId) await createNewConversation();
  if (isNew) { setTitle(currentConvId, text); loadConversations(); }
  inputEl.value = '';
  inputEl.style.height = 'auto';
  if (charCounter) charCounter.textContent = '';
  appendMessage('user', text);
  isStreaming = true;
  sendBtn.disabled = true;
  setStatus('Thinking…');
  const { group, bubble, content_el } = appendMessage('assistant', '');
  content_el.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
  bubble.dataset.streaming = 'true';
  let toolsWrap = null;

  try {
    const r = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: currentConvId, message: text, stream: true })
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let full = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const lines = decoder.decode(value, { stream: true }).split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') break;
        try {
          const evt = JSON.parse(raw);
          if (evt.text) {
            full += evt.text;
            content_el.innerHTML = renderMd(full);
            attachCopyBtns(bubble);
            scrollBottom();
          }
          if (evt.tool_call) {
            if (!toolsWrap) { toolsWrap = document.createElement('div'); toolsWrap.className = 'tool-calls-wrap'; bubble.appendChild(toolsWrap); }
            const row = document.createElement('div');
            row.className = 'tool-call-row';
            row.dataset.tool = evt.tool_call.name || evt.tool_call;
            row.innerHTML = `<svg width="12" height="12" viewBox="0 0 16 16" fill="none" style="color:var(--tx-3);flex-shrink:0"><path d="M10.5 1.5c.5.5.5 1.5 0 2l-7 7c-.5.5-1.5.5-2 0s-.5-1.5 0-2l7-7c.5-.5 1.5-.5 2 0z" stroke="currentColor" stroke-width="1.3"/><path d="M13 5l-2-2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg><span class="tool-call-name">${escHtml(evt.tool_call.name || evt.tool_call)}</span><span class="tool-status tool-status--running">running…</span>`;
            toolsWrap.appendChild(row);
            scrollBottom();
          }
          if (evt.tool_result && toolsWrap) {
            const row = toolsWrap.querySelector(`[data-tool="${escHtml(evt.tool_result.name)}"]`);
            if (row) {
              const s = row.querySelector('.tool-status');
              s.className = 'tool-status ' + (evt.tool_result.success ? 'tool-status--success' : 'tool-status--error');
              s.textContent = evt.tool_result.success ? '✓ done' : '✗ failed';
            }
          }
        } catch (_) {}
      }
    }

    content_el.innerHTML = renderMd(full || 'Done! Let me know if you need anything else.');
    attachCopyBtns(bubble);
    delete bubble.dataset.streaming;
    await loadConversations();
    setStatus('Ready');
  } catch (err) {
    console.error('Chat error:', err);
    content_el.innerHTML = renderMd(`⚠️ **Connection error** — couldn't reach the server.\n\nYour message: *"${escHtml(text)}"*\n\nPlease check your connection and try again.`);
    setStatus('Error', true);
  }

  isStreaming = false;
  sendBtn.disabled = false;
  inputEl?.focus();
  scrollBottom();
}

/* ══════════════════════════════
   HELPERS
══════════════════════════════ */
function sanitizeHtml(html) {
  const temp = document.createElement('div');
  temp.innerHTML = html;
  temp.querySelectorAll('script,iframe,object,embed,form,input,textarea,button,link,meta,style,base').forEach(el => el.remove());
  temp.querySelectorAll('*').forEach(el => {
    [...el.attributes].forEach(attr => {
      if (attr.name.startsWith('on')) el.removeAttribute(attr.name);
    });
  });
  return temp.innerHTML;
}

function renderMd(text) {
  const raw = typeof marked !== 'undefined'
    ? (() => { try { return marked.parse(String(text), { breaks: true, gfm: true }); } catch (_) { return null; } })()
    : null;
  if (raw) return sanitizeHtml(raw);
  return String(text).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/```[\w]*\n([\s\S]*?)```/g, '<pre><code>$1</code></pre>').replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\*(.+?)\*/g, '<em>$1</em>').replace(/\n/g, '<br>');
}

function escHtml(str) { const d = document.createElement('div'); d.textContent = String(str ?? ''); return d.innerHTML; }
function formatTime(date) { return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
function scrollBottom(force) { if (!messagesEl) return; const nearBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 150; if (force || nearBottom) messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: force ? 'instant' : 'smooth' }); }
function setStatus(text, isError) { if (!statusText) return; statusText.textContent = text; const dot = qs('.status-dot'); if (dot) dot.style.background = isError ? '#ef4444' : '#22c55e'; }
let _toastTimer;
function showToast(msg, duration = 2400) { const el = $('toast'); if (!el) return; el.textContent = msg; el.classList.remove('hidden'); clearTimeout(_toastTimer); _toastTimer = setTimeout(() => el.classList.add('hidden'), duration); }
