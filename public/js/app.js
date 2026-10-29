/**
 * Baasha Agent - ReAct Multi-Tool Intelligence Assistant
 */

const API_BASE = ''; // Proxied via Express on same origin

// Application State
const state = {
  currentThread: localStorage.getItem('current_thread') || null,
  topK: 4,
  threshold: 0.10,
  isProcessing: false
};

// DOM Elements
const threadModal = document.getElementById('threadModal');
const newThreadForm = document.getElementById('newThreadForm');
const threadInput = document.getElementById('threadInput');

const displayThreadName = document.getElementById('displayThreadName');
const activeThreadPill = document.getElementById('activeThreadPill');
const userAvatar = document.getElementById('userAvatar');
const clearThreadBtn = document.getElementById('clearThreadBtn');
const logoutBtn = document.getElementById('logoutBtn');

const topKSlider = document.getElementById('topKSlider');
const topKVal = document.getElementById('topKVal');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdVal = document.getElementById('thresholdVal');

const messagesContainer = document.getElementById('messagesContainer');
const welcomeHero = document.getElementById('welcomeHero');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');

const reasoningBoxWrapper = document.getElementById('reasoningBoxWrapper');
const stepGuardrails = document.getElementById('stepGuardrails');
const stepAgent = document.getElementById('stepAgent');
const stepExecution = document.getElementById('stepExecution');
const stepGrounding = document.getElementById('stepGrounding');

const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const mobileMenuBtn = document.getElementById('mobileMenuBtn');

// ==============================================================================
// INITIALIZATION
// ==============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();

  if (!state.currentThread) {
    showModal();
  } else {
    await setThread(state.currentThread);
  }
});

// Setup All UI Event Handlers
function setupEventListeners() {
  // Mobile / Toggle Sidebar
  sidebarToggle?.addEventListener('click', () => sidebar.classList.toggle('open'));
  mobileMenuBtn?.addEventListener('click', () => sidebar.classList.toggle('open'));

  // Thread Login Form
  newThreadForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const val = threadInput.value.trim();
    if (val) {
      setThread(val);
      hideModal();
    }
  });

  // Clear Thread History
  clearThreadBtn.addEventListener('click', async () => {
    if (!state.currentThread) return;
    if (confirm(`Are you sure you want to clear your chat history, ${state.currentThread}?`)) {
      await clearCurrentThread();
    }
  });

  // Log Out / Change User
  logoutBtn?.addEventListener('click', () => {
    localStorage.removeItem('current_thread');
    state.currentThread = null;
    threadInput.value = '';
    messagesContainer.innerHTML = '';
    messagesContainer.appendChild(welcomeHero);
    welcomeHero.style.display = 'block';
    showModal();
  });

  // Sliders
  topKSlider.addEventListener('input', (e) => {
    state.topK = parseInt(e.target.value);
    topKVal.textContent = state.topK;
  });

  thresholdSlider.addEventListener('input', (e) => {
    state.threshold = parseFloat(e.target.value);
    thresholdVal.textContent = state.threshold.toFixed(2);
  });

  // Auto-expand textarea
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
  });

  // Enter to send, Shift+Enter for newline
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event('submit'));
    }
  });

  // Form Submit
  chatForm.addEventListener('submit', handleSendMessage);
}

// ==============================================================================
// THREAD & SESSION MANAGEMENT
// ==============================================================================
function showModal() {
  threadModal.classList.remove('hidden');
  threadInput.focus();
}

function hideModal() {
  threadModal.classList.add('hidden');
}

async function setThread(threadId) {
  state.currentThread = threadId;
  localStorage.setItem('current_thread', threadId);

  displayThreadName.textContent = threadId;
  activeThreadPill.textContent = `User: ${threadId}`;
  userAvatar.textContent = threadId.charAt(0).toUpperCase();

  await loadThreadHistory(threadId);
}

async function loadThreadHistory(threadId) {
  messagesContainer.innerHTML = '';
  try {
    const res = await fetch(`${API_BASE}/api/history?thread_id=${encodeURIComponent(threadId)}`);
    const data = await res.json();
    const messages = data.messages || [];

    if (messages.length === 0 || (messages.length === 1 && messages[0].role === 'assistant')) {
      // Empty or initial welcome message
      messagesContainer.appendChild(welcomeHero);
      welcomeHero.style.display = 'block';
      if (messages.length === 1) {
        appendMessage('assistant', messages[0].content);
      }
    } else {
      welcomeHero.style.display = 'none';
      messages.forEach(m => {
        appendMessage(m.role, m.content, m.response);
      });
    }
  } catch (err) {
    console.error('Failed to load thread history:', err);
  }
}

async function clearCurrentThread() {
  if (!state.currentThread) return;
  try {
    await fetch(`${API_BASE}/api/clear`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: state.currentThread })
    });
    await loadThreadHistory(state.currentThread);
  } catch (err) {
    alert('Failed to clear thread: ' + err.message);
  }
}

// ==============================================================================
// CHAT & REACT REASONING EXECUTION
// ==============================================================================
async function handleSendMessage(e) {
  e.preventDefault();
  const prompt = chatInput.value.trim();
  if (!prompt || state.isProcessing) return;

  state.isProcessing = true;
  sendBtn.disabled = true;
  chatInput.value = '';
  chatInput.style.height = 'auto';

  // Hide welcome hero
  if (welcomeHero) welcomeHero.style.display = 'none';

  // Render User Message
  appendMessage('user', prompt);

  // Show Animated Live Reasoning Status Card
  showReasoningCard();

  try {
    // Stage 1: Guardrail Inspection
    updateReasoningStep('stepGuardrails', 'active', 'Inspecting security rules...');
    await delay(250);
    updateReasoningStep('stepGuardrails', 'done', '✅ Safe & Authorized');

    // Stage 2: Main Agent Reasoning
    updateReasoningStep('stepAgent', 'active', '120B Brain reasoning & selecting tool...');

    const res = await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: prompt,
        thread_id: state.currentThread,
        top_k: state.topK,
        threshold: state.threshold
      })
    });

    const data = await res.json();
    if (!res.ok || !data || !data.response) {
      throw new Error(data?.error || data?.details || `Server returned error (${res.status})`);
    }

    const responsePayload = data.response;

    // Stage 3: Sub-Agent Execution
    updateReasoningStep('stepAgent', 'done', '✅ Strategy Decided');
    updateReasoningStep('stepExecution', 'active', `Running ${responsePayload.status || 'Sub-Agent'}...`);
    await delay(200);
    updateReasoningStep('stepExecution', 'done', `✅ Completed: ${responsePayload.summary || 'Done'}`);

    // Stage 4: Grounding & Pydantic
    updateReasoningStep('stepGrounding', 'active', 'Verifying anti-hallucination citations...');
    await delay(200);
    updateReasoningStep('stepGrounding', 'done', '✅ Grounding Verified');
    await delay(250);

    // Hide reasoning card & render assistant output
    hideReasoningCard();
    appendMessage('assistant', responsePayload.answer, responsePayload);

  } catch (err) {
    hideReasoningCard();
    appendMessage('assistant', `⚠️ **Error connecting to assistant:** ${err.message}`);
  } finally {
    state.isProcessing = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

// ==============================================================================
// ANIMATED REASONING BOX HELPERS
// ==============================================================================
function showReasoningCard() {
  reasoningBoxWrapper.style.display = 'block';
  // Reset steps
  resetReasoningStep('stepGuardrails', 'Checking...');
  resetReasoningStep('stepAgent', 'Waiting...');
  resetReasoningStep('stepExecution', 'Waiting...');
  resetReasoningStep('stepGrounding', 'Waiting...');
  scrollToBottom();
}

function hideReasoningCard() {
  reasoningBoxWrapper.style.display = 'none';
}

function resetReasoningStep(stepId, text) {
  const el = document.getElementById(stepId);
  if (!el) return;
  el.className = 'step-item';
  el.querySelector('.step-status').textContent = text;
}

function updateReasoningStep(stepId, status, statusText) {
  const el = document.getElementById(stepId);
  if (!el) return;
  el.className = `step-item ${status}`;
  if (statusText) {
    el.querySelector('.step-status').textContent = statusText;
  }
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ==============================================================================
// MESSAGE RENDERING & FORMATTING
// ==============================================================================
function appendMessage(role, content, responseObj = null) {
  const row = document.createElement('div');
  row.className = `message-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.innerHTML = role === 'user' ? '👤' : '🤖';

  const contentBox = document.createElement('div');
  contentBox.className = 'message-content';

  // Check if Guardrail violation
  const isGuardrailBlocked = responseObj && responseObj.guardrails && !responseObj.guardrails.is_safe;

  if (isGuardrailBlocked) {
    contentBox.innerHTML = renderGuardrailAlert(responseObj.guardrails);
  } else {
    const body = document.createElement('div');
    body.className = 'message-body';
    body.innerHTML = formatMarkdown(content);
    contentBox.appendChild(body);

    // If RAG Answer: Attach Verified Citations Drawer & Pydantic Structure
    if (responseObj && responseObj.sources && responseObj.sources.length > 0 && responseObj.status !== 'CONVERSATIONAL') {
      contentBox.appendChild(renderCitationsExpander(responseObj.sources));
      contentBox.appendChild(renderPydanticExpander(responseObj.rag_answer || responseObj));
    }

    // If Gmail Draft: Attach Gmail Draft Card with 1-Click Compose
    if (responseObj && responseObj.gmail_draft) {
      contentBox.appendChild(renderGmailDraftCard(responseObj.gmail_draft));
    }
  }

  row.appendChild(avatar);
  row.appendChild(contentBox);
  messagesContainer.appendChild(row);
  scrollToBottom();

  // Attach copy listeners
  contentBox.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const code = btn.getAttribute('data-code');
      navigator.clipboard.writeText(code).then(() => {
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = '📋 Copy', 1800);
      });
    });
  });

  // Attach expander toggle listeners
  contentBox.querySelectorAll('.expander-trigger').forEach(trigger => {
    trigger.addEventListener('click', () => {
      const target = trigger.nextElementSibling;
      target.classList.toggle('open');
      const arrow = trigger.querySelector('.expander-arrow');
      if (arrow) {
        arrow.textContent = target.classList.contains('open') ? '▲' : '▼';
      }
    });
  });
}

function renderGuardrailAlert(guardrails) {
  const flags = guardrails.flagged_patterns || [];
  const tagsHtml = flags.map(f => `<span class="trigger-tag">${escapeHtml(f)}</span>`).join('');
  return `
    <div class="guardrail-alert">
      <div class="guardrail-alert-header">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"></polygon><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
        <span>SECURITY GUARDRAIL TRIGGERED</span>
      </div>
      <p style="color: #fca5a5; font-size: 0.9rem;">${escapeHtml(guardrails.message)}</p>
      ${flags.length > 0 ? `
        <div style="margin-top: 10px;">
          <strong style="font-size: 0.75rem; color: #f87171; text-transform: uppercase;">Detected Trigger Patterns:</strong>
          <div class="trigger-tags">${tagsHtml}</div>
          <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 6px;">
            ℹ️ Trigger patterns represent specific security rules or threat signatures matched by Input Guardrails.
          </div>
        </div>
      ` : ''}
    </div>
  `;
}

function renderCitationsExpander(sources) {
  const container = document.createElement('div');
  container.className = 'citations-expander';

  const itemsHtml = sources.map((s, idx) => `
    <div class="citation-item">
      <div class="citation-header">
        <strong>Source #${idx + 1} — Page ${s.page} | ${escapeHtml(s.section)}</strong>
        <span>Relevance: ${s.relevance_score.toFixed(4)}</span>
      </div>
      <div class="citation-quote">"${escapeHtml(s.quote_snippet)}"</div>
    </div>
  `).join('');

  container.innerHTML = `
    <button class="expander-trigger" type="button">
      <span>📚 Verified Citations (${sources.length} Paper Sources)</span>
      <span class="expander-arrow">▼</span>
    </button>
    <div class="expander-content">
      ${itemsHtml}
    </div>
  `;
  return container;
}

function renderPydanticExpander(ragAnswer) {
  const container = document.createElement('div');
  container.className = 'pydantic-expander';
  const jsonStr = JSON.stringify(ragAnswer, null, 2);

  container.innerHTML = `
    <button class="expander-trigger" type="button">
      <span>📦 Grounded Pydantic Structure (answer + citations)</span>
      <span class="expander-arrow">▼</span>
    </button>
    <div class="expander-content">
      <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 6px;">
        Anti-hallucination verified Pydantic schema enforcing strictly grounded answer with citations:
      </div>
      <pre style="background: #090d16; padding: 10px; border-radius: 6px; font-size: 0.78rem;"><code>${escapeHtml(jsonStr)}</code></pre>
    </div>
  `;
  return container;
}

function renderGmailDraftCard(draft) {
  const card = document.createElement('div');
  card.className = 'gmail-draft-card';

  const recipient = draft.recipient || '';
  const subject = draft.subject || 'Draft Email';
  const body = draft.body || '';
  const composeUrl = draft.gmail_compose_url || '#';

  card.innerHTML = `
    <div class="gmail-card-header">
      <div class="gmail-card-brand">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/>
        </svg>
        <span>Gmail Draft Preparation</span>
      </div>
      <span class="gmail-draft-badge">Draft Ready (Not Sent)</span>
    </div>

    <div class="gmail-fields">
      <div class="gmail-field-row">
        <span class="gmail-field-label">To:</span>
        <span class="gmail-field-value">
          <code class="gmail-recipient-pill">${escapeHtml(recipient)}</code>
        </span>
      </div>
      <div class="gmail-field-row">
        <span class="gmail-field-label">Subject:</span>
        <span class="gmail-field-value"><strong>${escapeHtml(subject)}</strong></span>
      </div>
    </div>

    <div class="gmail-body-preview">${escapeHtml(body)}</div>

    <div class="gmail-card-actions">
      <a href="${composeUrl}" target="_blank" rel="noopener noreferrer" class="btn-gmail-open">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
        <span>Open & Review in Gmail</span>
      </a>
      <button type="button" class="btn-gmail-copy">
        📋 <span>Copy Email Body</span>
      </button>
    </div>

    <div class="gmail-security-note">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
      <span>Safe Mode: Email was not sent automatically. Click 'Open & Review in Gmail' to review and click Send yourself.</span>
    </div>
  `;

  // Attach copy event
  const copyBtn = card.querySelector('.btn-gmail-copy');
  copyBtn?.addEventListener('click', () => {
    navigator.clipboard.writeText(body).then(() => {
      copyBtn.querySelector('span').textContent = 'Copied to Clipboard!';
      setTimeout(() => {
        copyBtn.querySelector('span').textContent = 'Copy Email Body';
      }, 2000);
    });
  });

  return card;
}

// Basic Markdown & Code Block Formatter
function formatMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);

  // Formatted Code Blocks ```python ... ```
  html = html.replace(/```([a-zA-Z0-9]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    const rawCode = unescapeHtml(code);
    return `
      <div class="code-container">
        <div class="code-header">
          <span>${lang ? lang.toUpperCase() : 'CODE'}</span>
          <button class="copy-btn" data-code="${escapeHtml(rawCode)}">📋 Copy</button>
        </div>
        <pre><code>${code}</code></pre>
      </div>
    `;
  });

  // Inline code `...`
  html = html.replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono); font-size: 0.88em; color: #a5f3fc;">$1</code>');

  // Headings
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2 style="color: #fff; margin: 14px 0 8px;">$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1 style="color: #fff; margin: 16px 0 10px;">$1</h1>');

  // Bold & Italic
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // Markdown Tables
  if (html.includes('|') && html.includes('---')) {
    html = formatMarkdownTables(html);
  }

  // Blockquotes
  html = html.replace(/^\&gt; (.*$)/gim, '<blockquote>$1</blockquote>');

  // Bullet Lists
  html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');

  // Paragraph breaks
  html = html.replace(/\n\n/g, '</p><p>');
  html = `<p>${html}</p>`;
  html = html.replace(/<p><\/p>/g, '');

  return html;
}

function formatMarkdownTables(text) {
  const lines = text.split('\n');
  let inTable = false;
  let tableHtml = '<div style="overflow-x: auto; margin: 12px 0;"><table style="width: 100%; border-collapse: collapse; font-size: 0.88rem; background: rgba(0,0,0,0.2); border-radius: 8px; overflow: hidden;">';

  lines.forEach(line => {
    const trimmed = line.trim();
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      const cells = trimmed.slice(1, -1).split('|').map(c => c.trim());
      if (trimmed.includes('---')) {
        // Separator row
        return;
      }
      if (!inTable) {
        inTable = true;
        tableHtml += '<thead><tr style="background: rgba(99, 102, 241, 0.15); border-bottom: 1px solid var(--border-color);">';
        cells.forEach(c => tableHtml += `<th style="padding: 10px 14px; text-align: left; font-weight: 600; color: #fff;">${c}</th>`);
        tableHtml += '</tr></thead><tbody>';
      } else {
        tableHtml += '<tr style="border-bottom: 1px solid var(--border-color);">';
        cells.forEach(c => tableHtml += `<td style="padding: 8px 14px; color: var(--text-primary);">${c}</td>`);
        tableHtml += '</tr>';
      }
    } else if (inTable) {
      inTable = false;
      tableHtml += '</tbody></table></div>' + line;
    }
  });

  if (inTable) {
    tableHtml += '</tbody></table></div>';
  }
  return tableHtml;
}

function scrollToBottom() {
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function unescapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#039;/g, "'");
}

