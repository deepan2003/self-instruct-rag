/**
 * Cursor iOS Experience - Autonomous Agent & Cloud Workspaces
 * Inspired by official Cursor iOS app curated by Mobbin
 */

const API_BASE = ''; // Express server on same origin

// Application State
const state = {
  currentThread: localStorage.getItem('current_thread') || null,
  activeView: 'welcome',
  topK: 4,
  threshold: 0.10,
  isProcessing: false,
  attachment: null, // { name, type, size, data }
  prMerged: false,
  speechRecognition: null,
  isListening: false
};

// DOM Cache
const dom = {
  // Presentation
  btnToggleFrameMode: document.getElementById('btnToggleFrameMode'),
  frameToggleIcon: document.getElementById('frameToggleIcon'),
  frameToggleLabel: document.getElementById('frameToggleLabel'),
  statusClock: document.getElementById('statusClock'),
  dynamicIsland: document.getElementById('dynamicIsland'),
  islandIndicator: document.getElementById('islandIndicator'),
  iosToast: document.getElementById('iosToast'),

  // Views
  viewWelcome: document.getElementById('viewWelcome'),
  viewAllAgents: document.getElementById('viewAllAgents'),
  viewThread: document.getElementById('viewThread'),
  viewPR: document.getElementById('viewPR'),

  // Welcome View
  welcomeAuthForm: document.getElementById('welcomeAuthForm'),
  welcomeThreadInput: document.getElementById('welcomeThreadInput'),

  // All Agents View
  btnAgentsBackToWelcome: document.getElementById('btnAgentsBackToWelcome'),
  btnAgentsMenu: document.getElementById('btnAgentsMenu'),
  currentWorkspaceName: document.getElementById('currentWorkspaceName'),
  workspaceItemRow: document.getElementById('workspaceItemRow'),
  btnConnectSourceControl: document.getElementById('btnConnectSourceControl'),
  connectBtnText: document.getElementById('connectBtnText'),
  homeInputForm: document.getElementById('homeInputForm'),
  homeInputText: document.getElementById('homeInputText'),
  homeAttachBtn: document.getElementById('homeAttachBtn'),
  homeMicBtn: document.getElementById('homeMicBtn'),

  // Thread View
  btnThreadBackToAgents: document.getElementById('btnThreadBackToAgents'),
  displayActiveThreadTitle: document.getElementById('displayActiveThreadTitle'),
  btnThreadMore: document.getElementById('btnThreadMore'),
  threadMessagesContainer: document.getElementById('threadMessagesContainer'),
  threadActionsBar: document.getElementById('threadActionsBar'),
  btnOpenPRView: document.getElementById('btnOpenPRView'),
  btnMarkReady: document.getElementById('btnMarkReady'),
  markReadyLabel: document.getElementById('markReadyLabel'),
  actionPillDiffStats: document.getElementById('actionPillDiffStats'),

  // Thread Input
  threadInputForm: document.getElementById('threadInputForm'),
  threadFileInput: document.getElementById('threadFileInput'),
  threadAttachBtn: document.getElementById('threadAttachBtn'),
  threadChatInput: document.getElementById('threadChatInput'),
  threadMicBtn: document.getElementById('threadMicBtn'),
  threadSendBtn: document.getElementById('threadSendBtn'),
  sendBtnIcon: document.getElementById('sendBtnIcon'),
  threadAttachmentPreview: document.getElementById('threadAttachmentPreview'),
  attachChipIcon: document.getElementById('attachChipIcon'),
  attachChipImg: document.getElementById('attachChipImg'),
  attachChipName: document.getElementById('attachChipName'),
  btnRemoveAttachment: document.getElementById('btnRemoveAttachment'),

  // PR View
  btnPRBackToThread: document.getElementById('btnPRBackToThread'),
  btnCopyPRLink: document.getElementById('btnCopyPRLink'),
  btnPRMenu: document.getElementById('btnPRMenu'),
  prMainTitle: document.getElementById('prMainTitle'),
  prStatusBadge: document.getElementById('prStatusBadge'),
  btnSquashAndMerge: document.getElementById('btnSquashAndMerge'),
  squashMergeLabel: document.getElementById('squashMergeLabel'),
  demoTempDisplay: document.getElementById('demoTempDisplay'),
  demoLocationName: document.getElementById('demoLocationName'),
  demoConditionLabel: document.getElementById('demoConditionLabel'),
  weatherDemoCard: document.getElementById('weatherDemoCard'),

  // Settings Sheet
  settingsSheet: document.getElementById('settingsSheet'),
  sheetBackdrop: document.getElementById('sheetBackdrop'),
  btnCloseSheet: document.getElementById('btnCloseSheet'),
  sheetThreadName: document.getElementById('sheetThreadName'),
  sheetAvatar: document.getElementById('sheetAvatar'),
  sheetTopKSlider: document.getElementById('sheetTopKSlider'),
  sheetTopKVal: document.getElementById('sheetTopKVal'),
  sheetThresholdSlider: document.getElementById('sheetThresholdSlider'),
  sheetThresholdVal: document.getElementById('sheetThresholdVal'),
  btnClearHistoryBtn: document.getElementById('btnClearHistoryBtn'),
  btnSwitchAccountBtn: document.getElementById('btnSwitchAccountBtn')
};

// ==============================================================================
// INITIALIZATION
// ==============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initClock();
  setupEventListeners();
  initSpeechRecognition();

  if (!state.currentThread) {
    switchView('welcome');
  } else {
    updateWorkspaceLabels(state.currentThread);
    switchView('agents');
    loadThreadHistory(state.currentThread);
  }
});

// Update iOS Status Bar Clock
function initClock() {
  function updateTime() {
    const now = new Date();
    let hours = now.getHours();
    let minutes = now.getMinutes();
    hours = hours % 12 || 12;
    minutes = minutes < 10 ? '0' + minutes : minutes;
    if (dom.statusClock) {
      dom.statusClock.textContent = `${hours}:${minutes}`;
    }
  }
  updateTime();
  setInterval(updateTime, 30000);
}

// ==============================================================================
// VIEW SWITCHER (Welcome -> All Agents -> Thread -> PR)
// ==============================================================================
function switchView(viewName) {
  state.activeView = viewName;

  dom.viewWelcome.style.display = viewName === 'welcome' ? 'flex' : 'none';
  dom.viewAllAgents.style.display = viewName === 'agents' ? 'flex' : 'none';
  dom.viewThread.style.display = viewName === 'thread' ? 'flex' : 'none';
  dom.viewPR.style.display = viewName === 'pr' ? 'flex' : 'none';

  if (viewName === 'thread') {
    scrollThreadToBottom();
    setTimeout(() => dom.threadChatInput?.focus(), 150);
  }
}

// ==============================================================================
// EVENT LISTENERS
// ==============================================================================
function setupEventListeners() {
  // Desktop Frame Toggle (iPhone Frame vs Fullscreen)
  dom.btnToggleFrameMode?.addEventListener('click', () => {
    const isFrame = document.body.classList.contains('desktop-frame-mode');
    if (isFrame) {
      document.body.classList.remove('desktop-frame-mode');
      document.body.classList.add('desktop-full-mode');
      dom.frameToggleIcon.textContent = '🖥️';
      dom.frameToggleLabel.textContent = 'Full Width Mode';
      showToast('Switched to Full Width View');
    } else {
      document.body.classList.remove('desktop-full-mode');
      document.body.classList.add('desktop-frame-mode');
      dom.frameToggleIcon.textContent = '📱';
      dom.frameToggleLabel.textContent = 'iPhone Frame View';
      showToast('Switched to iPhone Frame View');
    }
  });

  // Welcome Form Submit
  dom.welcomeAuthForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    const threadId = dom.welcomeThreadInput.value.trim() || 'web-client';
    setThread(threadId);
    switchView('agents');
    showToast(`Workspace connected: ${threadId}`);
  });

  // All Agents Navigation
  dom.btnAgentsBackToWelcome?.addEventListener('click', () => {
    switchView('welcome');
  });

  dom.workspaceItemRow?.addEventListener('click', () => {
    switchView('thread');
  });

  dom.btnAgentsMenu?.addEventListener('click', openSettingsSheet);
  dom.btnThreadMore?.addEventListener('click', openSettingsSheet);
  dom.btnPRMenu?.addEventListener('click', openSettingsSheet);

  // Connect Source Control Button
  dom.btnConnectSourceControl?.addEventListener('click', () => {
    const isConnected = dom.btnConnectSourceControl.classList.toggle('connected');
    if (isConnected) {
      dom.connectBtnText.textContent = '✓ Connected to GitHub';
      showToast('Source Control connected: GitHub @deepan2003');
    } else {
      dom.connectBtnText.textContent = 'Connect';
      showToast('Source Control disconnected');
    }
  });

  // Quick Spark Buttons on All Agents screen
  document.querySelectorAll('.spark-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      if (prompt) {
        switchView('thread');
        executeAgentQuery(prompt);
      }
    });
  });

  // Home Bottom Input Bar
  dom.homeInputForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = dom.homeInputText.value.trim();
    if (!text && !state.attachment) return;
    dom.homeInputText.value = '';
    switchView('thread');
    executeAgentQuery(text);
  });

  // Thread Header Navigation
  dom.btnThreadBackToAgents?.addEventListener('click', () => {
    switchView('agents');
  });

  // Thread Input Form Submit
  dom.threadInputForm?.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = dom.threadChatInput.value.trim();
    if (!text && !state.attachment) return;
    dom.threadChatInput.value = '';
    dom.threadChatInput.style.height = 'auto';
    executeAgentQuery(text);
  });

  // Auto-grow textarea
  dom.threadChatInput?.addEventListener('input', () => {
    dom.threadChatInput.style.height = 'auto';
    dom.threadChatInput.style.height = Math.min(dom.threadChatInput.scrollHeight, 100) + 'px';
  });

  // Enter to send (Shift+Enter for newline)
  dom.threadChatInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      dom.threadInputForm.dispatchEvent(new Event('submit'));
    }
  });

  // File Upload Handlers (Thread & Home)
  dom.homeAttachBtn?.addEventListener('click', () => dom.threadFileInput.click());
  dom.threadAttachBtn?.addEventListener('click', () => dom.threadFileInput.click());

  dom.threadFileInput?.addEventListener('change', handleFileSelected);
  dom.btnRemoveAttachment?.addEventListener('click', clearAttachment);

  // Bottom Action Buttons (View PR & Mark Ready)
  dom.btnOpenPRView?.addEventListener('click', () => {
    switchView('pr');
  });

  dom.btnMarkReady?.addEventListener('click', () => {
    const isReady = dom.btnMarkReady.classList.toggle('ready');
    if (isReady) {
      dom.markReadyLabel.textContent = '✓ Ready for Review';
      showToast('Task marked ready for review');
    } else {
      dom.markReadyLabel.textContent = 'Mark Ready';
    }
  });

  // PR Navigation
  dom.btnPRBackToThread?.addEventListener('click', () => {
    switchView('thread');
  });

  dom.btnCopyPRLink?.addEventListener('click', () => {
    navigator.clipboard.writeText(window.location.href);
    showToast('PR link copied to clipboard');
  });

  // Squash & Merge Button
  dom.btnSquashAndMerge?.addEventListener('click', () => {
    if (state.prMerged) return;
    state.prMerged = true;
    dom.btnSquashAndMerge.classList.add('merged');
    dom.squashMergeLabel.textContent = '✓ Merged and Deployed';
    dom.prStatusBadge.textContent = 'Merged';
    dom.prStatusBadge.style.backgroundColor = '#E0E7FF';
    dom.prStatusBadge.style.color = '#3730A3';
    showToast('Pull request #1 successfully squashed & merged!');
  });

  // Demo Interactive Condition Switcher in PR View
  document.querySelectorAll('.demo-switch-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.demo-switch-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const temp = btn.getAttribute('data-temp');
      const city = btn.getAttribute('data-city');
      const label = btn.getAttribute('data-label');
      const condition = btn.getAttribute('data-condition');

      dom.demoTempDisplay.textContent = temp;
      dom.demoLocationName.textContent = city;
      dom.demoConditionLabel.textContent = label;

      updateDemoVisuals(condition);
    });
  });

  // Voice Microphone Buttons
  dom.homeMicBtn?.addEventListener('click', toggleVoiceDictation);
  dom.threadMicBtn?.addEventListener('click', toggleVoiceDictation);

  // Settings Sheet Controls
  dom.btnCloseSheet?.addEventListener('click', closeSettingsSheet);
  dom.sheetBackdrop?.addEventListener('click', closeSettingsSheet);

  dom.sheetTopKSlider?.addEventListener('input', (e) => {
    state.topK = parseInt(e.target.value);
    dom.sheetTopKVal.textContent = state.topK;
  });

  dom.sheetThresholdSlider?.addEventListener('input', (e) => {
    state.threshold = parseFloat(e.target.value);
    dom.sheetThresholdVal.textContent = state.threshold.toFixed(2);
  });

  dom.btnClearHistoryBtn?.addEventListener('click', async () => {
    if (!state.currentThread) return;
    if (confirm('Are you sure you want to clear chat history for this workspace?')) {
      await clearCurrentThread();
      closeSettingsSheet();
      showToast('Chat history cleared');
    }
  });

  dom.btnSwitchAccountBtn?.addEventListener('click', () => {
    closeSettingsSheet();
    switchView('welcome');
    showToast('Switched to login screen');
  });
}

// ==============================================================================
// WEATHER DEMO CARD VISUAL SWITCHER
// ==============================================================================
function updateDemoVisuals(condition) {
  const card = dom.weatherDemoCard;
  const sunOrb = document.getElementById('sunOrbGlow');

  if (condition === 'sunny') {
    card.style.background = 'linear-gradient(180deg, #0284C7 0%, #38BDF8 60%, #BAE6FD 100%)';
    if (sunOrb) sunOrb.style.display = 'block';
  } else if (condition === 'rainy') {
    card.style.background = 'linear-gradient(180deg, #334155 0%, #475569 55%, #64748B 100%)';
    if (sunOrb) sunOrb.style.display = 'none';
  } else if (condition === 'storm') {
    card.style.background = 'linear-gradient(180deg, #1E1B4B 0%, #312E81 60%, #4338CA 100%)';
    if (sunOrb) sunOrb.style.display = 'none';
  } else {
    // Partly cloudy
    card.style.background = 'linear-gradient(180deg, #1E88E5 0%, #42A5F5 45%, #90CAF9 100%)';
    if (sunOrb) sunOrb.style.display = 'block';
  }
}

// ==============================================================================
// VOICE SPEECH RECOGNITION
// ==============================================================================
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      state.isListening = true;
      dom.threadMicBtn?.classList.add('listening');
      dom.homeMicBtn?.classList.add('listening');
      showToast('Listening... Speak now');
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      if (dom.threadChatInput) dom.threadChatInput.value = transcript;
      if (dom.homeInputText) dom.homeInputText.value = transcript;
      showToast(`Recognized: "${transcript}"`);
    };

    recognition.onerror = () => {
      state.isListening = false;
      dom.threadMicBtn?.classList.remove('listening');
      dom.homeMicBtn?.classList.remove('listening');
    };

    recognition.onend = () => {
      state.isListening = false;
      dom.threadMicBtn?.classList.remove('listening');
      dom.homeMicBtn?.classList.remove('listening');
    };

    state.speechRecognition = recognition;
  }
}

function toggleVoiceDictation() {
  if (!state.speechRecognition) {
    showToast('Voice dictation simulated (Type or speak in supported browsers)');
    return;
  }
  if (state.isListening) {
    state.speechRecognition.stop();
  } else {
    try {
      state.speechRecognition.start();
    } catch (e) {
      showToast('Mic active');
    }
  }
}

// ==============================================================================
// FILE ATTACHMENTS
// ==============================================================================
function handleFileSelected(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    state.attachment = {
      name: file.name,
      type: file.type || 'application/octet-stream',
      size: file.size,
      data: reader.result
    };

    // Show Chip Preview
    if (dom.threadAttachmentPreview) {
      dom.threadAttachmentPreview.style.display = 'flex';
      dom.attachChipName.textContent = file.name;

      if (file.type.startsWith('image/')) {
        dom.attachChipImg.src = reader.result;
        dom.attachChipImg.style.display = 'inline-block';
        dom.attachChipIcon.style.display = 'none';
      } else {
        dom.attachChipImg.style.display = 'none';
        dom.attachChipIcon.style.display = 'inline-block';
        dom.attachChipIcon.textContent = getFileEmoji(file.name);
      }
    }

    showToast(`Attached: ${file.name}`);
    dom.threadChatInput?.focus();
  };

  reader.readAsDataURL(file);
}

function clearAttachment() {
  state.attachment = null;
  if (dom.threadFileInput) dom.threadFileInput.value = '';
  if (dom.threadAttachmentPreview) dom.threadAttachmentPreview.style.display = 'none';
  if (dom.attachChipImg) dom.attachChipImg.src = '';
}

function getFileEmoji(name) {
  const ext = (name || '').split('.').pop().toLowerCase();
  if (ext === 'pdf') return '📕';
  if (['doc', 'docx'].includes(ext)) return '📘';
  if (['xls', 'xlsx', 'csv'].includes(ext)) return '📊';
  if (['py', 'js', 'json', 'ts'].includes(ext)) return '💻';
  return '📄';
}

// ==============================================================================
// THREAD & WORKSPACE SESSION
// ==============================================================================
function setThread(threadId) {
  state.currentThread = threadId;
  localStorage.setItem('current_thread', threadId);
  updateWorkspaceLabels(threadId);
}

function updateWorkspaceLabels(threadId) {
  if (dom.currentWorkspaceName) dom.currentWorkspaceName.textContent = threadId;
  if (dom.displayActiveThreadTitle) dom.displayActiveThreadTitle.textContent = `${threadId} · agent`;
  if (dom.sheetThreadName) dom.sheetThreadName.textContent = threadId;
  if (dom.sheetAvatar) dom.sheetAvatar.textContent = threadId.charAt(0).toUpperCase();
  if (dom.welcomeThreadInput) dom.welcomeThreadInput.value = threadId;
}

async function loadThreadHistory(threadId) {
  try {
    const res = await fetch(`${API_BASE}/api/history?thread_id=${encodeURIComponent(threadId)}`);
    const data = await res.json();
    const messages = data.messages || [];

    dom.threadMessagesContainer.innerHTML = '';

    if (messages.length === 0) {
      renderWelcomeThreadStarters();
    } else {
      messages.forEach(m => {
        appendChatMessage(m.role, m.content, m.response);
      });
      dom.threadActionsBar.style.display = 'flex';
    }
  } catch (err) {
    renderWelcomeThreadStarters();
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
    dom.threadMessagesContainer.innerHTML = '';
    renderWelcomeThreadStarters();
    dom.threadActionsBar.style.display = 'none';
  } catch (err) {
    showToast('Failed to clear thread');
  }
}

function renderWelcomeThreadStarters() {
  const welcomeWrap = document.createElement('div');
  welcomeWrap.className = 'assistant-msg-wrap';
  welcomeWrap.innerHTML = `
    <div class="assistant-body-text">
      <h2>Welcome to your Cursor Cloud Agent</h2>
      <p>I can draft emails, run python scripts, check live Tamil Nadu weather, retrieve research citations, and build visual components.</p>
    </div>
  `;
  dom.threadMessagesContainer.appendChild(welcomeWrap);
}

// ==============================================================================
// AGENT QUERY EXECUTION & LIVE REACT PIPELINE
// ==============================================================================
async function executeAgentQuery(queryText) {
  if (state.isProcessing) return;
  let prompt = queryText ? queryText.trim() : '';
  const currentAttachment = state.attachment;

  if (!prompt && !currentAttachment) return;

  if (!prompt && currentAttachment) {
    prompt = currentAttachment.type.startsWith('image/')
      ? "Analyze and describe the visual architecture and elements of this image."
      : "Summarize this document and extract its key takeaways.";
  }

  // Update header title to reflect active query
  if (dom.displayActiveThreadTitle) {
    dom.displayActiveThreadTitle.textContent = truncate(prompt, 26);
  }

  // Clear attachment
  clearAttachment();

  state.isProcessing = true;
  setRunningIndicator(true);

  // Append User Bubble
  appendUserMessage(prompt, currentAttachment);

  // Create Assistant Message Placeholder
  const assistantCard = createAssistantMessageCard();
  dom.threadMessagesContainer.appendChild(assistantCard.container);
  scrollThreadToBottom();

  const startTime = Date.now();
  const thoughtInterval = setInterval(() => {
    const elapsed = Math.round((Date.now() - startTime) / 1000);
    assistantCard.thoughtDuration.textContent = `${elapsed}s`;
  }, 1000);

  try {
    // Stage 1: Thinking & MCP tools exploration
    await delay(350);
    assistantCard.addTraceStep('thinking', 'Thinking');
    await delay(300);
    assistantCard.addTraceStep('command', 'Ran command ls -la /workspace && dev_env -v');
    await delay(350);
    assistantCard.addTraceStep('command', 'Explored · get_mcp_tools');
    await delay(300);
    assistantCard.addTraceStep('thinking', 'Planning architecture and component demo...');

    // Call Backend API
    const res = await fetch(`${API_BASE}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: prompt,
        thread_id: state.currentThread || 'web-client',
        top_k: state.topK,
        threshold: state.threshold,
        file_data: currentAttachment
      })
    });

    const data = await res.json();
    if (!res.ok || !data || !data.response) {
      throw new Error(data?.error || data?.details || `Backend status ${res.status}`);
    }

    const payload = data.response;
    clearInterval(thoughtInterval);
    const totalSec = Math.max(1, Math.round((Date.now() - startTime) / 1000));
    assistantCard.thoughtDuration.textContent = `${totalSec}s`;

    // Render Answer and Rich Artifacts
    assistantCard.renderContent(payload.answer, payload);

    // Show View PR & Ready action pills (Cursor iOS 43)
    dom.threadActionsBar.style.display = 'flex';

  } catch (err) {
    clearInterval(thoughtInterval);
    assistantCard.renderError(err.message);
  } finally {
    state.isProcessing = false;
    setRunningIndicator(false);
    scrollThreadToBottom();
  }
}

// Assistant Card Component Builder
function createAssistantMessageCard() {
  const container = document.createElement('div');
  container.className = 'assistant-msg-wrap';

  // Thought Pill Toggle (Cursor iOS 36)
  const thoughtToggle = document.createElement('button');
  thoughtToggle.className = 'thought-pill-toggle open';
  thoughtToggle.innerHTML = `
    <span class="thought-tag">Thought <span class="thought-duration">0s</span></span>
    <span class="thought-chevron">▼</span>
  `;

  // Reasoning Trace Box
  const traceBox = document.createElement('div');
  traceBox.className = 'reasoning-trace-box';

  // Toggle Thought dropdown
  thoughtToggle.addEventListener('click', () => {
    thoughtToggle.classList.toggle('open');
    traceBox.style.display = thoughtToggle.classList.contains('open') ? 'flex' : 'none';
  });

  // Body Content Container
  const bodyContent = document.createElement('div');
  bodyContent.className = 'assistant-body-text';

  container.appendChild(thoughtToggle);
  container.appendChild(traceBox);
  container.appendChild(bodyContent);

  const thoughtDuration = thoughtToggle.querySelector('.thought-duration');

  return {
    container,
    thoughtDuration,
    addTraceStep: (type, text) => {
      const step = document.createElement('div');
      step.className = `trace-step-item ${type}`;
      step.textContent = text;
      traceBox.appendChild(step);
      scrollThreadToBottom();
    },
    renderContent: (answerText, payload) => {
      bodyContent.innerHTML = formatMarkdown(answerText);

      // If code demo instructions exist, render Live Demo Card (Cursor iOS 43)
      if (answerText.includes('npm') || answerText.includes('run') || answerText.includes('cd')) {
        bodyContent.appendChild(renderLiveDemoTerminalCard());
      }

      // Changes Card (Cursor iOS 43)
      bodyContent.appendChild(renderChangesCard());

      // If Citations exist
      if (payload && payload.sources && payload.sources.length > 0) {
        bodyContent.appendChild(renderCitationsCard(payload.sources));
      }

      // If Gmail Draft exists
      if (payload && payload.gmail_draft) {
        bodyContent.appendChild(renderGmailDraftCard(payload.gmail_draft));
      }

      attachCopyListeners(container);
    },
    renderError: (errMsg) => {
      bodyContent.innerHTML = `<p style="color: #DC2626;">⚠️ <strong>Error:</strong> ${escapeHtml(errMsg)}</p>`;
    }
  };
}

// Append User Message to Thread
function appendUserMessage(content, attachment) {
  const row = document.createElement('div');
  row.className = 'user-msg-wrap';

  let attachHtml = '';
  if (attachment) {
    if (attachment.type.startsWith('image/')) {
      attachHtml = `<div style="margin-bottom: 8px;"><img src="${attachment.data}" style="max-width: 100%; border-radius: 12px; display: block;" alt="Uploaded" /></div>`;
    } else {
      attachHtml = `<div style="display: flex; align-items: center; gap: 6px; font-weight: 600; font-size: 13px; margin-bottom: 6px;">${getFileEmoji(attachment.name)} ${escapeHtml(attachment.name)}</div>`;
    }
  }

  row.innerHTML = `
    <div class="user-bubble">
      ${attachHtml}
      <span>${escapeHtml(content)}</span>
    </div>
  `;
  dom.threadMessagesContainer.appendChild(row);
  scrollThreadToBottom();
}

function appendChatMessage(role, content, payload) {
  if (role === 'user') {
    appendUserMessage(content, null);
  } else {
    const card = createAssistantMessageCard();
    card.thoughtDuration.textContent = '4s';
    card.addTraceStep('done', 'Agent reasoning verified');
    card.renderContent(content, payload);
    dom.threadMessagesContainer.appendChild(card.container);
  }
}

// ==============================================================================
// RICH CARDS (Live Demo, Changes, Citations, Gmail)
// ==============================================================================

// Live Demo Terminal Card (Cursor iOS 43)
function renderLiveDemoTerminalCard() {
  const card = document.createElement('div');
  card.className = 'code-terminal-card';
  card.innerHTML = `
    <div class="code-card-header">
      <button type="button" class="btn-code-copy" data-copy="cd weather-card-demo && npm install && npm run dev">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
        <span>Copy</span>
      </button>
    </div>
    <div class="code-card-body">
      <div class="code-line-row"><span class="line-no">1</span><span class="line-cmd">cd weather-card-demo</span></div>
      <div class="code-line-row"><span class="line-no">2</span><span class="line-cmd">npm install</span></div>
      <div class="code-line-row"><span class="line-no">3</span><span class="line-cmd">npm run dev</span></div>
    </div>
  `;
  return card;
}

// Changes Card (Cursor iOS 43)
function renderChangesCard() {
  const card = document.createElement('div');
  card.className = 'changes-card';
  card.innerHTML = `
    <div class="changes-card-header">
      <span>Changes 13</span>
    </div>
    <div class="changes-files-list">
      <div class="change-file-item">
        <div class="file-left-badge">
          <span class="badge-ftype">&lt;&gt;</span>
          <span>App.css</span>
        </div>
        <div class="change-stats">
          <span class="stat-add">+93</span>
          <span class="stat-del">-157</span>
        </div>
      </div>
      <div class="change-file-item">
        <div class="file-left-badge">
          <span class="badge-ftype ts">TS</span>
          <span>App.tsx</span>
        </div>
        <div class="change-stats">
          <span class="stat-add">+54</span>
          <span class="stat-del">-112</span>
        </div>
      </div>
      <div class="change-file-item">
        <div class="file-left-badge">
          <span class="badge-ftype md">M↓</span>
          <span>ARCHITECTURE.md</span>
        </div>
        <div class="change-stats">
          <span class="stat-add">+96</span>
          <span class="stat-del">-0</span>
        </div>
      </div>
      <div class="change-file-item">
        <div class="file-left-badge">
          <span class="badge-ftype ts">TS</span>
          <span>demoForecasts.ts</span>
        </div>
        <div class="change-stats">
          <span class="stat-add">+43</span>
          <span class="stat-del">-0</span>
        </div>
      </div>
      <div class="change-file-item">
        <div class="file-left-badge">
          <span class="badge-ftype">&lt;&gt;</span>
          <span>index.css</span>
        </div>
        <div class="change-stats">
          <span class="stat-add">+21</span>
          <span class="stat-del">-93</span>
        </div>
      </div>
    </div>
  `;
  return card;
}

// RAG Sources Card
function renderCitationsCard(sources) {
  const card = document.createElement('div');
  card.className = 'sources-card';

  const items = sources.map((s, idx) => `
    <div class="source-item">
      <strong>#${idx + 1} Page ${s.page} | ${escapeHtml(s.section)}</strong>
      <p style="margin-top: 4px; color: #4B5563;">"${escapeHtml(s.quote_snippet)}"</p>
    </div>
  `).join('');

  card.innerHTML = `
    <button class="sources-toggle" type="button">
      <span>📚 Verified ACL Paper Citations (${sources.length})</span>
      <span>▼</span>
    </button>
    <div class="sources-content" style="display: none;">
      ${items}
    </div>
  `;

  const btn = card.querySelector('.sources-toggle');
  const content = card.querySelector('.sources-content');
  btn.addEventListener('click', () => {
    const isShown = content.style.display !== 'none';
    content.style.display = isShown ? 'none' : 'flex';
  });

  return card;
}

// Gmail Draft Card in iOS style
function renderGmailDraftCard(draft) {
  const card = document.createElement('div');
  card.className = 'gmail-ios-card';
  card.innerHTML = `
    <div class="gmail-header">
      <div class="gmail-brand">
        <span>✉️ Gmail Draft</span>
      </div>
      <span class="gmail-badge">Ready for Review</span>
    </div>
    <div style="font-size: 13.5px; margin-bottom: 6px;">
      <strong>To:</strong> <code>${escapeHtml(draft.recipient || '')}</code>
    </div>
    <div style="font-size: 13.5px; margin-bottom: 8px;">
      <strong>Subject:</strong> ${escapeHtml(draft.subject || '')}
    </div>
    <div class="gmail-preview-body">${escapeHtml(draft.body || '')}</div>
    <div style="margin-top: 12px; display: flex; gap: 8px;">
      <a href="${draft.gmail_compose_url || '#'}" target="_blank" rel="noopener noreferrer" class="btn-gmail-compose">
        <span>Open & Review in Gmail ↗</span>
      </a>
      <button type="button" class="btn-code-copy" data-copy="${escapeHtml(draft.body || '')}" style="background: #F3F4F6; border-radius: 9999px; padding: 6px 14px;">
        <span>Copy Body</span>
      </button>
    </div>
  `;
  return card;
}

// ==============================================================================
// HELPERS & SETTINGS
// ==============================================================================
function setRunningIndicator(isRunning) {
  if (dom.islandIndicator) {
    if (isRunning) {
      dom.islandIndicator.classList.add('active');
    } else {
      dom.islandIndicator.classList.remove('active');
    }
  }
}

function openSettingsSheet() {
  dom.sheetBackdrop?.classList.add('active');
  dom.settingsSheet?.classList.add('open');
}

function closeSettingsSheet() {
  dom.sheetBackdrop?.classList.remove('active');
  dom.settingsSheet?.classList.remove('open');
}

function showToast(message) {
  if (!dom.iosToast) return;
  dom.iosToast.textContent = message;
  dom.iosToast.classList.add('show');
  setTimeout(() => dom.iosToast.classList.remove('show'), 2600);
}

function scrollThreadToBottom() {
  if (dom.threadMessagesContainer) {
    dom.threadMessagesContainer.scrollTop = dom.threadMessagesContainer.scrollHeight;
  }
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function truncate(str, max) {
  if (!str) return '';
  return str.length > max ? str.substring(0, max) + '...' : str;
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

function attachCopyListeners(parent) {
  parent.querySelectorAll('.btn-code-copy').forEach(btn => {
    btn.addEventListener('click', () => {
      const text = btn.getAttribute('data-copy');
      if (text) {
        navigator.clipboard.writeText(text);
        showToast('Copied to clipboard');
      }
    });
  });
}

// Basic Markdown Formatter
function formatMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);

  // Bold & Italic
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // Headings
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h2>$1</h2>');

  // Inline Code
  html = html.replace(/`([^`]+)`/g, '<code style="background: #F3F4F6; padding: 2px 6px; border-radius: 4px; font-family: var(--font-mono); font-size: 0.9em; color: #1F2937;">$1</code>');

  // Lists
  html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');

  // Line breaks
  html = html.replace(/\n\n/g, '</p><p>');
  html = `<p>${html}</p>`;
  html = html.replace(/<p><\/p>/g, '');

  return html;
}
