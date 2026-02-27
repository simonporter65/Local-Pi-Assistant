/* 
 * ui/heartbeat.js
 * Connects to /events SSE stream and updates the UI with background activity.
 * Injected into index.html at the bottom of <script>.
 */

// ─────────────────────────────────────────────
// Background events SSE connection
// ─────────────────────────────────────────────
let eventSource = null;
let taskPanelOpen = false;
let taskData = [];

function connectEvents() {
  if (eventSource) eventSource.close();
  
  eventSource = new EventSource('/events');
  
  eventSource.onmessage = (e) => {
    try {
      const event = JSON.parse(e.data);
      handleHeartbeatEvent(event);
    } catch (err) {}
  };
  
  eventSource.onerror = () => {
    updateAgentStatus('offline', 'Reconnecting...');
    setTimeout(connectEvents, 5000);
  };
  
  eventSource.onopen = () => {
    updateAgentStatus('online', 'Online · Local · Private');
  };
}

// ─────────────────────────────────────────────
// Handle heartbeat events
// ─────────────────────────────────────────────
function handleHeartbeatEvent(event) {
  const type = event.type;

  switch (type) {
    case 'connected':
      updateQueueBadge(event.queue_summary);
      break;

    case 'heartbeat_working':
      updateAgentStatus('working', `Working: ${event.task_title || '...'}`);
      updateActivityBar(event.task_title, event.task_type, 'active');
      break;

    case 'heartbeat_task_done':
      updateAgentStatus('online', 'Online · Local · Private');
      updateActivityBar(event.task_title, event.task_type, 'done');
      flashTaskBadge();
      loadTaskSummary();
      // Show a subtle notification in the chat
      if (event.summary) {
        showBackgroundCompletionToast(event.task_title, event.summary);
      }
      break;

    case 'heartbeat_task_failed':
      updateActivityBar(event.task_title, event.task_type, 'failed');
      loadTaskSummary();
      break;

    case 'heartbeat_reflecting':
      updateAgentStatus('thinking', 'Reflecting...');
      updateActivityBar('Reflection', 'reflect', 'active');
      break;

    case 'heartbeat_tasks_generated':
      updateAgentStatus('online', 'Online · Local · Private');
      loadTaskSummary();
      flashTaskBadge();
      break;

    case 'heartbeat_paused':
      updateAgentStatus('paused', 'Focused on you');
      clearActivityBar();
      break;

    case 'heartbeat_resuming':
      updateAgentStatus('online', 'Online · Local · Private');
      break;

    case 'heartbeat_idle':
      updateAgentStatus('online', 'Online · Local · Private');
      break;

    case 'heartbeat_skill_call':
      updateActivityBar(event.message, 'skill', 'active');
      break;
  }
}

// ─────────────────────────────────────────────
// Status bar updates
// ─────────────────────────────────────────────
function updateAgentStatus(state, text) {
  const dot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  const headerSub = document.getElementById('headerSub');
  
  if (statusText) statusText.textContent = text;
  if (headerSub) headerSub.textContent = text;
  
  const colors = {
    online:  '#3ecf8e',
    working: '#f59e0b',
    thinking:'#a78bfa',
    paused:  '#3b82f6',
    offline: '#ef4444',
  };
  
  if (dot) {
    dot.style.background = colors[state] || colors.online;
    dot.style.animation = state === 'working' || state === 'thinking'
      ? 'pulse 0.6s ease-in-out infinite'
      : 'pulse 2s ease-in-out infinite';
  }
}

// ─────────────────────────────────────────────
// Activity bar (bottom of sidebar)
// ─────────────────────────────────────────────
const TASK_TYPE_ICONS = {
  research:     '🔍',
  self_improve: '🔧',
  prepare:      '📋',
  remind:       '🔔',
  reflect:      '🧠',
  maintain:     '⚙️',
  custom:       '✨',
  skill:        '⚙',
};

function updateActivityBar(title, type, state) {
  let bar = document.getElementById('activityBar');
  if (!bar) return;
  
  const icon = TASK_TYPE_ICONS[type] || '•';
  const stateClass = state === 'active' ? 'activity-active' : 
                     state === 'done'   ? 'activity-done' : 'activity-failed';
  
  bar.innerHTML = `
    <div class="activity-row ${stateClass}">
      <span class="activity-icon">${icon}</span>
      <span class="activity-title">${escHtml(title || '').substring(0, 45)}</span>
      ${state === 'active' ? '<div class="activity-spinner"></div>' : ''}
    </div>
  `;
}

function clearActivityBar() {
  const bar = document.getElementById('activityBar');
  if (bar) bar.innerHTML = '';
}

// ─────────────────────────────────────────────
// Task badge on the queue button
// ─────────────────────────────────────────────
async function loadTaskSummary() {
  try {
    const r = await fetch('/tasks/summary');
    const data = await r.json();
    updateQueueBadge(data);
  } catch (e) {}
}

function updateQueueBadge(summary) {
  if (!summary) return;
  const pending = (summary.pending || 0);
  const running = (summary.running || 0);
  const badge = document.getElementById('taskBadge');
  if (!badge) return;
  const total = pending + running;
  badge.textContent = total > 0 ? total : '';
  badge.style.display = total > 0 ? 'flex' : 'none';
}

function flashTaskBadge() {
  const btn = document.getElementById('taskBtn');
  if (!btn) return;
  btn.style.borderColor = 'var(--accent)';
  btn.style.color = 'var(--accent)';
  setTimeout(() => {
    btn.style.borderColor = '';
    btn.style.color = '';
  }, 2000);
}

// ─────────────────────────────────────────────
// Subtle background completion toast
// ─────────────────────────────────────────────
function showBackgroundCompletionToast(title, summary) {
  // Only show if the task seems interesting enough
  if (!summary || summary.length < 30) return;
  
  const container = document.getElementById('messages');
  const toast = document.createElement('div');
  toast.style.cssText = `
    display: flex; align-items: flex-start; gap: 8px;
    opacity: 0; transition: opacity 0.3s; margin: 8px 0;
  `;
  toast.innerHTML = `
    <div style="width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,var(--accent),#2563eb);
         display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;">🤖</div>
    <div style="background:rgba(62,207,142,0.08);border:1px solid rgba(62,207,142,0.25);
         border-radius:18px;border-bottom-left-radius:4px;padding:10px 14px;max-width:68%;font-size:13px;line-height:1.5;">
      <div style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;
           color:var(--accent);margin-bottom:5px;">⚡ Background task complete</div>
      <div style="font-weight:500;color:var(--text);margin-bottom:4px;">${escHtml(title)}</div>
      <div style="color:var(--text-dim);font-size:12.5px;">${escHtml(summary.substring(0, 180))}${summary.length > 180 ? '...' : ''}</div>
    </div>
  `;
  container.appendChild(toast);
  scrollToBottom();
  
  // Fade in
  requestAnimationFrame(() => {
    toast.style.opacity = '1';
  });
}

// ─────────────────────────────────────────────
// Task queue panel (slide-in from right)
// ─────────────────────────────────────────────
function toggleTaskPanel() {
  taskPanelOpen = !taskPanelOpen;
  let panel = document.getElementById('taskPanel');
  
  if (!panel) {
    panel = createTaskPanel();
    document.body.appendChild(panel);
  }
  
  if (taskPanelOpen) {
    loadAndRenderTasks(panel);
    panel.style.transform = 'translateX(0)';
  } else {
    panel.style.transform = 'translateX(100%)';
  }
}

function createTaskPanel() {
  const panel = document.createElement('div');
  panel.id = 'taskPanel';
  panel.style.cssText = `
    position: fixed; right: 0; top: 0; bottom: 0;
    width: 380px; max-width: 90vw;
    background: #161a1f;
    border-left: 1px solid rgba(255,255,255,0.07);
    transform: translateX(100%);
    transition: transform 0.3s ease;
    z-index: 1000;
    display: flex; flex-direction: column;
    font-family: 'DM Sans', sans-serif;
  `;
  return panel;
}

async function loadAndRenderTasks(panel) {
  try {
    const r = await fetch('/tasks?status=pending');
    const data = await r.json();
    const pending = data.tasks || [];
    
    const rDone = await fetch('/tasks?status=done');
    const dataDone = await rDone.json();
    const done = (dataDone.tasks || []).slice(0, 10);
    
    const summary = data.summary || {};
    
    panel.innerHTML = `
      <div style="padding:20px 20px 16px;border-bottom:1px solid rgba(255,255,255,0.07);display:flex;justify-content:space-between;align-items:center;">
        <div>
          <div style="font-size:15px;font-weight:600;color:#e8eaed;">Background Tasks</div>
          <div style="font-size:12px;color:#6b7280;margin-top:2px;">
            ${summary.pending||0} pending · ${summary.done||0} completed
          </div>
        </div>
        <button onclick="toggleTaskPanel()" style="background:transparent;border:none;color:#6b7280;cursor:pointer;font-size:18px;">✕</button>
      </div>

      <div style="padding:12px 16px 8px;display:flex;gap:8px;">
        <button onclick="addTaskPrompt()" style="flex:1;padding:8px;background:rgba(62,207,142,0.15);
          border:1px solid rgba(62,207,142,0.4);color:#3ecf8e;border-radius:8px;
          cursor:pointer;font-size:12.5px;font-family:inherit;">+ Add Task</button>
        <button onclick="loadAndRenderTasks(document.getElementById('taskPanel'))"
          style="padding:8px 12px;background:transparent;border:1px solid rgba(255,255,255,0.1);
          color:#6b7280;border-radius:8px;cursor:pointer;font-size:12.5px;font-family:inherit;">↻</button>
      </div>

      <div style="flex:1;overflow-y:auto;padding:0 16px 16px;">
        ${pending.length ? `
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#4b5563;
               padding:8px 0 6px;font-weight:500;">Pending (${pending.length})</div>
          ${pending.map(renderTaskCard).join('')}
        ` : '<div style="padding:20px;color:#4b5563;font-size:13px;text-align:center;">No pending tasks</div>'}
        
        ${done.length ? `
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#4b5563;
               padding:12px 0 6px;font-weight:500;">Recently completed</div>
          ${done.map(t => renderTaskCard(t, true)).join('')}
        ` : ''}
      </div>
    `;
  } catch (e) {
    panel.innerHTML = `<div style="padding:20px;color:#ef4444;">Failed to load tasks: ${e.message}</div>`;
  }
}

function renderTaskCard(task, done = false) {
  const icon = TASK_TYPE_ICONS[task.task_type] || '•';
  const priorityColors = { high: '#f59e0b', normal: '#6b7280', low: '#374151', idle: '#1f2937' };
  const prioColor = priorityColors[task.priority_name] || '#6b7280';
  
  return `
    <div style="background:#1e2329;border:1px solid rgba(255,255,255,0.07);border-radius:10px;
         padding:11px 13px;margin-bottom:7px;opacity:${done ? '0.6' : '1'};">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
        <div style="display:flex;gap:7px;align-items:flex-start;flex:1;min-width:0;">
          <span style="flex-shrink:0;margin-top:1px;">${icon}</span>
          <div style="min-width:0;">
            <div style="font-size:13px;font-weight:500;color:#e8eaed;line-height:1.35;
                 white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
              ${escHtml(task.title)}
            </div>
            <div style="font-size:11px;color:#6b7280;margin-top:3px;">
              ${task.task_type} · <span style="color:${prioColor}">${task.priority_name}</span>
            </div>
          </div>
        </div>
        ${!done ? `
          <button onclick="cancelTask(${task.id})" 
            style="flex-shrink:0;background:transparent;border:none;color:#4b5563;
                   cursor:pointer;font-size:14px;padding:0;line-height:1;">✕</button>
        ` : '<span style="color:#3ecf8e;font-size:12px;flex-shrink:0;">✓</span>'}
      </div>
      ${task.result_summary && done ? `
        <div style="font-size:11.5px;color:#6b7280;margin-top:6px;line-height:1.4;
             border-top:1px solid rgba(255,255,255,0.05);padding-top:6px;">
          ${escHtml(task.result_summary.substring(0, 120))}${task.result_summary.length > 120 ? '...' : ''}
        </div>
      ` : ''}
    </div>
  `;
}

async function cancelTask(id) {
  try {
    await fetch(`/tasks/${id}`, { method: 'DELETE' });
    loadAndRenderTasks(document.getElementById('taskPanel'));
    loadTaskSummary();
  } catch (e) {}
}

async function addTaskPrompt() {
  const title = prompt('Task title:');
  if (!title) return;
  const description = prompt('Task description (what should the agent do?):') || title;
  
  try {
    await fetch('/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, description, priority_name: 'normal' })
    });
    loadAndRenderTasks(document.getElementById('taskPanel'));
    loadTaskSummary();
  } catch (e) {}
}

// ─────────────────────────────────────────────
// Init — connect event stream and poll tasks
// ─────────────────────────────────────────────
function initHeartbeatUI() {
  connectEvents();
  loadTaskSummary();
  setInterval(loadTaskSummary, 60 * 1000);
}
