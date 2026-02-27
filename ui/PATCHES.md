<!--
  ui/PATCHES.md
  
  Add the following snippets to index.html to complete the heartbeat UI.
  These are documented as patches rather than a full rewrite for clarity.
-->

<!-- 1. Add to <style> block: Activity bar and task button styles -->
<style>
/* Task queue button with badge */
.task-btn-wrap {
  position: relative;
  display: inline-flex;
}

.task-badge {
  position: absolute;
  top: -4px;
  right: -4px;
  min-width: 16px;
  height: 16px;
  background: var(--accent);
  color: #0d0f12;
  border-radius: 8px;
  font-size: 9px;
  font-weight: 700;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 0 3px;
}

/* Activity bar at bottom of sidebar */
.activity-bar {
  padding: 10px 12px;
  border-top: 1px solid var(--border);
  min-height: 42px;
}

.activity-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-muted);
}

.activity-row.activity-active { color: var(--status-think); }
.activity-row.activity-done   { color: var(--status-ok); }
.activity-row.activity-failed { color: var(--status-err); }

.activity-icon { flex-shrink: 0; }

.activity-title {
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
}

.activity-spinner {
  width: 10px;
  height: 10px;
  border: 1.5px solid var(--status-think);
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  flex-shrink: 0;
}
</style>

<!-- 2. Replace header-actions in the sidebar-header with: -->
<!-- (Shows the task queue button with badge) -->
<div class="header-actions">
  <div class="task-btn-wrap">
    <button class="header-btn" id="taskBtn" onclick="toggleTaskPanel()" title="Background tasks">
      📋
    </button>
    <div class="task-badge" id="taskBadge"></div>
  </div>
  <button class="header-btn" onclick="clearChat()" title="Clear chat">🗑</button>
  <button class="header-btn" onclick="showProfile()" title="Your profile">👤</button>
</div>

<!-- 3. Add before closing </aside> tag: -->
<div class="activity-bar" id="activityBar">
  <!-- Background task activity appears here -->
</div>

<!-- 4. Add before closing </body> tag: -->
<script src="/static/heartbeat.js"></script>
<script>
  // Extend init() to also start heartbeat
  const _originalInit = init;
  async function init() {
    await _originalInit();
    initHeartbeatUI();
  }
</script>
