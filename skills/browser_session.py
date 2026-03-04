"""
skills/browser_session.py
Persistent browser sessions for multi-step web automation.
Sessions survive across skill calls — browser stays open until you close it.
Auto-screenshots after click/scroll/press so the model can see the result.
"""

DESCRIPTION = (
    "Persistent browser sessions for multi-step web automation. "
    "Sessions stay open between calls — ideal for filling forms, clicking links, multi-step flows. "
    "Actions: open, goto, browse, click, type, scroll, press, screenshot, extract, close, list. "
    "Args: session_id (str, default='default'), action (str), url (str), "
    "selector (str), text (str), scroll_px (int). "
    "Always start with 'browse' (navigate + extract + screenshot). "
    "After click/type, check the screenshot path returned to see what changed."
)

import os
import time
from datetime import datetime

SCREENSHOT_DIR = os.environ.get("AGENT_SCREENSHOTS", "/mnt/nvme/agent/screenshots")

# Module-level session store — persists across skill calls while server is running
# {session_id: {"pw": ..., "browser": ..., "context": ..., "page": ..., "last_used": float}}
_sessions: dict = {}

SESSION_TIMEOUT = 600  # Close idle sessions after 10 minutes


def _cleanup_idle():
    now = time.time()
    to_close = [sid for sid, s in _sessions.items() if now - s["last_used"] > SESSION_TIMEOUT]
    for sid in to_close:
        _close_session(sid)


def _close_session(session_id: str):
    s = _sessions.pop(session_id, None)
    if s:
        try:
            s["browser"].close()
        except Exception:
            pass
        try:
            s["pw"].stop()
        except Exception:
            pass


def _get_or_create(session_id: str):
    _cleanup_idle()

    if session_id not in _sessions:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        pw = sync_playwright().start()
        browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        _sessions[session_id] = {
            "pw": pw,
            "browser": browser,
            "context": context,
            "page": page,
            "last_used": time.time(),
        }

    _sessions[session_id]["last_used"] = time.time()
    return _sessions[session_id]["page"]


def _screenshot(page, label="session") -> str:
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOT_DIR, f"{label}_{ts}.png")
    try:
        page.screenshot(path=path, full_page=False)
    except Exception as e:
        return f"(screenshot failed: {e})"
    return path


def _extract(page) -> str:
    try:
        page.evaluate("""() => {
            ['script','style','nav','footer','header','aside'].forEach(tag => {
                document.querySelectorAll(tag).forEach(el => el.remove())
            })
        }""")
        content = page.inner_text("body")
        if len(content) > 4000:
            content = content[:4000] + f"\n...[{len(content)} total chars, truncated]"
        return content
    except Exception as e:
        return f"(extract failed: {e})"


def run(
    action: str,
    session_id: str = "default",
    url: str = "",
    selector: str = "",
    text: str = "",
    scroll_px: int = 500,
) -> str:
    try:
        if action == "list":
            active = list(_sessions.keys())
            return f"Active sessions: {active}" if active else "No active sessions."

        if action == "open":
            _get_or_create(session_id)
            return f"Session '{session_id}' opened."

        if action == "close":
            _close_session(session_id)
            return f"Session '{session_id}' closed."

        if action == "browse":
            if not url:
                return "ERROR: url required for browse"
            page = _get_or_create(session_id)
            page.goto(url, timeout=25000, wait_until="domcontentloaded")
            content = _extract(page)
            shot = _screenshot(page, f"browse_{session_id}")
            title = page.title()
            return (
                f"[{session_id}] URL: {page.url}\nTitle: {title}\n"
                f"Screenshot: {shot}\n\n"
                f"--- PAGE TEXT ---\n{content}"
            )

        if action == "goto":
            if not url:
                return "ERROR: url required for goto"
            page = _get_or_create(session_id)
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            return f"[{session_id}] Navigated to: {page.title()} — {page.url}"

        if action == "screenshot":
            page = _get_or_create(session_id)
            shot = _screenshot(page, f"shot_{session_id}")
            return f"[{session_id}] Screenshot: {shot}\nURL: {page.url}"

        if action == "extract":
            page = _get_or_create(session_id)
            content = _extract(page)
            return f"[{session_id}] URL: {page.url}\n\n{content}"

        if action == "click":
            if not selector:
                return "ERROR: selector required for click"
            page = _get_or_create(session_id)
            page.click(selector, timeout=10000)
            page.wait_for_load_state("domcontentloaded", timeout=5000)
            shot = _screenshot(page, f"click_{session_id}")
            return f"[{session_id}] Clicked: {selector}\nURL now: {page.url}\nScreenshot: {shot}"

        if action == "type":
            if not selector or not text:
                return "ERROR: selector and text required for type"
            page = _get_or_create(session_id)
            page.fill(selector, text)
            shot = _screenshot(page, f"type_{session_id}")
            return f"[{session_id}] Typed '{text[:50]}' into {selector}\nScreenshot: {shot}"

        if action == "press":
            if not text:
                return "ERROR: text (key name) required for press. E.g. 'Enter', 'Tab', 'Escape'"
            page = _get_or_create(session_id)
            page.keyboard.press(text)
            try:
                page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass
            shot = _screenshot(page, f"press_{session_id}")
            return f"[{session_id}] Pressed {text}\nURL now: {page.url}\nScreenshot: {shot}"

        if action == "scroll":
            page = _get_or_create(session_id)
            page.mouse.wheel(0, scroll_px)
            shot = _screenshot(page, f"scroll_{session_id}")
            return f"[{session_id}] Scrolled {scroll_px}px\nScreenshot: {shot}"

        return (
            f"Unknown action: {action}. "
            "Use: open, close, list, goto, browse, screenshot, extract, click, type, press, scroll"
        )

    except Exception as e:
        return f"BrowserSession error [{session_id}/{action}]: {e}"
