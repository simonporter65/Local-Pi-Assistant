"""
skills/browser.py
Full headless browser control via Playwright.
Supports navigation, clicking, typing, screenshots, and content extraction.
"""

DESCRIPTION = (
    "Control a headless browser. "
    "Args: action (browse|goto|screenshot|extract|click|type|scroll|pdf), "
    "url (str), selector (str), text (str), save_path (str), scroll_px (int). "
    "'browse' = navigate + extract text + screenshot in one call, returns both."
)

import os
from datetime import datetime

SCREENSHOT_DIR = os.environ.get("AGENT_SCREENSHOTS", "/mnt/nvme/agent/screenshots")
WORKSPACE = os.environ.get("AGENT_WORKSPACE", "/mnt/nvme/agent/workspace")


def run(
    action: str,
    url: str = "",
    selector: str = "",
    text: str = "",
    save_path: str = "",
    scroll_px: int = 500,
) -> str:
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        return (
            "Playwright not installed. Run:\n"
            "pip install playwright && playwright install chromium"
        )

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        try:
            if action == "browse":
                # One-shot: navigate + extract readable text + take screenshot
                if not url:
                    return "ERROR: url required for browse"
                page.goto(url, timeout=25000, wait_until="domcontentloaded")
                # Extract clean text
                page.evaluate("""() => {
                    ['script','style','nav','footer','header','aside'].forEach(tag => {
                        document.querySelectorAll(tag).forEach(el => el.remove())
                    })
                }""")
                content = page.inner_text("body")
                if len(content) > 4000:
                    content = content[:4000] + f"\n...[{len(content)} total chars]"
                # Take screenshot
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                shot_path = os.path.join(SCREENSHOT_DIR, f"browse_{ts}.png")
                page.screenshot(path=shot_path, full_page=False)  # viewport only — faster
                title = page.title()
                return (
                    f"URL: {page.url}\nTitle: {title}\n"
                    f"Screenshot: {shot_path}\n\n"
                    f"--- PAGE TEXT ---\n{content}"
                )

            elif action == "goto":
                page.goto(url, timeout=20000, wait_until="domcontentloaded")
                return f"Loaded: {page.title()} — {page.url}"

            elif action == "screenshot":
                if url:
                    page.goto(url, timeout=20000, wait_until="networkidle")
                if not save_path:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(SCREENSHOT_DIR, f"browser_{ts}.png")
                page.screenshot(path=save_path, full_page=True)
                size = os.path.getsize(save_path)
                return f"Browser screenshot saved: {save_path} ({size:,} bytes)"

            elif action == "extract":
                page.goto(url, timeout=20000, wait_until="domcontentloaded")
                # Remove clutter
                page.evaluate("""() => {
                    ['script','style','nav','footer','header','aside'].forEach(tag => {
                        document.querySelectorAll(tag).forEach(el => el.remove())
                    })
                }""")
                content = page.inner_text("body")
                # Trim
                if len(content) > 5000:
                    content = content[:5000] + f"\n...[{len(content)} total chars]"
                return f"URL: {url}\n\n{content}"

            elif action == "click":
                if not selector:
                    return "ERROR: selector required for click"
                page.click(selector, timeout=10000)
                return f"Clicked: {selector}"

            elif action == "type":
                if not selector or not text:
                    return "ERROR: selector and text required for type"
                page.fill(selector, text)
                return f"Typed '{text[:50]}' into {selector}"

            elif action == "scroll":
                page.mouse.wheel(0, scroll_px)
                return f"Scrolled {scroll_px}px"

            elif action == "pdf":
                if url:
                    page.goto(url, timeout=20000, wait_until="networkidle")
                if not save_path:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(WORKSPACE, f"page_{ts}.pdf")
                page.pdf(path=save_path, format="A4")
                return f"PDF saved: {save_path}"

            else:
                return f"Unknown action: {action}. Use: goto, screenshot, extract, click, type, scroll, pdf"

        except PWTimeout:
            return f"Timeout: {action} on {url}"
        except Exception as e:
            return f"Browser error: {e}"
        finally:
            browser.close()
