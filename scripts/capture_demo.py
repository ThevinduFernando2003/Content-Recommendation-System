"""Capture full-page screenshots and a walkthrough video of the local UI.

Requires the API on port 8000 and Streamlit on port 8501, plus Playwright
Chromium (`python -m playwright install chromium`). This script is not part
of the test suite.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
VIDEO_DIR = ROOT / "docs" / "demo" / "video"
PACK = ROOT / "docs" / "demo" / "catalogue-demo.zip"
UI = os.environ.get("DEMO_UI", "http://127.0.0.1:8520")
DOCS = os.environ.get("DEMO_DOCS", "http://127.0.0.1:8020/docs")


def shot(page: Page, name: str) -> None:
    """Save the whole Streamlit page, including content below the fold."""
    page.evaluate(
        """() => {
          const main = document.querySelector('[data-testid="stMain"]');
          if (!main) return;
          const height = main.scrollHeight;
          let el = main;
          while (el) {
            el.style.overflow = 'visible';
            el.style.height = 'auto';
            el.style.maxHeight = 'none';
            el = el.parentElement;
          }
          document.documentElement.style.height = 'auto';
          document.body.style.height = 'auto';
          main.style.height = height + 'px';
        }"""
    )
    page.wait_for_timeout(300)
    page.screenshot(path=str(SHOTS / name), full_page=True)
    page.evaluate(
        """() => {
          const main = document.querySelector('[data-testid="stMain"]');
          if (!main) return;
          let el = main;
          while (el) {
            el.style.overflow = '';
            el.style.height = '';
            el.style.maxHeight = '';
            el = el.parentElement;
          }
        }"""
    )
    print("shot", name)


def scroll_page(page: Page, steps: int = 5) -> None:
    total = page.evaluate(
        """() => {
          const main = document.querySelector('[data-testid="stMain"]');
          return main ? Math.max(main.scrollHeight - main.clientHeight, 0) : 0;
        }"""
    )
    for index in range(steps):
        position = int(total * (index + 1) / steps)
        page.evaluate(
            """(y) => {
              const main = document.querySelector('[data-testid="stMain"]');
              if (main) main.scrollTop = y;
            }""",
            position,
        )
        page.wait_for_timeout(800)
    page.evaluate(
        """() => {
          const main = document.querySelector('[data-testid="stMain"]');
          if (main) main.scrollTop = 0;
        }"""
    )
    page.wait_for_timeout(400)


def _tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(1200)


def _open_filters(page: Page) -> None:
    page.get_by_text("Filters", exact=True).first.click()
    page.wait_for_timeout(800)


def main() -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for old in VIDEO_DIR.glob("*.webm"):
        old.unlink()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(VIDEO_DIR),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()
        page.set_default_timeout(90000)
        page.goto(UI, wait_until="networkidle")
        page.get_by_text("Find something to watch").wait_for()
        page.wait_for_timeout(800)
        shot(page, "01-recommend-home.png")

        search = page.get_by_label("Search the catalogue")
        search.click()
        search.fill("break bad")
        search.press("Tab")
        page.get_by_text("Closest catalogue title is Breaking Bad").wait_for()
        page.wait_for_timeout(800)
        shot(page, "02-near-search-break-bad.png")

        _open_filters(page)
        page.get_by_text("What the ratings mean").first.wait_for()
        page.wait_for_timeout(500)
        shot(page, "03-rating-guide.png")
        rating = page.get_by_role("tabpanel", name="Recommend").get_by_role(
            "combobox", name="Rating"
        )
        rating.click()
        page.keyboard.type("TV-MA")
        page.keyboard.press("Enter")
        page.wait_for_timeout(400)
        page.wait_for_timeout(400)

        page.get_by_role("button", name="Recommend", exact=True).click()
        page.get_by_text("More like Breaking Bad").wait_for()
        page.get_by_role("tabpanel", name="Recommend").locator("img").first.wait_for(
            state="visible", timeout=90000
        )
        page.wait_for_timeout(2500)
        scroll_page(page, 4)
        shot(page, "04-posters-and-descriptions.png")

        page.get_by_text("See more").first.click()
        page.wait_for_timeout(800)
        shot(page, "05-see-more-reason.png")

        _tab(page, "Describe your taste")
        taste = page.get_by_role("tabpanel", name="Describe your taste")
        for mood in ("crime", "drama", "thriller"):
            taste.get_by_role("button", name=mood, exact=True).click()
            page.wait_for_function(
                """(word) => {
                  const areas = [...document.querySelectorAll('textarea')];
                  return areas.some((el) => (el.value || '').toLowerCase().includes(word));
                }""",
                arg=mood,
            )
            page.wait_for_timeout(400)
        page.wait_for_timeout(600)
        shot(page, "06-taste-moods.png")
        page.get_by_role("button", name="Find titles", exact=True).click()
        page.get_by_text("Closest titles").wait_for()
        page.get_by_role("tabpanel", name="Describe your taste").locator("img").first.wait_for(
            state="visible", timeout=90000
        )
        page.wait_for_timeout(2500)
        scroll_page(page, 4)
        shot(page, "07-taste-results.png")
        page.get_by_role("tabpanel", name="Describe your taste").get_by_text(
            "See more"
        ).first.click()
        page.wait_for_timeout(700)
        shot(page, "08-taste-see-more.png")

        _tab(page, "Explore the catalogue")
        page.get_by_text("Titles added per year").wait_for()
        page.wait_for_timeout(1500)
        scroll_page(page, 6)
        shot(page, "09-explore.png")

        _tab(page, "About the model")
        page.get_by_text("How a recommendation is chosen").wait_for()
        page.wait_for_timeout(800)
        scroll_page(page, 3)
        shot(page, "10-about.png")

        page.goto(DOCS, wait_until="networkidle")
        page.wait_for_timeout(1200)
        scroll_page(page, 4)
        shot(page, "11-api-docs.png")

        video = page.video
        context.close()
        browser.close()
        if video is not None:
            saved = Path(video.path())
            target = VIDEO_DIR / "walkthrough.webm"
            if saved.resolve() != target.resolve():
                saved.replace(target)
            print("video", target)

    for junk in SHOTS.glob("_*.png"):
        junk.unlink()
    readme = (
        "Local walkthrough of the catalogue UI.\n\n"
        "screenshots/ holds one full-page image per step.\n"
        "video/walkthrough.webm plays the same steps in order:\n"
        "home, near search for break bad, rating guide and TV-MA filter,\n"
        "poster row with descriptions, See more, mood chips, taste results,\n"
        "explore, about, and the API docs.\n"
    )
    with zipfile.ZipFile(PACK, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", readme)
        for path in sorted(SHOTS.glob("*.png")):
            archive.write(path, f"screenshots/{path.name}")
        video_path = VIDEO_DIR / "walkthrough.webm"
        if video_path.exists():
            archive.write(video_path, "video/walkthrough.webm")
    print("zip", PACK)


if __name__ == "__main__":
    main()
