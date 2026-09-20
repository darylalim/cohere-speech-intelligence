# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright>=1.50"]
# ///
"""Capture the README screenshots from the running app.

    uv run streamlit run streamlit_app.py --server.headless true   # in another shell
    uv run --no-project docs/screenshots.py

Writes result-{dark,light}.png and chunks-{dark,light}.png into docs/screenshots/.
Playwright comes from the inline metadata above, never from the project: nothing
here imports it and ty would report it unresolved. --no-project keeps a bare
`uv run` from re-locking the project when uv.lock is stale. The first run needs
`uv run --no-project --with playwright playwright install chromium-headless-shell`
once, ~95 MB; or pass --browser with the path to any chrome-headless-shell.

The recording is synthesized on the way in -- say and ffmpeg, both on PATH --
unless --fixture names an mp3 to reuse. CLAUDE.md's Screenshots section carries
the reasoning behind every number here; this file carries only what it needs to.
"""

import argparse
import base64
import pathlib
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright  # ty: ignore[unresolved-import]

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "screenshots"

# window.innerHeight measured in Chrome on the 1920x1080 panel the layout was
# tuned for, menu bar and Dock showing. Measured, not derived: 1080 minus the
# browser chrome minus the menu bar is 934, which matches nothing on screen.
VIEWPORT = {"width": 1920, "height": 839}
# Empty main-area background: nothing hovered, no tooltip, no focus ring.
PARK = (1500, 800)
SUCCESS = '[data-testid="stMetric"]'
ERRORS = "Sign in required|Incompatible checkpoint|Transcription failed"

# The Bliss copy. 271 words; Samantha renders it in 80.3 s, which long-form
# decoding cuts at 30.0 and 61.4 s -- three rows in the chunk table. The
# default voice is a per-user setting and renders the same text at 88.4 s,
# so -v is not optional. Kept verbatim: `say -f` on this text reproduces the
# AIFF byte for byte.
TEXT = """\
Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.

Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.

But, in a larger sense, we can not dedicate, we can not consecrate, we can not hallow this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us, that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion, that we here highly resolve that these dead shall not have died in vain, that this nation, under God, shall have a new birth of freedom, and that government of the people, by the people, for the people, shall not perish from the earth.
"""


def synthesize(workdir: pathlib.Path) -> pathlib.Path:
    """Render TEXT with say and encode it as the mp3 the screenshots show."""
    for tool in ("say", "ffmpeg"):
        if shutil.which(tool) is None:
            sys.exit(
                f"{tool} is not on PATH; pass --fixture to reuse a recording instead"
            )
    text = workdir / "gettysburg.txt"
    aiff = workdir / "gettysburg.aiff"
    # Short on purpose: the uploader chip ellipsises gettysburg-address.mp3,
    # and the name is on screen twice.
    mp3 = workdir / "gettysburg.mp3"
    text.write_text(TEXT)
    subprocess.run(["say", "-v", "Samantha", "-f", text, "-o", aiff], check=True)
    # -q:a 4 because the chip shows the size: 0.6MB in the committed captures.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", aiff]
        + ["-codec:a", "libmp3lame", "-q:a", "4", mp3],
        check=True,
    )
    return mp3


def settle(page, ms: int = 1500) -> None:
    """Wait out the rerun, then clear every state a screenshot should not carry.

    Streamlit's "Running..." pill sits top-right while a rerun is in flight.
    Focus stays on the button that was clicked, and the mouse stays where it
    clicked: without the blur and the park, Transcribe is captured in its
    hover tone and the sidebar shows its collapse chevron.
    """
    page.wait_for_selector(
        '[data-testid="stStatusWidget"]', state="detached", timeout=600_000
    )
    page.evaluate("document.activeElement && document.activeElement.blur()")
    page.mouse.move(*PARK)
    page.wait_for_timeout(ms)


def pair(page, out: pathlib.Path, name: str, shoot) -> None:
    """One capture per theme, from one session.

    Streamlit's "System" theme follows a live prefers-color-scheme change, so
    both halves carry one transcript and one set of numbers. A second context
    would be a second session and a second transcription, with its own Elapsed
    and Speed.
    """
    for scheme in ("dark", "light"):
        page.emulate_media(color_scheme=scheme)
        page.wait_for_timeout(800)
        path = out / f"{name}-{scheme}.png"
        shoot(path)
        print("wrote", path)
    page.emulate_media(color_scheme="dark")
    page.wait_for_timeout(300)


def capture(
    url: str, fixture: pathlib.Path, out: pathlib.Path, browser_path: str | None
) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=browser_path, headless=True)
        ctx = browser.new_context(
            viewport=VIEWPORT, device_scale_factor=1, color_scheme="dark"
        )
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle")
        page.get_by_role("heading", name="Cohere Transcribe").wait_for()
        settle(page)

        page.set_input_files('input[type="file"]', fixture)
        page.wait_for_function(
            "() => [...document.querySelectorAll('button')]"
            ".some(b => b.textContent.includes('Transcribe') && !b.disabled)"
        )
        settle(page, 500)
        page.get_by_role("button", name="Transcribe").click()
        # Three metrics render from the result once the status reads "Done in".
        # "RTFx" is only in the Speed tooltip, so waiting on it hangs; the three
        # error labels fail fast rather than waiting out the timeout.
        page.wait_for_function(
            f"""() => {{
                const t = document.body.innerText;
                if (/{ERRORS}/.test(t)) throw new Error("transcription errored: " + t.slice(0, 400));
                return document.querySelectorAll('{SUCCESS}').length === 3;
            }}""",
            timeout=600_000,
        )
        settle(page, 2000)
        pair(
            page,
            out,
            "result",
            lambda path: page.screenshot(path=str(path), animations="disabled"),
        )

        # The chunk expander's label is its count, so select on the word; .first
        # is the sidebar's Advanced expander. Open, the table sits below the
        # 839 px fold with the main pane scrolling internally -- full_page is a
        # no-op there -- so this is a crop of the expander, not the viewport.
        chunks = page.locator('[data-testid="stExpander"]').filter(has_text="chunks")
        chunks.locator("summary").click()
        settle(page, 1500)
        chunks.scroll_into_view_if_needed()
        settle(page, 1000)
        # Rasterised at 2x: the README embeds the crop at its CSS width, and
        # most readers of an Apple-Silicon-only app are on Retina displays,
        # where a 1x raster goes soft beside the page text. Through CDP rather
        # than locator.screenshot(scale="device"), because Playwright re-applies
        # the context's 1x metrics before each of its own captures, so a
        # device-metrics override is undone; clip.scale re-renders the region.
        cdp = ctx.new_cdp_session(page)

        def crop(path: pathlib.Path) -> None:
            box = chunks.bounding_box()
            shot = cdp.send(
                "Page.captureScreenshot", {"format": "png", "clip": {**box, "scale": 2}}
            )
            path.write_bytes(base64.b64decode(shot["data"]))

        pair(page, out, "chunks", crop)
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--fixture",
        type=pathlib.Path,
        help="an mp3 to upload instead of synthesizing one",
    )
    parser.add_argument(
        "--browser", help="path to a chrome-headless-shell, instead of Playwright's own"
    )
    args = parser.parse_args()

    try:
        urllib.request.urlopen(args.url, timeout=5).close()
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"nothing is answering at {args.url} ({exc}); start the app first")

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fixture = args.fixture or synthesize(pathlib.Path(tmp))
        capture(args.url, fixture, args.out, args.browser)
    print(
        "re-read Elapsed and Speed off result-dark.png into README's alt text: they move every run"
    )


if __name__ == "__main__":
    main()
