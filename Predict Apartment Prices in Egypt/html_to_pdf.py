"""Render Project_Technical_Report.html to PDF using Chrome or Edge (headless)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HTML = Path(__file__).resolve().parent / "Project_Technical_Report.html"
PDF = Path(__file__).resolve().parent / "Project_Technical_Report.pdf"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path.home() / r"AppData\Local\Google\Chrome\Application\chrome.exe",
]

EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def find_browser() -> Path | None:
    for p in CHROME_CANDIDATES + EDGE_CANDIDATES:
        if p.is_file():
            return p
    return None


def main() -> int:
    if not HTML.is_file():
        print("Missing:", HTML, file=sys.stderr)
        return 1
    browser = find_browser()
    if not browser:
        print("No Chrome/Edge found for headless PDF.", file=sys.stderr)
        return 1
    if PDF.exists():
        PDF.unlink()
    uri = HTML.as_uri()
    cmd = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        uri,
    ]
    print("Running:", browser.name, "->", PDF.name)
    subprocess.run(cmd, check=True, timeout=120)
    if not PDF.is_file() or PDF.stat().st_size < 1000:
        print("PDF missing or too small.", file=sys.stderr)
        return 1
    print("OK:", PDF, "size=", PDF.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
