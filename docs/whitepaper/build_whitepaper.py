"""Render the whitepaper with existing Pandoc and Chrome, without model calls."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="Also export a PDF with local Chrome")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise SystemExit("Pandoc is required; this script does not install dependencies.")
    stem = "easyagent-whitepaper.zh-CN"
    html = root / f"{stem}.html"
    subprocess.run([
        pandoc, str(root / f"{stem}.md"), "--from=markdown-implicit_figures", "--to=html5",
        "--standalone", "--self-contained", "--toc", "--toc-depth=3", "--no-highlight",
        f"--lua-filter={root / 'reading-filter.lua'}",
        f"--template={root / 'print-template.html'}", f"--resource-path={root}",
        "--output", str(html),
    ], cwd=root, check=True)
    print(f"HTML: {html}")
    if args.pdf:
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        if not chrome:
            raise SystemExit("HTML built; Chrome/Chromium is required for PDF export.")
        pdf = root / f"{stem}.pdf"
        with tempfile.TemporaryDirectory(prefix="easyagent-whitepaper-chrome-") as profile:
            subprocess.run([
                chrome, "--headless", "--disable-gpu", "--disable-dev-shm-usage",
                "--no-first-run", "--no-default-browser-check", "--disable-background-networking",
                f"--user-data-dir={profile}", "--no-pdf-header-footer",
                "--timeout=30000", f"--print-to-pdf={pdf}", html.as_uri(),
            ], check=True, timeout=60, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not pdf.is_file() or pdf.stat().st_size == 0:
            raise SystemExit("Chrome did not produce a non-empty PDF.")
        print(f"PDF: {pdf}")


if __name__ == "__main__":
    main()
