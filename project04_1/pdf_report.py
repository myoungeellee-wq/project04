from __future__ import annotations

from html import escape
from pathlib import Path
import os
import subprocess
import tempfile


def markdown_to_report_html(markdown_text: str) -> str:
    """Convert SAR markdown to print-friendly HTML without extra packages."""

    lines = markdown_text.splitlines()
    html_lines: list[str] = []
    in_code = False
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            html_lines.append("</ul>")
            in_list = False

    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("```"):
            close_list()
            html_lines.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue

        if in_code:
            html_lines.append(escape(line) + "\n")
            continue

        if not line:
            close_list()
            html_lines.append("<div class=\"space\"></div>")
        elif line.startswith("### "):
            close_list()
            html_lines.append(f"<h3>{escape(line[4:])}</h3>")
        elif line.startswith("## "):
            close_list()
            html_lines.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("# "):
            close_list()
            html_lines.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{escape(line[2:])}</li>")
        else:
            close_list()
            html_lines.append(f"<p>{escape(line)}</p>")

    close_list()
    if in_code:
        html_lines.append("</code></pre>")

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <style>
    @page {{ size: A4; margin: 16mm 14mm; }}
    body {{
      font-family: "Malgun Gothic", "Segoe UI", Arial, sans-serif;
      color: #1f2d3d;
      font-size: 12px;
      line-height: 1.55;
    }}
    h1 {{
      margin: 0 0 16px;
      padding-bottom: 10px;
      border-bottom: 2px solid #245b82;
      color: #17324d;
      font-size: 24px;
    }}
    h2 {{
      margin: 18px 0 8px;
      color: #245b82;
      font-size: 17px;
    }}
    h3 {{
      margin: 14px 0 7px;
      color: #315b78;
      font-size: 14px;
    }}
    p {{ margin: 6px 0; }}
    ul {{ margin: 6px 0 8px 18px; padding: 0; }}
    li {{ margin: 4px 0; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      padding: 10px;
      border: 1px solid #d8e0e8;
      background: #f5f7fa;
      border-radius: 4px;
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 10px;
    }}
    .space {{ height: 4px; }}
  </style>
</head>
<body>
  {"".join(html_lines)}
</body>
</html>"""


def find_edge_executable() -> str:
    candidates = [
        os.getenv("EDGE_PATH", ""),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Microsoft Edge 실행 파일을 찾을 수 없습니다. EDGE_PATH 환경변수로 경로를 지정해주세요.")


def sar_report_to_pdf_bytes(markdown_text: str) -> bytes:
    """Render SAR markdown to PDF bytes through local Edge headless print."""

    html_text = markdown_to_report_html(markdown_text)
    with tempfile.TemporaryDirectory() as tmp_dir:
        html_path = Path(tmp_dir) / "sar_report.html"
        pdf_path = Path(tmp_dir) / "sar_report.pdf"
        html_path.write_text(html_text, encoding="utf-8")
        subprocess.run(
            [
                find_edge_executable(),
                "--headless",
                "--disable-gpu",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        return pdf_path.read_bytes()
