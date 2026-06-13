"""Render docs/index.html and docs/archive/YYYY-MM-DD.html from data JSON.

Input:  docs/data/*.json (latest by date used for index.html)
Output: docs/index.html, docs/archive/<date>.html, docs/archive/index.html
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"
DOCS = ROOT / "docs"
DATA_DIR = DOCS / "data"
ARCHIVE_DIR = DOCS / "archive"

CATEGORY_ORDER = ["domestic", "international", "hotel", "tech_ai"]
CATEGORY_LABEL = {
    "domestic": "国内",
    "international": "国際",
    "hotel": "ホテル・観光",
    "tech_ai": "Tech・AI",
}

JP_WEEKDAY = ["月", "火", "水", "木", "金", "土", "日"]


def jp_date(date_str: str) -> str:
    """2026-06-13 -> 2026-06-13 (土)"""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{date_str} ({JP_WEEKDAY[dt.weekday()]})"
    except ValueError:
        return date_str


def categories_for_template(data: dict) -> list[dict]:
    cats = data.get("categories", {})
    return [
        {
            "key": key,
            "label": CATEGORY_LABEL[key],
            "articles": cats.get(key, []),
        }
        for key in CATEGORY_ORDER
    ]


def render(env: Environment, data: dict, *, is_archive_page: bool) -> str:
    template = env.get_template("index.html.j2")
    date_str = data.get("date", "")
    generated_at = data.get("generated_at", "")
    try:
        gen_dt = datetime.fromisoformat(generated_at)
        updated_at = gen_dt.strftime("%H:%M JST")
    except (ValueError, TypeError):
        updated_at = generated_at or "-"

    title = (
        f"Daily News {date_str}" if is_archive_page else "Daily News - 最新ダイジェスト"
    )

    return template.render(
        title=title,
        display_date=jp_date(date_str),
        updated_at=updated_at,
        categories=categories_for_template(data),
        model=data.get("model", ""),
        is_archive_page=is_archive_page,
        asset_prefix="../" if is_archive_page else "",
        home_href="../" if is_archive_page else "./",
    )


def render_archive_index(env: Environment, dates: list[str]) -> str:
    template = env.get_template("archive_index.html.j2")
    return template.render(dates=sorted(dates, reverse=True))


def main() -> None:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    data_files = sorted(DATA_DIR.glob("*.json"))
    if not data_files:
        print("[ERROR] no data files in docs/data/", file=sys.stderr)
        sys.exit(1)

    all_dates: list[str] = []
    for f in data_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        date_str = data.get("date") or f.stem
        all_dates.append(date_str)
        archive_html = render(env, data, is_archive_page=True)
        (ARCHIVE_DIR / f"{date_str}.html").write_text(archive_html, encoding="utf-8")

    latest_file = max(data_files, key=lambda p: p.stem)
    latest_data = json.loads(latest_file.read_text(encoding="utf-8"))
    index_html = render(env, latest_data, is_archive_page=False)
    (DOCS / "index.html").write_text(index_html, encoding="utf-8")
    print(f"Wrote docs/index.html (latest: {latest_data.get('date')})", file=sys.stderr)

    archive_index_html = render_archive_index(env, all_dates)
    (ARCHIVE_DIR / "index.html").write_text(archive_index_html, encoding="utf-8")
    print(f"Wrote {len(all_dates)} archive pages", file=sys.stderr)


if __name__ == "__main__":
    main()
