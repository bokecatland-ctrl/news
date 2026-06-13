"""Fetch RSS feeds for each category and produce a raw articles JSON.

Output: tmp/raw_articles.json
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import feedparser
import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = ROOT / "scripts" / "sources.yaml"
OUTPUT_FILE = ROOT / "tmp" / "raw_articles.json"

JST = timezone(timedelta(hours=9))
LOOKBACK_HOURS = 28
HTTP_TIMEOUT = 20.0
MAX_ARTICLES_PER_SOURCE = 25
USER_AGENT = "DailyNewsBot/1.0 (+https://github.com/bokecatland-ctrl/news)"


def normalize_url(url: str) -> str:
    """Strip query params used for tracking; keep canonical form for dedup."""
    if not url:
        return url
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", "", ""))
    except Exception:
        return url


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_published(entry) -> datetime | None:
    for field in ("published_parsed", "updated_parsed"):
        t = entry.get(field)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def fetch_feed(source: dict) -> tuple[str, list[dict]]:
    """Fetch one feed; never raise — log and return empty list on failure."""
    name = source["name"]
    url = source["url"]
    try:
        resp = httpx.get(
            url,
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] fetch failed: {name} ({url}): {e}", file=sys.stderr)
        return name, []

    parsed = feedparser.parse(resp.content)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    out: list[dict] = []
    for entry in parsed.entries[:MAX_ARTICLES_PER_SOURCE]:
        link = entry.get("link", "")
        title = strip_html(entry.get("title", ""))
        if not link or not title:
            continue
        published = parse_published(entry)
        if published and published < cutoff:
            continue
        summary = strip_html(entry.get("summary", "") or entry.get("description", ""))
        if len(summary) > 800:
            summary = summary[:800] + "…"
        article = {
            "id": hashlib.sha1(normalize_url(link).encode()).hexdigest()[:12],
            "title": title,
            "url": link,
            "source": name,
            "published": published.isoformat() if published else None,
            "summary_raw": summary,
        }
        out.append(article)
    print(f"[OK]  {name}: {len(out)} articles", file=sys.stderr)
    return name, out


def main() -> None:
    sources_data = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8"))

    result: dict[str, list[dict]] = {}
    for category, sources in sources_data.items():
        all_articles: list[dict] = []
        seen_ids: set[str] = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for _, articles in ex.map(fetch_feed, sources):
                for a in articles:
                    if a["id"] in seen_ids:
                        continue
                    seen_ids.add(a["id"])
                    all_articles.append(a)

        all_articles.sort(key=lambda a: a.get("published") or "", reverse=True)
        result[category] = all_articles
        print(f"=> {category}: {len(all_articles)} unique articles", file=sys.stderr)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "fetched_at": datetime.now(JST).isoformat(),
                "categories": result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    total = sum(len(v) for v in result.values())
    print(f"Wrote {OUTPUT_FILE} ({total} articles total)", file=sys.stderr)


if __name__ == "__main__":
    main()
