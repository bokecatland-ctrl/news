"""Summarize raw articles into a per-category curated list via Gemini API.

Input:  tmp/raw_articles.json
Output: docs/data/YYYY-MM-DD.json

Uses Google Gemini 2.5 Flash via the google-genai SDK (free tier).
Free tier: 15 req/min, 1500 req/day — easily covers 4 calls/day.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "tmp" / "raw_articles.json"
DATA_DIR = ROOT / "docs" / "data"

MODEL = "gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 8000
MAX_ARTICLES_PER_CATEGORY_INPUT = 40
MAX_ARTICLES_PER_CATEGORY_OUTPUT = 4

JST = timezone(timedelta(hours=9))

CATEGORY_LABEL = {
    "domestic": "国内",
    "international": "国際",
    "hotel": "ホテル・観光",
    "tech_ai": "テクノロジー・AI",
}

CATEGORY_FOCUS = {
    "domestic": "日本国内の主要ニュース。政治、経済、社会、事件・事故、自然災害など。",
    "international": "国際ニュース。海外の政治・経済・紛争・外交・主要事件など。",
    "hotel": (
        "ホテル業界・観光業界・インバウンド関連。"
        "国内ホテル運営、新規開業、海外ラグジュアリーホテル、観光統計、インバウンド需要動向など。"
    ),
    "tech_ai": (
        "テクノロジー・AI業界。生成AI、大手テック企業、製品リリース、研究、規制、スタートアップなど。"
    ),
}

SYSTEM_PROMPT = """あなたは日本語ニュースダイジェストの編集者です。
複数のRSSフィードから取得した記事リストを受け取り、読者にとって重要な記事を選び、日本語で簡潔に要約します。

出力ルール:
- 重要度の高い記事を選び、同じトピックの記事は1つに統合する
- 各記事を日本語で2〜3文に要約する (元記事の本文転載はしない、独自の日本語で書く)
- 重要度 (importance) を 1〜5 で付与 (5が最重要)
- ソース名と元URLは入力されたものをそのまま使う
- 出力は必ず指定されたJSON形式のみ。前後に説明文を書かない
- 同一トピックを複数記事が報じている場合、最も信頼できる/詳細な1件を残し他は除外する"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "articles": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "summary": {"type": "STRING"},
                    "source": {"type": "STRING"},
                    "url": {"type": "STRING"},
                    "importance": {"type": "INTEGER"},
                    "published": {"type": "STRING", "nullable": True},
                },
                "required": ["title", "summary", "source", "url", "importance"],
            },
        },
    },
    "required": ["articles"],
}


def build_user_prompt(category: str, articles: list[dict], max_output: int) -> str:
    label = CATEGORY_LABEL.get(category, category)
    focus = CATEGORY_FOCUS.get(category, "")

    article_lines = []
    for i, a in enumerate(articles, 1):
        published = a.get("published") or "不明"
        article_lines.append(
            f"[{i}] タイトル: {a['title']}\n"
            f"    ソース: {a['source']}\n"
            f"    公開: {published}\n"
            f"    URL: {a['url']}\n"
            f"    概要: {a.get('summary_raw') or '(概要なし)'}\n"
        )
    articles_block = "\n".join(article_lines)

    return f"""カテゴリ: {label}
焦点: {focus}

以下は直近24時間の {label} 関連記事 ({len(articles)}件) です。
読者が短時間で全体像を把握できるよう、本当に重要な記事だけを **3件 (上限4件)** に厳選してください。
件数を埋めるための妥協は不要です。重要度の低い記事は迷わず削ってください。

選定基準:
- 影響範囲が大きい / 速報性が高い / 業界やトレンドの転換点となる出来事を優先
- 同じトピックを複数のソースが報じている場合は最も信頼できる1件に統合
- 重要度4以上の記事だけを残すのが理想

{articles_block}

出力は articles 配列のみのJSON。各要素は title / summary / source / url / importance / published を含むこと。
importance は 1〜5 の整数、published は入力をそのまま (なければ null)。"""


def summarize_category(
    client: genai.Client, category: str, articles: list[dict]
) -> list[dict]:
    if not articles:
        return []

    truncated = articles[:MAX_ARTICLES_PER_CATEGORY_INPUT]
    user_prompt = build_user_prompt(
        category, truncated, MAX_ARTICLES_PER_CATEGORY_OUTPUT
    )

    print(
        f"[summarize] {category}: {len(truncated)} articles -> Gemini",
        file=sys.stderr,
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.4,
            ),
        )
    except Exception as e:
        print(f"[ERROR] Gemini API failed for {category}: {e}", file=sys.stderr)
        return _fallback_articles(truncated)

    text = (response.text or "").strip()
    if not text:
        print(f"[WARN] empty response for {category}; falling back", file=sys.stderr)
        return _fallback_articles(truncated)

    try:
        data = json.loads(text)
        result = data.get("articles", [])
    except json.JSONDecodeError as e:
        print(
            f"[WARN] JSON parse failed for {category}: {e}; falling back",
            file=sys.stderr,
        )
        return _fallback_articles(truncated)

    out = []
    for item in result:
        if not isinstance(item, dict):
            continue
        if not item.get("title") or not item.get("url"):
            continue
        importance = item.get("importance", 3)
        try:
            importance = int(importance)
        except (ValueError, TypeError):
            importance = 3
        importance = max(1, min(5, importance))
        out.append({
            "title": str(item["title"]),
            "summary": str(item.get("summary", "")),
            "source": str(item.get("source", "")),
            "url": str(item["url"]),
            "importance": importance,
            "published": item.get("published"),
        })
    out.sort(key=lambda a: a["importance"], reverse=True)
    print(f"[summarize] {category}: produced {len(out)} summaries", file=sys.stderr)
    return out


def _fallback_articles(articles: list[dict]) -> list[dict]:
    """If the LLM fails, fall back to headlines-only (no summary)."""
    fallback = []
    for a in articles[:MAX_ARTICLES_PER_CATEGORY_OUTPUT]:
        fallback.append({
            "title": a["title"],
            "summary": a.get("summary_raw", "")[:200],
            "source": a["source"],
            "url": a["url"],
            "importance": 3,
            "published": a.get("published"),
        })
    return fallback


DAILY_DIGEST_SYSTEM = """あなたは日本語ニュースダイジェストの編集者です。
今日の主要ニュース (4カテゴリ横断) から、読者が30秒で全体像をつかめる「今日のポイント」を最大3点に絞って書いてください。

ルール:
- 各ポイントは1〜2文の日本語、合計3点 (内容が薄い日は2点でもよい)
- カテゴリ偏りなく重要なものを選ぶ
- 個々の見出しを羅列するのではなく、要点を抽出する
- 客観的・簡潔に
- 出力は指定JSON形式のみ
"""

DAILY_DIGEST_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "headline": {"type": "STRING"},
        "points": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["headline", "points"],
}


def build_daily_digest_prompt(categories: dict[str, list[dict]]) -> str:
    blocks = []
    for cat_key, label in CATEGORY_LABEL.items():
        articles = categories.get(cat_key, [])
        if not articles:
            continue
        lines = [f"## {label}"]
        for a in articles[:5]:
            summary = (a.get("summary") or "")[:200]
            lines.append(f"- 【重要度{a.get('importance', 3)}】{a['title']}: {summary}")
        blocks.append("\n".join(lines))

    return (
        "以下は本日選定された主要ニュースです:\n\n"
        + "\n\n".join(blocks)
        + "\n\nこれらを横断的に見て、本当に重要な動きを最大3点選び、要点を書いてください。\n"
        + "headline は全体を象徴する15字程度の見出し。"
    )


def summarize_daily_digest(
    client: genai.Client, categories: dict[str, list[dict]]
) -> dict | None:
    if not any(categories.values()):
        return None

    digest = _try_ai_digest(client, categories)
    if digest:
        return digest

    print("[summarize] AI digest unavailable; using heuristic fallback", file=sys.stderr)
    return _heuristic_digest(categories)


def _try_ai_digest(
    client: genai.Client, categories: dict[str, list[dict]]
) -> dict | None:
    prompt = build_daily_digest_prompt(categories)
    print("[summarize] generating daily digest via Gemini", file=sys.stderr)

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=DAILY_DIGEST_SYSTEM,
                # Gemini 2.5 Flash の thinking tokens は出力枠に含まれるため広めに確保
                max_output_tokens=4000,
                response_mime_type="application/json",
                response_schema=DAILY_DIGEST_SCHEMA,
                temperature=0.4,
            ),
        )
    except Exception as e:
        print(
            f"[WARN] daily digest API call failed: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return None

    # Diagnostic: log finish reason so the next run reveals MAX_TOKENS / SAFETY
    try:
        for i, cand in enumerate(getattr(response, "candidates", None) or []):
            print(
                f"[diag] digest candidate[{i}].finish_reason="
                f"{getattr(cand, 'finish_reason', None)}",
                file=sys.stderr,
            )
    except Exception:
        pass

    try:
        text = (response.text or "").strip()
    except Exception as e:
        print(
            f"[WARN] daily digest response.text failed: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return None

    if not text:
        print("[WARN] daily digest returned empty text", file=sys.stderr)
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[WARN] daily digest JSON parse failed: {e}", file=sys.stderr)
        print(f"[diag] raw text head: {text[:300]}", file=sys.stderr)
        return None

    points = [str(p).strip() for p in data.get("points", []) if str(p).strip()]
    if not points:
        return None

    return {
        "headline": str(data.get("headline", "")).strip() or "今日のピックアップ",
        "points": points[:3],
    }


def _heuristic_digest(categories: dict[str, list[dict]]) -> dict | None:
    """Fallback when AI digest fails — pick top-importance articles, prefer category diversity."""
    pool: list[tuple[str, dict]] = []
    for cat_key, articles in categories.items():
        for a in articles:
            pool.append((cat_key, a))

    if not pool:
        return None

    pool.sort(key=lambda x: x[1].get("importance", 0), reverse=True)

    picks: list[str] = []
    seen_cats: set[str] = set()
    # First pass: one per distinct category, ordered by importance
    for cat_key, a in pool:
        if len(picks) >= 3:
            break
        if cat_key in seen_cats:
            continue
        seen_cats.add(cat_key)
        label = CATEGORY_LABEL.get(cat_key, cat_key)
        picks.append(f"【{label}】{a.get('title', '').strip()}")

    # Second pass: fill up to 3 from the remaining pool if still short
    if len(picks) < 3:
        for cat_key, a in pool:
            if len(picks) >= 3:
                break
            label = CATEGORY_LABEL.get(cat_key, cat_key)
            line = f"【{label}】{a.get('title', '').strip()}"
            if line not in picks:
                picks.append(line)

    if not picks:
        return None

    return {"headline": "今日のピックアップ", "points": picks}


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    categories: dict[str, list[dict]] = raw["categories"]
    client = genai.Client(api_key=api_key)

    summarized: dict[str, list[dict]] = {}
    for category, articles in categories.items():
        summarized[category] = summarize_category(client, category, articles)

    daily_digest = summarize_daily_digest(client, summarized)

    now_jst = datetime.now(JST)
    date_str = now_jst.strftime("%Y-%m-%d")
    output = {
        "date": date_str,
        "generated_at": now_jst.isoformat(),
        "model": MODEL,
        "daily_digest": daily_digest,
        "categories": summarized,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{date_str}.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
