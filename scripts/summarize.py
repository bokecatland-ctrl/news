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
    "hotel": "ホテル",
    "tech_ai": "テクノロジー・AI",
}

CATEGORY_FOCUS = {
    "domestic": (
        "日本国内のニュース。**マクロ経済・金融政策・財政・主要政策**を最優先する。"
        "具体的には: 日銀の金融政策、為替、株価、金利、政府の主要政策発表 (税制、産業政策、エネルギー政策)、"
        "原発・脱炭素、財政、雇用・賃金など。"
        "単発の事件・事故・芸能・スポーツは原則除外。社会ネタは制度・産業・経済に影響する重大なもののみ。"
    ),
    "international": (
        "国際ニュース。**地政学・外交・国際経済・エネルギー市場**を最優先する。"
        "G7/G20、米中・中東情勢、戦争・停戦、サプライチェーン、原油・LNG市場、グローバル金融政策、"
        "主要国の選挙・政権交代など。"
        "個別の事件・事故は世界経済や国際秩序に影響するもののみ。芸能・スポーツは除外。"
    ),
    "hotel": (
        "**ホテル経営者・運営者の視点**でニュースを選ぶ。これがこのカテゴリの最重要ポイント。"
        "次の3軸を意識すること:"
        " (1) **業界直接**: 国内ホテル動向、新規開業、M&A、海外ラグジュアリーチェーン、観光統計、インバウンド需給。"
        " (2) **コスト要因**: エネルギー価格、為替、人件費、原材料、サプライチェーンなど運営コストに効くマクロ動向。"
        " (3) **需要・体験**: 天候・気候、テクノロジー (AI/予約アプリ/音声操作)、消費者行動の変化など宿泊需要や顧客体験に影響する動向。"
        "他カテゴリのニュースでも上記3軸に直結するなら積極的に拾うこと。"
        "単なる小ネタや広報リリースより「経営判断に効く」記事を優先。"
    ),
    "tech_ai": (
        "テクノロジー・AI業界。**大手テック企業の戦略、AI、半導体・市場**を最優先する。"
        "Apple/Google/Microsoft/Meta/Amazon/NVIDIA/OpenAI/Anthropic クラスの戦略的動向、"
        "AI製品の節目となるリリース、半導体株価・サプライチェーン、AIインフラ投資、"
        "規制 (EU AI法、米中半導体規制等)、業界構造を変えるスタートアップなど。"
        "機能追加レベルの小ネタや個別ガジェットレビューは除外。"
    ),
}

SYSTEM_PROMPT = """あなたはビジネスパーソン向け朝刊ブリーフィングの編集者です。
読者は経営者・管理職レベルで、自分の仕事や経営判断に効く情報を求めています。
複数のRSSフィードから取得した記事リストを受け取り、本当に価値のある記事を厳選して要約します。

═══════════════════════════════════════
【最重要原則: トピック選定】
═══════════════════════════════════════

✅ 優先するもの:
- マクロ経済・金融政策 (中央銀行、為替、株価、金利、財政)
- エネルギー・地政学 (中東、原油、LNG、原発、サプライチェーン、停戦/紛争)
- 大手テック企業の戦略・AI・半導体動向 (業界構造に影響するレベル)
- 政府の主要政策発表 (税制、産業政策、エネルギー政策、規制)
- 業界構造を変えるM&A・規制変更・技術破壊
- ホテル運営の判断材料となる動き (需要・コスト・テクノロジー面)

❌ 除外するもの:
- 芸能・スポーツ・有名人ゴシップ
- 単発の社会事件・事故 (大規模災害は除く)
- 政治家の細かい派閥争い・失言
- 機能追加レベルのIT小ネタ・個別ガジェットレビュー
- 単純な広報リリース・人事異動 (経営インパクトが薄いもの)

═══════════════════════════════════════
【要約の質】
═══════════════════════════════════════

- 単なる事実伝達では不十分。**「これが何を意味するか」「次の焦点はどこか」を1文加える**
- 数字・固有名詞・日付を具体的に入れる (例: "終値6万8402円", "6/8 G7開幕", "10億ドルでGemini採用")
- 同じトピックを複数記事が報じている場合は最も信頼できる1件に統合 (重複させない)
- ビジネスパーソンが「自分の仕事にどう関係するか」を判断できる粒度に

═══════════════════════════════════════
【タグ付け】
═══════════════════════════════════════

各記事に短いタグ (日本語8文字以内、または英語10文字以内) を付ける。
記事の本質を一語で表すラベル。

良い例:
- マーケット / 金融政策 / 財政 / エネルギー政策 / G7 / 中東情勢
- 需要回復 / コスト・市況 / インバウンド / 業界再編
- WWDC / Siri刷新 / 半導体・市場 / AI規制 / M&A

═══════════════════════════════════════
【出力ルール】
═══════════════════════════════════════

- 重要度 (importance) 1〜5 を付与 (5が最重要)
- ソース名と元URLは入力をそのまま使用 (改変禁止)
- 出力は指定JSON形式のみ。前後に説明文を書かない
- 件数を埋めるための妥協は不要。「これは絶対外せない」記事だけを選ぶ"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "articles": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "tag": {"type": "STRING"},
                    "summary": {"type": "STRING"},
                    "source": {"type": "STRING"},
                    "url": {"type": "STRING"},
                    "importance": {"type": "INTEGER"},
                    "published": {"type": "STRING", "nullable": True},
                },
                "required": ["title", "tag", "summary", "source", "url", "importance"],
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

    return f"""カテゴリ: 【{label}】

このカテゴリで選ぶべき記事の焦点:
{focus}

以下は直近24時間の {label} 関連記事 ({len(articles)}件) です。
ビジネスパーソンの経営判断に効く記事だけを **3件 (上限4件)** に厳選してください。
件数を埋めるための妥協は不要。重要度の低い記事は迷わず削ること。
理想は重要度4以上の記事だけを残すこと。

{articles_block}

【出力】
articles 配列のみのJSON。
各要素は title / tag / summary / source / url / importance / published を含むこと:
- tag: 8文字以内の短いラベル (例: マーケット、Siri刷新、G7)
- summary: 数字・固有名詞を含めた2〜3文。最後に「次の焦点」「意味するところ」を1文添えるのが望ましい
- importance: 1〜5 の整数
- published: 入力をそのまま (なければ null)"""


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
        tag = str(item.get("tag", "")).strip()
        if len(tag) > 12:
            tag = tag[:12]
        out.append({
            "title": str(item["title"]),
            "tag": tag or None,
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
            "tag": None,
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
