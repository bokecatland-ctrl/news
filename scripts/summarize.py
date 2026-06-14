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
MAX_OUTPUT_TOKENS = 16000
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
        "日本国内のニュース。**教養として知っておくべき内容** を最優先する。\n"
        "具体的には:\n"
        "- マクロ経済・金融政策 (日銀、為替、株価、金利、財政、GDP、CPI)\n"
        "- 政府の主要政策発表 (税制、産業政策、エネルギー政策、規制、制度改正)\n"
        "- 主要国会議・首相外交 (G7前後の動き、米中関係の日本への影響)\n"
        "- 科学技術の節目 (H3ロケット成功、ノーベル賞日本人受賞、新薬承認等)\n"
        "- 歴史的な政治イベント (政権交代、皇室、選挙、憲法改正論議)\n"
        "- エネルギー・原発・脱炭素の重要決定\n"
        "単発の事件・事故・芸能・スポーツは原則除外。"
        "社会ネタは制度・産業・国民生活に大きく影響する重大なもののみ。"
    ),
    "international": (
        "国際ニュース。**地政学・大型国際イベント・教養としての世界情勢** を最優先する。\n"
        "具体的には:\n"
        "- **大型国際イベントは徹底網羅**: G7/G20、APEC、米中首脳会談、国連総会、"
        "  COP気候会議、ダボス会議 (準備段階・当日・成果発表すべて)\n"
        "- 地政学・戦争・停戦 (中東、ウクライナ、台湾海峡、米中対立)\n"
        "- グローバル金融政策・国際経済 (Fed、ECB、IMF、世界貿易)\n"
        "- 主要国の選挙・政権交代・大型政策発表\n"
        "- エネルギー安全保障・サプライチェーン\n"
        "- 科学・宇宙・医療の世界的な節目\n"
        "個別事件は世界経済・国際秩序・歴史に影響するもののみ。芸能・スポーツは除外。\n"
        "(W杯・五輪のような国家規模スポーツイベントは「文化的事象」として可)"
    ),
    "hotel": (
        "ホテル業界・観光のニュース。**量より質を厳しく重視**。1日1〜2件で十分、ゼロでも可。\n"
        "\n"
        "✅ 拾うべき (一般教養として価値あり):\n"
        "- 国際ラグジュアリーチェーン (Marriott/Hilton/Hyatt/Accor等) の大規模戦略・大型開業\n"
        "- 業界を代表する大企業 (アパ、星野リゾート、東急、JR系、東宝) の大型M&A・グローバル展開\n"
        "- 政府の観光政策の根本的な変更 (JNTOの長期戦略、観光税、新ビザ制度、観光立国計画)\n"
        "- インバウンド統計の節目発表 (年間訪日数の記録、外国人消費額の動向)\n"
        "- 一般ニュースが旅行・観光に与える大きな影響 (大規模災害、地政学、為替、感染症)\n"
        "\n"
        "❌ 拾ってはいけない (ノイズ):\n"
        "- **業界セミナー・研修会・表彰式・小規模イベント** (絶対除外)\n"
        "- 中小事業者・地方の小規模ホテルの開業・運営ネタ\n"
        "- 個別施設のキャンペーン・新メニュー・季節企画\n"
        "- 内輪の人事異動・社内表彰\n"
        "- ホテル向けの細かいオペレーション・研修コンテンツ\n"
        "\n"
        "判断基準: 「ホテル業界に興味のない一般人が知っていてもいい話か?」を問うこと。\n"
        "業界内部の人しか興味ないネタは全て除外。"
    ),
    "tech_ai": (
        "テクノロジー・AI業界。**大手テック企業の戦略、AI、半導体・市場、大型発表会** を最優先する。\n"
        "\n"
        "対象企業: Apple/Google/Microsoft/Meta/Amazon/NVIDIA/OpenAI/Anthropic/Tesla/SpaceX クラス。\n"
        "\n"
        "✅ **大型発表会は徹底網羅** (これが特に重要):\n"
        "- Apple WWDC、iPhone発表会\n"
        "- Google I/O、Google Cloud Next\n"
        "- Microsoft Build、Ignite\n"
        "- OpenAI DevDay、新モデル発表\n"
        "- Anthropic の新モデル・発表 (Claude シリーズ等)\n"
        "- NVIDIA GTC、CES、MWC\n"
        "- Meta Connect、Samsung Unpacked\n"
        "これらは複数の発表が連続するため、1イベントで複数記事に分けて拾ってよい。\n"
        "例: Google I/O 当日は「Gemini新モデル」「Gemini Spark」「Android XR」など別記事で並べる。\n"
        "\n"
        "✅ その他:\n"
        "- AI製品の節目リリース、業界構造を変える発表\n"
        "- 半導体株価・サプライチェーン、AIインフラ投資\n"
        "- 規制 (EU AI法、米中半導体規制、SNS規制)\n"
        "- 業界激変級のスタートアップ動向 (大型IPO、評価額の節目)\n"
        "- 宇宙開発の節目 (SpaceX、Blue Origin、JAXAなど)\n"
        "\n"
        "❌ 除外: 機能追加レベルの小ネタ、個別ガジェットレビュー、社内人事"
    ),
}

# カテゴリ別の出力件数上限。hotel は厳選してノイズを減らす。
MAX_OUTPUT_BY_CATEGORY = {
    "domestic": 4,
    "international": 4,
    "hotel": 2,
    "tech_ai": 4,
}

SYSTEM_PROMPT = """あなたは「教養を育てる朝刊ブリーフィング」の編集者です。
読者は知的好奇心が強く、世の中の動きを把握して人として成長したいと考えています。
**一般常識・教養として知っておくべき出来事** を中心に選びます。

═══════════════════════════════════════
【最重要原則: トピック選定】
═══════════════════════════════════════

✅ 優先するもの (知識・教養として価値がある):
- マクロ経済・金融政策 (中央銀行、為替、株価、金利、財政)
- 地政学・国際情勢 (戦争、停戦、外交、エネルギー安全保障、米中、中東)
- 大手テック企業の戦略・AI・半導体動向 (業界構造に影響するレベル)
- 政府の主要政策発表 (税制、産業政策、エネルギー政策、規制、制度改正)
- 業界構造を変えるM&A・規制変更・技術破壊
- 科学・宇宙・医療の節目となる発見・成果
- 主要国の選挙・政権交代・歴史的政治イベント

✅ **大型イベントは徹底的に網羅すること** (これが特に重要):
- 国際: G7/G20、APEC、米中首脳会談、国連総会、COP気候会議、ダボス会議
- テック: Apple WWDC、Google I/O、Microsoft Build、OpenAI DevDay、Anthropic 発表、
        CES、MWC、NVIDIA GTC、Meta Connect、Samsung Galaxy Unpacked
- スポーツ・文化: オリンピック、W杯、ノーベル賞発表、アカデミー賞
- 宇宙: NASA/JAXA/SpaceX/Blue Origin の節目ミッション
これらは **当日 / 前日 / 翌日** にしっかり取り上げる。
複数の発表がある場合は別記事に分けて多めに拾ってよい (例: Google I/O で4記事連続)。

❌ 除外するもの (教養にならない、ノイズ):
- 芸能・有名人ゴシップ
- 単発の小規模社会事件・事故 (大規模災害は除く)
- 政治家の細かい派閥争い・失言
- 機能追加レベルのIT小ネタ・個別ガジェットレビュー
- 単純な広報リリース
- **セミナー・業界内イベント・小規模事業者の発表・社内表彰** (絶対除外)
- **中小企業や地方の細かい動き** (全国・国際的に影響しないもの)

═══════════════════════════════════════
【横断視点が極めて重要】
═══════════════════════════════════════

大きなマクロイベントは複数カテゴリにまたがって出てくる。たとえば:
- G7開催 → 国際 (本体) + 国内 (首相外交) + Tech (AI規制議題)
- WWDC → Tech (本体・複数記事) + 国際 (テック動向として)
- 米イラン情勢 → 国際 (本体) + 国内 (エネルギー安保政策)
- ノーベル賞 → 国内 (日本人受賞時) + 国際 (世界の動向として)

同じ事件でもカテゴリごとに **切り口を変えて** 書くこと。

═══════════════════════════════════════
【見出しの質】
═══════════════════════════════════════

見出しは情報密度を最大化する。単なる主語+述語ではなく、副題的な情報を含める:
- 良い例: 「米イラン、停戦60日延長で合意間近　「合意後30日でホルムズ機雷掃海、通航再開へ」」
- 良い例: 「Skyscanner調査「2026年の新トレンド」　徳島742%増・旭川476%増と地方都市に脚光」
- 良い例: 「Apple WWDC 2026プレビュー (6/8〜)　Geminiに追われるSiriの大刷新に注目」
- 悪い例: 「米イラン交渉が進む」(抽象的、数字なし、含意なし)

スペース2つで主見出しと副題を区切るパターン、引用符で副題を入れるパターン、いずれも可。

進行中の重大事案 (台風直撃中、首脳会談継続中等) は見出し末尾に「【随時更新】」を付けてよい。

═══════════════════════════════════════
【要約の質】
═══════════════════════════════════════

- 単なる事実伝達では不十分。**「これが何を意味するか」「次の焦点はどこか」を1文加える**
- **具体的な数字・固有名詞・日付・人名・場所を絶対に省略しない**
  (例: "終値6万8402円", "6/8 G7開幕", "10億ドルでGemini採用", "年間1,400億円",
   "徳島742%増", "和歌山県南部に午前4時半上陸", "Skyscanner調査")
- 同じトピックを複数記事が報じている場合は最も信頼できる1件に統合 (同一カテゴリ内では重複させない)
- ビジネスパーソンが「自分の仕事にどう関係するか」を判断できる粒度に

【文体】
- 国内・国際・Tech カテゴリ → 新聞記事調 (「〜となる」「〜という見通し」「〜が焦点」)
- ホテル カテゴリ → ホテル経営者向けに **提案調を一部混ぜる** (「〜を推奨」「〜したい」「〜が重要」)

═══════════════════════════════════════
【速報級の事案】
═══════════════════════════════════════

以下に該当する突発的な重大事案は **タグを「速報」または「速報・XX」(複合) にする**:
- 大型M&A・IPO・経営トップ交代 (例: AnthropicのIPO書類提出)
- 大規模災害・台風直撃・地震 (例: 台風6号上陸)
- 主要国首脳の電撃会談・電撃発表
- 緊急の制度・規制変更
- 業界構造を激変させる発表 (重要モデルリリース等)

速報タグはサイト上で赤系で強調表示されるため、本当に重要な突発事案だけに使うこと。
1日に複数あって構わない。なくても構わない。

═══════════════════════════════════════
【タグ付け】
═══════════════════════════════════════

各記事に短いタグ (8〜12文字) を付ける。記事の本質を一語で表すラベル。

タグの種類:
1. **通常タグ** (内容を表す): マーケット / 金融政策 / 財政 / エネルギー政策 / G7 / 中東 / 需要回復
2. **時間性タグ** (直近イベント): 「本日開幕」「明日」「今日から」「来週」「○○日後」
   → 当日・翌日・近日中の大型イベントには時間性タグを使うこと
3. **クロスタグ** (横断ジャンル): 「テック×ホテル」「AI×規制」「制度・運用」

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

    # hotel は特に件数を絞る
    if category == "hotel":
        count_instruction = (
            f"このカテゴリは **量より質**。上限{max_output}件まで。\n"
            f"教養として価値のあるものだけを厳選し、該当がなければ **0件でも構わない**。\n"
            f"特にセミナー・小規模事業者・地方の細かいネタは絶対に拾わないこと。"
        )
    else:
        target = max(1, max_output - 1)
        count_instruction = (
            f"**{target}件 (上限{max_output}件)** に厳選してください。\n"
            f"ただし大型イベント (G7、WWDC、Google I/O、ノーベル賞 等) の当日・前日・翌日は、\n"
            f"そのイベント関連記事を複数連続で拾ってよい (上限を超えてもよい、最大{max_output + 2}件まで)。\n"
            f"件数を埋めるための妥協は不要。重要度の低い記事は迷わず削ること。"
        )

    return f"""カテゴリ: 【{label}】

このカテゴリで選ぶべき記事の焦点:
{focus}

以下は直近24時間の {label} 関連記事 ({len(articles)}件) です。

{count_instruction}

{articles_block}

【出力】
articles 配列のみのJSON。
各要素は title / tag / summary / source / url / importance / published を含むこと:
- tag: 8文字以内の短いラベル (例: マーケット、Siri刷新、G7)
- summary: 数字・固有名詞を含めた2〜3文。最後に「次の焦点」「意味するところ」を1文添えるのが望ましい
- importance: 1〜5 の整数
- published: 入力をそのまま (なければ null)"""


def summarize_category(
    client: genai.Client,
    category: str,
    articles: list[dict],
    max_input: int = MAX_ARTICLES_PER_CATEGORY_INPUT,
) -> list[dict]:
    if not articles:
        return []

    max_output = MAX_OUTPUT_BY_CATEGORY.get(
        category, MAX_ARTICLES_PER_CATEGORY_OUTPUT
    )
    truncated = articles[:max_input]
    user_prompt = build_user_prompt(category, truncated, max_output)

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
        print(
            f"[ERROR] Gemini API failed for {category}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return _fallback_articles(truncated, max_output)

    # Diagnostic: log finish reason so next run reveals MAX_TOKENS / SAFETY / etc.
    try:
        for i, cand in enumerate(getattr(response, "candidates", None) or []):
            print(
                f"[diag] {category} candidate[{i}].finish_reason="
                f"{getattr(cand, 'finish_reason', None)}",
                file=sys.stderr,
            )
    except Exception:
        pass

    # response.text may raise if the response was blocked or empty
    try:
        text = (response.text or "").strip()
    except Exception as e:
        print(
            f"[WARN] response.text failed for {category}: "
            f"{type(e).__name__}: {e}; falling back",
            file=sys.stderr,
        )
        return _fallback_articles(truncated, max_output)

    if not text:
        print(f"[WARN] empty response for {category}; falling back", file=sys.stderr)
        return _fallback_articles(truncated, max_output)

    try:
        data = json.loads(text)
        result = data.get("articles", [])
    except json.JSONDecodeError as e:
        print(
            f"[WARN] JSON parse failed for {category}: {e}; falling back",
            file=sys.stderr,
        )
        print(f"[diag] {category} raw text head: {text[:300]}", file=sys.stderr)
        return _fallback_articles(truncated, max_output)

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
        if len(tag) > 14:
            tag = tag[:14]
        # 速報タグは常に最重要扱いにする
        if "速報" in tag:
            importance = 5
        out.append({
            "title": str(item["title"]),
            "tag": tag or None,
            "summary": str(item.get("summary", "")),
            "source": str(item.get("source", "")),
            "url": str(item["url"]),
            "importance": importance,
            "published": item.get("published"),
        })
    # 速報タグの記事を最優先、次に importance 順
    out.sort(
        key=lambda a: (
            0 if a.get("tag") and "速報" in a["tag"] else 1,
            -a["importance"],
        )
    )
    print(f"[summarize] {category}: produced {len(out)} summaries", file=sys.stderr)
    return out


def _fallback_articles(
    articles: list[dict],
    max_output: int = MAX_ARTICLES_PER_CATEGORY_OUTPUT,
) -> list[dict]:
    """If the LLM fails, fall back to headlines-only (no summary)."""
    fallback = []
    for a in articles[:max_output]:
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

【今日のポイント (points)】
- 各ポイントは1〜2文の日本語、合計3点 (内容が薄い日は2点でもよい)
- カテゴリ偏りなく重要なものを選ぶ
- 個々の見出しを羅列するのではなく、要点を抽出する
- 客観的・簡潔に

【見出し (headline)】
全体を象徴する15〜25字程度の見出し。今日の主軸テーマを端的に。
良い例: 「G7とWWDC、来週同日開幕」「台風6号上陸 + Anthropic IPO」「Google I/O 特別号」

【アラート (alert)】← 重要!
今日の最重要かつ突発的な1件 (速報級) があれば alert に短くまとめる。
画面最上部に強調表示される。基準:
- 大規模災害・台風直撃進行中 / 大型M&A・IPO電撃発表 / 主要国首脳会談電撃決定
- 大型イベント当日 (G7開幕日、WWDC開幕日等) のみ
- AnthropicやOpenAIなど業界激変級のリリース

alert の形式:
{ "icon": "🌀 等の絵文字1〜2文字", "text": "1行 (40字以内目安) の本文" }
例: { "icon": "🌀", "text": "台風6号 現在東海〜関東に最接近中　鉄道・航空に大規模影響" }
例: { "icon": "🔴", "text": "速報: Anthropic、SECにIPO書類を非公開提出　OpenAIに先行" }
例: { "icon": "🇫🇷", "text": "本日開幕: G7エヴィアン・サミット　高市首相・トランプ氏が初対面" }

該当する突発的事案がない日は alert を null にすること (毎日無理に作らない)。

【出力】
指定JSON形式のみ。前後に説明文を書かない。
"""

DAILY_DIGEST_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "headline": {"type": "STRING"},
        "points": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "alert": {
            "type": "OBJECT",
            "nullable": True,
            "properties": {
                "icon": {"type": "STRING"},
                "text": {"type": "STRING"},
            },
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

    # alert はオプション。形式が壊れていれば捨てる。
    alert_raw = data.get("alert")
    alert: dict | None = None
    if isinstance(alert_raw, dict):
        icon = str(alert_raw.get("icon", "")).strip()
        text = str(alert_raw.get("text", "")).strip()
        if icon and text:
            alert = {"icon": icon[:4], "text": text[:120]}

    return {
        "headline": str(data.get("headline", "")).strip() or "今日のピックアップ",
        "points": points[:3],
        "alert": alert,
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

    return {"headline": "今日のピックアップ", "points": picks, "alert": None}


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    categories: dict[str, list[dict]] = raw["categories"]
    client = genai.Client(api_key=api_key)

    # Hotel カテゴリは「他カテゴリの記事をホテル経営者目線で再解釈」も含むため、
    # 全カテゴリの記事を統合したプールから選定させる。
    # URL で重複排除し、業界専門ソース (hotel) の記事を先頭に並べることで
    # 業界直接ニュースが拾われやすくなるようにする。
    seen_urls: set[str] = set()
    hotel_pool: list[dict] = []
    # Step 1: 業界専門ソースの記事を先に
    for a in categories.get("hotel", []):
        url = a.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            hotel_pool.append(a)
    # Step 2: 他カテゴリの記事を追加 (重複は除く)
    for cat_key in ("domestic", "international", "tech_ai"):
        for a in categories.get(cat_key, []):
            url = a.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                hotel_pool.append(a)

    summarized: dict[str, list[dict]] = {}
    for category, articles in categories.items():
        if category == "hotel":
            # 拡張プールから 100 件まで渡す (hotel 業界 + 他カテゴリ横断)
            summarized[category] = summarize_category(
                client, category, hotel_pool, max_input=100
            )
        else:
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
