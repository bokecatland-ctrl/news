# Daily News Digest

国内・国際・ホテル/観光・テクノロジー/AI の主要ニュースを毎朝 7:00 JST に自動収集・要約して公開する静的サイト。

## 概要

- **トピック**: 国内 / 国際 / ホテル・観光 / テクノロジー・AI
- **更新**: 毎日 07:00 JST (GitHub Actions cron で自動実行)
- **公開**: GitHub Pages (`main` ブランチの `/docs`)
- **要約**: Claude API (`claude-haiku-4-5`) で日本語 2〜3 文に要約
- **記事配信**: 各ソースの RSS フィード経由 (要約 + 元記事リンクのみ、本文転載なし)

## ローカルでの動作確認

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

python scripts/fetch_news.py    # RSS 取得 -> tmp/raw_articles.json
python scripts/summarize.py     # Claude で要約 -> docs/data/YYYY-MM-DD.json
python scripts/build_site.py    # HTML 生成 -> docs/index.html

python -m http.server 8000 --directory docs
# http://localhost:8000 を開く
```

## 構成

```
news/
├── .github/workflows/daily-update.yml   # 毎日 07:00 JST cron
├── scripts/
│   ├── sources.yaml                     # RSS ソース定義 (カテゴリ別)
│   ├── fetch_news.py                    # RSS 収集
│   ├── summarize.py                     # Claude API 要約
│   └── build_site.py                    # 静的 HTML 生成
├── templates/
│   ├── index.html.j2                    # メイン Jinja2 テンプレート
│   └── archive_index.html.j2            # アーカイブ一覧
├── docs/                                # GitHub Pages 公開ルート
│   ├── index.html                       # 当日ダイジェスト (自動生成)
│   ├── assets/style.css
│   ├── archive/YYYY-MM-DD.html          # 日次アーカイブ
│   └── data/YYYY-MM-DD.json             # 構造化データ
└── requirements.txt
```

## 初回セットアップ

リポジトリ作成直後に GitHub 側で2点設定する必要があります:

1. **Settings → Secrets and variables → Actions** で `ANTHROPIC_API_KEY` を登録
2. **Settings → Pages** で公開ソースを `main` ブランチの `/docs` フォルダに設定
3. **Actions タブから "Daily News Update" を手動実行** (`workflow_dispatch`) して初回データを生成

数分後に `https://bokecatland-ctrl.github.io/news/` でサイトが見られるようになります。

## ソースの追加・削除

`scripts/sources.yaml` を編集してコミットするだけ。カテゴリ別に RSS の URL を並べています。1つのソースが落ちても他のソースで補完される設計です。

## 注意

- 各記事の著作権は元配信元に帰属します。サイトは要約と元記事リンクのみを掲載しています。
- 要約は Claude API による自動生成のため、ニュアンスや事実関係に誤りが含まれる可能性があります。重要な意思決定は必ず元記事を確認してください。
