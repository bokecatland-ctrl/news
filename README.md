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

## 初回セットアップ手順

### ステップ 1: Anthropic API キーを発行

1. `https://console.anthropic.com/` にログイン (アカウントがなければ作成)
2. **Settings → Billing** でクレジットカードを登録、$5〜$10 をチャージ
   - 本サイトは Claude Haiku 4.5 利用で 1 日あたり数円〜数十円程度
3. **Settings → API Keys → "Create Key"** をクリック
4. Name: `news-digest-github-actions` (任意)
5. 表示された `sk-ant-api03-...` で始まる文字列を**その場でコピー**
   - ⚠️ この画面を閉じると二度と表示されません

### ステップ 2: GitHub Secrets に登録

GitHub Actions の cron ジョブがこのキーを使って Claude API を呼びます。コードに書くと漏洩するので必ず Secrets に登録します。

1. `https://github.com/bokecatland-ctrl/news/settings/secrets/actions` を開く
2. 右上の緑ボタン **"New repository secret"** をクリック
3. 入力:
   - **Name**: `ANTHROPIC_API_KEY` (完全一致、大文字)
   - **Secret**: ステップ 1 でコピーしたキー
4. **"Add secret"** をクリック

登録後、Secrets 一覧に `ANTHROPIC_API_KEY` が出ていれば OK (値そのものは GitHub 上でも見えなくなります。これが正しい挙動です)。

### ステップ 3: GitHub Pages を有効化

1. `https://github.com/bokecatland-ctrl/news/settings/pages` を開く
2. **Source**: `Deploy from a branch`
3. **Branch**: `main` / フォルダ: `/docs`
4. **"Save"** をクリック

### ステップ 4: 初回ビルドを手動実行

cron 待ちせずに今すぐ動かして動作確認します。

1. `https://github.com/bokecatland-ctrl/news/actions` を開く
2. 左サイドから **"Daily News Update"** を選択
3. 右の **"Run workflow"** → Branch `main` → **"Run workflow"**
4. 2〜4 分待つと緑のチェックマークで完了

### ステップ 5: 公開 URL を開く

`https://bokecatland-ctrl.github.io/news/`

PC でもスマホでも閲覧できます。スマホでは Safari → 共有 → 「ホーム画面に追加」でアプリのように使えます。

### トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| `ANTHROPIC_API_KEY is not set` | Secret 未登録 / 名前が違う | ステップ 2 を再確認 |
| `401 Unauthorized` | キーが無効 | Console で新キー発行 → Secret 更新 |
| `429 / 403` | クレジット切れ・レート制限 | Billing でチャージ |
| Pages が 404 | `main` にマージされていない / Pages 設定の Branch がズレている | ステップ 3 を確認 |

### キーの定期ローテーション (推奨)

3〜6 ヶ月ごとに:

1. Console で新規キー発行
2. GitHub Secrets を "Update" で上書き
3. Console で旧キーを Revoke

## ソースの追加・削除

`scripts/sources.yaml` を編集してコミットするだけ。カテゴリ別に RSS の URL を並べています。1つのソースが落ちても他のソースで補完される設計です。

## 注意

- 各記事の著作権は元配信元に帰属します。サイトは要約と元記事リンクのみを掲載しています。
- 要約は Claude API による自動生成のため、ニュアンスや事実関係に誤りが含まれる可能性があります。重要な意思決定は必ず元記事を確認してください。
