# 普通话ナロー（精密）国際音声記号（IPA）音声認識

[![ModelScope](https://img.shields.io/badge/ModelScope-魔搭-blue)](https://www.modelscope.cn/models/QiGuanFuChen/mandarin-ipa-asr)

<p align="center">
<a href="README.md"><img alt="中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-blue"></a>
<a href="README_en.md"><img alt="English" src="https://img.shields.io/badge/English-blue"></a>
<a href="README_ja.md"><img alt="日本語" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-brightgreen"></a>
<a href="README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-blue"></a>
<a href="README_vi.md"><img alt="Tiếng Việt" src="https://img.shields.io/badge/Ti%E1%BA%BFng_Vi%E1%BB%87t-blue"></a>
<a href="README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Fran%C3%A7ais-blue"></a>
<a href="README_ru.md"><img alt="Русский" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue"></a>
</p>

[SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) エンコーダをベースとし、その上に**ナロー（精密）国際音声記号（IPA）CTC デコード・ヘッド**を実装し、全パラメータ微調整（full fine-tuning）によって、普通話の「音節＋声調」レベルの音声書き起こしを実現します。

モデルは空白区切りのナロー IPA 音節列を出力し、各音節には独自の声調値記号（例 `ɡ̊wa̠n̚˥`、`x̞wa̠ɪ̯˧˥`）が付与されます。すなわち、声母／韻母と声調が同時に得られ、音声学分析、普通話発音評価、声調指導などの用途に適しています。

> ライセンス：**CC BY-NC-SA 4.0**（Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International）。
> モデル重みと学習データはコードリポジトリには同梱されず、それぞれ ModelScope およびローカルのデータ構築フローから取得します。詳細は後述。

---

## 特徴

- **ナロー IPA 出力**：[nk2028/putonghua-ipa-converter](https://github.com/nk2028/putonghua-ipa-converter) の UntPhesoca ナロー方式を採用し、音素レベルで标注しつつ声調値を保持。
- **全パラメータ微調整**：SenseVoiceSmall の Conformer エンコーダを解凍し、新規 IPA ヘッドとともに学習；特徴フロントエンドは凍結、`dither=0` で再現性を保証。
- **ラベルに依存しない声調指標**：`ipa2tone` マッピング表（IPA 音節 → 調類 1–5）を用いて声調精度を計算。具体的な音素表記によらず公平な比較が可能。
- **大規模検証**：5918 件の統合検証セットで実測（[評価指標](#評価指標大規模検証) を参照）。
- **Gradio デモ**：音声のアップロード／録音による認識と、参照テキストとの差異の自動比較（音節レベルの声調アライメント強調）をサポート。

---

## ディレクトリ構成

```
mandarin-ipa-asr/
├── LICENSE                                  # CC BY-NC-SA 4.0 全文
├── README.md
├── requirements.txt                         # 実測済み依存バージョン
├── .gitignore
├── app.py                                   # Gradio デモ（認識＋差異自動比較）
├── configs/
│   └── example_train_config.json            # 学習設定例
├── vocab/
│   ├── vocab_mandarin_ipa_combined.json     # IPA 出力語彙（1427 クラス、公開）
│   └── vocab_mandarin_ipa_tone_combined.json# IPA 音節→調類 1-5 マッピング（公開）
├── src/
│   ├── model.py        # SenseVoiceIpa：エンコーダ＋IPA CTC ヘッド
│   ├── utils.py        # CTC デコード／指標／アライメント／ビーム探索／音声読み込み
│   ├── dataset.py      # 音声-テキストデータセット（wav/flac/ogg/mp3）
│   ├── train.py        # 全パラメータ微調整学習
│   ├── infer.py        # 単発推論＋バッチ評価（大規模検証）
│   └── prepare_data.py # 拼音テキストから IPA 学習セットを構築（前処理、音声なし）
└── results/
    └── metrics.md       # 実測検証指標と学習曲線
```

---

## ベースモデル

| 項目 | 説明 |
| --- | --- |
| 名称 | SenseVoiceSmall（FunAudioLLM / 阿里達摩院） |
| 出典 | GitHub: [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)；ModelScope: `iic/SenseVoiceSmall` |
| 構造 | `WavFrontend`（fbank 特徴＋任意の f0）＋ Conformer エンコーダ（512 次元）＋ 元の漢字 CTC ヘッド |
| 本プロジェクトの変更 | フロントエンドを凍結；その Conformer エンコーダ上に新たな **IPA CTC ヘッド** を実装（後述）；元の漢字ヘッドは使用しない |

> ベースモデル重みは初回実行時に ModelScope から自動ダウンロードしてキャッシュされ、手動準備は不要。

---

## 学習手法

1. **初期化**：事前学習済み SenseVoiceSmall の `WavFrontend` と `encoder` を読み込み（元の漢字 CTC ヘッドは破棄）、新たに IPA CTC ヘッド `ctc_head = Linear(512,512) → ReLU → Dropout(0.1) → Linear(512, vocab)` を作成。
2. **凍結戦略**：`WavFrontend` は常に凍結、`dither` は 0 に固定（ランダムノイズ注入を無効化し特徴の決定性を保証）；エンコーダと IPA ヘッドは**解凍して全パラメータ微調整**（ヘッドのみの学習も可）。
3. **損失と最適化**：`CTCLoss(blank=0)`；AdamW（`lr=1e-4`、`weight_decay=1e-4`）；`CosineAnnealingLR`（`T_max = steps × epochs`）；勾配クリップ 1.0。
4. **精度**：bf16 混合精度（`autocast` はエンコーダの順伝播のみに適用、特徴と損失は fp32 を維持）。
5. **重み保存**：`best.pt`（検証 TER 最低）、`best_tone.pt`（声調精度最高）。
6. **旧重みとの互換性**：推論／学習再開時に、旧重みのヘッド命名の残骸を自動的に `ctc_head.*` に再マッピングし、ヘッドが暗黙的に失われるのを防止。

---

## 学習データ

| 項目 | 数量 |
| --- | --- |
| 学習セット | **11,691** 件（普通話読み上げ；zhvoice 普通話サブセット約 8,993 件を 1.3× 再利用／拡張して構築） |
| 検証セット | **918** 件（普通話読み上げ、学習セットと排他） |
| ラベル | 各件は空白区切りのナロー IPA 音節列（拼音を nk2028/putonghua-ipa-converter で変換） |
| IPA 語彙 | 1,427 クラス（`<blank>`/`<unk>` を含む） |

> 元の zhvoice コーパスは約 **900 時間、3200+ 話者、約 112.98 万テキスト件**；
> 本プロジェクトはそのうち読み上げ品質の高い普通話サブセットのみを使用して学習／検証セットを構築し、ナロー IPA に変換。

---

## モデルパラメータ

| パラメータ | 値 |
| --- | --- |
| 総パラメータ数 | **234,993,874** |
| 全パラメータ微調整の学習可能パラメータ | **222,131,699**（エンコーダ＋IPA ヘッド；フロントエンドと元の漢字ヘッドは凍結） |
| 推論のみ（ヘッド学習可、エンコーダ凍結） | 994,707 |
| エンコーダ次元 | 512 |
| IPA 出力語彙 | 1,427 |
| 声調マッピング項目数（ipa2tone） | 1,425（`<blank>`/`<unk>` を除く） |

---

## データセットの出典とリンク

| データ／ツール | 用途 | ライセンス | リンク |
| --- | --- | --- | --- |
| **zhvoice** | 学習コーパス（普通話読み上げサブセット） | リポジトリ参照 | https://github.com/fighting41love/zhvoice |
| **putonghua-ipa-converter** | 拼音→ナロー IPA 変換（scheme 2, UntPhesoca） | CC0 | https://github.com/nk2028/putonghua-ipa-converter |
| **SenseVoiceSmall** | 事前学習ベースモデル | リポジトリ参照 | https://github.com/FunAudioLLM/SenseVoice（ModelScope `iic/SenseVoiceSmall`） |

---

## 評価指標（大規模検証）

デコードは学習と完全に一致（生波形入力、dither=0、bf16 エンコーダ順伝播）。指標の定義は [`results/metrics.md`](results/metrics.md) を参照。

### 918 件の一級甲等（一甲）普通話検証コーパス

| 指標 | 値 |
| --- | --- |
| TER（音節誤り率、低いほど良い） | **0.0955** |
| ACC（文単位完全一致） | 0.7059 |
| TACC（トークン正解率） | 0.9145 |
| 声調精度（ラベル非依存） | **0.9212** |

### 5918 件の統合検証セット（大規模検証）

＝ 5000 件の zhvoice 実シナリオ mp3 ＋ 918 件の普通話読み上げ。zhvoice は実シナリオで分野が雑多かつノイズが多く、そのため文単位 ACC が一甲普通話検証コーパスより低いのは想定内；声調認識は高い水準を維持。

| 指標 | 値 |
| --- | --- |
| TER（音節誤り率、低いほど良い） | **0.1089** |
| ACC（文単位完全一致） | 0.4439 |
| TACC（トークン正解率） | 0.8986 |
| 声調精度（ラベル非依存） | **0.8915** |

> 補足：声調精度は拼音方式の同位モデルと同等（同一フレームワークで拼音ヘッド声調約 0.92、本 IPA ヘッド 0.89–0.92）。「音素＋声調」の統合标注でも声調情報は失われていないことを示す。

---

## インストールと環境（いずれも実測）

| 依存 | バージョン | 説明 |
| --- | --- | --- |
| Python | **3.9.13** | 3.9 推奨（3.8–3.11 は動作可、3.9.13 を実測） |
| CUDA | **12.8** | 学習／推論には NVIDIA GPU が必要；CPU のみの推論も可だが遅い |
| torch | 2.7.0+cu128 | CUDA 12.8 に対応 |
| torchaudio | 2.7.0+cu128 | |
| funasr | 1.4.16 | SenseVoiceSmall の読み込み |
| modelscope | 1.32.0 | ベースモデル自動ダウンロード |
| transformers | 4.43.0 | |
| numpy | 1.23.4 | |
| editdistance | 0.6.2 | 指標計算 |
| tqdm | 4.64.1 | プログレスバー |
| soundfile | 0.12.1 | 音声読み書き |
| gradio | 4.24.0 | デモ UI |
| librosa | 0.9.2 | （任意）データ前処理 |
| ffmpeg | 2025-08-23 | mp3 デコード（PATH に必要） |

インストール：

```bash
pip install -r requirements.txt
# ffmpeg は別途インストールし PATH に追加（Windows は https://www.gyan.dev/ffmpeg/ や scoop/apt 等）
```

**デバイス要件**：学習は ≥ 16 GB VRAM 推奨（bf16 下で 234M パラメータ全微調整＋活性化）；推論は数 GB で足り、単発音声は CPU でも可（遅い）。

---

## クイックスタート

> すべてのスクリプトは `vocab/`、`data/`、`weights/`、`checkpoints/` などの相対パスを「プロジェクトルート」（リポジトリルート）基準で解決するため、**プロジェクトルートに `cd` しなくても任意の作業ディレクトリから起動可能**；絶対パスを渡した場合はそのまま使用。
> ベースモデル `iic/SenseVoiceSmall` は ModelScope のモデル id であり、初回実行時に自動ダウンロードしてキャッシュされる。

> ### 【重要】微調整重みを先に読み込まないと、出力は無意味な文字化けになる
>
> 本プロジェクトは**モデル重みをリポジトリに含めない**（ModelScope で別途公開）。デモや推論を起動する際に **`--ckpt` で重みを指定しない** と、
> プログラムは**ランダム初期化**されたモデルを暗黙的に使用する——音声に対して出力される IPA は無意味となり、しばしば音節が繰り返される。
> 「たくさん認識された」ように見えるが、実際の発音ではない。
> 例えば『静夜思』前半は約 10 音節だが、重みなしでは 30 件以上の繰り返し文字化け音節を出力する可能性がある。
>
> 正しい使い方（重みは先に `weights/best.pt` へダウンロード、次節参照）：
> ```bash
> python app.py --ckpt weights/best.pt
> ```
> `--ckpt` を指定しない場合、`app.py` は `weights/best.pt` の自動検索を試みる；両方とも無い場合は起動時に目立つ【警告】を出力。

### 1. モデル重みの取得

モデル重みは**本リポジトリには含まれない**；ModelScope からダウンロード（作者が別途公開）、例：

```bash
# モデル公開済みと仮定し、modelscope でローカルへダウンロード
modelscope download --model QiGuanFuChen/mandarin-ipa-asr --local_dir weights/
```

`weights/best.pt` を取得後、`--ckpt` で指定。（独自に学習した場合は追加で `checkpoints/best_tone.pt` が出力される；下の「学習ガイド」参照）

### 2. 単発音声の推論

```bash
# プロジェクトルートへの cd は不要：スクリプトは自身の位置基準で vocab/data 等の相対パスを解決；任意のディレクトリから起動可
python src/infer.py --wav path/to/audio.wav --ckpt weights/best.pt
# 出力：空白区切りのナロー IPA 音節列
```

### 3. バッチ評価（検証指標）

```bash
python src/infer.py --eval --ckpt weights/best.pt \
    --val_scp data/val.scp --val_text data/val.text --limit 0
# TER / ACC / TACC / 声調精度を出力
```

### 4. Gradio デモ（音声アップロード＋差異自動比較）

> `--ckpt` で重みを読み込むこと（上の【重要】参照）。そうでないと認識結果はランダム初期化の無意味な文字化けになる。
> 「認識」タブはデフォルトでビーム探索（`--beam`、デフォルト 12）を使用し、挿入／反復誤りを低減；`--beam 0` で貪欲デコードに戻る。

```bash
# 任意のディレクトリから起動可（相対パスはプロジェクトルート基準）；--ckpt は相対または絶対パス可
# --ckpt 省略時は weights/best.pt を自動試行
python app.py --ckpt weights/best.pt --port 7860
# ブラウザで http://127.0.0.1:7860 を開く
```

- **認識**タブ：音声をアップロードまたは録音 → ナロー IPA を出力。
- **比較（自動差異）**タブ：音声をアップロードし（任意で）参照 IPA テキストを入力 →
  音節レベルの声調アライメントを自動実行し差異を強調（正解／声調誤り／読み誤り／過剰／欠落）；
  参照テキストがなくても、「貪欲 vs ビーム探索」の 2 つのデコード結果を自動比較する。

---

## 学習ガイド

1. データを準備（下の「データ利用注意」参照）し、`train.scp` / `train.text` / `val.scp` / `val.text` を得る
   （形式：`uid 音声パス` と `uid 空白区切りIPA音節`）。
2. 実行：

```bash
python src/train.py \
    --train_scp data/train.scp --train_text data/train.text \
    --val_scp data/val.scp --val_text data/val.text \
    --vocab_path vocab/vocab_mandarin_ipa_combined.json \
    --ipa2tone_path vocab/vocab_mandarin_ipa_tone_combined.json \
    --output_dir checkpoints --epochs 10 --batch_size 16
```

パラメータを `configs/example_train_config.json` に書き込み、その後単に実行も可：

```bash
python src/train.py $(python -c "import json,sys; c=json.load(open('configs/example_train_config.json')); print(' '.join(f'--{k} {v}' for k,v in c.items()))")
```

- 拼音全微調整重みからウォームスタートする場合：`--warm_start weights/pinyin_ft.pt`（ヘッド次元が異なれば自動スキップしてランダム初期化）。
- `checkpoints/best.pt`（TER 最低）と `checkpoints/best_tone.pt`（声調最高）を出力。

---

## データ利用注意（データは非公開）

ライセンスと規模の理由から、**学習データは直接公開しない**。以下の手順で同等のデータセットを再構築できる：

1. **zhvoice** コーパス（https://github.com/fighting41love/zhvoice）をダウンロードし、解凍して音声と拼音テキストを取得。
2. **putonghua-ipa-converter**（https://github.com/nk2028/putonghua-ipa-converter）をクローンし、その `data/putonghua.js`（scheme 2 = UntPhesoca ナロー）で拼音をナロー IPA に変換。
3. 本リポジトリの `src/prepare_data.py` で学習用ファイルを生成：

```bash
# 1) zhvoice サブセット -> ナロー IPA テキスト + scp + 語彙
#    --conv_js はコンバータの data/putonghua.js を指す；--audio_root は音声ルート
python src/prepare_data.py build-zhvoice \
    --metadata zhvoice/metadata.csv \
    --audio_root zhvoice/wavs \
    --conv_js path/to/putonghua-ipa-converter/data/putonghua.js \
    --out data/zhvoice_ipa --train_n 8993 --val_n 918

# 2) 自身の声調付き拼音テキストを IPA に変換（ディレクトリ内に train/text、val/text が必要）
python src/prepare_data.py build-mandarin \
    --text_dir data/mandarin_pinyin \
    --conv_js path/to/putonghua-ipa-converter/data/putonghua.js \
    --out data/zhvoice_ipa

# 3) 複数ソースを統合し語彙を拡張、*_combined ファイルを出力
python src/prepare_data.py combine \
    --zhvoice_dir data/zhvoice_ipa --mandarin_ipa_dir data/zhvoice_ipa \
    --out data/combined

# 4) 倍数で学習セットを拡張（例 1.3×）：普通話サブセットを基に zhvoice データを混ぜる
python src/prepare_data.py scale \
    --mandarin_text data/combined/train_text --mandarin_scp data/combined/train_scp \
    --zhvoice_text data/zhvoice_ipa/train_text --zhvoice_scp data/zhvoice_ipa/train_scp \
    --factor 1.3 --out data/ipa130
```

生成された `*.scp`（音声パス）と `*.text`（IPA ラベル）を `train.py` / `infer.py` の入力として使用。
**生音声や第三者テキストを本リポジトリとともに公開しないこと**；公開するのは上記スクリプトと利用説明のみ。

---

## ライセンス

コードと語彙は **CC BY-NC-SA 4.0**（Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International；[LICENSE](LICENSE) 参照）で公開。
- **表示（BY / Attribution）**：利用時は原作者とプロジェクトの出典を保持すること。
- **非営利（NC / NonCommercial）**：商業目的での利用は不可。
- **継承（SA / ShareAlike）**：派生物は同一ライセンスで公開すること。

モデル重みと学習データは各出典ライセンス（ModelScope／データソースリポジトリ）で別途提供され、本リポジトリのライセンスを上書きするものではない。

---

## 引用と謝辞

- ベースモデル：FunAudioLLM, *SenseVoice*.
- ナロー IPA 変換：nk2028, *putonghua-ipa-converter*（CC0）.
- 学習コーパス：fighting41love, *zhvoice*.

---

<p align="center">
<a href="README.md"><img alt="%E4%B8%AD%E6%96%87" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-blue"></a>
<a href="README_en.md"><img alt="English" src="https://img.shields.io/badge/English-blue"></a>
<a href="README_ja.md"><img alt="%E6%97%A5%E6%9C%AC%E8%AA%9E" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-brightgreen"></a>
<a href="README_ko.md"><img alt="%ED%95%9C%EA%B5%AD%EC%96%B4" src="https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-blue"></a>
<a href="README_vi.md"><img alt="Ti%E1%BA%BFng_Vi%E1%BB%87t" src="https://img.shields.io/badge/Ti%E1%BA%BFng_Vi%E1%BB%87t-blue"></a>
<a href="README_fr.md"><img alt="Fran%C3%A7ais" src="https://img.shields.io/badge/Fran%C3%A7ais-blue"></a>
<a href="README_ru.md"><img alt="%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue"></a>
</p>
