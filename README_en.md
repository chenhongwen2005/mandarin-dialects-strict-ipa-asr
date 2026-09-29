# Mandarin Narrow/Phonetic IPA Speech Recognition

[![ModelScope](https://img.shields.io/badge/ModelScope-魔搭-blue)](https://www.modelscope.cn/models/QiGuanFuChen/mandarin-ipa-asr)

<p align="center">
<a href="README.md"><img alt="中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-blue"></a>
<a href="README_en.md"><img alt="English" src="https://img.shields.io/badge/English-brightgreen"></a>
<a href="README_ja.md"><img alt="日本語" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-blue"></a>
<a href="README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-blue"></a>
<a href="README_vi.md"><img alt="Tiếng Việt" src="https://img.shields.io/badge/Ti%E1%BA%BFng_Vi%E1%BB%87t-blue"></a>
<a href="README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Fran%C3%A7ais-blue"></a>
<a href="README_ru.md"><img alt="Русский" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue"></a>
</p>

Built on the [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) encoder, on top of which we attach a **narrow International Phonetic Alphabet (IPA) CTC decoding head**. Through full fine-tuning, it performs Mandarin "syllable + tone" level speech transcription.

The model outputs space-separated narrow IPA syllable sequences, each syllable carrying its own tone-value symbol (e.g. `ɡ̊wa̠n̚˥`, `x̞wa̠ɪ̯˧˥`) — i.e. giving both the initial/final and the tone at once. Suitable for phonetic analysis, Mandarin pronunciation assessment, tone teaching, and similar scenarios.

> License: **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International).
> Model weights and training data are not shipped with the code repository; they are obtained separately via ModelScope and a local data-build pipeline, respectively. See below.


## Model Pipeline

![pipeline](assets/pipeline.svg)

<p align="center">Figure: end-to-end pipeline from audio to narrow IPA syllable sequence.</p>

---

## Features

- **Narrow IPA output**: adopts the UntPhesoca narrow scheme from [nk2028/putonghua-ipa-converter](https://github.com/nk2028/putonghua-ipa-converter), labeling at the phoneme level while preserving tone values.
- **Full fine-tuning**: unfreezes the SenseVoiceSmall Conformer encoder and trains it together with the newly built IPA head; the feature frontend is frozen and `dither=0` guarantees reproducibility.
- **Label-agnostic tone metric**: tone accuracy is computed via an `ipa2tone` mapping table (IPA syllable → tone class 1–5), independent of the specific phoneme notation, facilitating fair comparison.
- **Large-scale validation**: measured on a 5918-sample merged validation set (see [Evaluation metrics](#evaluation-metrics-large-scale-validation)).
- **Gradio demo**: supports uploading/recording audio for recognition, and automatically compares differences against a reference text (syllable-level tone alignment with highlighting).

---

## Directory structure

```
mandarin-ipa-asr/
├── LICENSE                                  # Full text of CC BY-NC-SA 4.0
├── README.md
├── requirements.txt                         # Verified dependency versions
├── .gitignore
├── app.py                                   # Gradio demo (recognition + auto diff)
├── configs/
│   └── example_train_config.json            # Example training config
├── vocab/
│   ├── vocab_mandarin_ipa_combined.json     # IPA output vocabulary (1427 classes, openly released)
│   └── vocab_mandarin_ipa_tone_combined.json# IPA syllable -> tone class 1-5 mapping (openly released)
├── src/
│   ├── model.py        # SenseVoiceIpa: encoder + IPA CTC head
│   ├── utils.py        # CTC decode / metrics / alignment / beam search / audio loading
│   ├── dataset.py      # Audio-text dataset (wav/flac/ogg/mp3)
│   ├── train.py        # Full fine-tuning training
│   ├── infer.py        # Single-sample inference + batch evaluation (large-scale validation)
│   └── prepare_data.py # Build IPA training set from pinyin text (preprocessing, no audio)
└── results/
    └── metrics.md       # Measured validation metrics and training curves
```

---

## Base model

| Item | Description |
| --- | --- |
| Name | SenseVoiceSmall (FunAudioLLM / Alibaba DAMO Academy) |
| Source | GitHub: [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice); ModelScope: `iic/SenseVoiceSmall` |
| Structure | `WavFrontend` (fbank features + optional f0) + Conformer encoder (512-dim) + original Chinese-character CTC head |
| This project's modification | Frontend frozen; a new **IPA CTC head** is mounted on top of its Conformer encoder (see below); the original Chinese-character head is no longer used |

> The base model weights are automatically downloaded and cached by ModelScope on first run; no manual preparation needed.

---

## Training method

1. **Initialization**: load the pretrained SenseVoiceSmall `WavFrontend` and `encoder` (original Chinese-character CTC head discarded), then create a new IPA CTC head `ctc_head = Linear(512,512) → ReLU → Dropout(0.1) → Linear(512, vocab)`.
2. **Freezing strategy**: `WavFrontend` is always frozen, `dither` fixed at 0 (disables random noise injection, ensuring deterministic features); the encoder and IPA head are **unfrozen and fully fine-tuned** (training only the head is also supported).
3. **Loss and optimization**: `CTCLoss(blank=0)`; AdamW (`lr=1e-4`, `weight_decay=1e-4`); `CosineAnnealingLR` (`T_max = steps × epochs`); gradient clipping 1.0.
4. **Precision**: bf16 mixed precision (`autocast` applied only to the encoder forward pass; features and loss stay fp32).
5. **Weight saving**: `best.pt` (lowest validation TER), `best_tone.pt` (highest tone accuracy).
6. **Legacy-weight compatibility**: at inference / resume-training, residual old-weight head naming is automatically remapped to `ctc_head.*` to avoid silently dropping the head.

---

## Training data

| Item | Quantity |
| --- | --- |
| Training set | **11,691** samples (Mandarin read speech; derived from the ~8,993-sample Mandarin subset of zhvoice at 1.3× reuse/augmentation) |
| Validation set | **918** samples (Mandarin read speech, disjoint from the training set) |
| Labels | Each sample is a space-separated narrow IPA syllable sequence (converted from pinyin via nk2028/putonghua-ipa-converter) |
| IPA vocabulary | 1,427 classes (including `<blank>`/`<unk>`) |

> The raw zhvoice corpus is ~**900 hours, 3200+ speakers, ~1,129,800 text entries**;
> this project uses only its higher-quality Mandarin read-speech subset to build the training/validation sets, converted to narrow IPA.

---

## Model parameters

| Parameter | Value |
| --- | --- |
| Total parameters | **234,993,874** |
| Trainable parameters (full fine-tuning) | **222,131,699** (encoder + IPA head; frontend and original Chinese head frozen) |
| Inference-only (head trainable, encoder frozen) | 994,707 |
| Encoder dimension | 512 |
| IPA output vocabulary | 1,427 |
| Tone mapping entries (ipa2tone) | 1,425 (excluding `<blank>`/`<unk>`) |

---

## Dataset sources and links

| Data / Tool | Purpose | License | Link |
| --- | --- | --- | --- |
| **zhvoice** | Training corpus (Mandarin read-speech subset) | See repo | https://github.com/fighting41love/zhvoice |
| **putonghua-ipa-converter** | Pinyin → narrow IPA conversion (scheme 2, UntPhesoca) | CC0 | https://github.com/nk2028/putonghua-ipa-converter |
| **SenseVoiceSmall** | Pretrained base model | See repo | https://github.com/FunAudioLLM/SenseVoice (ModelScope `iic/SenseVoiceSmall`) |

---

## Evaluation metrics (large-scale validation)

Decoding is fully consistent with training (raw waveform input, dither=0, bf16 encoder forward). Metric definitions are in [`results/metrics.md`](results/metrics.md).

### 918-sample first-class (Yijia) Mandarin validation corpus

| Metric | Value |
| --- | --- |
| TER (syllable error rate, lower is better) | **0.0955** |
| ACC (exact full-sentence match) | 0.7059 |
| TACC (token accuracy) | 0.9145 |
| Tone accuracy (label-agnostic) | **0.9212** |

### 5918-sample merged validation set (large-scale validation)

= 5000 zhvoice real-scene mp3 + 918 Mandarin read-speech samples. zhvoice is real-scene, more diverse in domain, and noisier, so its full-sentence ACC being lower than the Yijia Mandarin validation corpus is expected; tone recognition still remains at a high level.

| Metric | Value |
| --- | --- |
| TER (syllable error rate, lower is better) | **0.1089** |
| ACC (exact full-sentence match) | 0.4439 |
| TACC (token accuracy) | 0.8986 |
| Tone accuracy (label-agnostic) | **0.8915** |

> Note: tone accuracy is comparable to similar models under a pinyin scheme (under the same framework the pinyin-head tone is ~0.92, this IPA head 0.89–0.92), demonstrating that under joint "phoneme + tone" annotation, tone information is not lost.

---

## Installation and environment (all verified)

| Dependency | Version | Notes |
| --- | --- | --- |
| Python | **3.9.13** | 3.9 recommended (3.8–3.11 should work; 3.9.13 verified) |
| CUDA | **12.8** | Training/inference needs NVIDIA GPU; pure CPU inference possible but slow |
| torch | 2.7.0+cu128 | matches CUDA 12.8 |
| torchaudio | 2.7.0+cu128 | |
| funasr | 1.4.16 | loads SenseVoiceSmall |
| modelscope | 1.32.0 | auto-downloads base model |
| transformers | 4.43.0 | |
| numpy | 1.23.4 | |
| editdistance | 0.6.2 | metric computation |
| tqdm | 4.64.1 | progress bar |
| soundfile | 0.12.1 | audio read/write |
| gradio | 4.24.0 | demo UI |
| librosa | 0.9.2 | (optional) data preprocessing |
| ffmpeg | 2025-08-23 | decode mp3 (must be on PATH) |

Installation:

```bash
pip install -r requirements.txt
# ffmpeg must be installed separately and added to PATH (Windows: https://www.gyan.dev/ffmpeg/ or scoop/apt)
```

**Device requirements**: training recommends ≥ 16 GB VRAM (234M params full fine-tuning + activations under bf16); inference needs only a few GB, and a single audio sample can run on CPU (slower).

---

## Quick start

> All scripts resolve `vocab/`, `data/`, `weights/`, `checkpoints/` and other relative paths relative to the "project root" (i.e. the repo root), so **you do NOT need to `cd` into the project root to launch from any working directory**; passing an absolute path uses it as-is.
> The base model `iic/SenseVoiceSmall` is a ModelScope model id and will be auto-downloaded and cached on first run.

> ### 【Important】You must load the fine-tuned weights first, otherwise the output is meaningless gibberish
>
> This project **does not put model weights in the repository** (released separately on ModelScope). If you start the demo or inference **without specifying weights via `--ckpt`**,
> the program will silently use a **randomly initialized** model — the IPA it emits for audio will be meaningless and often shows repeated syllables,
> looking like "a lot got recognized", but it is not the real pronunciation at all.
> For example, the first half of *Jingyesi* (静夜思) has only about 10 syllables, but without weights it may output 30+ repeated gibberish syllables.
>
> Correct usage (weights must first be downloaded to `weights/best.pt`, see next section):
> ```bash
> python app.py --ckpt weights/best.pt
> ```
> If `--ckpt` is not given, `app.py` will automatically try to locate `weights/best.pt`; if both are absent, startup prints a prominent 【warning】.

### 1. Obtain model weights

Model weights **are not in this repository**; download them from ModelScope (released separately by the author), e.g.:

```bash
# Assuming the model is published, use modelscope to download locally
modelscope download --model QiGuanFuChen/mandarin-ipa-asr --local_dir weights/
```

After obtaining `weights/best.pt`, specify it via `--ckpt`. (If you train your own, an additional `checkpoints/best_tone.pt` is produced; see "Training guide" below)

### 2. Single-audio inference

```bash
# No need to cd into project root: the script resolves vocab/data relative paths by its own location; launchable from any directory
python src/infer.py --wav path/to/audio.wav --ckpt weights/best.pt
# Output: space-separated narrow IPA syllable sequence
```

### 3. Batch evaluation (validation metrics)

```bash
python src/infer.py --eval --ckpt weights/best.pt \
    --val_scp data/val.scp --val_text data/val.text --limit 0
# Outputs TER / ACC / TACC / tone accuracy
```

### 4. Gradio demo (upload audio + auto diff)

> Be sure to load weights via `--ckpt` (see the 【Important】 note above), otherwise the recognition result is meaningless random-init gibberish.
> The "Recognize" tab uses beam search by default (`--beam`, default 12) to reduce insertion/repetition errors; `--beam 0` falls back to greedy decoding.

```bash
# Launchable from any directory (relative paths resolved against project root); --ckpt accepts relative or absolute paths
# If --ckpt is omitted, weights/best.pt is auto-tried
python app.py --ckpt weights/best.pt --port 7860
# Open http://127.0.0.1:7860 in a browser
```

- **Recognize** tab: upload or record audio → outputs narrow IPA.
- **Compare (auto diff)** tab: upload audio and (optionally) fill in reference IPA text →
  automatically performs syllable-level tone alignment and highlights differences (correct / wrong tone / misread / extra / missing);
  even without reference text, it auto-compares the two decoding results "greedy vs beam search" for differences.

---

## Training guide

1. Prepare data (see "Data usage notes" below) to get `train.scp` / `train.text` / `val.scp` / `val.text`
   (format: `uid audio_path` and `uid space-separated IPA syllables`).
2. Run:

```bash
python src/train.py \
    --train_scp data/train.scp --train_text data/train.text \
    --val_scp data/val.scp --val_text data/val.text \
    --vocab_path vocab/vocab_mandarin_ipa_combined.json \
    --ipa2tone_path vocab/vocab_mandarin_ipa_tone_combined.json \
    --output_dir checkpoints --epochs 10 --batch_size 16
```

You can also write parameters into `configs/example_train_config.json` and then simply run:

```bash
python src/train.py $(python -c "import json,sys; c=json.load(open('configs/example_train_config.json')); print(' '.join(f'--{k} {v}' for k,v in c.items()))")
```

- To warm-start from a pinyin full-fine-tuning weight: `--warm_start weights/pinyin_ft.pt` (different head dimension is auto-skipped and randomly initialized).
- Outputs `checkpoints/best.pt` (lowest TER) and `checkpoints/best_tone.pt` (highest tone).

---

## Data usage notes (data not public)

For licensing and size reasons, **training data is not released directly**. You can rebuild an equivalent dataset by following these steps:

1. Download the **zhvoice** corpus (https://github.com/fighting41love/zhvoice), unzip to get audio and pinyin text.
2. Clone **putonghua-ipa-converter** (https://github.com/nk2028/putonghua-ipa-converter), use its `data/putonghua.js` (scheme 2 = UntPhesoca narrow) to convert pinyin to narrow IPA.
3. Use this repo's `src/prepare_data.py` to generate the files needed for training:

```bash
# 1) zhvoice subset -> narrow IPA text + scp + vocabulary
#    --conv_js points to the converter's data/putonghua.js; --audio_root is the audio root dir
python src/prepare_data.py build-zhvoice \
    --metadata zhvoice/metadata.csv \
    --audio_root zhvoice/wavs \
    --conv_js path/to/putonghua-ipa-converter/data/putonghua.js \
    --out data/zhvoice_ipa --train_n 8993 --val_n 918

# 2) Convert your own tone-marked pinyin text to IPA (dir must contain train/text, val/text)
python src/prepare_data.py build-mandarin \
    --text_dir data/mandarin_pinyin \
    --conv_js path/to/putonghua-ipa-converter/data/putonghua.js \
    --out data/zhvoice_ipa

# 3) Merge multiple sources and expand vocabulary, output *_combined files
python src/prepare_data.py combine \
    --zhvoice_dir data/zhvoice_ipa --mandarin_ipa_dir data/zhvoice_ipa \
    --out data/combined

# 4) Expand training set by a factor (e.g. 1.3×): base on the Mandarin subset, mix in zhvoice data
python src/prepare_data.py scale \
    --mandarin_text data/combined/train_text --mandarin_scp data/combined/train_scp \
    --zhvoice_text data/zhvoice_ipa/train_text --zhvoice_scp data/zhvoice_ipa/train_scp \
    --factor 1.3 --out data/ipa130
```

The generated `*.scp` (audio paths) and `*.text` (IPA labels) can be used as input to `train.py` / `infer.py`.
**Do not publish the raw audio or third-party text with this repository**; only publish the scripts and usage notes above.

---

## License

Code and vocabulary are released under **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International; see [LICENSE](LICENSE)).
- **Attribution (BY)**: please retain the original author and project origin when using.
- **NonCommercial (NC)**: may not be used for commercial purposes.
- **ShareAlike (SA)**: derivative works must be released under the same license.

Model weights and training data are provided separately under their respective source licenses (ModelScope / data-source repos) and do not override this repository's license.

---

## Citation and acknowledgements

- Base model: FunAudioLLM, *SenseVoice*.
- Narrow IPA conversion: nk2028, *putonghua-ipa-converter* (CC0).
- Training corpus: fighting41love, *zhvoice*.

---

<p align="center">
<a href="README.md"><img alt="%E4%B8%AD%E6%96%87" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-blue"></a>
<a href="README_en.md"><img alt="English" src="https://img.shields.io/badge/English-brightgreen"></a>
<a href="README_ja.md"><img alt="%E6%97%A5%E6%9C%AC%E8%AA%9E" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-blue"></a>
<a href="README_ko.md"><img alt="%ED%95%9C%EA%B5%AD%EC%96%B4" src="https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-blue"></a>
<a href="README_vi.md"><img alt="Ti%E1%BA%BFng_Vi%E1%BB%87t" src="https://img.shields.io/badge/Ti%E1%BA%BFng_Vi%E1%BB%87t-blue"></a>
<a href="README_fr.md"><img alt="Fran%C3%A7ais" src="https://img.shields.io/badge/Fran%C3%A7ais-blue"></a>
<a href="README_ru.md"><img alt="%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue"></a>
</p>
