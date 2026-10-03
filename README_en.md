# Mandarin Chinese and dialects strict IPA speech recognition
# 汉语普通话与方言严式IPA语音识别

[![ModelScope](https://img.shields.io/badge/ModelScope-blue)](https://www.modelscope.cn/models/QiGuanFuChen/mandarin-ipa-asr)

<p align="center">
<a href="README.md"><img alt="中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-brightgreen"></a>
<a href="README_en.md"><img alt="English" src="https://img.shields.io/badge/English-blue"></a>
<a href="README_ja.md"><img alt="日本語" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-blue"></a>
<a href="README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-blue"></a>
<a href="README_vi.md"><img alt="Tiếng Việt" src="https://img.shields.io/badge/Ti%E1%BA%BFng_Vi%E1%BB%87t-blue"></a>
<a href="README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Fran%C3%A7ais-blue"></a>
<a href="README_ru.md"><img alt="Русский" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue"></a>
</p>

A strict-IPA (International Phonetic Alphabet) speech recognition project built on the
[SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) encoder with a **strict-IPA CTC head**,
supporting **Mandarin, Cantonese, and Sichuanese** at the "syllable + tone" level.

- **Mandarin**: full fine-tuning (frozen frontend, full encoder + IPA-head fine-tuning), weights `weights/best.pt`.
- **Cantonese / Sichuanese**: **LoRA adapters** on top of the frozen SenseVoiceSmall (only LoRA + dialect CTC head
  trained), weights `out_canto/best.pt` / `out_sichuan/best.pt`.

Output is a space-separated sequence of strict-IPA syllables, each carrying its own tone letters
(e.g. `ɡ̊wa̠n̚˥`, `x̞wa̠ɪ̯˧˥`), giving onset/nucleus and tone together — suited to phonetic analysis,
pronunciation assessment, and tone teaching.

> License: **CC BY-NC-SA 4.0**. Model weights and training data are not shipped with the code repo; see below.


## Three-language metrics (measured)

Decoded on each language's **own validation set** (greedy), identical to training (raw waveform, `dither=0`,
bf16 encoder). Full definitions and reproduction commands: [results/metrics.md](results/metrics.md).

| Language | Train | Val | Vocab | TER↓ | token acc | tone acc | exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Mandarin** | 11,691 | 918 | 1,427 | **0.0955** | **0.9145** | **0.9212** | 0.7059 |
| **Cantonese** | 8,426 | 1,999 | 1,580 | **0.0881** | 0.8914 | 0.8953 | 0.4612 |
| **Sichuanese** | 5,869 | 653 | 912 | **0.0829** | 0.9187 | 0.9131 | 0.4196 |

> Mandarin also has a 5,918-item merged (large-scale) validation set: TER 0.1089 / token 0.8986 / tone 0.8915 / exact 0.4439.
> All three languages sit at TER 0.083–0.096 with tone accuracy ≥ 0.89, showing tone information is preserved
> under joint "phoneme + tone" labeling.


## Quick start

```bash
# Gradio demo — switch language at launch
python app.py --ckpt weights/best.pt --port 7860            # Mandarin (default)
python app.py --language cantonese --port 7860              # Cantonese (LoRA)
python app.py --language sichuan  --port 7860               # Sichuanese (LoRA)

# Single-file inference
python src/infer.py --wav audio.wav --ckpt weights/best.pt   # Mandarin
bash infer_cantonese_lora.sh "audio.wav"                    # Cantonese
bash infer_sichuan_lora.sh  "audio.wav"                     # Sichuanese

# Batch evaluation
bash infer_cantonese_lora.sh eval
bash infer_sichuan_lora.sh eval
```

Weights are not in the repo. Download the Mandarin model from ModelScope:

```bash
modelscope download --model QiGuanFuChen/mandarin-ipa-asr --local_dir weights/
```

See [README.md](README.md) (Chinese) for the full guide: training methods, data preparation, environment,
and license. Other language versions (ja/ko/vi/fr/ru) carry the same project name and pointer.


## License

Code and vocab are released under **CC BY-NC-SA 4.0** (see [LICENSE](LICENSE)).
- **BY**: retain authorship and project origin.
- **NC**: not for commercial use.
- **SA**: derivatives must use the same license.

Model weights and training data follow their respective source licenses and are not covered by this repo's license.


## Credits

- Base model: FunAudioLLM, *SenseVoice*.
- Mandarin strict IPA: nk2028, *putonghua-ipa-converter* (CC0).
- Dialect IPA: diamoe dialect IPA scheme.
- Training corpora: fighting41love, *zhvoice*; plus Cantonese / Sichuanese speech corpora.


<p align="center">
<a href="README.md"><img alt="%E4%B8%AD%E6%96%87" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-brightgreen"></a>
<a href="README_en.md"><img alt="English" src="https://img.shields.io/badge/English-blue"></a>
<a href="README_ja.md"><img alt="日本語" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-blue"></a>
<a href="README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-blue"></a>
<a href="README_vi.md"><img alt="Tiếng Việt" src="https://img.shields.io/badge/Ti%E1%BA%BFng_Vi%E1%BB%87t-blue"></a>
<a href="README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Fran%C3%A7ais-blue"></a>
<a href="README_ru.md"><img alt="Русский" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue"></a>
</p>
