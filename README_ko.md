# 표준어 내로(정밀) 국제음성기호(IPA) 음성 인식

[![ModelScope](https://img.shields.io/badge/ModelScope-魔搭-blue)](https://www.modelscope.cn/models/QiGuanFuChen/mandarin-ipa-asr)

<p align="center">
<a href="README.md"><img alt="中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-blue"></a>
<a href="README_en.md"><img alt="English" src="https://img.shields.io/badge/English-blue"></a>
<a href="README_ja.md"><img alt="日本語" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-blue"></a>
<a href="README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-brightgreen"></a>
<a href="README_vi.md"><img alt="Tiếng Việt" src="https://img.shields.io/badge/Ti%E1%BA%BFng_Vi%E1%BB%87t-blue"></a>
<a href="README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Fran%C3%A7ais-blue"></a>
<a href="README_ru.md"><img alt="Русский" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue"></a>
</p>

[SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) 인코더를 기반으로, 그 위에 **내로(정밀) 국제음성기호(IPA) CTC 디코딩 헤드**를 부착하고 전체 미세조정(full fine-tuning)을 통해 표준어의 “음절 + 성조” 수준 음성 전사를 수행합니다.

모델은 공백으로 구분된 내로 IPA 음절 시퀀스를 출력하며, 각 음절에는 고유한 성조값 기호(예 `ɡ̊wa̠n̚˥`, `x̞wa̠ɪ̯˧˥`)가 붙습니다. 즉 초성/운모와 성조가 동시에 주어지므로 음성학 분석, 표준어 발음 평가, 성조 교육 등에 적합합니다.

> 라이선스: **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International).
> 모델 가중치와 학습 데이터는 코드 저장소에 포함되지 않으며, 각각 ModelScope와 로컬 데이터 구축 절차를 통해 별도로 얻습니다. 자세한 내용은 아래를 참조하세요.

---

## 특징

- **내로 IPA 출력**: [nk2028/putonghua-ipa-converter](https://github.com/nk2028/putonghua-ipa-converter)의 UntPhesoca 내로 체계를 채택하여 음소 수준으로 표기하면서 성조값을 보존합니다.
- **전체 미세조정**: SenseVoiceSmall의 Conformer 인코더를 해제(unfreeze)하고 새로 만든 IPA 헤드와 함께 학습합니다. 특징 프론트엔드는 동결(freeze)하며, `dither=0`으로 재현성을 보장합니다.
- **라벨 무관 성조 지표**: `ipa2tone` 매핑 테이블(IPA 음절 → 성조류 1–5)을 통해 성조 정확도를 계산하므로 구체적인 음소 표기와 무관하게 공정하게 비교할 수 있습니다.
- **대규모 검증**: 5918개 병합 검증 세트에서 실측했습니다(아래 [평가 지표](README.md#效果指标大规模验证) 참조).
- **Gradio 데모**: 오디오 업로드/녹음 인식과 참조 텍스트와의 차이 자동 비교(음절 수준 성조 정렬 하이라이트)를 지원합니다.

---

## 디렉터리 구조

```
mandarin-ipa-asr/
├── LICENSE                                  # CC BY-NC-SA 4.0 전문
├── README.md
├── requirements.txt                         # 실측 검증된 의존성 버전
├── .gitignore
├── app.py                                   # Gradio 데모 (인식 + 차이 자동 비교)
├── configs/
│   └── example_train_config.json            # 학습 설정 예시
├── vocab/
│   ├── vocab_mandarin_ipa_combined.json     # IPA 출력 어휘 (1427류, 공개)
│   └── vocab_mandarin_ipa_tone_combined.json# IPA 음절 → 성조류 1-5 매핑 (공개)
├── src/
│   ├── model.py        # SenseVoiceIpa: 인코더 + IPA CTC 헤드
│   ├── utils.py        # CTC 디코드 / 지표 / 정렬 / 빔 탐색 / 오디오 로드
│   ├── dataset.py      # 오디오-텍스트 데이터셋 (wav/flac/ogg/mp3)
│   ├── train.py        # 전체 미세조정 학습
│   ├── infer.py        # 단일 추론 + 배치 평가 (대규모 검증)
│   └── prepare_data.py # 병음(pinyin) 텍스트에서 IPA 학습셋 구축 (전처리, 오디오 없음)
└── results/
    └── metrics.md       # 실측 검증 지표와 학습 곡선
```

---

## 기본 모델

| 항목 | 설명 |
| --- | --- |
| 이름 | SenseVoiceSmall (FunAudioLLM / 알리바바 DAMO 아카데미) |
| 출처 | GitHub: [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice); ModelScope: `iic/SenseVoiceSmall` |
| 구조 | `WavFrontend` (fbank 특징 + 선택적 f0) + Conformer 인코더 (512차원) + 원래 한자 CTC 헤드 |
| 본 프로젝트의 변경 | 프론트엔드 동결; 그 Conformer 인코더 위에 새 **IPA CTC 헤드**를 부착(아래 참조); 원래 한자 헤드는 사용하지 않음 |

> 기본 모델 가중치는 첫 실행 시 ModelScope에서 자동 다운로드되어 캐시되므로 수동 준비 불필요.

---

## 학습 방법

1. **초기화**: 사전학습된 SenseVoiceSmall의 `WavFrontend`와 `encoder`를 불러오고(원래 한자 CTC 헤드는 폐기), 새 IPA CTC 헤드 `ctc_head = Linear(512,512) → ReLU → Dropout(0.1) → Linear(512, vocab)`를 생성합니다.
2. **동결 전략**: `WavFrontend`는 항상 동결, `dither`는 0으로 고정(무작위 노이즈 주입 비활성화로 특징의 결정성 보장); 인코더와 IPA 헤드는 **해제되어 전체 미세조정**(헤드만 학습하는 것도 지원).
3. **손실과 최적화**: `CTCLoss(blank=0)`; AdamW (`lr=1e-4`, `weight_decay=1e-4`); `CosineAnnealingLR` (`T_max = steps × epochs`); 그래디언트 클리핑 1.0.
4. **정밀도**: bf16 혼합 정밀도(`autocast`는 인코더 순방향에만 적용, 특징과 손실은 fp32 유지).
5. **가중치 저장**: `best.pt` (검증 TER 최저), `best_tone.pt` (성조 정확도 최고).
6. **구버전 가중치 호환**: 추론/이어학습 시 구버전 가중치에 남은 헤드 명명을 자동으로 `ctc_head.*`로 재매핑하여 헤드가 조용히 누락되는 것을 방지.

---

## 학습 데이터

| 항목 | 수량 |
| --- | --- |
| 학습 세트 | **11,691**개 (표준어 낭독; zhvoice 표준어 하위집합 약 8,993개를 1.3× 재사용/증강하여 구성) |
| 검증 세트 | **918**개 (표준어 낭독, 학습 세트와 배타적) |
| 라벨 | 각 항목은 공백 구분 내로 IPA 음절 시퀀스 (병음을 nk2028/putonghua-ipa-converter로 변환) |
| IPA 어휘 | 1,427류 (`<blank>`/`<unk>` 포함) |

> 원본 zhvoice 코퍼스는 약 **900시간, 3200+ 화자, 약 112.98만 텍스트 항목**;
> 본 프로젝트는 그중 낭독 품질이 높은 표준어 하위집합만 사용하여 학습/검증 세트를 구성하고 내로 IPA로 변환.

---

## 모델 파라미터

| 파라미터 | 값 |
| --- | --- |
| 총 파라미터 | **234,993,874** |
| 전체 미세조정 학습 가능 파라미터 | **222,131,699** (인코더 + IPA 헤드; 프론트엔드와 원래 한자 헤드 동결) |
| 추론만 (헤드 학습 가능, 인코더 동결) | 994,707 |
| 인코더 차원 | 512 |
| IPA 출력 어휘 | 1,427 |
| 성조 매핑 항목 수 (ipa2tone) | 1,425 (`<blank>`/`<unk>` 제외) |

---

## 데이터셋 출처 및 링크

| 데이터 / 도구 | 용도 | 라이선스 | 링크 |
| --- | --- | --- | --- |
| **zhvoice** | 학습 코퍼스 (표준어 낭독 하위집합) | 저장소 참조 | https://github.com/fighting41love/zhvoice |
| **putonghua-ipa-converter** | 병음 → 내로 IPA 변환 (scheme 2, UntPhesoca) | CC0 | https://github.com/nk2028/putonghua-ipa-converter |
| **SenseVoiceSmall** | 사전학습 기본 모델 | 저장소 참조 | https://github.com/FunAudioLLM/SenseVoice (ModelScope `iic/SenseVoiceSmall`) |

---

## 평가 지표 (대규모 검증)

디코딩은 학습과 완전히 일치 (생 파형 입력, dither=0, bf16 인코더 순방향). 지표 정의는 [`results/metrics.md`](results/metrics.md) 참조.

### 918개 1급(일갑) 표준어 검증 코퍼스

| 지표 | 값 |
| --- | --- |
| TER (음절 오류율, 낮을수록 좋음) | **0.0955** |
| ACC (문장 전체 정확 일치) | 0.7059 |
| TACC (토큰 정확도) | 0.9145 |
| 성조 정확도 (라벨 무관) | **0.9212** |

### 5918개 병합 검증 세트 (대규모 검증)

= 5000개 zhvoice 실제 환경 mp3 + 918개 표준어 낭독. zhvoice는 실제 환경이며 도메인이 더 다양하고 잡음이 많으므로 문장 전체 ACC가 일갑 표준어 검증 코퍼스보다 낮은 것은 예상된 것임; 성조 인식은 여전히 높은 수준 유지.

| 지표 | 값 |
| --- | --- |
| TER (음절 오류율, 낮을수록 좋음) | **0.1089** |
| ACC (문장 전체 정확 일치) | 0.4439 |
| TACC (토큰 정확도) | 0.8986 |
| 성조 정확도 (라벨 무관) | **0.8915** |

> 참고: 성조 정확도는 병음 체계의 유사 모델과 대등함(동일 프레임워크에서 병음 헤드 성조 약 0.92, 본 IPA 헤드 0.89–0.92). “음소 + 성조” 공동 표기에서도 성조 정보가 손실되지 않았음을 입증.

---

## 설치 및 환경 (모두 실측)

| 의존성 | 버전 | 설명 |
| --- | --- | --- |
| Python | **3.9.13** | 3.9 권장 (3.8–3.11 작동 가능, 3.9.13 실측) |
| CUDA | **12.8** | 학습/추론에 NVIDIA GPU 필요; CPU 단독 추론 가능하나 느림 |
| torch | 2.7.0+cu128 | CUDA 12.8 대응 |
| torchaudio | 2.7.0+cu128 | |
| funasr | 1.4.16 | SenseVoiceSmall 로드 |
| modelscope | 1.32.0 | 기본 모델 자동 다운로드 |
| transformers | 4.43.0 | |
| numpy | 1.23.4 | |
| editdistance | 0.6.2 | 지표 계산 |
| tqdm | 4.64.1 | 진행률 표시 |
| soundfile | 0.12.1 | 오디오 읽기/쓰기 |
| gradio | 4.24.0 | 데모 UI |
| librosa | 0.9.2 | (선택) 데이터 전처리 |
| ffmpeg | 2025-08-23 | mp3 디코딩 (PATH에 필요) |

설치:

```bash
pip install -r requirements.txt
# ffmpeg은 별도 설치 후 PATH에 추가 (Windows: https://www.gyan.dev/ffmpeg/ 또는 scoop/apt)
```

**장치 요구사항**: 학습은 ≥ 16 GB VRAM 권장 (bf16에서 234M 파라미터 전체 미세조정 + 활성화); 추론은 수 GB면 충분하며 단일 오디오는 CPU에서도 가능(느림).

---

## 빠른 시작

> 모든 스크립트는 `vocab/`, `data/`, `weights/`, `checkpoints/` 등 상대 경로를 “프로젝트 루트”(저장소 루트) 기준으로 해석하므로 **프로젝트 루트로 `cd`하지 않아도 어떤 작업 디렉터리에서나 실행 가능**; 절대 경로를 주면 그대로 사용. 기본 모델 `iic/SenseVoiceSmall`은 ModelScope 모델 id이며 첫 실행 시 자동 다운로드되어 캐시됨.

> ### 【중요】미세조정 가중치를 먼저 로드하지 않으면 출력이 무의미한 난잡물(gibberish)이 됩니다
>
> 본 프로젝트는 **모델 가중치를 저장소에 넣지 않음**(ModelScope에 별도 공개). 데모나 추론을 시작할 때 **`--ckpt`로 가중치를 지정하지 않으면**,
> 프로그램이 **무작위 초기화**된 모델을 조용히 사용함 — 오디오에 대해 출력되는 IPA는 무의미하며 종종 음절이 반복됨.
> “많이 인식된” 것처럼 보이지만 실제 발음이 아님. 예: 《정야사(静夜思)》 전반은 약 10음절이지만 가중치 없으면 30개 이상의 반복된 난잡 음절을 출력할 수 있음.
>
> 올바른 사용법 (가중치를 먼저 `weights/best.pt`로 다운로드, 다음 절 참조):
> ```bash
> python app.py --ckpt weights/best.pt
> ```
> `--ckpt`를 지정하지 않으면 `app.py`가 `weights/best.pt` 자동 탐지를 시도; 둘 다 없으면 시작 시 눈에 띄는 [경고] 출력.

### 1. 모델 가중치 얻기

모델 가중치는 **이 저장소에 없음**; ModelScope에서 다운로드 (저자가 별도 공개), 예:

```bash
# 모델이 발행되었다고 가정, modelscope로 로컬 다운로드
modelscope download --model QiGuanFuChen/mandarin-ipa-asr --local_dir weights/
```

`weights/best.pt` 확보 후 `--ckpt`로 지정. (직접 학습하면 추가로 `checkpoints/best_tone.pt`가 생성됨; 아래 “학습 가이드” 참조)

### 2. 단일 오디오 추론

```bash
# 프로젝트 루트로 cd 불필요: 스크립트는 자기 위치 기준으로 vocab/data 상대경로 해석; 어떤 디렉터리에서나 실행 가능
python src/infer.py --wav path/to/audio.wav --ckpt weights/best.pt
# 출력: 공백 구분 내로 IPA 음절 시퀀스
```

### 3. 배치 평가 (검증 지표)

```bash
python src/infer.py --eval --ckpt weights/best.pt \
    --val_scp data/val.scp --val_text data/val.text --limit 0
# TER / ACC / TACC / 성조 정확도 출력
```

### 4. Gradio 데모 (오디오 업로드 + 차이 자동 비교)

> 반드시 `--ckpt`로 가중치 로드 (위 [중요] 참조), 그렇지 않으면 인식 결과가 무작위 초기화 무의미 난잡물.
> “인식” 탭은 기본적으로 빔 탐색(`--beam`, 기본 12) 사용하여 삽입/반복 오류 감소; `--beam 0`은 탐욕(greedy) 디코딩으로 복귀.

```bash
# 어떤 디렉터리에서나 실행 가능 (상대경로는 프로젝트 루트 기준); --ckpt는 상대 또는 절대 경로 가능
# --ckpt 생략 시 weights/best.pt 자동 시도
python app.py --ckpt weights/best.pt --port 7860
# 브라우저에서 http://127.0.0.1:7860 열기
```

- **인식** 탭: 오디오 업로드 또는 녹음 → 내로 IPA 출력.
- **비교 (자동 차이)** 탭: 오디오 업로드 및 (선택적) 참조 IPA 텍스트 입력 →
  음절 수준 성조 정렬 자동 수행 및 차이 하이라이트(정답/성조 오류/오독/과다/누락);
  참조 텍스트가 없어도 “탐욕 vs 빔 탐색” 두 디코딩 결과를 자동 비교.

---

## 학습 가이드

1. 데이터 준비 (아래 “데이터 사용 안내” 참조)하여 `train.scp` / `train.text` / `val.scp` / `val.text` 확보
   (형식: `uid 오디오경로` 및 `uid 공백구분IPA음절`).
2. 실행:

```bash
python src/train.py \
    --train_scp data/train.scp --train_text data/train.text \
    --val_scp data/val.scp --val_text data/val.text \
    --vocab_path vocab/vocab_mandarin_ipa_combined.json \
    --ipa2tone_path vocab/vocab_mandarin_ipa_tone_combined.json \
    --output_dir checkpoints --epochs 10 --batch_size 16
```

매개변수를 `configs/example_train_config.json`에 작성 후 단순 실행도 가능:

```bash
python src/train.py $(python -c "import json,sys; c=json.load(open('configs/example_train_config.json')); print(' '.join(f'--{k} {v}' for k,v in c.items()))")
```

- 병음 전체 미세조정 가중치에서 웜스타트: `--warm_start weights/pinyin_ft.pt` (헤드 차원 다르면 자동 건너뛰고 무작위 초기화).
- `checkpoints/best.pt` (TER 최저) 및 `checkpoints/best_tone.pt` (성조 최고) 출력.

---

## 데이터 사용 안내 (데이터 비공개)

라이선스와 규모상 **학습 데이터는 직접 공개하지 않음**. 다음 절차로 동등한 데이터셋 재구성 가능:

1. **zhvoice** 코퍼스 (https://github.com/fighting41love/zhvoice) 다운로드 후 해제하여 오디오와 병음 텍스트 확보.
2. **putonghua-ipa-converter** (https://github.com/nk2028/putonghua-ipa-converter) 클론, 그 `data/putonghua.js` (scheme 2 = UntPhesoca 내로)로 병음을 내로 IPA로 변환.
3. 본 저장소 `src/prepare_data.py`로 학습용 파일 생성:

```bash
# 1) zhvoice 하위집합 -> 내로 IPA 텍스트 + scp + 어휘
#    --conv_js는 변환기의 data/putonghua.js를 가리킴; --audio_root는 오디오 루트
python src/prepare_data.py build-zhvoice \
    --metadata zhvoice/metadata.csv \
    --audio_root zhvoice/wavs \
    --conv_js path/to/putonghua-ipa-converter/data/putonghua.js \
    --out data/zhvoice_ipa --train_n 8993 --val_n 918

# 2) 자신의 성조 표기 병음 텍스트를 IPA로 변환 (디렉터리에 train/text, val/text 필요)
python src/prepare_data.py build-mandarin \
    --text_dir data/mandarin_pinyin \
    --conv_js path/to/putonghua-ipa-converter/data/putonghua.js \
    --out data/zhvoice_ipa

# 3) 여러 소스 병합 및 어휘 확장, *_combined 파일 출력
python src/prepare_data.py combine \
    --zhvoice_dir data/zhvoice_ipa --mandarin_ipa_dir data/zhvoice_ipa \
    --out data/combined

# 4) 배수로 학습세트 확장 (예 1.3×): 표준어 하위집합을 기준으로 zhvoice 데이터 혼합
python src/prepare_data.py scale \
    --mandarin_text data/combined/train_text --mandarin_scp data/combined/train_scp \
    --zhvoice_text data/zhvoice_ipa/train_text --zhvoice_scp data/zhvoice_ipa/train_scp \
    --factor 1.3 --out data/ipa130
```

생성된 `*.scp` (오디오 경로)와 `*.text` (IPA 라벨)를 `train.py` / `infer.py` 입력으로 사용.
**원본 오디오나 제3자 텍스트를 본 저장소와 함께 게시하지 마세요**; 게시는 위 스크립트와 사용 안내만.

---

## 라이선스

코드와 어휘는 **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International; [LICENSE](LICENSE) 참조)로 공개.
- **저작표시 (BY / Attribution)**: 사용 시 원저자와 프로젝트 출처 유지.
- **비영리 (NC / NonCommercial)**: 상업 목적 사용 불가.
- **동일조건변경허락 (SA / ShareAlike)**: 2차 저작물은 동일 라이선스로 공개.

모델 가중치와 학습 데이터는 각 출처 라이선스(ModelScope/데이터 출처 저장소)로 별도 제공되며 본 저장소 라이선스를 대체하지 않음.

---

## 인용 및 감사

- 기본 모델: FunAudioLLM, *SenseVoice*.
- 내로 IPA 변환: nk2028, *putonghua-ipa-converter* (CC0).
- 학습 코퍼스: fighting41love, *zhvoice*.
