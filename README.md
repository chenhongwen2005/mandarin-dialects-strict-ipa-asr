# 汉语普通话与方言严式IPA语音识别
# Mandarin Chinese and dialects strict IPA speech recognition

[![ModelScope](https://img.shields.io/badge/ModelScope-魔搭-blue)](https://www.modelscope.cn/models/QiGuanFuChen/mandarin-ipa-asr)

<p align="center">
<a href="README.md"><img alt="中文" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-brightgreen"></a>
<a href="README_en.md"><img alt="English" src="https://img.shields.io/badge/English-blue"></a>
<a href="README_ja.md"><img alt="日本語" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-blue"></a>
<a href="README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-blue"></a>
<a href="README_vi.md"><img alt="Tiếng Việt" src="https://img.shields.io/badge/Ti%E1%BA%BFng_Vi%E1%BB%87t-blue"></a>
<a href="README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Fran%C3%A7ais-blue"></a>
<a href="README_ru.md"><img alt="Русский" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue"></a>
</p>

基于 [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) 编码器，挂载**严式国际音标（IPA）CTC 解码头**，
实现**普通话、粤语、四川话**三种语言的「音节 + 声调」级语音转写。

- **普通话**：完整微调（冻结前端，全量微调编码器 + IPA 头），权重 `weights/base.pt`。
- **粤语 / 四川话**：在冻结的 SenseVoiceSmall 上叠加 **LoRA 适配器 + 方言 CTC 头**（仅训 LoRA 与方言头），
  权重分别 `out_canto/cantonese.pt`、`out_sichuan/sichuan.pt`。

模型输出为空格分隔的严式 IPA 音节序列，每个音节自带调值符号（如 `ɡ̊wa̠n̚˥`、`x̞wa̠ɪ̯˧˥），
即同时给出声母 / 韵母与声调，适合语音学分析、发音评测、声调教学等场景。

> 许可证：**CC BY-NC-SA 4.0**（知识共享署名-非商业性使用-相同方式共享 4.0 国际）。
> 模型权重与训练数据不随代码仓库发布，分别通过 ModelScope 与本地数据构建流程获取，详见下文。


## 三语效果指标（实测）

各语言在**各自验证集**上解码（贪心），与训练保持完全一致口径（原始波形、`dither=0`、bf16 编码器前向）。
详细定义与复现命令见 [results/metrics.md](results/metrics.md)。

| 语言 | 训练集 | 验证集 | 词表类数 | TER↓ | token 准确率 | 声调准确率 | 整句完全匹配 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **普通话** | 11,691 | 918 | 1,427 | **0.0955** | **0.9145** | **0.9212** | 0.7059 |
| **粤语** | 8,426 | 1,999 | 1,580 | **0.0881** | 0.8914 | 0.8953 | 0.4612 |
| **四川话** | 5,869 | 653 | 912 | **0.0829** | 0.9187 | 0.9131 | 0.4196 |

> 普通话另有 5918 条合并（大规模）验证集：TER 0.1089 / token 0.8986 / 声调 0.8915 / 整句 0.4439。
> 三语 TER 均处于 0.083–0.096，声调准确率均 ≥ 0.89，证明「音素 + 声调」联合标注下声调信息无丢失。


## 特性

- **三语支持**：普通话（完整微调）+ 粤语 / 四川话（LoRA 适配器），通过 `--language` 一键切换。
- **严式 IPA 输出**：普通话采用 [nk2028/putonghua-ipa-converter](https://github.com/nk2028/putonghua-ipa-converter)
  的 UntPhesoca 严式方案；粤语 / 四川话采用 diamoe 方言 IPA 方案（汉字 → 方言罗马字 → 严式 IPA）。
- **标签无关声调指标**：通过 `ipa2tone` 映射表（IPA 音节 → 调类）计算声调准确率，与具体音素写法无关，便于公平比较。
- **Gradio 演示**：支持上传 / 录制音频识别，并与参考文本自动比对差异（逐音节声调对齐高亮）。


## 目录结构

```
mandarin-ipa-asr/
├── LICENSE                                  # CC BY-NC-SA 4.0 全文
├── README.md / README_*.md                 # 多语言说明
├── requirements.txt
├── .gitignore
├── app.py                                   # Gradio 演示（--language 切换三语）
├── configs/example_train_config.json        # 训练配置示例
├── vocab/
│   ├── vocab_mandarin_ipa_combined.json     # 普通话 IPA 输出词表（1427 类）
│   └── vocab_mandarin_ipa_tone_combined.json# 普通话 IPA 音节 -> 调类 1-5
├── src/
│   ├── model.py                # SenseVoiceIpa：编码器 + IPA CTC 头（普通话全量微调）
│   ├── model_cantonese.py     # SenseVoiceIpaLora：LoRA 适配器（粤语/四川话共用）
│   ├── utils.py                # CTC 解码 / 指标 / 对齐 / 音频加载 / 模型构建
│   ├── dataset.py              # 音频-文本数据集
│   ├── train.py / infer.py     # 普通话全量微调训练 / 推理评测
│   ├── train_cantonese_lora.py # LoRA 训练（方言无关，粤语/四川话共用）
│   ├── infer_cantonese_lora.py # LoRA 推理评测（方言无关，粤语/四川话共用）
│   ├── cantonese_g2p.py / sichuan_g2p.py  # 方言汉字 → 严式 IPA
│   ├── diamoe_dialect_ipa.py  # diamoe 方言 IPA 转换后端
│   └── prepare_cantonese_data.py / prepare_sichuan_data.py  # 方言数据构建
├── data/
│   ├── cantonese_ipa/          # 粤语 scp/text/vocab/ipa2tone
│   └── sichuan_ipa/            # 四川话 scp/text/vocab/ipa2tone
├── weights/base.pt             # 普通话微调权重（不入库，魔搭(ModelScope) 下载）
├── out_canto/cantonese.pt       # 粤语 LoRA 权重
├── out_sichuan/sichuan.pt       # 四川话 LoRA 权重
└── results/metrics.md          # 三语实测指标与训练曲线
```


## 基础模型

| 项 | 说明 |
| --- | --- |
| 名称 | SenseVoiceSmall（FunAudioLLM / 阿里达摩院） |
| 来源 | GitHub: [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)；ModelScope: `iic/SenseVoiceSmall` |
| 结构 | `WavFrontend`（fbank 特征 + 可选 f0） + Conformer 编码器（512 维）+ 原始汉字 CTC 头 |
| 本项目的改造 | 冻结前端；在 Conformer 编码器之上挂载新的 **IPA CTC 头**；粤语/四川话再叠加 **LoRA 适配器** |

> 基础模型权重在首次运行时由 ModelScope 自动下载并缓存，无需手动准备。


## 训练方法

### 普通话：完整微调
1. 加载预训练 SenseVoiceSmall 的 `WavFrontend` 与 `encoder`（原汉字 CTC 头丢弃），新建 IPA CTC 头。
2. `WavFrontend` 始终冻结、`dither=0`；编码器与 IPA 头解冻并全量微调。
3. `CTCLoss(blank=0)`；AdamW（`lr=1e-4）；`CosineAnnealingLR`；梯度裁剪 1.0；bf16 混合精度。
4. 保存 `base.pt`（TER 最低）、`base_tone.pt`（声调最高）。

### 粤语 / 四川话：LoRA 适配器
1. 复用 SenseVoiceSmall 的 `WavFrontend` + `encoder` 前向（均冻结）。
2. 在编码器 4 类 Linear（qk-v / out / ff-w1 / ff-w2）上注入 LoRA（rank=32, alpha=32）。
3. 新建**方言 IPA CTC 头**（`vocab_size` = 方言词表类数）。
4. 仅训练 LoRA 低秩参数 + 方言 CTC 头（底座全程冻结），显存占用小、训练快。


## 训练数据

| 语言 | 训练集 | 验证集 | IPA 词表 | 说明 |
| --- | --- | --- | --- | --- |
| 普通话 | 11,691 | 918（一甲） / 5,918（合并） | 1,427 | zhvoice 普通话子集 → nk2028 UntPhesoca 严式 IPA |
| 粤语 | 8,426 | 1,999 | 1,580 | 粤语口语句库 → diamoe 方言 IPA（宽/严式） |
| 四川话 | 5,869 | 653 | 912 | 四川话（成渝片）口语句库 → diamoe 方言 IPA |

> 原始语料规模与授权见各数据源仓库；本项目仅使用其中朗读/口语句库子集，转换为严式 IPA。


## 安装与环境（均实测）

| 依赖 | 版本 |
| --- | --- |
| Python | 3.9.13（3.8–3.11 应可运行） |
| CUDA | 12.8 |
| torch / torchaudio | 2.7.0+cu128 |
| funasr | 1.4.16 |
| modelscope | 1.32.0 |
| transformers | 4.43.0 |
| numpy | 1.23.4 |
| editdistance | 0.6.2 |
| tqdm | 4.64.1 |
| soundfile | 0.12.1 |
| gradio | 4.24.0 |
| librosa | 0.9.2（可选） |
| ffmpeg | 2025-08-23（需在 PATH 中） |

```bash
pip install -r requirements.txt
# ffmpeg 需单独安装并加入 PATH（Windows 可用 https://www.gyan.dev/ffmpeg/ 或 scoop/apt 等）
```

**设备需求**：普通话全量微调建议 ≥ 16 GB 显存；LoRA 训练数 GB 即可；推理仅需数 GB，单条音频 CPU 亦可。


## 快速开始

> 所有脚本均按「项目根目录」解析 `vocab/`、`data/`、`weights/`、`out_*` 等相对路径，无需 `cd` 到项目根。
> 基础模型 `iic/SenseVoiceSmall` 为 ModelScope 模型 id，首次运行会自动下载并缓存。

### 1. 获取模型权重

模型权重**不放入本仓库**，请从 ModelScope（QiGuanFuChen/mandarin-ipa-asr）下载（由作者单独发布）：

```bash
# 从魔搭(ModelScope)下载权重仓库（含 base.pt / cantonese.pt / sichuan.pt）
modelscope download --model QiGuanFuChen/mandarin-ipa-asr --local_dir .
# 或直接运行 app.py：缺失权重时会自动从魔搭下载
```

### 2. 单条音频推理

```bash
# 普通话
python src/infer.py --wav path/to/audio.wav --ckpt weights/base.pt

# 粤语（LoRA，复用通用推理脚本）
bash infer_cantonese_lora.sh "path/to/audio.wav"

# 四川话（LoRA）
bash infer_sichuan_lora.sh "path/to/audio.wav"
```

### 3. 批量评测（验证指标）

```bash
python src/infer.py --eval --ckpt weights/base.pt --val_scp data/val.scp --val_text data/val.text
bash infer_cantonese_lora.sh eval      # -> out_canto/preds_val.txt + TER/token/tone
bash infer_sichuan_lora.sh eval        # -> out_sichuan/preds_val.txt + TER/token/tone
```

### 4. Gradio 演示（三语切换）

```bash
# 默认以普通话启动（也可显式 --language 指定初始语言）
python app.py --ckpt weights/base.pt --port 7860
python app.py --language cantonese --port 7860   # 初始即粤语 LoRA
python app.py --language sichuan  --port 7860   # 初始即四川话 LoRA
```

**界面内实时切换语言**：启动后，在页面顶部「识别语言」单选框即可在
**普通话 / 粤语 / 四川话** 之间即时切换，**无需重启服务**。切换时自动卸载旧模型显存、
加载对应权重，并联动更新「比对」标签页的参考来源选项（方言自动隐藏「中文→IPA 转换器」，
仅保留直接填 IPA）。

- **识别**标签页：上传或录制音频 → 输出严式 IPA。
- **比对**标签页：上传音频并填入参考 → 逐音节声调对齐并高亮差异（普通话支持「中文→IPA 转换器」真值；
  方言请直接填空格分隔的方言严式 IPA）。
- **IPA 语音特征分析**标签页：上传音频或粘贴 IPA → 逐音节拆解并标注调值与附加符号的语音学特征。


## 训练指南

### 普通话（完整微调）
```bash
python src/train.py \
    --train_scp data/train.scp --train_text data/train.text \
    --val_scp data/val.scp --val_text data/val.text \
    --vocab_path vocab/vocab_mandarin_ipa_combined.json \
    --ipa2tone_path vocab/vocab_mandarin_ipa_tone_combined.json \
    --output_dir checkpoints --epochs 10 --batch_size 16
```

### 粤语 / 四川话（LoRA）
```bash
# 粤语
./train_cantonese_lora.sh            # 训练（如需续训，显式传 --resume out_canto/cantonese_last.pt）
./train_cantonese_lora.sh fresh      # 清空旧权重，从头训练
./train_cantonese_lora.sh eval       # 只评测

# 四川话（复用同一通用 LoRA 训练脚本，仅数据/输出目录不同）
./train_sichuan_lora.sh
./train_sichuan_lora.sh fresh
./train_sichuan_lora.sh eval
```


## 数据使用说明（数据不公开）

出于授权与体量考虑，**训练数据不直接发布**。你可按以下步骤自行重建等价数据集：

1. **普通话**：下载 [zhvoice](https://github.com/fighting41love/zhvoice)，用
   [putonghua-ipa-converter](https://github.com/nk2028/putonghua-ipa-converter)（scheme 2 = UntPhesoca 严式）
   将拼音转为严式 IPA，再用 `src/prepare_data.py` 生成 `train/val.scp` 与 `text`。
2. **粤语 / 四川话**：用 `src/prepare_cantonese_data.py` / `src/prepare_sichuan_data.py` 从各自的
   口语句库（汉字列）经 `diamoe_dialect_ipa.py` 转为方言严式 IPA，产出 `data/*_ipa/` 下的
   `scp/text/vocab/ipa2tone`。

**请不要将原始音频或第三方文本随本仓库发布**，仅发布上述脚本与使用说明。


## 测试

仓库含一个**不依赖模型权重 / 基础模型**的轻量冒烟测试，用于守护路径解析、词表加载与 CTC 解码等关键逻辑：

```bash
python tests/smoke.py
```

并通过 GitHub Actions 在每次 push / PR 到 `main` 时自动运行（见 `.github/workflows/ci.yml）。


## 许可证

代码与词表以 **CC BY-NC-SA 4.0** 发布（见 [LICENSE](LICENSE)）。
- **署名（BY）**：使用时请保留原作者与项目出处。
- **非商业（NC）**：不得用于商业目的。
- **相同方式共享（SA）**：衍生作品须以相同许可证发布。

模型权重与训练数据按各自来源许可单独提供（ModelScope / 数据源仓库），不构成对本仓库许可证的覆盖。


## 引用与致谢

- 基础模型：FunAudioLLM, *SenseVoice*.
- 普通话严式 IPA 转换：nk2028, *putonghua-ipa-converter*（CC0）.
- 方言 IPA 转换：diamoe 方言 IPA 方案.
- 训练语料：fighting41love, *zhvoice*；及粤语 / 四川话口语句库.


<p align="center">
<a href="README.md"><img alt="%E4%B8%AD%E6%96%87" src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-brightgreen"></a>
<a href="README_en.md"><img alt="English" src="https://img.shields.io/badge/English-blue"></a>
<a href="README_ja.md"><img alt="日本語" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-blue"></a>
<a href="README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/%ED%95%9C%EA%B5%AD%EC%96%B4-blue"></a>
<a href="README_vi.md"><img alt="Tiếng Việt" src="https://img.shields.io/badge/Ti%E1%BA%BFng_Vi%E1%BB%87t-blue"></a>
<a href="README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Fran%C3%A7ais-blue"></a>
<a href="README_ru.md"><img alt="Русский" src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-blue"></a>
</p>
