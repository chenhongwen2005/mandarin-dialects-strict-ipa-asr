# 普通话严式国际音标（IPA）语音识别

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

基于 [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) 编码器，在其上挂载一个
**严式国际音标（IPA）CTC 解码头**，通过全量微调，实现普通话的「音节 + 声调」级语音转写。

模型输出为空格分隔的严式 IPA 音节序列，每个音节自带调值符号（如 `ɡ̊wa̠n̚˥`、`x̞wa̠ɪ̯˧˥`），
即同时给出声母 / 韵母与声调，适合语音学分析、普通话发音评测、声调教学等场景。

> 许可证：**CC BY-NC-SA 4.0**（知识共享署名-非商业性使用-相同方式共享 4.0 国际）。
> 模型权重与训练数据不随代码仓库发布，分别通过 ModelScope 与本地数据构建流程获取，详见下文。

---

## 特性

- **严式 IPA 输出**：采用 [nk2028/putonghua-ipa-converter](https://github.com/nk2028/putonghua-ipa-converter)
  的 UntPhesoca 严式方案，标注到音素级并保留声调调值。
- **全量微调**：解冻 SenseVoiceSmall 的 Conformer 编码器与新建 IPA 头一起训练；特征前端冻结、
  `dither=0` 保证可复现。
- **标签无关声调指标**：通过 `ipa2tone` 映射表（IPA 音节 → 调类 1–5）计算声调准确率，与具体
  音素写法无关，便于公平比较。
- **大规模验证**：在 5918 条合并验证集上实测（见[效果指标](#效果指标大规模验证)）。
- **Gradio 演示**：支持上传 / 录制音频识别，并与参考文本自动比对差异（逐音节声调对齐高亮）。

---

## 目录结构

```
mandarin-ipa-asr/
├── LICENSE                                  # CC BY-NC-SA 4.0 全文
├── README.md
├── requirements.txt                         # 实测依赖版本
├── .gitignore
├── app.py                                   # Gradio 演示（识别 + 自动比对差异）
├── configs/
│   └── example_train_config.json            # 训练配置示例
├── vocab/
│   ├── vocab_mandarin_ipa_combined.json     # IPA 输出词表（1427 类，开放发布）
│   └── vocab_mandarin_ipa_tone_combined.json# IPA 音节 -> 调类 1-5 映射（开放发布）
├── src/
│   ├── model.py        # SenseVoiceIpa：编码器 + IPA CTC 头
│   ├── utils.py        # CTC 解码 / 指标 / 对齐 / beam search / 音频加载
│   ├── dataset.py      # 音频-文本数据集（wav/flac/ogg/mp3）
│   ├── train.py        # 全量微调训练
│   ├── infer.py        # 单条推理 + 批量评测（大规模验证）
│   └── prepare_data.py # 由拼音文本构建 IPA 训练集（数据预处理，不含音频）
└── results/
    └── metrics.md       # 实测验证指标与训练曲线
```

---

## 基础模型

| 项 | 说明 |
| --- | --- |
| 名称 | SenseVoiceSmall（FunAudioLLM / 阿里达摩院） |
| 来源 | GitHub: [FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)；ModelScope: `iic/SenseVoiceSmall` |
| 结构 | `WavFrontend`（fbank 特征 + 可选 f0） + Conformer 编码器（512 维）+ 原始汉字 CTC 头 |
| 本项目的改造 | 冻结前端；在其 Conformer 编码器之上挂载新的 **IPA CTC 头**（见下），原汉字头不再使用 |

> 基础模型权重在首次运行时由 ModelScope 自动下载并缓存，无需手动准备。

---

## 训练方法

1. **初始化**：加载预训练 SenseVoiceSmall 的 `WavFrontend` 与 `encoder`（原汉字 CTC 头丢弃），
   新建 IPA CTC 头 `ctc_head = Linear(512,512) → ReLU → Dropout(0.1) → Linear(512, vocab)`。
2. **冻结策略**：`WavFrontend` 始终冻结、`dither` 固定为 0（关闭随机加噪，保证特征确定性）；
   编码器与 IPA 头**解冻并全量微调**（亦支持仅训练头）。
3. **损失与优化**：`CTCLoss(blank=0)`；AdamW（`lr=1e-4`, `weight_decay=1e-4`）；
   `CosineAnnealingLR`（`T_max = steps × epochs`）；梯度裁剪 1.0。
4. **精度**：bf16 混合精度（`autocast` 仅作用于编码器前向，特征与损失保持 fp32）。
5. **权重保存**：`best.pt`（验证集 TER 最低）、`best_tone.pt`（声调准确率最高）。
6. **兼容旧权重**：推理 / 续训时自动将旧版权重中残留的头命名映射为 `ctc_head.*`，避免静默丢头。

---

## 训练数据

| 项目 | 数量 |
| --- | --- |
| 训练集 | **11,691** 条（普通话朗读；由 zhvoice 普通话子集约 8,993 条按 1.3× 复用/扩充得到） |
| 验证集 | **918** 条（普通话朗读，与训练集互斥） |
| 标签 | 每条为空格分隔的严式 IPA 音节序列（由拼音经 nk2028/putonghua-ipa-converter 转换） |
| IPA 词表 | 1,427 类（含 `<blank>`/`<unk>`） |

> 原始 zhvoice 语料规模约 **900 小时、3200+ 说话人、约 112.98 万条文本**；
> 本项目仅使用其中朗读质量较高的普通话子集构造训练/验证集，并转换为严式 IPA。

---

## 模型参数

| 参数 | 数值 |
| --- | --- |
| 总参数量 | **234,993,874** |
| 全量微调可训练参数 | **222,131,699**（编码器 + IPA 头；前端与原汉字头冻结） |
| 仅推理（头可训练、编码器冻结） | 994,707 |
| 编码器维度 | 512 |
| IPA 输出词表 | 1,427 |
| 声调映射条目（ipa2tone） | 1,425（`<blank>`/`<unk>` 除外） |

---

## 数据集来源与链接

| 数据 / 工具 | 用途 | 许可 | 链接 |
| --- | --- | --- | --- |
| **zhvoice** | 训练语料（普通话朗读子集） | 见仓库 | https://github.com/fighting41love/zhvoice |
| **putonghua-ipa-converter** | 拼音 → 严式 IPA 转换（scheme 2, UntPhesoca） | CC0 | https://github.com/nk2028/putonghua-ipa-converter |
| **SenseVoiceSmall** | 预训练基础模型 | 见仓库 | https://github.com/FunAudioLLM/SenseVoice（ModelScope `iic/SenseVoiceSmall`） |

---

## 效果指标（大规模验证）

解码与训练完全一致（原始波形输入、dither=0、bf16 编码器前向）。指标定义见
[`results/metrics.md`](results/metrics.md)。

### 918 条一级甲等（一甲）普通话验证语料

| 指标 | 数值 |
| --- | --- |
| TER（音节错误率，越低越好） | **0.0955** |
| ACC（整句完全匹配） | 0.7059 |
| TACC（token 正确率） | 0.9145 |
| 声调准确率（标签无关） | **0.9212** |

### 5918 条合并验证集（大规模验证）

= 5000 条 zhvoice 真实场景 mp3 + 918 条普通话朗读。zhvoice 为真实场景、领域更杂、噪声更多，
故整句 ACC 低于一甲普通话验证语料属预期；声调识别仍保持高水平。

| 指标 | 数值 |
| --- | --- |
| TER（音节错误率，越低越好） | **0.1089** |
| ACC（整句完全匹配） | 0.4439 |
| TACC（token 正确率） | 0.8986 |
| 声调准确率（标签无关） | **0.8915** |

> 说明：声调准确率与拼音方案下的同类模型相当（同框架下拼音头声调约 0.92，本 IPA 头 0.89–0.92），
> 证明在「音素 + 声调」联合标注下，声调信息并未丢失。

---

## 安装与环境（均实测）

| 依赖 | 版本 | 说明 |
| --- | --- | --- |
| Python | **3.9.13** | 推荐 3.9（3.8–3.11 应可运行，已实测 3.9.13） |
| CUDA | **12.8** | 训练/推理需 NVIDIA GPU；纯 CPU 可推理但较慢 |
| torch | 2.7.0+cu128 | 与 CUDA 12.8 对应 |
| torchaudio | 2.7.0+cu128 | |
| funasr | 1.4.16 | 加载 SenseVoiceSmall |
| modelscope | 1.32.0 | 自动下载基础模型 |
| transformers | 4.43.0 | |
| numpy | 1.23.4 | |
| editdistance | 0.6.2 | 指标计算 |
| tqdm | 4.64.1 | 进度条 |
| soundfile | 0.12.1 | 音频读写 |
| gradio | 4.24.0 | 演示界面 |
| librosa | 0.9.2 | （可选）数据预处理 |
| ffmpeg | 2025-08-23 | 解码 mp3（需在 PATH 中） |

安装：

```bash
pip install -r requirements.txt
# ffmpeg 需单独安装并加入 PATH（Windows 可用 https://www.gyan.dev/ffmpeg/ 或 scoop/apt 等）
```

**设备需求**：训练建议 ≥ 16 GB 显存（bf16 下 234M 参数全量微调 + 激活）；推理仅需数 GB，
单条音频 CPU 亦可（速度较慢）。

---

## 快速开始

> 所有脚本均按「项目根目录」（即仓库根）解析 `vocab/`、`data/`、`weights/`、`checkpoints/` 等相对路径，
> 因此**无需 `cd` 到项目根即可从任意工作目录启动**；传入绝对路径则原样使用。
> 基础模型 `iic/SenseVoiceSmall` 为 ModelScope 模型 id，首次运行会自动下载并缓存。

> ### 【重要】必须先加载微调权重，否则输出是无意义乱码
>
> 本项目**不把模型权重放入仓库**（单独发布于 ModelScope）。若启动演示或推理时**未通过 `--ckpt` 指定权重**，
> 程序会静默使用一个**随机初始化**的模型——对音频吐出的 IPA 将是无意义的，且常出现音节反复重复，
> 看起来"识别出来很多"，但根本不是真实读音。
> 例如《静夜思》上阕只有约 10 个音节，未加载权重时却可能输出 30+ 个重复乱码音节。
>
> 正确做法（权重需先下载到 `weights/best.pt`，见下节）：
> ```bash
> python app.py --ckpt weights/best.pt
> ```
> 若未指定 `--ckpt`，`app.py` 会自动尝试定位 `weights/best.pt`；两者皆无时，启动会打印醒目的【警告】提示。

### 1. 获取模型权重

模型权重**不放入本仓库**，请从 ModelScope 下载（由作者单独发布），例如：

```bash
# 假设模型已发布，使用 modelscope 下载到本地
modelscope download --model QiGuanFuChen/mandarin-ipa-asr --local_dir weights/
```

得到 `weights/best.pt` 后，通过 `--ckpt` 指定。（若你自行训练，还会额外产出 `checkpoints/best_tone.pt`，见下「训练指南」）

### 2. 单条音频推理

```bash
# 无需 cd 到项目根：脚本会按自身位置解析 vocab/data 等相对路径，可从任意目录启动
python src/infer.py --wav path/to/audio.wav --ckpt weights/best.pt
# 输出：空格分隔的严式 IPA 音节序列
```

### 3. 批量评测（验证指标）

```bash
python src/infer.py --eval --ckpt weights/best.pt \
    --val_scp data/val.scp --val_text data/val.text --limit 0
# 输出 TER / ACC / TACC / 声调准确率
```

### 4. Gradio 演示（上传音频 + 自动比对差异）

> 务必通过 `--ckpt` 加载权重（见上方【重要】提示），否则识别结果为随机初始化的无意义乱码。
> 「识别」页默认使用集束搜索（`--beam`，默认 12）以降低插入/重复错误；`--beam 0` 可退回贪心解码。

```bash
# 可从任意目录启动（相对路径均按项目根解析）；--ckpt 可传相对或绝对路径
# 未传 --ckpt 时会自动尝试 weights/best.pt
python app.py --ckpt weights/best.pt --port 7860
# 浏览器打开 http://127.0.0.1:7860
```

- **识别**标签页：上传或录制音频 → 输出严式 IPA。
- **比对（自动差异）**标签页：上传音频并（可选）填入参考 IPA 文本 →
  自动做逐音节声调对齐并高亮差异（正确/调错/错读/多读/漏读）；
  即使不填参考文本，也会自动用「贪心 vs 集束搜索」两套解码结果互比差异。

---

## 训练指南

1. 准备数据（见下「数据使用说明」），得到 `train.scp` / `train.text` / `val.scp` / `val.text`
   （格式：`uid 音频路径` 与 `uid 空格分隔的IPA音节`）。
2. 运行：

```bash
python src/train.py \
    --train_scp data/train.scp --train_text data/train.text \
    --val_scp data/val.scp --val_text data/val.text \
    --vocab_path vocab/vocab_mandarin_ipa_combined.json \
    --ipa2tone_path vocab/vocab_mandarin_ipa_tone_combined.json \
    --output_dir checkpoints --epochs 10 --batch_size 16
```

也可将参数写入 `configs/example_train_config.json` 后直接：

```bash
python src/train.py $(python -c "import json,sys; c=json.load(open('configs/example_train_config.json')); print(' '.join(f'--{k} {v}' for k,v in c.items()))")
```

- 如需从拼音全量微调权重热启动：`--warm_start weights/pinyin_ft.pt`（头维度不同会自动跳过、随机初始化）。
- 输出 `checkpoints/best.pt`（TER 最低）与 `checkpoints/best_tone.pt`（声调最高）。

---

## 数据使用说明（数据不公开）

出于授权与体量考虑，**训练数据不直接发布**。你可按以下步骤自行重建等价数据集：

1. 下载 **zhvoice** 语料（https://github.com/fighting41love/zhvoice），解压得到音频与拼音文本。
2. 克隆 **putonghua-ipa-converter**（https://github.com/nk2028/putonghua-ipa-converter），
   用其 `data/putonghua.js`（scheme 2 = UntPhesoca 严式）将拼音转为严式 IPA。
3. 用本仓库 `src/prepare_data.py` 生成训练所需文件：

```bash
# 1) zhvoice 子集 -> 严式 IPA 文本 + scp + 词表
#    --conv_js 指向转换器的 data/putonghua.js；--audio_root 为音频根目录
python src/prepare_data.py build-zhvoice \
    --metadata zhvoice/metadata.csv \
    --audio_root zhvoice/wavs \
    --conv_js path/to/putonghua-ipa-converter/data/putonghua.js \
    --out data/zhvoice_ipa --train_n 8993 --val_n 918

# 2) 将自有带调拼音文本转为 IPA（目录内需含 train/text、val/text）
python src/prepare_data.py build-mandarin \
    --text_dir data/mandarin_pinyin \
    --conv_js path/to/putonghua-ipa-converter/data/putonghua.js \
    --out data/zhvoice_ipa

# 3) 合并多来源并扩充词表，输出 *_combined 文件
python src/prepare_data.py combine \
    --zhvoice_dir data/zhvoice_ipa --mandarin_ipa_dir data/zhvoice_ipa \
    --out data/combined

# 4) 按倍数扩充训练集（如 1.3×）：以普通话子集为基，混入 zhvoice 数据
python src/prepare_data.py scale \
    --mandarin_text data/combined/train_text --mandarin_scp data/combined/train_scp \
    --zhvoice_text data/zhvoice_ipa/train_text --zhvoice_scp data/zhvoice_ipa/train_scp \
    --factor 1.3 --out data/ipa130
```

生成的 `*.scp`（音频路径）与 `*.text`（IPA 标签）即可作为 `train.py` / `infer.py` 的输入。
**请不要将原始音频或第三方文本随本仓库发布**，仅发布上述脚本与使用说明。

---

## 许可证

代码与词表以 **CC BY-NC-SA 4.0** 发布（见 [LICENSE](LICENSE)）。
- **署名（BY）**：使用时请保留原作者与项目出处。
- **非商业（NC）**：不得用于商业目的。
- **相同方式共享（SA）**：衍生作品须以相同许可证发布。

模型权重与训练数据按各自来源许可单独提供（ModelScope / 数据源仓库），不构成对本仓库许可证的覆盖。

---

## 引用与致谢

- 基础模型：FunAudioLLM, *SenseVoice*.
- 严式 IPA 转换：nk2028, *putonghua-ipa-converter*（CC0）.
- 训练语料：fighting41love, *zhvoice*.
