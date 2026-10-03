#!/usr/bin/env bash
# ================================================================
#  粤语 IPA 适配：冻结 SenseVoice 普通话底座 + LoRA 适配器 + 粤语 CTC 头
#  只训练 LoRA 低秩参数(约18.4M) + 粤语 IPA 头(约1.1M)，底座全程冻结
#
#  用法（在 D:/mandarin-ipa-asr-cu128 目录下，用 Git Bash 运行）:
#    ./train_cantonese_lora.sh          # 训练
#    ./train_cantonese_lora.sh eval     # 只评测（加载已训 cantonese.pt）
#
#  依赖环境: D:/mandarin-ipa-asr-cu128/runtime/python.exe
#            （已含 torch2.7+cu128 / funasr1.4 / ToJyutping / pyjyutping）
#  数据:     data/cantonese_ipa/{train_scp,train_text,val_scp,val_text,
#                               vocab_cantonese_ipa_combined.json,
#                               ipa2tone_cantonese.json}   （已生成）
#  输出:     out_canto/{cantonese.pt,cantonese_tone.pt,cantonese_last.pt,metrics.csv,preds_val.txt}
# ================================================================
set -e

# 用 pwd -W 拿到 Windows 原生路径(D:/...), 避免 MSYS 把 /d/... 转义成 D:\d\... 导致 python.exe 找不到文件
ROOT="$(cd "$(dirname "$0")" && pwd -W)"
cd "$ROOT"
PY="$ROOT/runtime/python.exe"
SRC="$ROOT/src"
DATA="$ROOT/data/cantonese_ipa"
OUT="$ROOT/out_canto"

# ---------- 训练超参（可按机器显存/速度调整）----------
EPOCHS=30
BATCH=8          # 显存不够就调小（如 4），并用 GRAD_ACCUM 补回等效 batch
GRAD_ACCUM=2
LR=2e-4
NUM_WORKERS=2
MAX_SECONDS=20   # 单条音频最长秒数，超长裁剪；粤语样本多为短句，20s 足够

if [ "$1" = "eval" ]; then
  echo ">>> 评测模式：加载 $OUT/cantonese.pt"
  "$PY" "$SRC/train_cantonese_lora.py" --eval \
    --resume     "$OUT/cantonese.pt" \
    --best_name  cantonese \
    --val_scp    "$DATA/val_scp" \
    --val_text   "$DATA/val_text" \
    --output_dir "$OUT"
else
  echo ">>> 训练模式：输出到 $OUT"
  "$PY" "$SRC/train_cantonese_lora.py" \
    --train_scp   "$DATA/train_scp" \
    --train_text  "$DATA/train_text" \
    --val_scp     "$DATA/val_scp" \
    --val_text    "$DATA/val_text" \
    --output_dir  "$OUT" \
    --best_name   cantonese \
    --epochs      $EPOCHS \
    --batch_size  $BATCH \
    --grad_accum  $GRAD_ACCUM \
    --lr          $LR \
    --num_workers $NUM_WORKERS \
    --max_seconds $MAX_SECONDS
fi
