#!/usr/bin/env bash
# ================================================================
#  粤语 LoRA 模型推理：wav -> 严式 IPA（含调值字母）
#  用法（D:/mandarin-ipa-asr-cu128 下 Git Bash）:
#    单条 : bash infer_cantonese_lora.sh <wav绝对路径>
#    批量 : bash infer_cantonese_lora.sh list <列表txt(每行一wav)> [输出txt]
#    评测 : bash infer_cantonese_lora.sh eval
#  权重默认取 out_canto/best.pt（需先训练）。
# ================================================================
set -e

# 用 pwd -W 拿到 Windows 原生路径(D:/...), 避免 MSYS 把 /d/... 转义成 D:\d\... 导致 python.exe 找不到文件
ROOT="$(cd "$(dirname "$0")" && pwd -W)"
cd "$ROOT"
PY="$ROOT/runtime/python.exe"
SRC="$ROOT/src"
CKPT="$ROOT/out_canto/best.pt"

case "$1" in
  eval)
    "$PY" "$SRC/infer_cantonese_lora.py" --eval --ckpt "$CKPT"
    ;;
  list)
    LIST="${2:?用法: bash infer_cantonese_lora.sh list <列表txt> [输出txt]}"
    OUT="${3:-preds_canto.txt}"
    "$PY" "$SRC/infer_cantonese_lora.py" --list "$LIST" --out "$OUT" --ckpt "$CKPT"
    echo "已写出 $OUT"
    ;;
  *)
    WAV="${1:?用法: bash infer_cantonese_lora.sh <wav> | list <txt> [out] | eval}"
    "$PY" "$SRC/infer_cantonese_lora.py" --wav "$WAV" --ckpt "$CKPT"
    ;;
esac
