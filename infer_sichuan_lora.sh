#!/usr/bin/env bash
# ================================================================
#  四川话 LoRA 模型推理：语音波形 -> 严式 IPA(含 Chao 调值字母)
#  复用通用推理脚本 infer_cantonese_lora.py (方言无关)
#
#  用法(在 D:/mandarin-ipa-asr-cu128 目录下, Git Bash):
#    # 单条 wav -> 严式 IPA
#    ./infer_sichuan_lora.sh "路径/xxx.wav"
#    # 批量(每行一个 wav 绝对路径)
#    ./infer_sichuan_lora.sh list wavs.txt preds_sichuan.txt
#    # 评测(TER / token_acc / tone_acc / exact, 写出 preds_val.txt)
#    ./infer_sichuan_lora.sh eval
#
#  权重: out_sichuan/best.pt (由 train_sichuan_lora.sh 训练得到)
# ================================================================
set -e

# 用 pwd -W 拿到 Windows 原生路径(D:/...), 避免 MSYS 把 /d/... 转义成 D:\d\... 导致 python.exe 找不到文件
ROOT="$(cd "$(dirname "$0")" && pwd -W)"
cd "$ROOT"
PY="$ROOT/runtime/python.exe"
SRC="$ROOT/src"
DATA="$ROOT/data/sichuan_ipa"
OUT="$ROOT/out_sichuan"
CKPT="$OUT/best.pt"

if [ "$1" = "eval" ]; then
  "$PY" "$SRC/infer_cantonese_lora.py" --eval \
    --ckpt "$CKPT" \
    --vocab "$DATA/vocab_sichuan_ipa_combined.json" \
    --ipa2tone "$DATA/ipa2tone_sichuan.json" \
    --val_scp "$DATA/val_scp" --val_text "$DATA/val_text" \
    --output_dir "$OUT"
elif [ "$1" = "list" ]; then
  "$PY" "$SRC/infer_cantonese_lora.py" --list "$2" --out "${3:-preds_sichuan.txt}" \
    --ckpt "$CKPT" \
    --vocab "$DATA/vocab_sichuan_ipa_combined.json" \
    --ipa2tone "$DATA/ipa2tone_sichuan.json"
else
  "$PY" "$SRC/infer_cantonese_lora.py" --wav "$1" \
    --ckpt "$CKPT" \
    --vocab "$DATA/vocab_sichuan_ipa_combined.json" \
    --ipa2tone "$DATA/ipa2tone_sichuan.json"
fi
