#!/usr/bin/env bash
# ================================================================
#  四川话(成都 / 西南官话成渝片) IPA 适配
#  冻结 SenseVoice 普通话底座 + LoRA 适配器 + 四川话 CTC 头
#  只训练 LoRA 低秩参数(~18.8M) + 四川话 IPA 头, 底座全程冻结
#
#  数据由 src/prepare_sichuan_data.py 生成于 data/sichuan_ipa/
#    (脚本集 6522 单句 + 对话集按时间戳切段, 共约 1.2 万条)
#  复用通用 LoRA 训练脚本 train_cantonese_lora.py (方言无关)
#
#  用法(在 D:/mandarin-ipa-asr-cu128 目录下, Git Bash):
#    ./train_sichuan_lora.sh            # 训练(若 sichuan_last.pt 存在则自动续训)
#    ./train_sichuan_lora.sh fresh      # 清空旧权重, 从头训练
#    ./train_sichuan_lora.sh eval       # 只评测(加载已训 sichuan.pt)
#
#  依赖环境: D:/mandarin-ipa-asr-cu128/runtime/python.exe
#  输出:     out_sichuan/{sichuan.pt,sichuan_tone.pt,sichuan_last.pt,metrics.csv,preds_val.txt}
# ================================================================
set -e

# 用 pwd -W 拿到 Windows 原生路径(D:/...), 避免 MSYS 把 /d/... 转义成 D:\d\... 导致 python.exe 找不到文件
ROOT="$(cd "$(dirname "$0")" && pwd -W)"
cd "$ROOT"
PY="$ROOT/runtime/python.exe"
SRC="$ROOT/src"
DATA="$ROOT/data/sichuan_ipa"
OUT="$ROOT/out_sichuan"

# ---------- 训练超参(可按机器显存/速度调整) ----------
EPOCHS=40
BATCH=8          # 显存不够就调小(如 4), 并用 GRAD_ACCUM 补回等效 batch
GRAD_ACCUM=2
LR=2e-4
NUM_WORKERS=2
MAX_SECONDS=15   # 单条音频最长秒数, 超长裁剪(对话集长句较多, 15s 足够且省显存)

# ---------- 稳定性 / 收敛性增强 ----------
MAX_GRAD_NORM=1.0     # 梯度裁剪: 防长音频 batch 梯度爆炸 / OOM 崩溃
WARMUP_EPOCHS=3       # LR 线性 warmup 轮数
EARLY_STOP_PATIENCE=8 # TER 连续 N 轮无提升则早停(0=关闭)
MIN_EPOCHS=10         # 早停最少训练轮数

case "$1" in
  eval)
    echo ">>> 评测模式：加载 $OUT/sichuan.pt"
    "$PY" "$SRC/train_cantonese_lora.py" --eval \
      --resume     "$OUT/sichuan.pt" \
      --best_name  sichuan \
      --vocab      "$DATA/vocab_sichuan_ipa_combined.json" \
      --ipa2tone   "$DATA/ipa2tone_sichuan.json" \
      --val_scp    "$DATA/val_scp" \
      --val_text   "$DATA/val_text" \
      --output_dir "$OUT"
    exit 0
    ;;
  fresh)
    echo ">>> 全新训练: 清空旧权重"
    rm -f "$OUT"/sichuan_last.pt "$OUT"/sichuan.pt "$OUT"/sichuan_tone.pt "$OUT"/metrics.csv "$OUT"/preds_val.txt
    TRAIN_RESUME=""
    ;;
  *)
    if [ -f "$OUT/sichuan_last.pt" ]; then
      echo ">>> 检测到 sichuan_last.pt, 自动续训(崩溃/被杀后重跑可继续, 不丢进度)"
      TRAIN_RESUME="--resume $OUT/sichuan_last.pt"
    else
      echo ">>> 全新训练"
      TRAIN_RESUME=""
    fi
    ;;
esac

echo ">>> 训练模式：输出到 $OUT"
"$PY" "$SRC/train_cantonese_lora.py" \
  --vocab       "$DATA/vocab_sichuan_ipa_combined.json" \
  --ipa2tone    "$DATA/ipa2tone_sichuan.json" \
  --train_scp   "$DATA/train_scp" \
  --train_text  "$DATA/train_text" \
  --val_scp     "$DATA/val_scp" \
  --val_text    "$DATA/val_text" \
  --output_dir  "$OUT" \
  --best_name   sichuan \
  --epochs      $EPOCHS \
  --batch_size  $BATCH \
  --grad_accum  $GRAD_ACCUM \
  --lr          $LR \
  --num_workers $NUM_WORKERS \
  --max_seconds $MAX_SECONDS \
  --max_grad_norm $MAX_GRAD_NORM \
  --warmup_epochs $WARMUP_EPOCHS \
  --early_stop_patience $EARLY_STOP_PATIENCE \
  --min_epochs  $MIN_EPOCHS \
  $TRAIN_RESUME
