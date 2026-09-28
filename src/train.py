# -*- coding: utf-8 -*-
"""严式 IPA 全量微调：解冻编码器 + IPA CTC 头，bf16 混合精度训练。

输入:
  - vocab_path / ipa2tone_path : IPA 词表与音节->调类映射
  - train_scp / train_text     : 训练音频与标签
  - val_scp   / val_text       : 验证音频与标签（同集评测声调）
  - warm_start                 : 可选，从拼音全量微调权重热启动（编码器形状匹配则载入，
                                 IPA 头维度不同则保持随机初始化）

保存:
  - best.pt      : 验证集 TER 最低的权重（转写错误率最优）
  - best_tone.pt : 验证集声调准确率最高的权重（声调识别最优）

用法:
  python train.py --train_scp train.scp --train_text train.text \
                  --val_scp val.scp --val_text val.text --output_dir checkpoints
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import SenseVoiceIpa
from utils import (
    ctc_greedy_decode,
    compute_ter,
    compute_exact_match,
    compute_token_acc,
    load_vocab,
)
from dataset import WavDataset, collate_wav

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 兼容旧版权重中残留的头命名 -> 当前 ctc_head.*
_COMPAT_HEAD_PREFIXES = ("sichuan_ctc_head.", "ipa_ctc_head.", "mandarin_ctc_head.")


def _compat_remap(state: dict) -> dict:
    """将旧权重中的头命名映射为当前模型使用的 ctc_head.*。"""
    out = {}
    for k, v in state.items():
        for old in _COMPAT_HEAD_PREFIXES:
            if k.startswith(old):
                out["ctc_head." + k[len(old):]] = v
                break
        else:
            out[k] = v
    return out


DEFAULT_CONFIG = dict(
    vocab_path="vocab/vocab_mandarin_ipa_combined.json",
    ipa2tone_path="vocab/vocab_mandarin_ipa_tone_combined.json",
    model_dir="iic/SenseVoiceSmall",
    warm_start="",
    train_scp="",
    train_text="",
    val_scp="",
    val_text="",
    output_dir="checkpoints",
    epochs=10,
    batch_size=16,
    learning_rate=1e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    max_seconds=20,
    precision="bf16",
    num_workers=0,
    seed=42,
)


@torch.no_grad()
def eval_split(model, loader, ctc, device, tag="val", verbose=False,
               id2token=None, ipa2tone=None):
    model.eval()
    preds, refs = [], []
    loss_sum, nb = 0.0, 0
    for wp, wlens, tp, tlens in tqdm(loader, desc=f"Eval {tag}"):
        wp, wlens = wp.to(device), wlens.to(device)
        tp, tlens = tp.to(device), tlens.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lp, enc_lens = model(wp, wlens, autocast_bf16=True)
            loss_sum += float(ctc(lp, tp, enc_lens, tlens))
        nb += 1
        preds.extend(ctc_greedy_decode(lp))
        refs.extend([t[:l].tolist() for t, l in zip(tp, tlens)])
    if id2token is None:
        _, id2token = load_vocab(DEFAULT_CONFIG["vocab_path"])
    ter = compute_ter(preds, refs)
    acc = compute_exact_match(preds, refs)
    tacc = compute_token_acc(preds, refs)
    tone = compute_tone_accuracy_ipa(preds, refs, id2token, ipa2tone) if ipa2tone else 0.0
    res = dict(ter=ter, acc=acc, tacc=tacc, tone_acc=tone,
               loss=loss_sum / max(1, nb), n=len(preds))
    if verbose:
        print(f"  TER {ter:.4f} | ACC {acc:.4f} | TACC {tacc:.4f} | "
              f"声调 {tone:.4f} | loss {res['loss']:.4f} | n={len(preds)}")
    return res


def main():
    cfg = dict(DEFAULT_CONFIG)
    ap = argparse.ArgumentParser()
    for k, v in cfg.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", action="store_true", default=v)
        else:
            typ = str if isinstance(v, str) else (int if isinstance(v, int) else float)
            ap.add_argument(f"--{k}", type=typ, default=v)
    args = ap.parse_args()
    for k, v in vars(args).items():
        if k in cfg and v is not None:
            cfg[k] = v

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    vocab_size = len(json.load(open(cfg["vocab_path"], encoding="utf-8")))
    ipa2tone = json.load(open(cfg["ipa2tone_path"], encoding="utf-8"))
    print(f"vocab_size={vocab_size} | ipa2tone={len(ipa2tone)}")

    base = SenseVoiceIpa(vocab_size=vocab_size, model_dir=cfg["model_dir"],
                         freeze_encoder=True)
    if cfg["warm_start"] and os.path.exists(cfg["warm_start"]):
        w = torch.load(cfg["warm_start"], map_location="cpu", weights_only=False)
        sd = w.get("model_state_dict", w)
        sd = _compat_remap(sd)
        cur = base.state_dict()
        sd_filtered = {k: v for k, v in sd.items()
                       if k in cur and v.shape == cur[k].shape}
        missing = [k for k in cur if k not in sd_filtered]
        base.load_state_dict(sd_filtered, strict=False)
        print(f"[warm-start] {cfg['warm_start']} 载入 {len(sd_filtered)}/{len(cur)} 参数 "
              f"(头维度不同跳过 {len(missing)} 项, IPA 头 random init)")
    base.unfreeze_encoder_and_head()
    model = base.to(DEVICE)

    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"可训参数 {n_tr:,} / {n_all:,}")

    train_ds = WavDataset(cfg["train_scp"], cfg["train_text"], cfg["vocab_path"],
                          max_seconds=cfg["max_seconds"])
    val_ds = WavDataset(cfg["val_scp"], cfg["val_text"], cfg["vocab_path"],
                        max_seconds=cfg["max_seconds"])
    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          collate_fn=collate_wav, num_workers=cfg["num_workers"])
    val_ld = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                        collate_fn=collate_wav, num_workers=cfg["num_workers"])
    _, id2token = load_vocab(cfg["vocab_path"])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    steps = len(train_ld)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, steps * cfg["epochs"]), eta_min=cfg["learning_rate"] * 0.01)
    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")
    csv_path = out_dir / "metrics.csv"
    csv_path.write_text("epoch,loss,ter,acc,tacc,tone_acc\n", encoding="utf-8")

    best_ter = float("inf")
    best_tone = -1.0
    use_amp = (cfg["precision"] == "bf16" and DEVICE == "cuda")
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        pbar = tqdm(train_ld, desc=f"Epoch {epoch}")
        for wp, wlens, tp, tlens in pbar:
            wp, wlens = wp.to(DEVICE), wlens.to(DEVICE)
            tp, tlens = tp.to(DEVICE), tlens.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    lp, enc_lens = model(wp, wlens, autocast_bf16=True)
                    loss = ctc(lp, tp, enc_lens, tlens)
            else:
                lp, enc_lens = model(wp, wlens)
                loss = ctc(lp, tp, enc_lens, tlens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
            if nb % 20 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                  "lr": f"{opt.param_groups[0]['lr']:.2e}"})
        train_loss = tot / max(1, nb)
        dt = time.time() - t0
        print(f"[EPOCH {epoch}] loss={train_loss:.4f}  ({dt:.1f}s)")
        metrics = eval_split(model, val_ld, ctc, DEVICE, "val", verbose=True,
                             id2token=id2token, ipa2tone=ipa2tone)
        for k in ("ter", "acc", "tacc", "tone_acc"):
            writer.add_scalar(f"val/{k}", metrics[k], epoch)
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.4f},{metrics['ter']:.4f},{metrics['acc']:.4f},"
                    f"{metrics['tacc']:.4f},{metrics['tone_acc']:.4f}\n")
        if metrics["ter"] < best_ter:
            best_ter = metrics["ter"]
            torch.save(_ckpt(model, epoch, metrics, vocab_size, cfg),
                       out_dir / "best.pt")
            print(f"  -> best.pt 更新 (TER={best_ter:.4f})")
        if metrics["tone_acc"] > best_tone:
            best_tone = metrics["tone_acc"]
            torch.save(_ckpt(model, epoch, metrics, vocab_size, cfg),
                       out_dir / "best_tone.pt")
            print(f"  -> best_tone.pt 更新 (声调={best_tone:.4f})")

    print(f"\n微调完成, best TER = {best_ter:.4f}, best 声调 = {best_tone:.4f}")
    print(f"推理: python infer.py --eval --ckpt {out_dir / 'best.pt'}")
    writer.close()


def _ckpt(model, epoch, metrics, vocab_size, cfg):
    return {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "vocab_size": vocab_size,
        "vocab_path": cfg["vocab_path"],
        "finetune": True,
        "ipa": True,
        "config": cfg,
    }


if __name__ == "__main__":
    main()
