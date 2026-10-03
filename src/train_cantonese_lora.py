# -*- coding: utf-8 -*-
"""粤语 IPA 适配训练：冻结 SenseVoice 底座 + LoRA 适配器 + 粤语 CTC 头。

只优化 LoRA 低秩参数与 CTC 头，底座（encoder + frontend）全程冻结。

训练:
    runtime/python.exe train_cantonese_lora.py --epochs 30 --batch_size 8

评测（加载已训权重，仅跑验证集）:
    runtime/python.exe train_cantonese_lora.py --eval --resume out_canto/best.pt

输出（--output_dir，默认 out_canto/）:
    best.pt        TER 最优的 LoRA+头权重
    best_tone.pt   声调准确率最优的权重
    last.pt        最近一轮
    metrics.csv    每轮 train_loss / val_TER / val_token_acc / val_tone_acc / val_exact
    preds_val.txt  最近一轮验证集解码结果（uid | ref | pred）
"""

import argparse
import csv
import json
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WavDataset, collate_wav
from model_cantonese import SenseVoiceIpaLora
from utils import (
    ctc_greedy_decode,
    compute_ter,
    compute_token_acc,
    compute_exact_match,
    compute_tone_accuracy_ipa,
    load_vocab,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cantonese_ipa")


def build_model(args, device):
    vocab = json.load(open(args.vocab, encoding="utf-8"))
    model = SenseVoiceIpaLora(
        vocab_size=len(vocab),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    ).to(device)
    return model, vocab


def load_checkpoint(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("model_state_dict", sd)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] 载入 {ckpt_path}  (missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"[ckpt] 未载入键(前10): {missing[:10]}")
    return sd


def run_epoch(model, loader, optimizer, criterion, device, train=True, log_every=50,
              desc="train", max_grad_norm=0.0):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_refs = [], []
    grad_accum = getattr(optimizer, "grad_accum", 1) if optimizer is not None else 1

    with torch.set_grad_enabled(train):
        iterator = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True)
        for step, (wp, wlens, tp, tlens) in enumerate(iterator):
            wp = wp.to(device)
            wlens = wlens.to(device)
            tp = tp.to(device)
            tlens = tlens.to(device)

            log_probs, enc_lens = model(wp, wlens, autocast_bf16=True)
            # CTC: log_probs (T,B,V), targets (B,S)
            loss = criterion(
                log_probs, tp, enc_lens, tlens
            )
            # enc_lens / tlens 已被 CTC 需要为整数；CTCLoss 期望 (T,B,V),(B,S)
            if train:
                (loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    if max_grad_norm and max_grad_norm > 0 and optimizer is not None:
                        _params = [p for g in optimizer.param_groups for p in g["params"]]
                        torch.nn.utils.clip_grad_norm_(_params, max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

            if not train:
                preds = ctc_greedy_decode(log_probs.detach(), blank_id=0)
                for p, r, rl in zip(preds, tp, tlens):
                    all_preds.append(p)
                    all_refs.append(r[: rl.item()].tolist())

            if train:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    if train and optimizer is not None and (step + 1) % grad_accum != 0:
        if max_grad_norm and max_grad_norm > 0:
            _params = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(_params, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    return avg_loss, all_preds, all_refs


@torch.no_grad()
def evaluate(model, loader, device, id2token, ipa2tone, criterion, desc="val"):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_refs = [], []
    with torch.set_grad_enabled(False):
        iterator = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True)
        for wp, wlens, tp, tlens in iterator:
            wp = wp.to(device)
            wlens = wlens.to(device)
            tp = tp.to(device)
            tlens = tlens.to(device)
            log_probs, enc_lens = model(wp, wlens, autocast_bf16=True)
            loss = criterion(log_probs, tp, enc_lens, tlens)
            total_loss += loss.item()
            n += 1
            preds = ctc_greedy_decode(log_probs.detach(), blank_id=0)
            for p, r, rl in zip(preds, tp, tlens):
                all_preds.append(p)
                all_refs.append(r[: rl.item()].tolist())
    avg_loss = total_loss / max(n, 1)
    ter = compute_ter(all_preds, all_refs)
    tok = compute_token_acc(all_preds, all_refs)
    exact = compute_exact_match(all_preds, all_refs)
    tone = compute_tone_accuracy_ipa(all_preds, all_refs, id2token, ipa2tone) if ipa2tone else None
    return {
        "loss": avg_loss,
        "ter": ter,
        "token_acc": tok,
        "exact": exact,
        "tone_acc": tone,
    }, all_preds, all_refs


def write_preds(path, uids, preds, refs, id2token):
    with open(path, "w", encoding="utf-8") as f:
        for u, p, r in zip(uids, preds, refs):
            pt = " ".join(id2token.get(i, "?") for i in p)
            rt = " ".join(id2token.get(i, "?") for i in r)
            f.write(f"{u}\t{rt}\t{pt}\n")


def collect_uids(scp_path):
    uids = []
    with open(scp_path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(" ", 1)
            if p:
                uids.append(p[0])
    return uids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=os.path.join(DATA_DIR, "vocab_cantonese_ipa_combined.json"))
    ap.add_argument("--ipa2tone", default=os.path.join(DATA_DIR, "ipa2tone_cantonese.json"))
    ap.add_argument("--train_scp", default=os.path.join(DATA_DIR, "train_scp"))
    ap.add_argument("--train_text", default=os.path.join(DATA_DIR, "train_text"))
    ap.add_argument("--val_scp", default=os.path.join(DATA_DIR, "val_scp"))
    ap.add_argument("--val_text", default=os.path.join(DATA_DIR, "val_text"))
    ap.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "out_canto"))
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_seconds", type=float, default=20.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0,
                    help="梯度裁剪范数(0=不裁剪); 防止长音频 batch 梯度爆炸/OOM 崩溃")
    ap.add_argument("--warmup_epochs", type=int, default=3,
                    help="LR 线性 warmup 轮数(0=恒定 LR)")
    ap.add_argument("--early_stop_patience", type=int, default=8,
                    help="TER 连续 N 轮无提升则早停(0=关闭)")
    ap.add_argument("--min_epochs", type=int, default=10,
                    help="早停最少训练轮数(避免过早停止)")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--resume", default="")
    ap.add_argument("--eval", action="store_true", help="只评测，不训练")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  amp={args.amp}")

    # 词表 + 调类映射
    token2id, id2token = load_vocab(args.vocab)
    if args.ipa2tone and os.path.exists(args.ipa2tone):
        ipa2tone = json.load(open(args.ipa2tone, encoding="utf-8"))
        print(f"[train] vocab={len(token2id)}  ipa2tone={len(ipa2tone)}")
    else:
        ipa2tone = None
        print(f"[train] vocab={len(token2id)}  ipa2tone=(未提供, 跳过声调ACC)")

    # 数据集
    train_ds = WavDataset(args.train_scp, args.train_text, args.vocab, max_seconds=args.max_seconds)
    val_ds = WavDataset(args.val_scp, args.val_text, args.vocab, max_seconds=args.max_seconds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_wav, num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_wav, num_workers=args.num_workers)

    model, _ = build_model(args, device)

    # CTC loss
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    if args.resume:
        _sd = load_checkpoint(model, args.resume)
        start_epoch = int(_sd.get("epoch", 0)) + 1
        print(f"[resume] 从 epoch {start_epoch} 继续训练")
    else:
        start_epoch = 1

    if args.eval:
        metrics, preds, refs = evaluate(model, val_loader, device, id2token, ipa2tone, criterion,
                                        desc="eval")
        print(f"[eval] TER={metrics['ter']:.4f}  token_acc={metrics['token_acc']:.4f}  "
              f"exact={metrics['exact']:.4f}  tone_acc={metrics['tone_acc']:.4f}")
        uids = collect_uids(args.val_scp)
        write_preds(os.path.join(args.output_dir, "preds_val.txt"), uids, preds, refs, id2token)
        print(f"[eval] 已写出 {os.path.join(args.output_dir, 'preds_val.txt')}")
        return

    # 优化器：只优化可训参数（LoRA + CTC 头）
    opt_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] 可训参数: {sum(p.numel() for p in opt_params):,}")
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer.grad_accum = args.grad_accum

    # LR 调度: 线性 warmup + cosine 衰减(末尾保留 10% 峰值 LR)
    if args.warmup_epochs > 0:
        def lr_lambda(ep):
            ep = ep + 1  # 1-based, 与训练 epoch 对齐
            if ep <= args.warmup_epochs:
                return ep / args.warmup_epochs
            progress = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            progress = min(progress, 1.0)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"[train] LR 调度: warmup={args.warmup_epochs}ep + cosine 衰减")
    else:
        scheduler = None
    # 续训时把调度器快进到对应 epoch, 使 LR 与从头训练一致
    if scheduler is not None and start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"[resume] 调度器快进至 epoch {start_epoch} "
              f"(lr={optimizer.param_groups[0]['lr']:.2e})")

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    _resume_metrics = bool(args.resume) and os.path.exists(metrics_path)
    with open(metrics_path, "a" if _resume_metrics else "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not _resume_metrics:
            w.writerow(["epoch", "train_loss", "val_loss", "val_TER", "val_token_acc",
                        "val_exact", "val_tone_acc", "time_s", "lr"])

    best_ter = float("inf")
    best_tone = -1.0
    epochs_no_improve = 0
    epoch_iter = tqdm(range(start_epoch, args.epochs + 1), desc="Training",
                      unit="epoch", dynamic_ncols=True)
    for epoch in epoch_iter:
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, train=True,
                                  desc=f"E{epoch}/{args.epochs} train",
                                  max_grad_norm=args.max_grad_norm)
        val, preds, refs = evaluate(model, val_loader, device, id2token, ipa2tone, criterion,
                                    desc=f"E{epoch}/{args.epochs} val")
        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_iter.set_postfix(TER=f"{val['ter']:.4f}", tok=f"{val['token_acc']:.3f}",
                               tone=f"{val['tone_acc']:.3f}", lr=f"{cur_lr:.1e}")
        print(f"[epoch {epoch}/{args.epochs}] train_loss={tr_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  TER={val['ter']:.4f}  "
              f"token_acc={val['token_acc']:.4f}  exact={val['exact']:.4f}  "
              f"tone_acc={val['tone_acc']:.4f}  lr={cur_lr:.2e}  ({dt:.0f}s)")

        # 保存
        sd = model.lora_state_dict()
        torch.save({"model_state_dict": sd, "args": vars(args), "epoch": epoch},
                   os.path.join(args.output_dir, "last.pt"))
        improved = val["ter"] < best_ter - 1e-4
        if improved:
            best_ter = val["ter"]
            epochs_no_improve = 0
            torch.save({"model_state_dict": sd, "args": vars(args), "epoch": epoch},
                       os.path.join(args.output_dir, "best.pt"))
            print(f"  -> 保存 best.pt (TER={best_ter:.4f})")
        else:
            epochs_no_improve += 1
        if val["tone_acc"] > best_tone:
            best_tone = val["tone_acc"]
            torch.save({"model_state_dict": sd, "args": vars(args), "epoch": epoch},
                       os.path.join(args.output_dir, "best_tone.pt"))
            print(f"  -> 保存 best_tone.pt (tone_acc={best_tone:.4f})")

        # 写出验证集预测
        uids = collect_uids(args.val_scp)
        write_preds(os.path.join(args.output_dir, "preds_val.txt"), uids, preds, refs, id2token)

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{tr_loss:.4f}", f"{val['loss']:.4f}", f"{val['ter']:.4f}",
                        f"{val['token_acc']:.4f}", f"{val['exact']:.4f}",
                        f"{val['tone_acc']:.4f}", f"{dt:.1f}", f"{cur_lr:.2e}"])

        if scheduler is not None:
            scheduler.step()

        # 早停: 连续 N 轮 TER 无提升且已过最小训练轮数
        if (args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience
                and epoch >= args.min_epochs):
            print(f"\n[early stop] TER 连续 {epochs_no_improve} 轮无提升, "
                  f"在第 {epoch} 轮提前停止")
            break

    print("\n=== 训练完成 ===")
    print(f"最佳 TER={best_ter:.4f}  最佳声调ACC={best_tone:.4f}")
    print(f"权重目录: {args.output_dir}")


if __name__ == "__main__":
    main()
