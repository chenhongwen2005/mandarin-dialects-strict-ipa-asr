# -*- coding: utf-8 -*-
"""粤语 LoRA 模型推理：语音波形 -> 严式 IPA（含调值字母）。

复用训练管线一致的音频加载与 CTC 贪心解码，保证训练/推理特征口径一致。

用法（在 D:/mandarin-ipa-asr-cu128 下用 Git Bash）:
    # 单条 wav
    runtime/python.exe src/infer_cantonese_lora.py --wav path.wav --ckpt out_canto/cantonese.pt

    # 批量（每行一个 wav 绝对路径）
    runtime/python.exe src/infer_cantonese_lora.py --list wavs.txt --out preds.txt --ckpt out_canto/cantonese.pt

    # 评测（与训练一致口径：TER / token_acc / tone_acc / exact）
    runtime/python.exe src/infer_cantonese_lora.py --eval --ckpt out_canto/cantonese.pt

未传 --ckpt（或文件不存在）时退化为随机初始化权重，仅用于验证链路。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

from dataset import WavDataset, collate_wav
from model_cantonese import SenseVoiceIpaLora
from utils import (
    load_audio,
    load_vocab,
    ctc_greedy_decode,
    compute_ter,
    compute_token_acc,
    compute_exact_match,
    compute_tone_accuracy_ipa,
)
from train_cantonese_lora import collect_uids, write_preds

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cantonese_ipa")


def load_checkpoint(model, ckpt_path):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[ckpt] 未找到权重 {ckpt_path} -> 随机初始化（仅验证链路）")
        return
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("model_state_dict", sd)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] 载入 {ckpt_path}  (missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"[ckpt] 未载入键(前10): {missing[:10]}")


def ids_to_ipa(ids, id2token):
    """token id 序列 -> 空格分隔的 IPA 音节串，跳过 <unk>(id=1)。"""
    return " ".join(id2token.get(i, "?") for i in ids if i != 1)


@torch.no_grad()
def decode_batch(model, wav_paths, device, id2token):
    """把若干 wav 组成一个 batch 跑前向，返回 IPA 字符串列表。"""
    waves, lengths = [], []
    for p in wav_paths:
        w = load_audio(p)
        if w.dim() == 1:
            w = w.unsqueeze(0)  # (1, T)
        waves.append(w[0])
        lengths.append(w.shape[-1])

    max_len = max(lengths)
    speech = torch.zeros(len(waves), max_len)
    slens = torch.tensor(lengths, dtype=torch.long)
    for i, w in enumerate(waves):
        speech[i, : w.shape[0]] = w

    speech = speech.to(device)
    slens = slens.to(device)
    log_probs, _ = model(speech, slens, autocast_bf16=(device == "cuda"))
    preds = ctc_greedy_decode(log_probs.detach(), blank_id=0)
    return [ids_to_ipa(pr, id2token) for pr in preds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--vocab", default=os.path.join(DATA_DIR, "vocab_cantonese_ipa_combined.json"))
    ap.add_argument("--ipa2tone", default=os.path.join(DATA_DIR, "ipa2tone_cantonese.json"))
    ap.add_argument("--wav", default="")
    ap.add_argument("--list", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--val_scp", default=os.path.join(DATA_DIR, "val_scp"))
    ap.add_argument("--val_text", default=os.path.join(DATA_DIR, "val_text"))
    ap.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "out_canto"))
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[infer] device={device}")

    token2id, id2token = load_vocab(args.vocab)
    model = SenseVoiceIpaLora(
        vocab_size=len(token2id), lora_rank=args.lora_rank, lora_alpha=args.lora_alpha
    ).to(device)
    load_checkpoint(model, args.ckpt)
    model.eval()
    if args.ipa2tone and os.path.exists(args.ipa2tone):
        ipa2tone = json.load(open(args.ipa2tone, encoding="utf-8"))
    else:
        ipa2tone = None

    if args.eval:
        val_ds = WavDataset(args.val_scp, args.val_text, args.vocab, max_seconds=20.0)
        loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_wav, num_workers=0,
        )
        all_preds, all_refs = [], []
        with torch.no_grad():
            for wp, wlens, tp, tlens in loader:
                wp, wlens = wp.to(device), wlens.to(device)
                tp, tlens = tp.to(device), tlens.to(device)
                lp, _ = model(wp, wlens, autocast_bf16=(device == "cuda"))
                preds = ctc_greedy_decode(lp, blank_id=0)
                for p, r, rl in zip(preds, tp, tlens):
                    all_preds.append(p)
                    all_refs.append(r[: rl.item()].tolist())
        ter = compute_ter(all_preds, all_refs)
        tok = compute_token_acc(all_preds, all_refs)
        exact = compute_exact_match(all_preds, all_refs)
        tone = compute_tone_accuracy_ipa(all_preds, all_refs, id2token, ipa2tone) if ipa2tone else None
        print(f"[eval] TER={ter:.4f}  token_acc={tok:.4f}  exact={exact:.4f}  tone_acc={tone:.4f}")
        uids = collect_uids(args.val_scp)
        write_preds(os.path.join(args.output_dir, "preds_val.txt"), uids, all_preds, all_refs, id2token)
        print(f"[eval] 已写出 {os.path.join(args.output_dir, 'preds_val.txt')}")
        return

    if args.list:
        paths = [ln.strip() for ln in open(args.list, encoding="utf-8") if ln.strip()]
        outs = decode_batch(model, paths, device, id2token)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                for p, o in zip(paths, outs):
                    f.write(f"{p}\t{o}\n")
            print(f"[infer] 已写出 {len(outs)} 条 -> {args.out}")
        else:
            for p, o in zip(paths, outs):
                print(f"{p}\t{o}")
        return

    if args.wav:
        out = decode_batch(model, [args.wav], device, id2token)[0]
        print(out)
        return

    print("请在 --wav / --list / --eval 中选一种模式")


if __name__ == "__main__":
    main()
