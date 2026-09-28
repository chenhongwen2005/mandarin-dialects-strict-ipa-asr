# -*- coding: utf-8 -*-
"""推理与评测：单条音频解码、批量评测、逐音节声调评分与参考文本比对。

用法:
  # 单条音频 -> 打印 IPA 转写
  python infer.py --wav audio.wav --ckpt checkpoints/best.pt

  # 批量评测（默认读 val_scp / val_text）-> TER / ACC / TACC / 声调准确率
  python infer.py --eval --ckpt checkpoints/best.pt --limit 0

  # 单条 + 提供参考文本 -> 逐音节声调对齐
  python infer.py --wav audio.wav --ckpt checkpoints/best.pt --ref "ipa syllables space separated"
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from utils import (
    ctc_greedy_decode,
    ctc_beam_search,
    build_ngram_lm,
    compute_ter,
    compute_exact_match,
    compute_token_acc,
    load_vocab,
    load_audio,
    build_model,
    score_utterance,
    compute_tone_accuracy_ipa,
    resolve_local_path,
)

MODEL_DIR = "iic/SenseVoiceSmall"
DEFAULTS = dict(
    vocab_path="vocab/vocab_mandarin_ipa_combined.json",
    ipa2tone_path="vocab/vocab_mandarin_ipa_tone_combined.json",
    model_dir=MODEL_DIR,
    ckpt="",
    wav="",
    eval=False,
    val_scp="",
    val_text="",
    limit=0,
    ref="",
    beam=0,
    lm_weight=0.0,
    lm_order=2,
    lm_data="",
)


def build(vocab_path: str, ckpt: str, model_dir: str):
    return build_model(vocab_path, ckpt, model_dir, freeze_encoder=True)


@torch.no_grad()
def decode_wav(model, dev, wav_path: str, beam: int = 0, lm=None,
              id2token=None, lm_weight: float = 0.0, beta: float = 0.0):
    w = load_audio(wav_path).unsqueeze(0).to(dev)
    sl = torch.tensor([w.shape[1]], device=dev)
    lp, _ = model(w, sl, autocast_bf16=True)
    if beam and beam > 1:
        logp = lp[:, 0, :].cpu().numpy()
        return ctc_beam_search(logp, beam_size=beam, lm=lm, id2token=id2token,
                              alpha=lm_weight, beta=beta)
    return ctc_greedy_decode(lp)[0]


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", action="store_true", default=v)
        else:
            typ = str if isinstance(v, str) else int
            ap.add_argument(f"--{k}", type=typ, default=v)
    args = ap.parse_args()
    for _k in ("vocab_path", "ipa2tone_path", "ckpt", "wav", "val_scp", "val_text", "lm_data"):
        setattr(args, _k, resolve_local_path(getattr(args, _k)))
    if args.model_dir and not os.path.isabs(args.model_dir) and os.path.isdir(resolve_local_path(args.model_dir)):
        args.model_dir = resolve_local_path(args.model_dir)

    model, token2id, dev = build(args.vocab_path, args.ckpt, args.model_dir)
    id2tok = {v: k for k, v in token2id.items()}
    ipa2tone = json.load(open(args.ipa2tone_path, encoding="utf-8")) if args.ipa2tone_path else {}

    lm = None
    if args.lm_weight > 0 and args.beam > 1 and args.lm_data:
        seqs = []
        for line in open(args.lm_data, encoding="utf-8"):
            parts = line.strip().split()
            if len(parts) > 1:
                seqs.append(parts[1:])
        lm = build_ngram_lm(seqs, order=args.lm_order)
        print(f"[lm] {args.lm_order} 元 LM 已构建: 句数={len(seqs)} 词表={lm['vocab_size']}")
    deco = "beam=" + str(args.beam) if args.beam > 1 else "greedy"
    if lm is not None:
        deco += f" + LM(alpha={args.lm_weight})"
    print(f"[decode] {deco}")

    if args.wav:
        ids = decode_wav(model, dev, args.wav, beam=args.beam, lm=lm,
                        id2token=id2tok, lm_weight=args.lm_weight)
        print("\n==== 单条音频 ====")
        print("wav :", args.wav)
        print("IPA :", " ".join(id2tok.get(i, "?") for i in ids))
        if args.ref:
            res = score_utterance(ids, args.ref.split(), id2tok, ipa2tone)
            print("参考 :", res["ref_str"])
            mark = {"correct": "✓", "tone_wrong": "调错", "wrong": "✗",
                    "extra": "多读", "missing": "漏读"}
            print("\n--- 逐音节对齐 (预测 / 参考 / 状态) ---")
            for pt, rt, st, lb in res["states"]:
                print(f"  {pt:10s} {rt:10s} {mark[st]}")
            print(f"\n音节准确率: {res['syllable_acc']:.4f}  ({res['n_pred']} 预测 / {res['n_ref']} 参考)")
            print(f"声调准确率: {res['tone_acc']:.4f}")
        return

    # 批量评测
    if not args.val_scp or not args.val_text:
        print("批量评测需要 --val_scp 与 --val_text")
        return
    scp = {}
    for line in open(args.val_scp, encoding="utf-8"):
        p = line.strip().split(" ", 1)
        if len(p) == 2:
            scp[p[0]] = p[1]
    pairs = []
    for line in open(args.val_text, encoding="utf-8"):
        p = line.strip().split(" ", 1)
        if len(p) == 2 and p[0] in scp:
            pairs.append((p[0], scp[p[0]], p[1].split()))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[eval] {len(pairs)} 条")

    preds, refs, samples = [], [], []
    for i, (uid, wav, ref_toks) in enumerate(pairs):
        ids = decode_wav(model, dev, wav, beam=args.beam, lm=lm,
                         id2token=id2tok, lm_weight=args.lm_weight)
        preds.append(ids)
        refs.append([token2id.get(t, 1) for t in ref_toks])
        if i < 30:
            samples.append((" ".join(id2tok.get(x, "?") for x in ids), " ".join(ref_toks)))

    ter = compute_ter(preds, refs)
    acc = compute_exact_match(preds, refs)
    tacc = compute_token_acc(preds, refs)
    tone = compute_tone_accuracy_ipa(preds, refs, id2tok, ipa2tone)

    print("\n==== 样例 (预测 vs 参考) ====")
    for p, r in samples[:20]:
        print("  预测:", p)
        print("  参考:", r)
    print("\n==== 整体指标 (n=%d) ====" % len(preds))
    print(f"TER  (音节错误率)  : {ter:.4f}")
    print(f"ACC  (整句匹配)    : {acc:.4f}")
    print(f"TACC (token 正确率): {tacc:.4f}")
    print(f"声调准确率         : {tone:.4f}")


if __name__ == "__main__":
    main()
