# -*- coding: utf-8 -*-
"""工具函数：CTC 解码、指标计算、Levenshtein 对齐、n-gram 语言模型与 beam search。"""

from typing import Dict, List, Optional, Tuple

import json
import math
import os

import numpy as np
import torch


def ctc_greedy_decode(
    log_probs: torch.Tensor, blank_id: int = 0
) -> List[List[int]]:
    """标准 CTC 贪心解码：先去 blank，再合并相邻重复 token。"""
    indices = log_probs.argmax(dim=-1)  # (T, B)
    results = []
    batch_size = indices.shape[1]
    for b in range(batch_size):
        seq = indices[:, b].tolist()
        no_blank = [t for t in seq if t != blank_id]
        decoded = []
        prev = None
        for t in no_blank:
            if t != prev:
                decoded.append(t)
            prev = t
        results.append(decoded)
    return results


def compute_ter(
    predictions: List[List[int]], references: List[List[int]]
) -> float:
    import editdistance

    total_dist = 0
    total_ref_len = 0
    for pred, ref in zip(predictions, references):
        total_dist += editdistance.eval(pred, ref)
        total_ref_len += len(ref)
    return total_dist / total_ref_len if total_ref_len else 0.0


def compute_exact_match(
    predictions: List[List[int]], references: List[List[int]]
) -> float:
    """整句精确匹配率（预测 token 序列完全等于参考 token 序列的样本占比）。"""
    if not predictions:
        return 0.0
    exact = sum(1 for p, r in zip(predictions, references) if p == r)
    return exact / len(predictions)


def compute_token_acc(
    predictions: List[List[int]], references: List[List[int]]
) -> float:
    """归一化 token 正确率：1 - 逐样本编辑距离 / max(预测长, 参考长)，再取平均。"""
    import editdistance

    if not predictions:
        return 0.0
    tot = 0.0
    for p, r in zip(predictions, references):
        d = editdistance.eval(p, r)
        denom = max(len(p), len(r), 1)
        tot += 1.0 - d / denom
    return tot / len(predictions)


def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    token2id = vocab
    id2token = {v: k for k, v in vocab.items()}
    return token2id, id2token


def tone_class_of(token: str, ipa2tone: Optional[Dict[str, int]] = None):
    """返回 token 的声调类别（1-5，轻声=5；无调则 None）。

    IPA 音节通过 ipa2tone 映射表查询；若为 None 则回退到取 token 末位阿拉伯数字。
    """
    if ipa2tone is not None:
        t = ipa2tone.get(token)
        return int(t) if isinstance(t, (int, float)) else None
    if token and token[-1].isdigit():
        return int(token[-1])
    return None


def levenshtein_align(pred, ref):
    """Levenshtein 对齐（带回溯）。op 语义（相对预测 pred / 参考 ref）：
    match: 两者相同；sub: 不同；del: 仅消费 pred（多读）；ins: 仅消费 ref（漏读）。"""
    n, m = len(pred), len(ref)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bp = [[0] * (m + 1) for _ in range(n + 1)]  # 0=match 1=sub 2=ins 3=del
    for i in range(1, n + 1):
        dp[i][0] = i
        bp[i][0] = 3
    for j in range(1, m + 1):
        dp[0][j] = j
        bp[0][j] = 2
    for i in range(1, n + 1):
        pi = pred[i - 1]
        for j in range(1, m + 1):
            if pi == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bp[i][j] = 0
            else:
                best = dp[i - 1][j] + 1
                b = 3
                if dp[i][j - 1] + 1 < best:
                    best = dp[i][j - 1] + 1
                    b = 2
                if dp[i - 1][j - 1] + 1 < best:
                    best = dp[i - 1][j - 1] + 1
                    b = 1
                dp[i][j] = best
                bp[i][j] = b
    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        b = bp[i][j]
        if b == 0:
            ops.append(("match", i - 1, j - 1))
            i -= 1
            j -= 1
        elif b == 1:
            ops.append(("sub", i - 1, j - 1))
            i -= 1
            j -= 1
        elif b == 3:
            ops.append(("del", i - 1, -1))
            i -= 1
        else:
            ops.append(("ins", -1, j - 1))
            j -= 1
    ops.reverse()
    return ops


def classify_tone_state(pred_tok, ref_tok, ipa2tone=None):
    """将一对 (预测 token, 参考 token) 分类为声调评分状态。

    返回 (state, label):
      correct   '✓'    完全正确（音节 + 声调）
      tone_wrong '调错' 同基音节、不同调
      wrong     '✗'    音节不同
      extra     '多读' 预测多出
      missing   '漏读' 参考多出
    """
    if pred_tok is None:
        return "missing", "漏读"
    if ref_tok is None:
        return "extra", "多读"
    if pred_tok == ref_tok:
        return "correct", "✓"
    # 基音节 = 去掉末位声调标记（IPA 末位为调值字母；pinyin 末位为数字）
    if ipa2tone is not None:
        pb = pred_tok
        rb = ref_tok
    else:
        pb = pred_tok[:-1] if (pred_tok and pred_tok[-1].isdigit()) else pred_tok
        rb = ref_tok[:-1] if (ref_tok and ref_tok[-1].isdigit()) else ref_tok
    if pb == rb:
        return "tone_wrong", "调错"
    return "wrong", "✗"


def score_utterance(
    pred_ids: List[int],
    ref_tokens: List[str],
    id2token: Dict[int, str],
    ipa2tone: Optional[Dict[str, int]] = None,
) -> Dict:
    """单条语音的逐音节声调评分。

    返回 dict: {states, tone_acc, syllable_acc, n_ref, n_pred, pred_str, ref_str}
    states 每项 = (pred_tok, ref_tok, state, label)
    """
    pred_tokens = [id2token.get(i, "?") for i in pred_ids]
    ref_seq = list(ref_tokens)
    ops = levenshtein_align(pred_tokens, ref_seq)
    states = []
    tone_correct = 0
    tone_total = 0
    syl_correct = 0
    for op, ip, ir in ops:
        if op in ("match", "sub"):
            pt, rt = pred_tokens[ip], ref_seq[ir]
            st, lb = classify_tone_state(pt, rt, ipa2tone)
            states.append((pt, rt, st, lb))
            if tone_class_of(rt, ipa2tone) is not None:
                tone_total += 1
                if st == "correct":
                    tone_correct += 1
            if st in ("correct", "tone_wrong"):
                syl_correct += 1
        elif op == "del":
            pt = pred_tokens[ip]
            st, lb = classify_tone_state(pt, None, ipa2tone)
            states.append((pt, "—", st, lb))
        else:
            rt = ref_seq[ir]
            st, lb = classify_tone_state(None, rt, ipa2tone)
            states.append(("—", rt, st, lb))
            if tone_class_of(rt, ipa2tone) is not None:
                tone_total += 1
    n_ref = len(ref_seq)
    return {
        "states": states,
        "tone_acc": (tone_correct / tone_total) if tone_total else 1.0,
        "syllable_acc": (syl_correct / n_ref) if n_ref else 1.0,
        "n_ref": n_ref,
        "n_pred": len(pred_tokens),
        "pred_str": " ".join(pred_tokens),
        "ref_str": " ".join(ref_seq),
    }


def compute_tone_accuracy_ipa(
    preds: List[List[int]],
    refs: List[List[int]],
    id2token: Dict[int, str],
    ipa2tone: Dict[str, int],
) -> float:
    """用 ipa2tone（IPA 音节 -> 调类 1-5）逐位比声调，标签无关的公平指标。"""
    correct = 0
    total = 0
    for pred, ref in zip(preds, refs):
        mn = min(len(pred), len(ref))
        for i in range(mn):
            pv = ipa2tone.get(id2token.get(pred[i], ""))
            rv = ipa2tone.get(id2token.get(ref[i], ""))
            if pv is None or rv is None:
                continue
            if pv == rv:
                correct += 1
            total += 1
    return correct / total if total else 0.0


# ── n-gram 语言模型（纯 Python，可选用于 beam search shallow fusion）─────────────
def build_ngram_lm(
    token_sequences, order: int = 2, smooth: float = 1.0
):
    from collections import Counter, defaultdict

    ngrams = defaultdict(Counter)
    ctx_count = defaultdict(int)
    unigram = Counter()
    for seq in token_sequences:
        for t in seq:
            unigram[t] += 1
        padded = ["<s>"] * (order - 1) + seq + ["</s>"]
        for n in range(1, order):
            for i in range(len(padded) - n):
                ctx = tuple(padded[i : i + n])
                nxt = padded[i + n]
                ngrams[ctx][nxt] += 1
                ctx_count[ctx] += 1
    return {
        "ngrams": ngrams,
        "ctx_count": ctx_count,
        "unigram": unigram,
        "vocab_size": len(unigram),
        "total_uni": sum(unigram.values()),
        "order": order,
        "smooth": smooth,
    }


def lm_logprob(lm, history, token):
    order = lm["order"]
    s = lm["smooth"]
    for n in range(min(order - 1, len(history)), 0, -1):
        ctx = tuple(history[-n:])
        cnts = lm["ngrams"].get(ctx)
        if cnts:
            denom = lm["ctx_count"][ctx] + s * lm["vocab_size"]
            return math.log((cnts.get(token, 0) + s) / denom)
    denom = lm["total_uni"] + s * lm["vocab_size"]
    return math.log((lm["unigram"].get(token, 0) + s) / denom)


def log_sum_exp(a, b):
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


# ── 音频加载（与训练一致：原始波形，不做峰值归一化）──────────────────────────────
import subprocess

FFMPEG = "ffmpeg"
TARGET_SR = 16000


def load_audio(path: str, sr: int = TARGET_SR) -> torch.Tensor:
    """加载音频为单声道 16kHz 张量。

    与训练数据管线 WavDataset.load_audio 保持完全一致：返回原始波形，
    不做峰值（max-abs）归一化。归一化会改变 WavFrontend 输出的 fbank 能量尺度，
    导致训练/推理特征分布不一致、识别率下降。
    """
    low = path.lower()
    if low.endswith((".wav", ".flac", ".ogg")):
        try:
            w, s = torchaudio.load(path)
            if s != sr:
                w = torchaudio.functional.resample(w, s, sr)
            if w.shape[0] > 1:
                w = w.mean(0, keepdim=True)
            return w.squeeze(0)
        except Exception:
            pass
    # 退回 ffmpeg 解码（mp3 等）
    try:
        out = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
             "-f", "f32le", "-ar", str(sr), "-ac", "1", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        if out.returncode == 0 and out.stdout:
            return torch.from_numpy(
                np.frombuffer(out.stdout, dtype=np.float32).copy()
            )
    except Exception:
        pass
    return torch.zeros(sr, dtype=torch.float32)


# 兼容旧版权重中残留的头命名 -> 当前 ctc_head.*
_COMPAT_HEAD_PREFIXES = ("sichuan_ctc_head.", "ipa_ctc_head.", "mandarin_ctc_head.")


def compat_remap_state(state: dict) -> dict:
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


def build_model(
    vocab_path: str,
    ckpt: str = "",
    model_dir: str = "iic/SenseVoiceSmall",
    freeze_encoder: bool = True,
    device: str = None,
):
    """构建 SenseVoiceIpa 并载入微调权重（自动兼容旧命名）。

    返回 (model, token2id, device)。
    """
    from model import SenseVoiceIpa

    token2id, _ = load_vocab(vocab_path)
    model = SenseVoiceIpa(
        vocab_size=len(token2id), model_dir=model_dir, freeze_encoder=freeze_encoder
    )
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = sd.get("model_state_dict", sd)
        state = compat_remap_state(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ckpt] loaded {ckpt}  (missing={len(missing)}, unexpected={len(unexpected)})")
        if missing:
            print(f"[ckpt] 未载入键: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    else:
        print("[ckpt] 未找到权重 -> 随机初始化（仅验证链路）")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    return model, token2id, dev


def ctc_beam_search(
    logp: np.ndarray,
    beam_size: int = 10,
    blank: int = 0,
    topk: int = 30,
    lm=None,
    id2token=None,
    alpha: float = 0.0,
    beta: float = 0.0,
    return_topk: int = None,
):
    """CTC 前缀 beam search（log 空间），可选语言模型 shallow fusion。

    logp: (T, V) numpy float。
    lm: build_ngram_lm 返回值；id2token: {id:str} 用于将 token id 转成 LM 字符串。
    alpha: LM 权重（0 即无 LM）；beta: 长度奖励（按 token 数）。
    返回解码出的 token id 列表。
    """
    T, V = logp.shape
    topk = min(topk, V)
    eff = beam_size if return_topk is None else max(beam_size, return_topk)
    beam = {(): {"pnb": -float("inf"), "pb": 0.0, "hist": []}}
    for t in range(T):
        lp = logp[t]
        top_idx = set(np.argpartition(lp, -topk)[-topk:].tolist())
        top_idx.add(blank)
        next_beam = {}
        for prefix, st in beam.items():
            Pnb, Pb, hist = st["pnb"], st["pb"], st["hist"]
            p_blank = lp[blank]
            nb = next_beam.setdefault(
                prefix, {"pnb": -float("inf"), "pb": -float("inf"), "hist": hist}
            )
            nb["pb"] = log_sum_exp(
                nb["pb"], log_sum_exp(Pb + p_blank, Pnb + p_blank)
            )
            last = prefix[-1] if prefix else -1
            for l in top_idx:
                if l == blank:
                    continue
                p = lp[l]
                new_prefix = prefix + (l,)
                nb2 = next_beam.setdefault(
                    new_prefix,
                    {"pnb": -float("inf"), "pb": -float("inf"), "hist": hist + [l]},
                )
                lm_s = 0.0
                if lm is not None and id2token is not None:
                    hstr = [id2token.get(h, "") for h in hist]
                    lm_s = alpha * lm_logprob(lm, hstr, id2token.get(l, ""))
                if l == last:
                    nb2["pnb"] = log_sum_exp(nb2["pnb"], (Pb + p) + lm_s)
                    nb["pnb"] = log_sum_exp(nb["pnb"], Pnb + p)
                else:
                    nb2["pnb"] = log_sum_exp(nb2["pnb"], (Pnb + p) + lm_s)
                    nb2["pnb"] = log_sum_exp(nb2["pnb"], (Pb + p) + lm_s)
        items = sorted(
            next_beam.items(),
            key=lambda kv: log_sum_exp(kv[1]["pnb"], kv[1]["pb"])
            + beta * len(kv[0]),
            reverse=True,
        )
        beam = {k: v for k, v in items[:eff]}
    if not beam:
        return []
    if return_topk is not None:
        scored = [
            (list(k), log_sum_exp(v["pnb"], v["pb"]) + beta * len(k))
            for k, v in beam.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:return_topk]
    best = max(
        beam.items(),
        key=lambda kv: log_sum_exp(kv[1]["pnb"], kv[1]["pb"]) + beta * len(kv[0]),
    )
    return list(best[0])
