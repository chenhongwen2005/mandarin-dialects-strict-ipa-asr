# -*- coding: utf-8 -*-
"""准备四川话 LoRA 训练数据: 两个语料集 -> wav.scp / text_ipa / vocab。

输入(解压后):
  data/sichuan_raw/scripted/UTTRANSINFO.txt + WAV/<spk>/<utt>.wav     (6522 单句)
  data/sichuan_raw/conv/TXT/*.txt + WAV/*.wav                          (12 对话, 按时间戳切段)

输出(data/sichuan_ipa/):
  train_scp / train_text / val_scp / val_text   (每行 <uid> <path|IPA音节>)
  vocab_sichuan_ipa_combined.json               (blank=0, unk=1, 其余 IPA 音节)

G2P: src/sichuan_g2p.hanzi_to_ipa_syllables (已知字走 chengdu 方言表, 未知字回退 pinyin->chengdu 音节表)
切段: 对话集用 ffmpeg 按 [start,end] 切 16k 单声道小片段; 过滤非语音标记与 speaker=0。

用法:
  runtime/python.exe src/prepare_sichuan_data.py                 # 全量(含对话切段, 较慢)
  runtime/python.exe src/prepare_sichuan_data.py --no-conv      # 仅脚本集(快, 用于先验证管道)
  runtime/python.exe src/prepare_sichuan_data.py --limit 200    # 各集取前 N 条(调试)
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sichuan_g2p import hanzi_to_ipa_syllables  # noqa: E402

RAW = os.path.join(HERE, "..", "data", "sichuan_raw")
OUT = os.path.join(HERE, "..", "data", "sichuan_ipa")
SEG_DIR = os.path.join(RAW, "conv_segments")
SPLIT_SEED = 42
TRAIN_RATIO = 0.9

_SEG_RE = re.compile(r"\[([\d.]+),([\d.]+)\]\s+(\S+)\s+(\S+)\s+(.*)")

# Chao 调值字母范围(U+02E5~U+02E9), 用于从 IPA 音节末尾提取调值轮廓
_CHAO = set("\u02e5\u02e6\u02e7\u02e8\u02e9")


def parse_scripted():
    rows = []
    p = os.path.join(RAW, "scripted", "UTTRANSINFO.txt")
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            uid = (row.get("UTTRANS_ID") or "").strip()
            spk = (row.get("SPEAKER_ID") or "").strip()
            txt = (row.get("TRANSCRIPTION") or "").strip()
            if not uid or not txt:
                continue
            wav = os.path.join(RAW, "scripted", "WAV", spk, uid)
            if not os.path.exists(wav):
                continue
            rows.append((uid, wav, txt))
    return rows


def _cut(wav_in, start, end, out_wav):
    if os.path.exists(out_wav):
        return True
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", wav_in, "-vn", "-ac", "1", "-ar", "16000", out_wav,
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and os.path.exists(out_wav)


def _vad_runs(wav_path, sr_target=16000, frame_hop=0.010, frame_win=0.025,
              energy_thr=0.004, min_run=0.05, merge_gap=0.25):
    """能量 VAD: 返回 [(start, end), ...] 语音段(秒)。纯 numpy/soundfile, 离线可用。
    阈值偏松(宁多勿漏), 仅用于"延长"切段, 不会裁剪语音。"""
    import numpy as np, soundfile as sf
    data, sr = sf.read(wav_path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != sr_target:
        n = int(round(len(data) * sr_target / sr))
        xp = np.linspace(0.0, 1.0, len(data))
        data = np.interp(np.linspace(0.0, 1.0, n), xp, data).astype(np.float32)
        sr = sr_target
    hop = max(1, int(frame_hop * sr))
    win = max(1, int(frame_win * sr))
    n_frames = 1 + max(0, len(data) - win) // hop if len(data) >= win else 1
    energy = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = data[i * hop:i * hop + win]
        energy[i] = np.sqrt(np.mean(seg * seg) + 1e-10)
    k = max(1, int(0.02 / frame_hop))
    if k > 1:
        energy = np.convolve(energy, np.ones(k) / k, mode="same")
    thr = max(float(energy_thr), float(np.percentile(energy, 8)) * 2.0)
    speech = energy > thr
    runs = []
    i, N = 0, len(speech)
    while i < N:
        if speech[i]:
            j = i
            while j < N and speech[j]:
                j += 1
            runs.append([i * frame_hop, j * frame_hop])
            i = j
        else:
            i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < merge_gap:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r[:])
    merged = [r for r in merged if r[1] - r[0] >= min_run]
    return merged


# 尾部固定余量: 补回被语料 end 时间戳截掉的尾字
_END_MARGIN = 0.45


def _snap_boundaries(utts, runs):
    """utts: 已按 s 排序的 [(s,e,spk,trans),...]; runs: VAD 语音段(仅用于延长)。
    策略: 起点略前探; 终点 = 原 end + 余量(并允许 VAD 在语音续接时进一步延长);
    终点封顶在下一句起点之前, 绝不裁剪语音、绝不吞下一句。"""
    res = []
    prev_end = -1.0
    for idx, (s, e, spk, trans) in enumerate(utts):
        ns = s - 0.05                      # 轻微前探, 补 onset
        ne = e + _END_MARGIN               # 扩展尾部, 修句末截字
        # VAD 仅用于"延长": 若 e 之后语音仍在继续, 延长到该语音段末尾
        for (rs, re) in runs:
            if rs <= e <= re:              # 覆盖 e 的语音段 -> 延长到其末
                ne = max(ne, re)
                break
            if e < rs <= e + 0.30:         # e 之后紧接另一段语音 -> 桥接延长
                ne = max(ne, re)
                break
        ns = max(ns, prev_end + 0.02)      # 不与上一句重叠
        if idx + 1 < len(utts):
            nxt = utts[idx + 1][0]
            ne = min(ne, nxt - 0.02)       # 不越界到下一句(含其前静音)
        if ne <= ns:
            ns, ne = s, e                  # 退化(相邻重叠)则退回原始时间戳
        res.append((ns, ne))
        prev_end = ne
    return res


def parse_conv(segment=True, limit=None, vad=False):
    rows = []
    txtdir = os.path.join(RAW, "conv", "TXT")
    wavdir = os.path.join(RAW, "conv", "WAV")
    if segment:
        os.makedirs(SEG_DIR, exist_ok=True)
    count = 0
    for tf in sorted(os.listdir(txtdir)):
        if not tf.endswith(".txt"):
            continue
        base = tf[:-4]
        wav_in = os.path.join(wavdir, base + ".wav")
        if not os.path.exists(wav_in):
            continue
        if not vad:
            # 原逻辑: 逐行按语料时间戳切
            with open(os.path.join(txtdir, tf), encoding="utf-8") as f:
                for ln in f:
                    if not ln.startswith("["):
                        continue
                    m = _SEG_RE.match(ln.rstrip("\n"))
                    if not m:
                        continue
                    start, end, spk, _gender, trans = m.groups()
                    trans = trans.strip()
                    if not trans or trans.startswith("["):
                        continue
                    if spk == "0":
                        continue
                    try:
                        s, e = float(start), float(end)
                    except ValueError:
                        continue
                    if e - s < 0.3:
                        continue
                    uid = f"{base}_{int(round(s * 100)):07d}"
                    if segment:
                        out_wav = os.path.join(SEG_DIR, uid + ".wav")
                        if not _cut(wav_in, s, e, out_wav):
                            continue
                        rows.append((uid, out_wav, trans))
                    else:
                        rows.append((uid, f"{wav_in}@{s:.2f}-{e:.2f}", trans))
                    count += 1
                    if limit and count >= limit:
                        return rows
            continue
        # --- VAD 对齐模式: 先收集该文件全部 utt, 再统一对齐边界 ---
        utts = []
        with open(os.path.join(txtdir, tf), encoding="utf-8") as f:
            for ln in f:
                if not ln.startswith("["):
                    continue
                m = _SEG_RE.match(ln.rstrip("\n"))
                if not m:
                    continue
                start, end, spk, _gender, trans = m.groups()
                trans = trans.strip()
                if not trans or trans.startswith("["):
                    continue
                if spk == "0":
                    continue
                try:
                    s, e = float(start), float(end)
                except ValueError:
                    continue
                if e - s < 0.3:
                    continue
                utts.append((s, e, spk, trans))
        if not utts:
            continue
        try:
            runs = _vad_runs(wav_in)
        except Exception:
            runs = []
        snapped = _snap_boundaries(utts, runs) if runs else [(s, e) for (s, e, _, _) in utts]
        for (s, e, spk, trans), (ns, ne) in zip(utts, snapped):
            uid = f"{base}_{int(round(ns * 100)):07d}_{count:05d}"
            if segment:
                out_wav = os.path.join(SEG_DIR, uid + ".wav")
                if not _cut(wav_in, ns, ne, out_wav):
                    continue
                rows.append((uid, out_wav, trans))
            else:
                rows.append((uid, f"{wav_in}@{ns:.2f}-{ne:.2f}", trans))
            count += 1
            if limit and count >= limit:
                return rows
    return rows


def build(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # G2P
    data = []          # (uid, wav, ipa_str)
    all_tokens = set()
    skipped = 0
    for uid, wav, txt in rows:
        syl, _un = hanzi_to_ipa_syllables(txt)
        if not syl:
            skipped += 1
            continue
        ipa = " ".join(syl)
        for t in syl:
            all_tokens.add(t)
        data.append((uid, wav, ipa))
    print(f"  G2P 后有效样本: {len(data)} (跳过空结果 {skipped})  不同 IPA 音节: {len(all_tokens)}")

    # 划分
    random.seed(SPLIT_SEED)
    idx = list(range(len(data)))
    random.shuffle(idx)
    n_train = int(len(data) * TRAIN_RATIO)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    def write(name, idlist):
        with open(os.path.join(out_dir, name + "_scp"), "w", encoding="utf-8") as fs, \
             open(os.path.join(out_dir, name + "_text"), "w", encoding="utf-8") as ft:
            for i in idlist:
                uid, wav, ipa = data[i]
                fs.write(f"{uid} {wav}\n")
                ft.write(f"{uid} {ipa}\n")

    write("train", train_idx)
    write("val", val_idx)

    # 词表(基于全部数据)
    vocab = {"<blank>": 0, "<unk>": 1}
    for i, tok in enumerate(sorted(all_tokens), start=2):
        vocab[tok] = i
    with open(os.path.join(out_dir, "vocab_sichuan_ipa_combined.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    print(f"  词表大小: {len(vocab)} (train={len(train_idx)} val={len(val_idx)})")

    # 声调映射: 取每个 IPA 音节末尾的 Chao 调值轮廓 -> 调类 id(同轮廓=同调类)
    contours = {}
    cid = 1
    ipa2tone = {}
    for tok in sorted(all_tokens):
        i = len(tok)
        while i > 0 and tok[i - 1] in _CHAO:
            i -= 1
        contour = tok[i:]
        if contour not in contours:
            contours[contour] = cid
            cid += 1
        ipa2tone[tok] = contours[contour]
    with open(os.path.join(out_dir, "ipa2tone_sichuan.json"), "w", encoding="utf-8") as f:
        json.dump(ipa2tone, f, ensure_ascii=False, indent=1)
    print(f"  调类数: {len(contours)} (ipa2tone_sichuan.json)")
    print(f"  写出 -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-conv", action="store_true", help="仅脚本集(不切对话)")
    ap.add_argument("--vad", action="store_true", help="对话集用能量 VAD 对齐切段边界(修句末截字)")
    ap.add_argument("--limit", type=int, default=None, help="每个语料集最多取 N 条(调试)")
    args = ap.parse_args()

    print("解析脚本集...")
    rows = parse_scripted()
    if args.limit:
        rows = rows[: args.limit]
    print(f"  脚本集样本: {len(rows)}")

    if not args.no_conv:
        mode = "VAD对齐" if args.vad else "按时间戳"
        print(f"解析对话集({mode} ffmpeg 切段, 较慢)...")
        conv = parse_conv(segment=True, limit=args.limit, vad=args.vad)
        print(f"  对话集有效段: {len(conv)}")
        rows += conv
    else:
        print("  跳过对话集(--no-conv)")

    print(f"合计样本: {len(rows)}")
    build(rows, OUT)


if __name__ == "__main__":
    main()
