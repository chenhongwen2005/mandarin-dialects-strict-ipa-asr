# -*- coding: utf-8 -*-
"""音频数据集：从 text（uid 标签序列）+ scp（uid 音频路径）读取，支持 wav/flac/ogg/mp3。

数据格式（训练/评测通用）:
  text : "<uid> <tok1> <tok2> ... <tokN>"   每行一条，token 为 IPA 音节
  scp  : "<uid> <audio_absolute_path>"       每行一条，路径可为 wav 或 mp3
两个文件的 uid 集合须一致。
"""

import json
import os
import subprocess

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

# ffmpeg 可执行文件路径（用于解码 mp3）；如不在 PATH 中请改为绝对路径或设为 "ffmpeg"
FFMPEG = "ffmpeg"
TARGET_SR = 16000


def decode_mp3(path: str) -> Optional[np.ndarray]:
    """用 ffmpeg 将压缩音频解码为 16kHz 单声道 float32 PCM。"""
    try:
        out = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
             "-f", "f32le", "-ar", str(TARGET_SR), "-ac", "1", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if out.returncode != 0 or not out.stdout:
            return None
        return np.frombuffer(out.stdout, dtype=np.float32)
    except Exception:
        return None


def load_audio(path: str) -> Optional[np.ndarray]:
    low = path.lower()
    if low.endswith((".wav", ".flac", ".ogg")):
        try:
            w, s = torchaudio.load(path)
            if s != TARGET_SR:
                w = torchaudio.functional.resample(w, s, TARGET_SR)
            if w.shape[0] > 1:
                w = w.mean(0)
            return w.squeeze(0).numpy().astype(np.float32)
        except Exception:
            return None
    return decode_mp3(path)


class WavDataset(Dataset):
    def __init__(self, scp_path, text_path, vocab_path, max_seconds=20, sr=16000):
        self.sr = sr
        self.max_samples = int(max_seconds * sr)
        t2i = json.load(open(vocab_path, encoding="utf-8"))
        self.unk = t2i.get("<unk>", 1)
        tok = {}
        for line in open(text_path, encoding="utf-8"):
            p = line.strip().split(" ", 1)
            if len(p) == 2 and p[1].strip():
                tok[p[0]] = [t2i.get(t, self.unk) for t in p[1].split()]
        base_dir = os.path.dirname(os.path.abspath(scp_path))
        self.rows = []
        for line in open(scp_path, encoding="utf-8"):
            p = line.strip().split(" ", 1)
            if len(p) == 2 and p[0] in tok:
                wav = p[1].strip()
                if not os.path.isabs(wav):
                    wav = os.path.join(base_dir, wav)
                self.rows.append((wav, tok[p[0]]))
        print(f"[WavDataset] {scp_path} -> {len(self.rows)} 条有效")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        wav, ids = self.rows[i]
        w = load_audio(wav)
        if w is None:
            w = np.zeros(self.max_samples, dtype=np.float32)
        w = torch.from_numpy(w)
        if w.numel() > self.max_samples:
            w = w[: self.max_samples]
        return w, torch.tensor(ids, dtype=torch.long)


def collate_wav(batch):
    wavs, toks = zip(*batch)
    wlens = torch.tensor([w.numel() for w in wavs], dtype=torch.long)
    tlens = torch.tensor([t.numel() for t in toks], dtype=torch.long)
    wp = torch.zeros(len(wavs), wlens.max().item())
    for i, w in enumerate(wavs):
        wp[i, : w.numel()] = w
    tp = torch.zeros(len(toks), tlens.max().item(), dtype=torch.long)
    for i, t in enumerate(toks):
        tp[i, : t.numel()] = t
    return wp, wlens, tp, tlens
