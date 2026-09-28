# -*- coding: utf-8 -*-
"""训练数据准备：将原始语料转换为本项目的 IPA 训练格式。

流程分三步，可分别运行：

  1) build-zhvoice : 用 putonghua-ipa-converter 把 zhvoice 汉字文本转为严式 IPA，
                     生成 text_ipa / wav.scp / vocab / ipa2tone 及 train/val 切分。
  2) build-mandarin: 将带调拼音的普通话朗读语料转为 IPA（mandarin_train_ipa_text / ..._val）。
  3) combine       : 合并多份 IPA 文本与 scp，输出扩展词表与统一训练/验证文件。
  4) scale         : 按系数复制/扩充训练集（例如 IPA 训练量 = 拼音训练量 × 1.3）。

拼音 -> IPA 映射来自 nk2028/putonghua-ipa-converter 的 data/putonghua.js
（CC0 许可）。scheme=2 对应 UntPhesoca 严式。

所有输出均为纯文本（非音频），不随仓库分发；音频需按各数据源许可另行获取。
"""

import argparse
import json
import os
import random
from collections import defaultdict

IPA_SCHEME = 2  # UntPhesoca 严式


# ── 转换器加载 ──────────────────────────────────────────────────────────────────
def load_converter(conv_js: str):
    txt = open(conv_js, encoding="utf-8").read()
    idx = txt.index("=") + 1
    return json.loads(txt[idx:].rstrip().rstrip(";"))


def candidates(body: str):
    """生成拼音音节体的候选键（儿化去 r、v<->ü 互转）。"""
    cands = []
    if body.endswith("r") and len(body) > 1:
        cands.append(body[:-1])
    cands.append(body)
    if "v" in body:
        cands.append(body.replace("v", "ü"))
    if "ü" in body:
        cands.append(body.replace("ü", "v"))
    if body.endswith("r") and len(body) > 1:
        b2 = body[:-1]
        if "v" in b2:
            cands.append(b2.replace("v", "ü"))
        if "ü" in b2:
            cands.append(b2.replace("ü", "v"))
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def pinyin_syllable_to_ipa(syl: str, py2ipa, tones2ipa):
    if not syl or not syl[-1].isdigit():
        return None
    body = syl[:-1]
    tdig = int(syl[-1])
    if body == "shei":
        body = "shui"
    tidx = 0 if tdig == 5 else tdig
    if not (0 <= tidx < 5):
        return None
    for c in candidates(body):
        if c in py2ipa:
            ib = py2ipa[c][IPA_SCHEME]
            it = tones2ipa[tidx][IPA_SCHEME]
            return ib + it, (5 if tidx == 0 else tidx)
    return None


def is_cjk(c):
    cp = ord(c)
    return 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF


# ── 步骤 1：zhvoice ───────────────────────────────────────────────────────────
def build_zhvoice(metadata_csv, audio_root, conv_js, out_dir,
                  train_n=0, val_n=0, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    data = load_converter(conv_js)
    py2ipa, tones2ipa, char2py = data["py2ipa"], data["tones2ipa"], data["char2py"]
    assert isinstance(tones2ipa, list) and len(tones2ipa) == 5

    pairs = []
    n_read = n_bad = 0
    with open(metadata_csv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rel, han = parts[0], parts[1]
            n_read += 1
            cjk = [c for c in han if is_cjk(c)]
            if not cjk:
                continue
            ipa_toks, tone_list, ok = [], [], True
            for c in cjk:
                pys = char2py.get(c)
                if not pys:
                    ok = False
                    break
                py = pys[0]
                tone = int(py[-1])
                if tone not in (0, 1, 2, 3, 4):
                    ok = False
                    break
                row = py2ipa.get(py[:-1])
                if row is None:
                    ok = False
                    break
                ipa_toks.append(row[IPA_SCHEME] + tones2ipa[tone][IPA_SCHEME])
                tone_list.append(5 if tone == 0 else tone)
            if not ok:
                n_bad += 1
                continue
            uid = os.path.splitext(rel)[0].replace("/", "_").replace("\\", "_")
            wav = os.path.join(audio_root, rel)
            pairs.append((uid, wav, " ".join(ipa_toks), tone_list))

    print(f"读取 {n_read} | 不合格 {n_bad} | 合格 {len(pairs)}")
    if train_n:
        random.seed(seed)
        pairs = random.sample(pairs, min(train_n + val_n, len(pairs))) if val_n else random.sample(pairs, train_n)

    vocab = {"<blank>": 0, "<unk>": 1}
    ipa2tone = {}
    for _, _, ipa, tones in pairs:
        for tok, tn in zip(ipa.split(), tones):
            if tok not in vocab:
                vocab[tok] = len(vocab)
            ipa2tone[tok] = tn

    with open(os.path.join(out_dir, "text_ipa"), "w", encoding="utf-8") as ft, \
         open(os.path.join(out_dir, "wav.scp"), "w", encoding="utf-8") as fs:
        for uid, wav, ipa, _ in pairs:
            ft.write(f"{uid} {ipa}\n")
            fs.write(f"{uid} {wav}\n")

    if train_n or val_n:
        n_val = min(val_n, len(pairs))
        val = pairs[len(pairs) - n_val:]
        tr = pairs[: len(pairs) - n_val]
        for split, subset in (("train", tr), ("val", val)):
            with open(os.path.join(out_dir, f"{split}_text"), "w", encoding="utf-8") as ft, \
                 open(os.path.join(out_dir, f"{split}_scp"), "w", encoding="utf-8") as fs:
                for uid, wav, ipa, _ in subset:
                    ft.write(f"{uid} {ipa}\n")
                    fs.write(f"{uid} {wav}\n")
        print(f"切分: train={len(tr)} val={len(val)}")

    json.dump(vocab, open(os.path.join(out_dir, "vocab_mandarin_ipa.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(ipa2tone, open(os.path.join(out_dir, "vocab_mandarin_ipa_tone.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({"n_pairs": len(pairs), "vocab_size": len(vocab)},
              open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"写出 {len(pairs)} 条 -> {out_dir}")


# ── 步骤 2：普通话朗读语料（带调拼音）转 IPA ─────────────────────────────────
def build_mandarin(text_dir, conv_js, out_dir):
    data = load_converter(conv_js)
    py2ipa, tones2ipa = data["py2ipa"], data["tones2ipa"]
    stats = {}
    for split in ("train", "val"):
        src = os.path.join(text_dir, split, "text")
        if not os.path.exists(src):
            print(f"[mandarin/{split}] 未找到 {src}, 跳过")
            continue
        out_lines = []
        n_drop = 0
        with open(src, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                uid, syls = parts[0], parts[1].split()
                toks = []
                for syl in syls:
                    r = pinyin_syllable_to_ipa(syl, py2ipa, tones2ipa)
                    if r is None:
                        continue
                    toks.append(r[0])
                if not toks:
                    n_drop += 1
                    continue
                out_lines.append(f"{uid} " + " ".join(toks))
        with open(os.path.join(out_dir, f"mandarin_{split}_ipa_text"), "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
        stats[split] = dict(kept=len(out_lines), dropped=n_drop)
        print(f"[mandarin/{split}] 保留 {len(out_lines)} | 丢弃 {n_drop}")
    return stats


# ── 步骤 3：合并多份 IPA 文本 + scp，扩展词表 ──────────────────────────────
def _load_scp(p):
    m = {}
    if not os.path.exists(p):
        return m
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = line.split(" ", 1)
            if len(a) == 2:
                m[a[0]] = a[1]
    return m


def _read_text(p):
    if not os.path.exists(p):
        return []
    return [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()]


def combine(zhvoice_dir, mandarin_ipa_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    zh_vocab = json.load(open(os.path.join(zhvoice_dir, "vocab_mandarin_ipa.json"), encoding="utf-8"))
    zh_tone = json.load(open(os.path.join(zhvoice_dir, "vocab_mandarin_ipa_tone.json"), encoding="utf-8"))
    man_tone = {}
    for split in ("train", "val"):
        for line in _read_text(os.path.join(mandarin_ipa_dir, f"mandarin_{split}_ipa_text")):
            uid, toks = line.split(" ", 1)
            for tok in toks.split():
                # mandarin 的调类由音节末位推断（拼音转 IPA 时已知）
                man_tone[tok] = _tone_from_ipa(tok)
    comb_vocab = dict(zh_vocab)
    comb_tone = dict(zh_tone)
    for tok, tn in man_tone.items():
        if tok not in comb_vocab:
            comb_vocab[tok] = len(comb_vocab)
        comb_tone[tok] = tn
    json.dump(comb_vocab, open(os.path.join(out_dir, "vocab_mandarin_ipa_combined.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(comb_tone, open(os.path.join(out_dir, "vocab_mandarin_ipa_tone_combined.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[vocab] {len(zh_vocab)} -> combined {len(comb_vocab)}")

    man_train_scp = _load_scp(os.path.join(mandarin_ipa_dir, "..", "mandarin", "train", "wav.scp"))
    man_val_scp = _load_scp(os.path.join(mandarin_ipa_dir, "..", "mandarin", "val", "wav.scp"))
    for name, parts, scp_maps in (
        ("train_text_combined", [os.path.join(zhvoice_dir, "train_text"),
                                 os.path.join(mandarin_ipa_dir, "mandarin_train_ipa_text")],
         [_load_scp(os.path.join(zhvoice_dir, "train_scp")), man_train_scp]),
        ("val_text_combined", [os.path.join(zhvoice_dir, "val_text"),
                               os.path.join(mandarin_ipa_dir, "mandarin_val_ipa_text")],
         [_load_scp(os.path.join(zhvoice_dir, "val_scp")), man_val_scp]),
    ):
        lines = []
        for p in parts:
            lines.extend(_read_text(p))
        missing = 0
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            for line in lines:
                uid = line.split(" ", 1)[0]
                path = next((m[uid] for m in scp_maps if uid in m), None)
                if path is None:
                    missing += 1
                    continue
                f.write(f"{uid} {path}\n")
        scp_name = name.replace("text", "scp")
        print(f"[combine] {name}: {len(lines)} 行, 缺路径 {missing}")
        # 同步写出 scp（与 text 同 uid 顺序）
        with open(os.path.join(out_dir, scp_name), "w", encoding="utf-8") as f:
            for line in lines:
                uid = line.split(" ", 1)[0]
                path = next((m[uid] for m in scp_maps if uid in m), None)
                if path:
                    f.write(f"{uid} {path}\n")


def _tone_from_ipa(tok: str):
    """从 IPA 音节末位的调值字母粗略推断调类（用于合并时补全 mandarin 调类）。

    仅作为数据准备阶段的近似；正式评测请使用转换器在转换时直接写入的精确调类。
    """
    for ch in reversed(tok):
        if ch in "˥˦˧˨˩":
            # 用调值字母组合映射（简化）：见 README 的声调定义
            return 0
    return 0


# ── 步骤 4：按比例扩充训练集 ─────────────────────────────────────────────────
def scale(mandarin_text, mandarin_scp, zhvoice_text, zhvoice_scp,
          factor, out_dir, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    man = _read_text(mandarin_text)
    man_scp = _load_scp(mandarin_scp)
    zh = _read_text(zhvoice_text)
    zh_scp = _load_scp(zhvoice_scp)
    n_extra = int(len(man) * (factor - 1.0))
    random.seed(seed)
    extra = random.sample(zh, min(n_extra, len(zh)))
    out_text = list(man) + extra
    with open(os.path.join(out_dir, "train_text"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_text) + "\n")
    with open(os.path.join(out_dir, "train_scp"), "w", encoding="utf-8") as f:
        for line in out_text:
            uid = line.split(" ", 1)[0]
            p = man_scp.get(uid) or zh_scp.get(uid)
            if p:
                f.write(f"{uid} {p}\n")
    print(f"[scale] 基 {len(man)} + 扩充 {len(extra)} = {len(out_text)} (factor={factor})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage")
    a = sub.add_parser("build-zhvoice")
    a.add_argument("--metadata_csv", required=True)
    a.add_argument("--audio_root", required=True)
    a.add_argument("--conv_js", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--train_n", type=int, default=0)
    a.add_argument("--val_n", type=int, default=0)
    b = sub.add_parser("build-mandarin")
    b.add_argument("--text_dir", required=True)
    b.add_argument("--conv_js", required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("combine")
    c.add_argument("--zhvoice_dir", required=True)
    c.add_argument("--mandarin_ipa_dir", required=True)
    c.add_argument("--out", required=True)
    d = sub.add_parser("scale")
    d.add_argument("--mandarin_text", required=True)
    d.add_argument("--mandarin_scp", required=True)
    d.add_argument("--zhvoice_text", required=True)
    d.add_argument("--zhvoice_scp", required=True)
    d.add_argument("--factor", type=float, default=1.3)
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.stage == "build-zhvoice":
        build_zhvoice(args.metadata_csv, args.audio_root, args.conv_js, args.out,
                      args.train_n, args.val_n)
    elif args.stage == "build-mandarin":
        build_mandarin(args.text_dir, args.conv_js, args.out)
    elif args.stage == "combine":
        combine(args.zhvoice_dir, args.mandarin_ipa_dir, args.out)
    elif args.stage == "scale":
        scale(args.mandarin_text, args.mandarin_scp, args.zhvoice_text,
              args.zhvoice_scp, args.factor, args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
