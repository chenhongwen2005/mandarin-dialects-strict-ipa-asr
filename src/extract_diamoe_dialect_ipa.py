# -*- coding: utf-8 -*-
"""从 DiaMoE-TTS 的 dialect_frontend 抽取 12 个方言的严式 IPA 注音表 -> 单一 JSON。

源目录:
  D:/DiaMoE-TTS-main/DiaMoE-TTS-main/dialect_frontend/frontend/dialect/<dialect>/
    syllable.xlsx  (或 base_syllable.xlsx): 列 pinyin,syllable,initial,final
    tone.xlsx      (或 base_tone.xlsx):     列 contour,mark
    hanzi.xlsx:    列 hanzi,standard_chinese_pinyin,dialect_pinyin,...
    word.xlsx      (或 base_word.xlsx):     列 word,standard_chinese_pinyin,dialect_pinyin,...

输出:
  D:/mandarin-ipa-asr-cu128/data/dialect_ipa_diamoe/dialect_ipa.json

JSON 结构:
  {
    "_meta": {...},
    "dialects": {
      "<dialect>": {
        "syllables": { "a1": {"ipa":"ˈɑ⁵⁵","initial":"∅","final":"ˈɑ⁵⁵"}, ... },
        "tones":     { "²¹⁴":"ᴹᴸ", ... },
        "hanzi":     { "多": ["du1"], ... },       # 汉字 -> 方言拼音（多音字为列表）
        "words":     { "快眼": "kua7 nngee3", ... }
      }, ...
    }
  }

运行:
  runtime/python.exe src/extract_diamoe_dialect_ipa.py
"""
import json
import os

import openpyxl

SRC = r"D:/DiaMoE-TTS-main/DiaMoE-TTS-main/dialect_frontend/frontend/dialect"
OUT_DIR = r"D:/mandarin-ipa-asr-cu128/data/dialect_ipa_diamoe"
OUT = os.path.join(OUT_DIR, "dialect_ipa.json")


def load_rows(path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return [], []
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    data = []
    for r in rows[1:]:
        if all(c is None for c in r):
            continue
        data.append({header[i]: ("" if r[i] is None else str(r[i]).strip()) for i in range(len(header))})
    return header, data


def col(header, candidates):
    for c in candidates:
        if c in header:
            return c
    return None


def extract_dialect(dname):
    dd = os.path.join(SRC, dname)
    out = {"syllables": {}, "tones": {}, "hanzi": {}, "words": {}}

    # ---- syllable / base_syllable ----
    syl_file = os.path.join(dd, "base_syllable.xlsx") \
        if os.path.exists(os.path.join(dd, "base_syllable.xlsx")) \
        else os.path.join(dd, "syllable.xlsx")
    if os.path.exists(syl_file):
        h, rows = load_rows(syl_file)
        pk = col(h, ["pinyin"])
        ik = col(h, ["initial"])
        fk = col(h, ["final"])
        sk = col(h, ["syllable"])
        for r in rows:
            p = r.get(pk, "")
            if not p:
                continue
            ipa = r.get(sk, "") or f"{r.get(ik, '')},{r.get(fk, '')}"
            out["syllables"][p] = {
                "ipa": ipa,
                "initial": r.get(ik, ""),
                "final": r.get(fk, ""),
            }

    # ---- tone / base_tone ----
    tone_file = os.path.join(dd, "base_tone.xlsx") \
        if os.path.exists(os.path.join(dd, "base_tone.xlsx")) \
        else os.path.join(dd, "tone.xlsx")
    if os.path.exists(tone_file):
        h, rows = load_rows(tone_file)
        ck = col(h, ["contour"])
        mk = col(h, ["mark"])
        for r in rows:
            c = r.get(ck, "")
            if c == "" or c is None:
                continue
            out["tones"][c] = r.get(mk, "")

    # ---- hanzi ----
    hz_file = os.path.join(dd, "hanzi.xlsx")
    if os.path.exists(hz_file):
        h, rows = load_rows(hz_file)
        hzk = col(h, ["hanzi"])
        dpk = col(h, ["dialect_pinyin"])
        for r in rows:
            ch = r.get(hzk, "")
            dp = r.get(dpk, "").replace("#", "").strip()
            if not ch or dp == "":
                continue
            out["hanzi"].setdefault(ch, [])
            if dp not in out["hanzi"][ch]:
                out["hanzi"][ch].append(dp)

    # ---- word / base_word ----
    wd_file = os.path.join(dd, "word.xlsx") \
        if os.path.exists(os.path.join(dd, "word.xlsx")) \
        else os.path.join(dd, "base_word.xlsx")
    if os.path.exists(wd_file):
        h, rows = load_rows(wd_file)
        wk = col(h, ["word"])
        dpk = col(h, ["dialect_pinyin"])
        for r in rows:
            w = r.get(wk, "").replace("#", "").strip()
            dp = r.get(dpk, "").replace("#", "").strip()
            if not w or dp == "":
                continue
            out["words"][w] = dp

    return out


def main():
    dialects = sorted([d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d))])
    result = {
        "_meta": {
            "source": SRC,
            "note": "DiaMoE-TTS dialect_frontend 严式 IPA 注音抽取（汉字->方言拼音->严式IPA->调域标记）",
        },
        "dialects": {},
    }
    summary = {}
    for d in dialects:
        ex = extract_dialect(d)
        result["dialects"][d] = ex
        summary[d] = {
            "syllables": len(ex["syllables"]),
            "tones": len(ex["tones"]),
            "hanzi": len(ex["hanzi"]),
            "words": len(ex["words"]),
        }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("抽取完成 ->", OUT)
    for d, s in summary.items():
        print(f"  {d:14s} syllables={s['syllables']:5d}  tones={s['tones']:3d}  "
              f"hanzi={s['hanzi']:5d}  words={s['words']:4d}")


if __name__ == "__main__":
    main()
