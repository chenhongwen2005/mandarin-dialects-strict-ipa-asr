# -*- coding: utf-8 -*-
"""使用从 DiaMoE-TTS 抽取的 12 方言严式 IPA 注音表。

提供方言「汉字/拼音 -> 严式 IPA」的能力，可直接用于：
  - 方言语音数据的 IPA 真值标注（G2P）
  - 扩充 mandarin-ipa-asr 的音素/词表
  - 与现有普通话/粤语 IPA 转换器对照

接口:
  list_dialects()                       -> ['chengdu', 'gaoxiong', ...]
  pinyin_to_ipa(dialect, pinyin_str)   -> (ipa_units:list, unmatched:list)
  apply_tone_marks(dialect, ipa_units) -> 把数字调值轮廓(⁵⁵)转成 Chao 调值字母(˥˥)
  hanzi_to_dialect_pinyin(text, dial)  -> 汉字 -> 方言拼音（空格分隔，含 word 级贪心匹配）
  hanzi_to_dialect_ipa(text, dial, tone_marks=True)
                                        -> 汉字 -> 严式 IPA 串

数据来自 data/dialect_ipa_diamoe/dialect_ipa.json（由 extract_diamoe_dialect_ipa.py 生成）。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(HERE, "..", "data", "dialect_ipa_diamoe", "dialect_ipa.json")

_cache = None


def _load():
    global _cache
    if _cache is None:
        with open(os.path.abspath(JSON_PATH), encoding="utf-8") as f:
            _cache = json.load(f)
    return _cache


def list_dialects():
    return sorted(_load()["dialects"].keys())


def get_dialect(dialect):
    d = _load()["dialects"].get(dialect)
    if d is None:
        raise KeyError(f"未知方言 {dialect}，可选: {list_dialects()}")
    return d


def pinyin_to_ipa(dialect, pinyin_units):
    """方言拼音串（空格分隔 或 list）-> 严式 IPA 单元 list（syllable 列已含调值轮廓）。

    返回 (ipa_units, unmatched)，unmatched 为未能查表的拼音。
    """
    d = get_dialect(dialect)
    syl = d["syllables"]
    if isinstance(pinyin_units, str):
        units = pinyin_units.split()
    else:
        units = list(pinyin_units)
    out, unmatched = [], []
    for u in units:
        u0 = u.strip().strip("#")
        if u0 in syl:
            out.append(syl[u0]["ipa"])
        else:
            unmatched.append(u0)
            out.append(u0)
    return out, unmatched


# Chao 调值字母（与项目此前 putonghua-ipa-converter scheme2 严 / ToJyutping 完全一致）：
# 数字调值 1~5（低->高）映射到 IPA 调值字母 ˩˨˧˦˥。
_SUPERSCRIPT_TO_CHAO = {
    "\u00b9": "\u02e9",  # ¹ -> ˩  (1 最低)
    "\u00b2": "\u02e8",  # ² -> ˨
    "\u00b3": "\u02e7",  # ³ -> ˧
    "\u2074": "\u02e6",  # ⁴ -> ˦
    "\u2075": "\u02e5",  # ⁵ -> ˥  (5 最高)
}


def apply_tone_marks(dialect, ipa_units):
    """把 IPA 单元里的数字调值轮廓（如 ⁵⁵）转成 Chao 调值字母（如 ˥˥）。

    与项目此前 putonghua-ipa-converter(scheme2 严) / ToJyutping 的调值符号完全一致，
    不再使用 HML(ᴴᴹᴸ) 调域标记。该转换对手方无关，直接按上标数字逐字符映射。
    """
    res = []
    for ipa in ipa_units:
        s = "".join(_SUPERSCRIPT_TO_CHAO.get(ch, ch) for ch in ipa)
        res.append(s)
    return res


def hanzi_to_dialect_pinyin(text, dialect):
    """汉字 -> 方言拼音（空格分隔）。逐字查 hanzi 表，word 表做贪心最长匹配；
    未登录字/标点原样保留。"""
    d = get_dialect(dialect)
    hanzi = d.get("hanzi", {})
    words = d.get("words", {})
    result = []
    i, s = 0, text
    while i < len(s):
        matched = None
        mlen = 0
        for L in range(min(4, len(s) - i), 0, -1):
            seg = s[i:i + L]
            if seg in words:
                matched = words[seg]
                mlen = L
                break
        if matched is not None:
            result.append(matched)
            i += mlen
            continue
        ch = s[i]
        if ch in hanzi and hanzi[ch]:
            result.append(hanzi[ch][0])
        else:
            result.append(ch)
        i += 1
    return " ".join(result)


def hanzi_to_dialect_ipa(text, dialect, tone_marks=True):
    """汉字 -> 该方言严式 IPA 串。tone_marks=True 时把数字调值轮廓转成 Chao 调值字母（˥˦˧˨˩）。"""
    py = hanzi_to_dialect_pinyin(text, dialect)
    units, unmatched = pinyin_to_ipa(dialect, py)
    if tone_marks:
        units = apply_tone_marks(dialect, units)
    return " ".join(units), unmatched


if __name__ == "__main__":
    print("支持方言:", list_dialects())
    for dial, sample in [("putonghua", "今天天气很好"), ("shanghai", "今天天气很好"),
                         ("chengdu", "今天天气很好"), ("gaoxiong", "今天天气真好")]:
        ipa, un = hanzi_to_dialect_ipa(sample, dial)
        print(f"[{dial}] {sample} -> {ipa}  (未匹配: {un})")
