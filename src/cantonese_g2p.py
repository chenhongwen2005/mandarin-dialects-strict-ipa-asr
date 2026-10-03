# -*- coding: utf-8 -*-
"""粤语汉字 -> 严式 IPA（带调值字母）G2P。

使用 ToJyutping (CanCLID, v3.2.0) 的 get_ipa_text 一步产出带 Chao 调值字母的
严式粤语 IPA，音节以 '.' 分隔。例如：

    get_ipa_text("两个人") -> "lœːŋ˩˧.kɔː˧.jɐn˨˩"

本模块按 '.' 拆成音节 token，并解析每个音节末尾的调值字母到规范调类 (1-6)：

    1 = ˥ (55)     高平/阴平
    2 = ˧˥ (35)   高升/阴上
    3 = ˧ (33)    中平/阴去
    4 = ˩ / ˨˩    低平/阳平
    5 = ˩˧ / ˨˧   低升/阳上
    6 = ˨ (22)    低平/阳去

许可证说明：ToJyutping 为 CanCLID 出品的宽松协议库，可在本项目内直接调用。
"""

# Chao tone letters (U+02E5 - U+02E9): ˥˦˧˨˩
_TONE_LETTERS = "˥˦˧˨˩"
_TONE_SET = set(_TONE_LETTERS)

# 粤语 6 调（调值字母串 -> 调类 1-6）
TONE_LETTER_TO_CLASS = {
    "˥": 1,
    "˧˥": 2,
    "˧": 3,
    "˩": 4,
    "˨˩": 4,
    "˩˧": 5,
    "˨˧": 5,
    "˨": 6,
}

# ToJyutping 对未知/乱码字符返回的占位符
_PLACEHOLDER = "⸨"


def _is_ipa_syllable(s: str) -> bool:
    """有效粤语 IPA 音节须含至少一个调值字母。"""
    return any(c in _TONE_SET for c in s)


def _extract_tone_letters(token: str) -> str:
    """取 token 末尾连续的调值字母串。"""
    i = len(token)
    while i > 0 and token[i - 1] in _TONE_SET:
        i -= 1
    return token[i:]


def ipa_syllable_to_tone_class(token: str):
    """从 IPA 音节末尾的调值字母解析调类 (1-6)；无法解析返回 None。"""
    tl = _extract_tone_letters(token)
    if not tl:
        return None
    return TONE_LETTER_TO_CLASS.get(tl)


def _safe_get_ipa(text: str):
    """调用 ToJyutping.get_ipa_text，捕获其内部异常（部分字符会触发其 IndexError）。"""
    import ToJyutping

    try:
        return ToJyutping.get_ipa_text(text)
    except Exception:
        return None


def hanzi_to_ipa_syllables(text: str):
    """汉字串 -> 粤语 IPA 音节列表（含调值字母）。

    输入可为空格分隔的粤语汉字（HF CSV 格式）。遇到未知/乱码字符时按字回退并跳过，
    保证单条样本不会因为个别坏字而整体作废。ToJyutping 对个别字符会抛异常，这里统一吞掉。

    返回音节字符串列表，例如 ['lœːŋ˩˧', 'kɔː˧', 'jɐn˨˩']。
    """
    text = "".join(text.split())  # 去掉所有空白
    if not text:
        return []

    # 快速路径：整句转换（绝大多数样本走这里）
    out = _safe_get_ipa(text)
    if out is not None and _PLACEHOLDER not in out:
        return [s for s in out.split(".") if s and _is_ipa_syllable(s)]

    # 回退：逐字转换，跳过无法转换/会崩溃的字（乱码/拉丁/数字等）
    syls = []
    for ch in text:
        ipa = _safe_get_ipa(ch)
        if ipa is None or _PLACEHOLDER in ipa or not ipa:
            continue
        for s in ipa.split("."):
            if s and _is_ipa_syllable(s):
                syls.append(s)
    return syls


def build_ipa2tone(vocab: dict):
    """从词表 (token->id) 构建 ipa2tone (token->调类) 映射，供声调准确率指标使用。"""
    ipa2tone = {}
    for tok in vocab:
        if tok in ("<blank>", "<unk>"):
            continue
        c = ipa_syllable_to_tone_class(tok)
        if c is not None:
            ipa2tone[tok] = c
    return ipa2tone


if __name__ == "__main__":
    tests = ["两 个 人 扯 猫 尾 唔 认 数", "廣東話", "ä½", "ABC123"]
    for t in tests:
        print(f"{t!r:>20} -> {hanzi_to_ipa_syllables(t)}")
    # tone class sanity
    for s in hanzi_to_ipa_syllables("两 个 人 扯 猫 尾 唔 认 数"):
        print(s, "->", ipa_syllable_to_tone_class(s))
