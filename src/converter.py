# -*- coding: utf-8 -*-
"""内置「中文 → 严式 IPA」转换器。

数据来源：nk2028/putonghua-ipa-converter（app/data/putonghua.js），许可证 CC0-1.0。
已将所需数据 vendor 到本项目 vendor/putonghua_converter/putonghua.js，
与训练时使用的 UntPhesoca 严式（scheme index=2）同源，故转换结果可直接作为
ASR 识别结果的「真值」参与逐音节比对。

注：此为第三方词典资源（CC0），与本项目训练/评测语料（依授权不公开）无关。
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "vendor", "putonghua_converter", "putonghua.js")

# 训练词表使用的记音方案：UntPhesoca 严式
SCHEME_UNT_PHESOCA_NARROW = 2

_DATA = None


def load_converter_data(path=None):
    """读取 putonghua.js（window.DATA_PUTONGHUA = {...} 形式）为 dict。"""
    p = path or DEFAULT_DATA
    with open(p, encoding="utf-8") as f:
        txt = f.read()
    # 去掉 "window.DATA_PUTONGHUA = " 前缀与结尾 ";"
    idx = txt.index("=") + 1
    return json.loads(txt[idx:].rstrip().rstrip(";"))


def _data():
    global _DATA
    if _DATA is None:
        _DATA = load_converter_data()
    return _DATA


def scheme_names():
    """返回所有可选记音方案名称。"""
    return _data()["schemes"]


def is_cjk(ch):
    cp = ord(ch)
    return 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF


def char_to_ipa(ch, scheme=SCHEME_UNT_PHESOCA_NARROW, pick=0):
    """单个汉字 → 严式 IPA 字符串；无读音返回 None。

    pick: 多音字取第几个读音（默认第 0 个，即最常用读音）。
    """
    d = _data()
    pys = d["char2py"].get(ch)
    if not pys:
        return None
    py = pys[min(pick, len(pys) - 1)]
    body = py[:-1]
    tone = int(py[-1])
    return d["py2ipa"][body][scheme] + d["tones2ipa"][tone][scheme]


def text_to_ipa(text, scheme=SCHEME_UNT_PHESOCA_NARROW, pick=0):
    """中文文本 → 严式 IPA。

    返回 (tokens, ipa_str, oov_chars)
      tokens:     空格分隔的 IPA 音节列表（仅含汉字，标点/非汉字被跳过）
      ipa_str:    " ".join(tokens)
      oov_chars:  词典中查不到读音的汉字列表
    """
    toks, oov = [], []
    for ch in text:
        if not is_cjk(ch):
            continue
        ipa = char_to_ipa(ch, scheme, pick)
        if ipa is None:
            oov.append(ch)
            continue
        toks.append(ipa)
    return toks, " ".join(toks), oov


if __name__ == "__main__":
    s, ipa, oov = text_to_ipa("床前明月光，疑是地上霜。")
    print("IPA:", ipa)
    print("OOV:", oov)
    print("schemes:", scheme_names())
