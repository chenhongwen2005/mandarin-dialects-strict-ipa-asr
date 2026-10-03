# -*- coding: utf-8 -*-
"""四川话(成都 / 西南官话成渝片) G2P: 汉字 -> 严式 IPA(已转 Chao 调值字母 ˥˦˧˨˩)。

设计:
  - 已知字: 优先走 diamoe_dialect_ipa 的 chengdu hanzi 表, 保留方言特有读音(如 我->ŋo)。
  - 未知字: 回退 pypinyin(数字调) -> chengdu syllable 表。西南方言音系与普通话高度一致,
            绝大多数标准汉字音节可直接在 chengdu 音节表查到严式 IPA, 且符号空间与已知字一致。
  - 每个拼音查 chengdu syllable 表得严式 IPA(含数字调值轮廓), 再 apply_tone_marks 转 Chao 字母。
  - 标点/数字/拉丁字母/未登录音节一律丢弃, 绝不把中文原字漏进 IPA 目标。

依赖: diamoe_dialect_ipa(本项目抽取) + pypinyin(runtime 已装)。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diamoe_dialect_ipa as D  # noqa: E402
from pypinyin import pinyin, Style  # noqa: E402

_DIALECT = "chengdu"
_HANZI = D.get_dialect(_DIALECT).get("hanzi", {})
_SYL = D.get_dialect(_DIALECT)["syllables"]

# 常见标点(含中文)直接丢弃
_PUNCT = set(
    "，。！？、；：“”‘’（）《》〈〉…—,.!?;:\"'()[]{}<>-—~· \t\n"
    "【】「」『』“”·•※★☆○●◇◆■□▲△▼▽※"
)

# 去除「意义注释」类括号(如 README 示例 "啷个【怎么】" 里的【怎么】并非读音)
_BRACKET_RE = re.compile(r"[【\[][^】\]]*[】\]]")


def _clean_text(text: str) -> str:
    text = _BRACKET_RE.sub("", text)
    return text


def _char_pinyin(ch: str):
    """单字 -> 成都方言拼音(数字调)。优先 hanzi 表, 否则 pypinyin。无法解析返回 None。"""
    if ch in _HANZI and _HANZI[ch]:
        return _HANZI[ch][0]
    try:
        py = pinyin(ch, style=Style.TONE3, heteronym=False, errors="default")[0][0]
    except Exception:
        return None
    if not py:
        return None
    return py


def hanzi_to_ipa_syllables(text: str):
    """汉字串 -> 成都严式 IPA 音节 list(已转 Chao 调值字母)。

    返回 (syllables:list[str], unmatched:list[str])：
      - syllables: 成功解析的 IPA 音节(空格分隔即一句 IPA)。
      - unmatched: 被丢弃的字符/拼音(仅供诊断, 不进入目标序列)。
    若整句无任一可解析音节, 返回 ([], unmatched)。
    """
    text = _clean_text(text or "")
    ipa_units = []
    unmatched = []
    for ch in text:
        if ch in _PUNCT:
            continue
        if ch.isdigit() or (ch.isascii() and ch.isalnum()):
            # 数字/拉丁字母: 口语读法多样, 暂不进入词表
            continue
        py = _char_pinyin(ch)
        if py is None:
            unmatched.append(ch)
            continue
        if py in _SYL:
            ipa_units.append(_SYL[py]["ipa"])
        else:
            # 去掉声调尾再试一次(中性调/音节表仅收录无调项)
            base = re.sub(r"[0-9]$", "", py)
            if base in _SYL:
                ipa_units.append(_SYL[base]["ipa"])
            else:
                unmatched.append(py)
    ipa_units = D.apply_tone_marks(_DIALECT, ipa_units)
    # 清理: diamoe 的 chengdu 音节表在每个音节前预置了重音符号 ˈ(U+02C8)/ˌ(U+02CC)。
    # 这是词级韵律标记, 落在"每个音节"上是噪声 —— 既无信息量, 又虚增序列长度与词表规模,
    # 会显著拉高 CER、拖慢收敛。作为 CTC 音节级训练目标必须去掉(粤语 ToJyutping 目标即无此符号)。
    ipa_units = [u.replace("\u02c8", "").replace("\u02cc", "").strip()
                 for u in ipa_units]
    ipa_units = [u for u in ipa_units if u]
    return ipa_units, unmatched


if __name__ == "__main__":
    samples = [
        "我今天去超市买东西然后回家吃饭",
        "三国吴国顾雍是个什么样的人",
        "耍花招比谁都勇敢是什么个歌",
        "这个事情咋个办嘛",
    ]
    for s in samples:
        syl, un = hanzi_to_ipa_syllables(s)
        print(f"{s}\n  -> {' '.join(syl)}\n  (丢弃: {un})\n")
