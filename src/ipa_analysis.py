# -*- coding: utf-8 -*-
"""严式 IPA 音节语音特征分析。

依据本项目词表（vocab_mandarin_ipa_combined.json，1427 类）实际使用的附加符号 /
修饰符库存，将每个 IPA 音节拆解为「基础段 + 调值 + 附加符号」，并把每个附加符号
翻译成中文语音学特征（舌位前/后移、送气、唇化、清化、唯闭/入声、元音中央化/开闭等），
最后给出逐音节明细表与全局特征统计。

所有符号解释均对照词表真实出现的符号（U+031A ̚、U+031F ̟、U+0320 ̠、U+033A ̺、
U+02B0 ʰ、U+02B7 ʷ、U+0361 ͡、调值字母 ˥˦˧˨˩ 等），不臆造。
"""

import unicodedata
from collections import Counter, OrderedDict

# ── 调值字母（U+02E5..U+02E9）──
TONE_BARS = {
    0x2E5: "5",  # ˥ 特高
    0x2E6: "4",  # ˦ 半高
    0x2E7: "3",  # ˧ 中
    0x2E8: "2",  # ˨ 半低
    0x2E9: "1",  # ˩ 特低
}

# ── 附加符号 / 修饰符 → 中文语音特征（对照词表真实库存）──
# key = 码点, value = (符号, 中文名, 语音学解释)
DIACRITICS = OrderedDict([
    (0x031A, ("̚", "唯闭/不除阻", "入声韵尾（如 -p/-t/-k）成阻后不爆破，仅做闭合，是入声的典型标记")),
    (0x031F, ("̟", "舌位前移", "元音/辅音舌位较前（advanced），如 ɑ̟")),
    (0x0320, ("̠", "舌位后移", "元音/辅音舌位较后（retracted），如 ɛ̠")),
    (0x0325, ("̥", "清化（下圈）", "浊音清化（voiceless，ring below），如 ʐ̥")),
    (0x032F, ("̯", "非音节化/滑音", "下弯连符，表非音节元音或滑音过渡，如 ɪ̯")),
    (0x02B0, ("ʰ", "送气", "aspiration，如 tʰ / kʰ")),
    (0x0361, ("͡", "破擦连写", "双下弯连符，表破擦音连写，如 t͡ʂ / t͡ɕ")),
    (0x033A, ("̺", "舌尖/齿舌尖", "下桥形，标 apical/linguodental 特征，区分舌尖/舌叶")),
    (0x033D, ("̽", "元音中央化", "上叉号，centralized，元音向中央偏移")),
    (0x030A, ("̊", "清化（上圈）", "voiceless（ring above），如 n̊")),
    (0x02B7, ("ʷ", "唇化", "labialization，如 kʷ / ɑʷ")),
    (0x0311, ("̑", "元音音质微调（央/高化）", "上弯连符，标记元音音色的细微偏移")),
    (0x031E, ("̞", "开元音", "下钉号，more open，舌位降低")),
    (0x031C, ("̜", "闭元音", "下半环，more close，舌位升高")),
    (0x0357, ("͗", "元音音质修饰（右半圆上）", "元音音色微调标记")),
    (0x0339, ("̹", "加圆唇", "下半环（右），圆唇度增加")),
    (0x0351, ("͑", "元音音质修饰（左半圆上）", "元音音色微调标记")),
    (0x030D, ("̍", "音节化/卷舌元音", "上竖线，常标 ɨ/ɯ 类音节化或卷舌元音")),
    (0x0308, ("̈", "非音节/分音", "分音符（diaeresis），表非音节化")),
    (0x1DA8, ("ᶨ", "龈腭/卷舌修饰（小 j）", "带尾小 j 修饰符，标记龈腭/卷舌辅音音色")),
    (0x02D5, ("˕", "元音降低修饰", "下钉号修饰符，元音降低")),
    (0x1DB9, ("ᶹ", "唇齿化修饰（小 v）", "带钩小 v 修饰符，标记唇齿化")),
])

# 调值轮廓 → 声调类说明（仅作展示辅助，运行时优先用 ipa2tone 的真实映射）
CONTOURS = {
    "55": "阴平 · 高平",
    "35": "阳平 · 中升",
    "214": "上声 · 降升",
    "51": "去声 · 全降",
    "53": "高降",
    "24": "中升",
    "31": "低降",
    "213": "降升",
    "5": "高平（短）",
    "4": "半高",
    "3": "中",
    "2": "半低",
    "1": "低",
}

# 调类数字 → 名称（与 ipa2tone 的 1-4 对应普通话四声）
TONE_CLASS_NAMES = {
    "1": "阴平",
    "2": "阳平",
    "3": "上声",
    "4": "去声",
    "0": "轻声",
    "5": "轻声",
}


def _is_tone_bar(ch):
    return ord(ch) in TONE_BARS


def extract_tone(token):
    """返回 (调值数字串, 调值字母列表)。例：'˧˥' -> ('35', ['˧','˥'])。"""
    bars = [c for c in token if _is_tone_bar(c)]
    return "".join(TONE_BARS[ord(c)] for c in bars), bars


def analyze_token(token, ipa2tone=None):
    """分析单个 IPA 音节。返回 dict。"""
    tone_digits, tone_marks = extract_tone(token)
    diacritics = []
    for ch in token:
        o = ord(ch)
        if o in DIACRITICS:
            sym, zh, desc = DIACRITICS[o]
            diacritics.append((sym, zh, desc))
    rec = {
        "raw": token,
        "tone_digits": tone_digits,
        "tone_marks": "".join(tone_marks),
        "diacritics": diacritics,
        "tone_class": None,
    }
    if ipa2tone:
        rec["tone_class"] = ipa2tone.get(token)
    return rec


def analyze_ipa(ipa_str, ipa2tone=None):
    """分析整段空格分隔的 IPA 序列。

    返回 dict：
      tokens:       逐音节分析记录列表
      feature_counts: 各附加符号特征出现次数（Counter，key=中文名）
      tone_dist:    调值轮廓分布（Counter，key=调值数字串）
      tone_class_dist: 调类分布（若有 ipa2tone）
      n:            音节数
    """
    toks = [t for t in ipa_str.split() if t]
    records = [analyze_token(t, ipa2tone) for t in toks]
    feat = Counter()
    for r in records:
        for _, zh, _ in r["diacritics"]:
            feat[zh] += 1
    tone_dist = Counter(r["tone_digits"] for r in records if r["tone_digits"])
    tone_class_dist = Counter()
    if ipa2tone:
        for r in records:
            if r["tone_class"] is not None:
                tone_class_dist[str(r["tone_class"])] += 1
    return {
        "tokens": records,
        "feature_counts": feat,
        "tone_dist": tone_dist,
        "tone_class_dist": tone_class_dist,
        "n": len(records),
    }


def _tone_label(rec):
    """调值显示：优先 ipa2tone 调类，否则用轮廓描述。"""
    if rec.get("tone_class") is not None:
        tc = str(rec["tone_class"])
        name = TONE_CLASS_NAMES.get(tc, tc)
        return f"{rec['tone_digits']}（{name}）" if rec["tone_digits"] else name
    if rec["tone_digits"]:
        return f"{rec['tone_digits']}（{CONTOURS.get(rec['tone_digits'], '调值')}）"
    return "—"


def to_html(analysis):
    """渲染为 Gradio gr.HTML 可显示的表格 + 统计。"""
    rows = []
    for i, r in enumerate(analysis["tokens"], 1):
        dia = " ".join(f"{sym}" for sym, _, _ in r["diacritics"]) or "—"
        fea = "<br>".join(f"{zh}" for _, zh, _ in r["diacritics"]) or "—"
        tone = _tone_label(r)
        rows.append(
            f"<tr><td style='text-align:center;color:#57606a'>{i}</td>"
            f"<td style='font-family:monospace;font-size:15px;color:#1f2328'>{r['raw']}</td>"
            f"<td style='color:#bf3989'>{tone}</td>"
            f"<td style='font-family:monospace;color:#0969da'>{dia}</td>"
            f"<td style='color:#374151'>{fea}</td></tr>"
        )
    table = (
        "<table style='border-collapse:collapse;width:100%;font-size:14px'>"
        "<thead><tr style='background:#f0f3f6'>"
        "<th style='text-align:center;padding:4px 8px'>#</th>"
        "<th style='text-align:left;padding:4px 8px'>音节(IPA)</th>"
        "<th style='text-align:left;padding:4px 8px'>调值</th>"
        "<th style='text-align:left;padding:4px 8px'>附加符号</th>"
        "<th style='text-align:left;padding:4px 8px'>语音特征</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )

    # 特征统计
    feat_lines = []
    for zh, n in analysis["feature_counts"].most_common():
        feat_lines.append(f"<li><b>{zh}</b>：{n} 处</li>")
    feat_html = (
        "<details open><summary style='cursor:pointer;font-weight:600;color:#1f2328'>"
        f"附加符号特征统计（共 {analysis['n']} 音节）</summary>"
        "<ul style='margin:6px 0 0 0;color:#374151'>" +
        ("".join(feat_lines) if feat_lines else "<li>无附加符号（均为基础段 + 调值）</li>") +
        "</ul></details>"
    )

    # 调值/调类分布
    tone_lines = []
    for td, n in analysis["tone_dist"].most_common():
        tone_lines.append(f"<li><b>{td}</b>（{CONTOURS.get(td,'调值')}）：{n} 个</li>")
    tc_lines = []
    if analysis["tone_class_dist"]:
        for tc, n in analysis["tone_class_dist"].most_common():
            tc_lines.append(f"<li><b>{TONE_CLASS_NAMES.get(tc, tc)}</b>：{n} 个</li>")
    tone_html = (
        "<details open><summary style='cursor:pointer;font-weight:600;color:#1f2328'>"
        "调值 / 调类分布</summary>"
        "<ul style='margin:6px 0 0 0;color:#374151'>" +
        ("".join(tone_lines) if tone_lines else "<li>未检出调值字母</li>") +
        ("".join(tc_lines) if tc_lines else "") +
        "</ul></details>"
    )

    return table + "<div style='height:10px'></div>" + feat_html + tone_html
