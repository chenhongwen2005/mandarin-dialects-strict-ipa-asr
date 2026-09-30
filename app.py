# -*- coding: utf-8 -*-
"""Gradio 演示：普通话严式国际音标（IPA）语音识别。

功能：
  1. 识别：上传或录制音频 -> 输出严式 IPA 音节序列（含声调调值）。
  2. 比对：上传音频并填入中文/参考 IPA -> 与转换器生成的真值逐音节声调对齐与差异高亮；
     无参考时自动用「贪心 / 集束搜索」两套解码结果互相比对差异。
  3. 语音特征分析：逐音节拆解 IPA，标注调值与每个附加符号的语音学特征并汇总统计。

用法（在项目根目录）:
  python app.py --ckpt <权重路径> [--share] [--port 7860]

说明：
  - 推理与训练保持完全一致：原始波形输入、frontend.dither=0、bf16 编码器前向。
  - 权重不随仓库发布，请从 ModelScope 下载后通过 --ckpt 指定。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import torch

# 关闭 Gradio 遥测外呼：离线/受限网络环境下，后台上报线程会因无法连接服务器而
# 抛出 httpx.ConnectTimeout（Thread-3），刷红屏但不影响功能。设为 False 即可抑制。
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

from utils import (
    ctc_greedy_decode,
    ctc_beam_search,
    levenshtein_align,
    load_vocab,
    load_audio,
    build_model,
    score_utterance,
    resolve_local_path,
)
from converter import text_to_ipa
from ipa_analysis import analyze_ipa, to_html

MAX_SECONDS = 30.0
BEAM_SIZE = 12


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="微调权重路径（.pt）。缺省则随机初始化演示链路。")
    ap.add_argument("--vocab_path", default="vocab/vocab_mandarin_ipa_combined.json")
    ap.add_argument("--ipa2tone_path", default="vocab/vocab_mandarin_ipa_tone_combined.json")
    ap.add_argument("--model_dir", default="iic/SenseVoiceSmall")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--max_seconds", type=float, default=MAX_SECONDS)
    ap.add_argument("--beam", type=int, default=BEAM_SIZE,
                    help="识别页集束搜索宽度；设为 0 则退回贪心解码。")
    return ap.parse_args()


STATE = {}


def load():
    a = parse_args()
    a.vocab_path = resolve_local_path(a.vocab_path)
    a.ipa2tone_path = resolve_local_path(a.ipa2tone_path)
    # 未显式指定 --ckpt 时，自动尝试定位 weights/best.pt（README 指定的权重位置），
    # 避免静默跑一个随机初始化的无效模型。
    if not a.ckpt:
        cand = resolve_local_path("weights/best.pt")
        if os.path.exists(cand):
            a.ckpt = cand
            print(f"[app] 未指定 --ckpt，自动使用 {a.ckpt}")
    a.ckpt = resolve_local_path(a.ckpt)
    if a.model_dir and not os.path.isabs(a.model_dir) and os.path.isdir(resolve_local_path(a.model_dir)):
        a.model_dir = resolve_local_path(a.model_dir)
    if not a.ckpt:
        print("=" * 64)
        print("【警告】未加载任何微调权重：--ckpt 为空且 weights/best.pt 不存在")
        print("    当前运行的是【随机初始化】模型，识别结果将是无意义的乱码。")
        print("    请先下载权重：")
        print("      modelscope download --model QiGuanFuChen/mandarin-ipa-asr --local_dir weights/")
        print("    再以权重启动：")
        print("      python app.py --ckpt weights/best.pt")
        print("=" * 64)
    model, token2id, dev = build_model(a.vocab_path, a.ckpt, a.model_dir, freeze_encoder=True)
    id2tok = {v: k for k, v in token2id.items()}
    ipa2tone = json.load(open(a.ipa2tone_path, encoding="utf-8")) if a.ipa2tone_path else {}
    STATE.update(
        model=model, token2id=token2id, id2tok=id2tok,
        ipa2tone=ipa2tone, dev=dev, max_seconds=a.max_seconds,
        beam=a.beam, vocab_set=set(token2id.keys()),
    )
    tag = os.path.basename(a.ckpt) if a.ckpt else "随机初始化(无权重)"
    print(f"[app] 模型就绪 device={dev} ckpt={tag}")


@torch.no_grad()
def decode_ids(audio_path: str, beam: int = 0):
    """返回 (ids, str)。beam>1 走集束搜索，否则贪心。"""
    model = STATE["model"]
    dev = STATE["dev"]
    id2tok = STATE["id2tok"]
    w = load_audio(audio_path)
    sr = 16000
    max_len = int(STATE["max_seconds"] * sr)
    if w.numel() > max_len:
        w = w[:max_len]
    x = w.unsqueeze(0).to(dev)
    sl = torch.tensor([x.shape[1]], device=dev)
    lp, _ = model(x, sl, autocast_bf16=True)
    if beam and beam > 1:
        logp = lp[:, 0, :].cpu().numpy()
        ids = ctc_beam_search(logp, beam_size=beam, id2token=id2tok)
    else:
        ids = ctc_greedy_decode(lp)[0]
    s = " ".join(id2tok.get(i, "?") for i in ids)
    return ids, s


def predict_ipa(audio_path: str):
    if audio_path is None:
        return "", "未提供音频"
    beam = STATE.get("beam", BEAM_SIZE) or 0
    ids, s = decode_ids(audio_path, beam)
    mode = f"集束({beam})" if beam and beam > 1 else "贪心"
    return s, f"共识别 {len(ids)} 个 IPA 音节（{mode}）"


# ── 差异高亮配色 ────────────────────────────────────────────────────────────────
COLOR = {
    "correct": "#2e7d32",
    "tone_wrong": "#ef6c00",
    "wrong": "#c62828",
    "extra": "#8e24aa",
    "missing": "#1565c0",
}


def alignment_html(states):
    rows = []
    for pt, rt, st, lb in states:
        c = COLOR.get(st, "#444")
        rows.append(
            f"<tr><td style='color:{c};font-weight:600'>{pt}</td>"
            f"<td style='color:{c};font-weight:600'>{rt}</td>"
            f"<td style='color:{c}'>{lb}</td></tr>"
        )
    return (
        "<table style='border-collapse:collapse;font-family:monospace;font-size:15px'>"
        "<thead><tr><th style='text-align:left;padding:2px 10px'>预测</th>"
        "<th style='text-align:left;padding:2px 10px'>参考</th>"
        "<th style='text-align:left;padding:2px 10px'>状态</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def diff_html(a_toks, b_toks, a_label, b_label):
    """对两组 token 序列做 Levenshtein 对齐并高亮差异（用于贪心 vs 集束）。"""
    ops = levenshtein_align(a_toks, b_toks)
    rows = []
    for op, ia, ib in ops:
        if op == "match":
            pt, rt = a_toks[ia], b_toks[ib]
            if pt == rt:
                c, lb = "#2e7d32", "✓"
            else:
                c, lb = "#ef6c00", "≠"
            rows.append(f"<tr><td style='color:{c}'>{pt}</td><td style='color:{c}'>{rt}</td><td style='color:{c}'>{lb}</td></tr>")
        elif op == "del":
            c, lb = "#8e24aa", "仅A"
            rows.append(f"<tr><td style='color:{c}'>{a_toks[ia]}</td><td style='color:#999'>—</td><td style='color:{c}'>{lb}</td></tr>")
        else:
            c, lb = "#1565c0", "仅B"
            rows.append(f"<tr><td style='color:#999'>—</td><td style='color:{c}'>{b_toks[ib]}</td><td style='color:{c}'>{lb}</td></tr>")
    return (
        f"<div style='margin-bottom:4px;font-size:13px;color:#555'>{a_label} ↔ {b_label}："
        f"共 {len(ops)} 步，差异 {sum(1 for o in ops if o[0]!='match' or (o[1]>=0 and o[2]>=0 and a_toks[o[1]]!=b_toks[o[2]]))} 处</div>"
        "<table style='border-collapse:collapse;font-family:monospace;font-size:15px'>"
        f"<thead><tr><th style='text-align:left;padding:2px 10px'>{a_label}</th>"
        f"<th style='text-align:left;padding:2px 10px'>{b_label}</th>"
        f"<th style='text-align:left;padding:2px 10px'>状态</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def compare(audio_path, ref_source, ref_text):
    """自动比对差异：与转换器生成的真值（中文文本）或直接填入的 IPA 对齐；
    无参考则贪心 vs 集束自动互比。"""
    if audio_path is None:
        return "", "", "请先上传音频", "—", ""
    greedy_ids, greedy_str = decode_ids(audio_path, 0)
    beam_ids, beam_str = decode_ids(audio_path, BEAM_SIZE)

    if ref_text and ref_text.strip():
        if ref_source == "转换器(中文)":
            ref_tokens_full, ref_ipa, oov = text_to_ipa(ref_text)
            # 过滤不在模型词表内的参考音节，避免把「超出输出空间」误判为识别错误
            ref_tokens = [t for t in ref_tokens_full if t in STATE["vocab_set"]]
            oov_n = len(ref_tokens_full) - len(ref_tokens)
            oov_note = ""
            if oov_n or oov:
                oov_note = (f"\n注：{oov_n} 个参考音节不在模型词表内（已排除统计）；"
                            f"查无读音的汉字：{' '.join(oov) or '无'}")
        else:
            ref_tokens = ref_text.strip().split()
            ref_ipa = " ".join(ref_tokens)
            oov_note = ""
        res = score_utterance(greedy_ids, ref_tokens, STATE["id2tok"], STATE["ipa2tone"])
        align = alignment_html(res["states"])
        counts = {"correct": 0, "tone_wrong": 0, "wrong": 0, "extra": 0, "missing": 0}
        for _, _, st, _ in res["states"]:
            if st in counts:
                counts[st] += 1
        summary = (
            f"音节准确率: {res['syllable_acc']:.4f}  ({res['n_pred']} 预测 / {res['n_ref']} 参考)  |  "
            f"声调准确率: {res['tone_acc']:.4f}\n"
            f"细分 — 完全正确:{counts['correct']}  调错:{counts['tone_wrong']}  "
            f"错读:{counts['wrong']}  多读:{counts['extra']}  漏读:{counts['missing']}"
            f"{oov_note}"
        )
        auto_diff = diff_html(greedy_str.split(), beam_str.split(), "贪心", f"集束({BEAM_SIZE})")
        ref_display = ref_ipa if ref_source == "转换器(中文)" else res["ref_str"]
        return greedy_str, ref_display, align, summary, auto_diff

    # 无参考：自动用贪心 vs 集束两套解码互比差异
    auto_diff = diff_html(greedy_str.split(), beam_str.split(), "贪心", f"集束({BEAM_SIZE})")
    summary = f"未提供参考文本，已自动用「贪心 vs 集束({BEAM_SIZE})」两套解码互比差异（见下方表格）。"
    return greedy_str, beam_str, "", summary, auto_diff


def analyze_ipa_features(audio_path, ipa_text):
    """IPA 语音特征分析：上传音频识别后分析，或直接粘贴 IPA 分析。"""
    if audio_path is None and not (ipa_text and ipa_text.strip()):
        return "", "请提供音频，或在右侧文本框直接粘贴严式 IPA。"
    if audio_path is not None:
        beam = STATE.get("beam", BEAM_SIZE) or 0
        _, ipa = decode_ids(audio_path, beam)
        src = "音频识别结果"
    else:
        ipa = ipa_text.strip()
        src = "粘贴的 IPA"
    a = analyze_ipa(ipa, STATE.get("ipa2tone"))
    html = to_html(a)
    parts = [f"分析对象：{src}（共 {a['n']} 个音节）"]
    if a["feature_counts"]:
        top = "、".join(f"{zh}×{n}" for zh, n in a["feature_counts"].most_common(6))
        parts.append(f"主要附加符号特征：{top}")
    else:
        parts.append("未检出附加符号（均为基础段 + 调值）")
    return html, "\n".join(parts)


def build_ui():
    with gr.Blocks(title="普通话严式IPA语音识别") as demo:
        gr.Markdown(
            "# 普通话严式国际音标（IPA）语音识别\n\n"
            "基于 SenseVoiceSmall 编码器 + 严式 IPA CTC 头微调。支持上传/录制音频识别，"
            "并可与**转换器生成的真值**自动比对差异（逐音节声调对齐高亮），"
            "以及逐音节分析 IPA 的附加符号与语音特征。"
        )
        with gr.Tab("识别"):
            audio_in = gr.Audio(label="上传或录制音频", type="filepath",
                               sources=["upload", "microphone"])
            btn1 = gr.Button("识别", variant="primary")
            with gr.Row():
                out_ipa = gr.Textbox(label="识别结果（严式 IPA）", lines=3)
                info1 = gr.Textbox(label="信息", lines=1)
            btn1.click(predict_ipa, [audio_in], [out_ipa, info1])
        with gr.Tab("比对（转换器真值）"):
            audio_in2 = gr.Audio(label="上传或录制音频", type="filepath",
                                sources=["upload", "microphone"])
            ref_source = gr.Radio(
                choices=["转换器(中文)", "直接填IPA"], value="转换器(中文)",
                label="参考来源：转换器自动把中文转成严式 IPA 真值（与模型同源）；或直接填空格分隔的 IPA")
            ref_in = gr.Textbox(
                label="参考文本",
                lines=2,
                placeholder="转换器模式：填中文，如「床前明月光，疑是地上霜」  |  直接IPA模式：填空格分隔的 IPA 音节")
            btn2 = gr.Button("识别并比对", variant="primary")
            with gr.Row():
                out_ipa2 = gr.Textbox(label="识别结果（严式 IPA）", lines=2)
                out_ref = gr.Textbox(label="参考真值（转换器 IPA / 填入 IPA）", lines=2)
            out_summary = gr.Textbox(label="统计 / 说明", lines=3)
            out_align = gr.HTML(label="逐音节声调对齐")
            out_auto = gr.HTML(label="自动差异（贪心 vs 集束）")
            btn2.click(compare, [audio_in2, ref_source, ref_in],
                       [out_ipa2, out_ref, out_align, out_summary, out_auto])
        with gr.Tab("IPA 语音特征分析"):
            gr.Markdown("逐音节拆解严式 IPA，标注调值与每个附加符号的语音学特征"
                        "（舌位前/后移、送气、唇化、清化、唯闭/入声、元音中央化/开闭等），并汇总统计。")
            with gr.Row():
                audio_in3 = gr.Audio(label="上传或录制音频（自动识别后分析）", type="filepath",
                                    sources=["upload", "microphone"])
                ipa_in = gr.Textbox(label="或直接粘贴严式 IPA（空格分隔）", lines=2,
                                    placeholder="t͡ʂ̺ʰwɑ̟ŋ̚˧˥ t͡ɕʰjɛ̠n̚˧˥ ...")
            btn3 = gr.Button("分析", variant="primary")
            out_feat_html = gr.HTML(label="逐音节特征明细 + 统计")
            out_feat_summary = gr.Textbox(label="概要", lines=3)
            btn3.click(analyze_ipa_features, [audio_in3, ipa_in],
                       [out_feat_html, out_feat_summary])
        gr.Markdown(
            "注：模型权重不随代码发布，请从 ModelScope 下载后通过 `python app.py --ckpt <路径>` 加载。"
            "「转换器」数据来自 nk2028/putonghua-ipa-converter（CC0），与训练词表同源；"
            "训练数据依据授权不公开，详见 README。"
        )
    return demo


def main():
    load()
    demo = build_ui()
    a = parse_args()
    demo.launch(server_port=a.port, share=a.share, inbrowser=False)


if __name__ == "__main__":
    main()
