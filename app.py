# -*- coding: utf-8 -*-
"""Gradio 演示：普通话严式国际音标（IPA）语音识别。

功能：
  1. 识别：上传或录制音频 -> 输出严式 IPA 音节序列（含声调调值）。
  2. 比对：上传音频并（可选）填入参考 IPA 文本 -> 自动做逐音节声调对齐与差异高亮；
     即使不填参考文本，也会自动用「贪心 / 集束搜索」两套解码结果互相比对差异。

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
)

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
    return ap.parse_args()


STATE = {}


def load():
    a = parse_args()
    model, token2id, dev = build_model(a.vocab_path, a.ckpt, a.model_dir, freeze_encoder=True)
    id2tok = {v: k for k, v in token2id.items()}
    ipa2tone = json.load(open(a.ipa2tone_path, encoding="utf-8")) if a.ipa2tone_path else {}
    STATE.update(
        model=model, token2id=token2id, id2tok=id2tok,
        ipa2tone=ipa2tone, dev=dev, max_seconds=a.max_seconds,
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
    ids, s = decode_ids(audio_path, 0)
    return s, f"共识别 {len(ids)} 个 IPA 音节"


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


def compare(audio_path, ref_text):
    """自动比对差异：有参考则与参考对齐；无参考则贪心 vs 集束自动互比。"""
    if audio_path is None:
        return "", "", "请先上传音频", "—", ""
    greedy_ids, greedy_str = decode_ids(audio_path, 0)
    beam_ids, beam_str = decode_ids(audio_path, BEAM_SIZE)

    if ref_text and ref_text.strip():
        ref_tokens = ref_text.strip().split()
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
        )
        auto_diff = diff_html(greedy_str.split(), beam_str.split(), "贪心", f"集束({BEAM_SIZE})")
        return greedy_str, res["ref_str"], align, summary, auto_diff

    # 无参考：自动用贪心 vs 集束两套解码互比差异
    auto_diff = diff_html(greedy_str.split(), beam_str.split(), "贪心", f"集束({BEAM_SIZE})")
    summary = f"未提供参考文本，已自动用「贪心 vs 集束({BEAM_SIZE})」两套解码互比差异（见下方表格）。"
    return greedy_str, beam_str, "", summary, auto_diff


def build_ui():
    with gr.Blocks(title="普通话严式IPA语音识别") as demo:
        gr.Markdown(
            "# 普通话严式国际音标（IPA）语音识别\n\n"
            "基于 SenseVoiceSmall 编码器 + 严式 IPA CTC 头微调。支持上传/录制音频识别，"
            "并可与参考 IPA 文本自动比对差异（逐音节声调对齐高亮）。"
        )
        with gr.Tab("识别"):
            audio_in = gr.Audio(label="上传或录制音频", type="filepath",
                               sources=["upload", "microphone"])
            btn1 = gr.Button("识别", variant="primary")
            with gr.Row():
                out_ipa = gr.Textbox(label="识别结果（严式 IPA）", lines=3)
                info1 = gr.Textbox(label="信息", lines=1)
            btn1.click(predict_ipa, [audio_in], [out_ipa, info1])
        with gr.Tab("比对（自动差异）"):
            audio_in2 = gr.Audio(label="上传或录制音频", type="filepath",
                                sources=["upload", "microphone"])
            ref_in = gr.Textbox(
                label="参考文本（可选，空格分隔的 IPA 音节，例如：ɡ̊wa̠n̚˥ x̞wa̠ɪ̯˧˥）",
                lines=2, placeholder="ɡ̊wa̠n̚˥ x̞wa̠ɪ̯˧˥  （留空则自动用贪心 vs 集束比对）")
            btn2 = gr.Button("识别并比对", variant="primary")
            with gr.Row():
                out_ipa2 = gr.Textbox(label="识别结果（严式 IPA）", lines=2)
                out_ref = gr.Textbox(label="参考文本（回声）", lines=2)
            out_summary = gr.Textbox(label="统计 / 说明", lines=2)
            out_align = gr.HTML(label="逐音节声调对齐（有参考时）")
            out_auto = gr.HTML(label="自动差异（贪心 vs 集束）")
            btn2.click(compare, [audio_in2, ref_in],
                       [out_ipa2, out_ref, out_align, out_summary, out_auto])
        gr.Markdown(
            "注：模型权重不随代码发布，请从 ModelScope 下载后通过 `python app.py --ckpt <路径>` 加载。"
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
