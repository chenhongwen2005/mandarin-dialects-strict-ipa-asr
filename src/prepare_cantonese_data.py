# -*- coding: utf-8 -*-
"""把 HuggingFace 的粤语方言 CSV 数据集转换为本项目 WavDataset 所需的
wav.scp / text_ipa 格式，并生成粤语 IPA 词表与 ipa2tone 映射。

输入 CSV（HuggingFace 导出）:
    Audio:FILE,Text:LABEL
    cantonese_dialect_small/common_voice_zh-HK_22235680.wav,两 个 人 扯 猫 尾 唔 认 数

注意：Audio 列为相对路径，且真实 wav 存在于「双层嵌套」目录
    <data_dir>/cantonese_dialect_small/cantonese_dialect_small/*.wav
（CSV 中只写单层 cantonese_dialect_small/...）。脚本会自动补一层并回退。

输出（--out_dir，默认 data/cantonese_ipa/）:
    train_scp, train_text, val_scp, val_text
    vocab_cantonese_ipa_combined.json   （<blank>:0, <unk>:1, 其余 IPA 音节）
    ipa2tone_cantonese.json             （IPA 音节 -> 调类 1-6）

用法:
    runtime/python.exe prepare_cantonese_data.py
    runtime/python.exe prepare_cantonese_data.py --data_dir "C:/Users/HP/Downloads/粤语" --out_dir data/cantonese_ipa
"""

import argparse
import csv
import json
import os
import sys

# 允许从任意目录启动
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from cantonese_g2p import hanzi_to_ipa_syllables, build_ipa2tone  # noqa: E402


def resolve_audio(col: str, base: str):
    """将 CSV 的相对音频路径解析为真实绝对路径。

    优先用原样拼接；若不存在，则自动补一层首段目录（应对双层嵌套）。
    """
    p = os.path.join(base, col)
    if os.path.isfile(p):
        return p
    parts = col.replace("\\", "/").split("/")
    if len(parts) > 1:
        p2 = os.path.join(base, parts[0], col)
        if os.path.isfile(p2):
            return p2
    return None


def parse_csv(csv_path: str, base: str):
    """返回 [(uid, audio_abs, ipa_syllables_str), ...] 与跳过计数。"""
    rows = []
    skipped = 0
    missing_audio = 0
    empty_g2p = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # 兼容有/无表头：若首行是表头则跳过
        if header and header[0].strip().startswith("Audio"):
            pass
        else:
            f.seek(0)
        for row in reader:
            if len(row) < 2:
                skipped += 1
                continue
            audio_col = row[0].strip()
            text = row[1].strip()
            if not audio_col or not text:
                skipped += 1
                continue
            audio = resolve_audio(audio_col, base)
            if audio is None:
                missing_audio += 1
                skipped += 1
                continue
            syls = hanzi_to_ipa_syllables(text)
            if not syls:
                empty_g2p += 1
                skipped += 1
                continue
            uid = os.path.splitext(os.path.basename(audio_col))[0]
            rows.append((uid, audio, " ".join(syls)))
    return rows, dict(skipped=skipped, missing_audio=missing_audio, empty_g2p=empty_g2p)


def write_split(rows, scp_path, text_path):
    with open(scp_path, "w", encoding="utf-8") as fs, \
         open(text_path, "w", encoding="utf-8") as ft:
        for uid, audio, syls in rows:
            fs.write(f"{uid} {audio}\n")
            ft.write(f"{uid} {syls}\n")


def main():
    ap = argparse.ArgumentParser()
    default_data = "C:/Users/HP/Downloads/粤语"
    ap.add_argument("--data_dir", default=default_data, help="含 CSV 与音频根目录")
    ap.add_argument("--train_csv", default=None, help="训练 CSV（默认 <data_dir>/cantonese_dialect_trainsets.csv）")
    ap.add_argument("--val_csv", default=None, help="验证 CSV（默认 <data_dir>/cantonese_dialect_valsets.csv）")
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "data", "cantonese_ipa"),
                    help="输出目录")
    args = ap.parse_args()

    data_dir = args.data_dir
    train_csv = args.train_csv or os.path.join(data_dir, "cantonese_dialect_trainsets.csv")
    val_csv = args.val_csv or os.path.join(data_dir, "cantonese_dialect_valsets.csv")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[prepare] data_dir = {data_dir}")
    print(f"[prepare] train_csv= {train_csv}  exists={os.path.isfile(train_csv)}")
    print(f"[prepare] val_csv  = {val_csv}  exists={os.path.isfile(val_csv)}")

    train_rows, train_stat = parse_csv(train_csv, data_dir)
    val_rows, val_stat = parse_csv(val_csv, data_dir)

    write_split(train_rows, os.path.join(out_dir, "train_scp"),
                os.path.join(out_dir, "train_text"))
    write_split(val_rows, os.path.join(out_dir, "val_scp"),
                os.path.join(out_dir, "val_text"))

    # 构建词表：train 优先，再补 val 独有 token
    vocab = {"<blank>": 0, "<unk>": 1}
    for uid, _, syls in train_rows:
        for t in syls.split():
            if t not in vocab:
                vocab[t] = len(vocab)
    val_only = 0
    for uid, _, syls in val_rows:
        for t in syls.split():
            if t not in vocab:
                vocab[t] = len(vocab)
                val_only += 1

    ipa2tone = build_ipa2tone(vocab)

    with open(os.path.join(out_dir, "vocab_cantonese_ipa_combined.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "ipa2tone_cantonese.json"), "w", encoding="utf-8") as f:
        json.dump(ipa2tone, f, ensure_ascii=False, indent=2)

    print("\n=== 结果 ===")
    print(f"训练样本 : {len(train_rows):>6}  (跳过 {train_stat['skipped']}: "
          f"缺音频 {train_stat['missing_audio']}, G2P空 {train_stat['empty_g2p']})")
    print(f"验证样本 : {len(val_rows):>6}  (跳过 {val_stat['skipped']}: "
          f"缺音频 {val_stat['missing_audio']}, G2P空 {val_stat['empty_g2p']})")
    print(f"词表大小 : {len(vocab):>6}  (val 独有 {val_only})")
    print(f"有调类映射的 token 数: {len(ipa2tone)}")
    print(f"输出目录 : {out_dir}")


if __name__ == "__main__":
    main()
