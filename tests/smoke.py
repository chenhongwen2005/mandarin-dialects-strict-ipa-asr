# -*- coding: utf-8 -*-
"""Smoke test —— 不依赖模型权重 / 基础模型的轻量校验。

覆盖：
  - 相对路径按项目根解析（修复从任意目录启动的坑）
  - 词表加载（1427 类严式 IPA）
  - CTC 贪心 / 集束解码（修复随机初始化无权重时输出乱码的对照逻辑）
  - Levenshtein 对齐

运行：python tests/smoke.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import torch

import utils  # noqa: E402
from utils import (  # noqa: E402
    ctc_beam_search,
    ctc_greedy_decode,
    levenshtein_align,
    load_vocab,
    resolve_local_path,
)


def test_resolve_local_path():
    base = "/tmp/proj"
    assert resolve_local_path("a/b.json", base=base) == os.path.join(base, "a/b.json")
    assert resolve_local_path("/abs/c.json") == "/abs/c.json"
    assert resolve_local_path("") == ""
    # 默认以 PROJECT_ROOT 为基准
    expected = os.path.normpath(os.path.join(ROOT, "vocab", "x.json"))
    assert os.path.normpath(resolve_local_path("vocab/x.json")) == expected
    print("PASS resolve_local_path")


def test_load_vocab():
    path = os.path.join(ROOT, "vocab", "vocab_mandarin_ipa_combined.json")
    token2id, id2token = load_vocab(path)
    assert len(token2id) == 1427, len(token2id)
    # 双向映射一致
    for k, v in token2id.items():
        assert id2token[v] == k
    print("PASS load_vocab (%d entries)" % len(token2id))


def _make_logits():
    # T=6, B=1, V=5 ; argmax over T -> [1, 0, 2, 0, 3, 0]
    # 相邻时刻用不同标签、以 blank 分隔，使贪心与集束结果一致。
    T, B, V = 6, 1, 5
    lp = torch.full((T, B, V), -10.0)
    lp[0, 0, 1] = 5.0
    lp[1, 0, 0] = 5.0  # blank
    lp[2, 0, 2] = 5.0
    lp[3, 0, 0] = 5.0  # blank
    lp[4, 0, 3] = 5.0
    lp[5, 0, 0] = 5.0  # blank
    return lp


def test_greedy():
    out = ctc_greedy_decode(_make_logits())[0]
    assert out == [1, 2, 3], out
    print("PASS ctc_greedy_decode ->", out)


def test_beam():
    id2tok = {0: "<blank>", 1: "a", 2: "b", 3: "c", 4: "d"}
    lp = _make_logits().squeeze(1)  # 集束搜索接收单样本 (T, V)
    ids = ctc_beam_search(lp, beam_size=4, id2token=id2tok)
    assert ids == [1, 2, 3], ids
    print("PASS ctc_beam_search ->", ids)


def test_align():
    ops = levenshtein_align(["a", "b"], ["a", "c"])
    assert any(o[0] == "match" for o in ops)
    print("PASS levenshtein_align")


if __name__ == "__main__":
    test_resolve_local_path()
    test_load_vocab()
    test_greedy()
    test_beam()
    test_align()
    print("\nALL SMOKE TESTS PASSED")
