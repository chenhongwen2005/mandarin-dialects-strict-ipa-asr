# -*- coding: utf-8 -*-
"""粤语适配：在冻结的 SenseVoiceSmall Conformer 编码器上注入 LoRA 适配器，
并挂载一个新的粤语 IPA CTC 头。

设计要点：
  * 复用 SenseVoiceIpa 的 frontend + encoder 前向（fp32 frontend 无梯度，bf16 encoder）。
  * 编码器底座（含 frontend）完全冻结，只训练 LoRA 低秩残差 + 粤语 CTC 头。
  * LoRA 注入目标为编码器里 4 类 Linear（按后缀匹配，自动兼容 encoders0/encoders 两套命名）：
        self_attn.linear_q_k_v, self_attn.linear_out,
        feed_forward.w_1,      feed_forward.w_2
  * 词表为粤语 IPA 音节（vocab_cantonese_ipa_combined.json，<blank>:0, <unk>:1）。

用法：
    from model_cantonese import SenseVoiceIpaLora
    model = SenseVoiceIpaLora(vocab_size=1580, lora_rank=32)
    model.unfreeze_lora_and_head()   # 仅 LoRA + 头可训
"""

import os
import sys

# 保证 src/ 在路径中（直接运行或作为模块导入都可用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List

import torch
import torch.nn as nn

from model import SenseVoiceIpa


# ── LoRA 线性层 ────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """把普通 Linear 包成 冻结底座 + 低秩残差 的形式。

    y = base(x) + (x @ A^T @ B^T) * scale,  scale = alpha / rank
    其中 base 权重冻结、不参与梯度；A/B 为可训低秩参数。
    """

    def __init__(self, linear: nn.Linear, rank: int = 32, alpha: int = 32):
        super().__init__()
        self.linear = linear
        # 冻结底座
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        # Linear.weight 形状为 (out_features, in_features)
        out_f, in_f = linear.weight.shape
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        # 初始化：A kaiming，B 置零 -> 初始 LoRA 残差=0，等价于原底座
        nn.init.kaiming_uniform_(self.lora_A, a=5.0 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = self.linear(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + lora


# 目标 Linear 的后缀（兼容 encoders0 / encoders 两套命名）
LORA_TARGET_SUFFIXES = (
    "self_attn.linear_q_k_v",
    "self_attn.linear_out",
    "feed_forward.w_1",
    "feed_forward.w_2",
)


def inject_lora(encoder: nn.Module, rank: int = 32, alpha: int = 32) -> List[str]:
    """在 encoder 中所有匹配目标后缀的 Linear 上注入 LoRA。

    按对象 id 去重，避免同一模块被两个命名路径重复包装。
    返回被注入的完整参数名列表（用于校验/日志）。
    """
    applied = []
    seen_ids = set()
    # 先收集再替换，避免迭代中修改
    targets = []
    for name, mod in encoder.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not any(name.endswith(s) for s in LORA_TARGET_SUFFIXES):
            continue
        if id(mod) in seen_ids:
            continue
        seen_ids.add(id(mod))
        targets.append((name, mod))

    for name, mod in targets:
        parent_name, child = name.rsplit(".", 1)
        parent = encoder.get_submodule(parent_name)
        wrapper = LoRALinear(mod, rank=rank, alpha=alpha)
        setattr(parent, child, wrapper)
        applied.append(name)
    return applied


class SenseVoiceIpaLora(SenseVoiceIpa):
    def __init__(
        self,
        vocab_size: int,
        model_dir: str = "iic/SenseVoiceSmall",
        encoder_dim: int = 512,
        lora_rank: int = 32,
        lora_alpha: int = 32,
        frontend_dither: float = 0.0,
    ):
        # 冻住底座（frontend + encoder 全部 requires_grad=False）
        super().__init__(
            vocab_size=vocab_size,
            model_dir=model_dir,
            encoder_dim=encoder_dim,
            freeze_encoder=True,
            frontend_dither=frontend_dither,
        )
        # 注入 LoRA（在已冻结的 encoder 之上）
        applied = inject_lora(self.sense_voice.encoder, rank=lora_rank, alpha=lora_alpha)
        self.lora_rank = lora_rank
        self.lora_applied = applied
        print(f"[SenseVoiceIpaLora] 注入 LoRA 到 {len(applied)} 个 Linear "
              f"(rank={lora_rank}, alpha={lora_alpha})")
        self.unfreeze_lora_and_head()
        self.count_params()

    def unfreeze_lora_and_head(self):
        """仅 LoRA 适配器 + 粤语 CTC 头可训；底座（含 frontend）保持冻结。"""
        # 先全部冻住底座
        for p in self.sense_voice.parameters():
            p.requires_grad = False
        if self.frontend is not None:
            for p in self.frontend.parameters():
                p.requires_grad = False
        # 解冻 LoRA（只解冻 lora_A / lora_B，绝不动底座 weight）
        for m in self.sense_voice.encoder.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.requires_grad = True
                m.lora_B.requires_grad = True
        # 解冻 CTC 头
        for p in self.ctc_head.parameters():
            p.requires_grad = True
        print("[SenseVoiceIpaLora] 仅 LoRA + 方言 CTC 头可训；底座冻结")

    def lora_state_dict(self):
        """只取可训参数（LoRA + CTC 头），用于轻量存盘。"""
        sd = {}
        for k, p in self.named_parameters():
            if p.requires_grad:
                sd[k] = p.detach().clone()
        return sd


if __name__ == "__main__":
    print("=== SenseVoiceIpaLora smoke (build only) ===")
    # 仅构建 + 前向需真实 vocab；这里只验证构造与参数统计
    import json
    vocab = json.load(open("../data/cantonese_ipa/vocab_cantonese_ipa_combined.json",
                           encoding="utf-8"))
    m = SenseVoiceIpaLora(vocab_size=len(vocab))
    print("lora applied count:", len(m.lora_applied))
    print("OK")
