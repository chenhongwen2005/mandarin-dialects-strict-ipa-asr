# -*- coding: utf-8 -*-
"""SenseVoiceIpa — 在预训练 SenseVoiceSmall 的 Conformer 编码器上挂载一个
严式 IPA（International Phonetic Alphabet）CTC 头，用于普通话音节级音素 + 声调识别。

- 基础模型：FunASR 的 SenseVoiceSmall（含 WavFrontend 特征提取 + Conformer 编码器）。
- 输出词表：vocab_mandarin_ipa_combined.json（1427 类，含 <blank>/<unk> 与严式 IPA 音节）。
- 训练策略：可冻结编码器只训头，也可解冻编码器 + 头做全量微调（本项目最终采用）。
- frontend.dither 固定为 0：避免每次前向随机加噪导致特征非确定性（对可复现训练/推理必需）。

用法（在 src/ 目录下）:
    from model import SenseVoiceIpa
    model = SenseVoiceIpa(vocab_size=1427, model_dir="iic/SenseVoiceSmall", freeze_encoder=True)
"""

from typing import Tuple

import torch
import torch.nn as nn


class SenseVoiceIpa(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dir: str = "iic/SenseVoiceSmall",
        encoder_dim: int = 512,
        freeze_encoder: bool = True,
        frontend_dither: float = 0.0,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size

        from funasr import AutoModel

        print(f"[SenseVoiceIpa] loading base model from {model_dir} ...")
        am = AutoModel(model=model_dir, disable_update=True, device="cpu")
        self.sense_voice: nn.Module = am.model
        self.frontend: nn.Module = am.kwargs["frontend"]

        # FunASR 的 WavFrontend 默认 dither=1.0，会在每次前向给波形叠加随机噪声，
        # 导致同一输入两次前向结果不同。固定为 0 以获得确定性特征。
        if self.frontend is not None and hasattr(self.frontend, "dither"):
            old = self.frontend.dither
            self.frontend.dither = float(frontend_dither)
            print(f"[SenseVoiceIpa] frontend.dither: {old} -> {self.frontend.dither}")

        if freeze_encoder:
            self.freeze_encoder()

        self.ctc_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, vocab_size),
        )

        print("[SenseVoiceIpa] ready.")
        self.count_params()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.frontend is not None:
            self.frontend = self.frontend.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        if self.frontend is not None:
            self.frontend = self.frontend.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        if self.frontend is not None:
            self.frontend = self.frontend.cpu()
        return self

    def freeze_encoder(self):
        for p in self.sense_voice.parameters():
            p.requires_grad = False
        if self.frontend is not None:
            for p in self.frontend.parameters():
                p.requires_grad = False

    def unfreeze_encoder_and_head(self):
        for p in self.sense_voice.encoder.parameters():
            p.requires_grad = True
        for p in self.ctc_head.parameters():
            p.requires_grad = True
        if self.frontend is not None:
            for p in self.frontend.parameters():
                p.requires_grad = False
        print("[SenseVoiceIpa] encoder + IPA head 解冻, frontend 冻结")

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        autocast_bf16: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = speech.device
        if self.frontend is not None:
            self.frontend = self.frontend.to(device)

        # 1) Frontend: 始终 fp32, 无梯度（特征提取, 不训练）
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            with torch.set_grad_enabled(False):
                feats, feats_lens = self.frontend(speech, speech_lengths)
        feats = feats.to(device)
        feats_lens = feats_lens.to(device)

        # 2) Encoder
        enc_grad = any(p.requires_grad for p in self.sense_voice.encoder.parameters())
        use_amp = autocast_bf16 and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            with torch.set_grad_enabled(enc_grad):
                enc_result = self.sense_voice.encoder(feats, feats_lens)
                encoder_out = enc_result[0]
                encoder_out_lens = enc_result[1]

        # 3) IPA CTC 头（可训）
        logits = self.ctc_head(encoder_out)
        logits = logits.float()  # 防 bf16 log_softmax 下溢 -> nan
        log_probs = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # (T, B, V) 供 CTC
        return log_probs, encoder_out_lens

    def count_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Trainable : {trainable:>12,}")
        print(f"  Total     : {total:>12,}")
        return trainable, total


if __name__ == "__main__":
    print("=== SenseVoiceIpa smoke test ===\n")
    model = SenseVoiceIpa(vocab_size=1427)
    model.eval()
    B, SR = 2, 16000
    speech = torch.randn(B, SR * 3)
    speech_lengths = torch.tensor([SR * 3, SR * 3 - 8000])
    with torch.no_grad():
        lp, enc_lens = model(speech, speech_lengths)
        print(f"log_probs      : {tuple(lp.shape)}  (T,B,V)")
        print(f"encoder_lens   : {enc_lens.tolist()}")
        assert lp.shape[1] == B
        assert lp.shape[2] == 1427
        print("\nOK smoke test passed")
