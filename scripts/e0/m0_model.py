"""M0-CausalDense-v1 — frozen architecture for E0 Q2 qualification.

Exact geometry per docs/experiments/e0/Q2_Q3_EXECUTION_RELEASE.md:
  CMDR-Lex-v1 token IDs -> learned embedding 512 -> N pre-norm causal blocks
  -> final RMSNorm -> post-norm state at <DECIDE> -> linear 512->3 classifier.

Frozen candidates differ only by depth:
  C0 15 layers 53,232,643 params
  C1 18 layers 63,459,331 params
  C2 21 layers 73,686,019 params

Per-block parameter arithmetic (verified against the frozen totals):
  attn q/k/v/o: 4 * 512*512               = 1,048,576
  SwiGLU gate/up/down: 3 * 512*1536       = 2,359,296
  two RMSNorm scales: 2 * 512             =     1,024
                                            --------
                                            3,408,896
Total(N) = 4096*512 + N*3,408,896 + 512 (final norm) + 512*3 + 3 (classifier)
C0: 2,097,152 + 15*3,408,896 + 1,539 = 53,232,643  (matches frozen value)

Implementation notes (recorded in run manifests):
  - RoPE uses the rotate_half (GPT-NeoX / Llama-style) convention, rotary dim 64.
  - Attention is manual matmul-softmax-matmul (deterministic under
    torch.use_deterministic_algorithms(True); SDPA kernels are not used).
  - Right padding; with a causal mask the <DECIDE> position attends only
    real tokens, so no key-padding mask is required for loss correctness.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 4096
HIDDEN = 512
NUM_Q_HEADS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 64
FFN_HIDDEN = 1536
RMS_EPS = 1e-5
ROPE_THETA = 10000.0
ROPE_DIM = 64
MAX_LEN = 384
NUM_CLASSES = 3
PAD_TOKEN_ID = 0

CANDIDATES: dict[str, dict[str, int]] = {
    "C0": {"layers": 15, "params": 53_232_643},
    "C1": {"layers": 18, "params": 63_459_331},
    "C2": {"layers": 21, "params": 73_686_019},
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = RMS_EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def build_rope_tables(max_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=device) / ROPE_DIM)
    )
    positions = torch.arange(max_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


class M0Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, NUM_Q_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(NUM_Q_HEADS * HEAD_DIM, HIDDEN, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        scores = scores + causal_mask
        attn = F.softmax(scores.float(), dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, seq_len, NUM_Q_HEADS * HEAD_DIM)
        return self.o_proj(out)


class M0MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, FFN_HIDDEN, bias=False)
        self.up_proj = nn.Linear(HIDDEN, FFN_HIDDEN, bias=False)
        self.down_proj = nn.Linear(FFN_HIDDEN, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class M0Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(HIDDEN)
        self.attention = M0Attention()
        self.post_attention_layernorm = RMSNorm(HIDDEN)
        self.mlp = M0MLP()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attention(self.input_layernorm(x), cos, sin, causal_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class M0CausalDenseV1(nn.Module):
    """M0 student: no LM head; classification from post-final-norm state at <DECIDE>."""

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.layers = nn.ModuleList(M0Block() for _ in range(num_layers))
        self.norm = RMSNorm(HIDDEN)
        self.classifier = nn.Linear(HIDDEN, NUM_CLASSES, bias=True)
        self.num_layers = num_layers

    def _init_parameters(self) -> None:
        # Frozen init: Xavier uniform gain 1 on embedding / attn / MLP / classifier
        # matrix; RMSNorm scales exactly 1; classifier bias exactly 0.
        nn.init.xavier_uniform_(self.embed_tokens.weight, gain=1.0)
        for layer in self.layers:
            for module in (
                layer.attention.q_proj,
                layer.attention.k_proj,
                layer.attention.v_proj,
                layer.attention.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                with torch.no_grad():
                    norm.weight.fill_(1.0)
        with torch.no_grad():
            self.norm.weight.fill_(1.0)
            self.classifier.bias.zero_()
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

    def forward(self, token_ids: torch.Tensor, decide_index: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, T) right-padded; decide_index: (B,) position of <DECIDE>."""
        bsz, seq_len = token_ids.shape
        device = token_ids.device
        cos, sin = build_rope_tables(seq_len, device)
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        x = self.embed_tokens(token_ids)
        for layer in self.layers:
            x = layer(x, cos, sin, causal_mask)
        x = self.norm(x)
        decide_state = x[torch.arange(bsz, device=device), decide_index]
        return self.classifier(decide_state)


def build_model(candidate: str, seed: int, device: torch.device) -> M0CausalDenseV1:
    """Deterministic construction: seed the CPU RNG, build in fixed module order."""
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate}")
    gen_state = torch.get_rng_state()
    torch.manual_seed(seed)
    model = M0CausalDenseV1(CANDIDATES[candidate]["layers"])
    model._init_parameters()
    torch.set_rng_state(gen_state)
    model.to(device)
    expected = CANDIDATES[candidate]["params"]
    actual = sum(p.numel() for p in model.parameters())
    if actual != expected:
        raise AssertionError(
            f"{candidate} parameter count {actual} != frozen {expected}"
        )
    return model
