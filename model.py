"""
model.py — Looped MoE Transformer with per-loop differentiation
=================================================================
Supports arbitrary looping topologies:
  - No looping (standard baseline)
  - Single looped core with optional pre/post unique layers
  - Multiple looped groups with independent params, e.g.:
    2 pre + [2×5] + [2×5] + 2 post = 24 effective depth, 8 unique layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1024
    n_experts: int = 8
    top_k: int = 2
    seq_len: int = 512
    vocab_size: int = 100277
    dropout: float = 0.0

    # Topology: a list of (n_layers, n_loops) tuples. Each tuple defines a group of
    # n_layers unique transformer blocks that are run n_loops times in sequence.
    # Examples (effective depth = sum of n_layers * n_loops):
    #   [(2, 1), (4, 5), (2, 1)]         -> 2 + 20 + 2 = 24 effective depth
    #   [(2, 1), (2, 5), (2, 5), (2, 1)] -> 2 + 10 + 10 + 2 = 24 effective depth
    #   [(20, 1)]                        -> 20 unique layers, no looping (baseline)
    topology: List[Tuple[int, int]] = field(default_factory=lambda: [(4, 5)])

    # Per-loop differentiation (applied to looped groups only)
    per_loop_routers: bool = False
    per_loop_attn: bool = False     # True = separate attention Q/K/V/O per loop (only experts shared)
    use_attn_lora: bool = False
    lora_rank: int = 32
    qk_norm: bool = True            # RMSNorm on Q,K per head before RoPE

    @property
    def effective_depth(self):
        return sum(n_layers * n_loops for n_layers, n_loops in self.topology)

    @property
    def n_unique_layers(self):
        return sum(n_layers for n_layers, _ in self.topology)

    @property
    def max_loops(self):
        return max(n_loops for _, n_loops in self.topology)

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    # Exposed for compatibility with the older single-group logging interface.
    @property
    def n_loops(self):
        return self.max_loops

    @property
    def n_pre_layers(self):
        if self.topology[0][1] == 1:
            return self.topology[0][0]
        return 0

    @property
    def n_post_layers(self):
        if self.topology[-1][1] == 1:
            return self.topology[-1][0]
        return 0


# ============================================================
# Building blocks
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, d_head, max_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos.unsqueeze(0).unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)


class LoRAAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, rank, n_copies):
        super().__init__()
        self.A = nn.Parameter(torch.randn(n_copies, rank, in_dim) * (1.0 / math.sqrt(in_dim)))
        self.B = nn.Parameter(torch.zeros(n_copies, out_dim, rank))

    def forward(self, x, copy_idx):
        A = self.A[copy_idx]
        B = self.B[copy_idx]
        return x @ A.t() @ B.t()


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, n_loops_for_lora: int = 1):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model

        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.rope = RotaryEmbedding(cfg.d_head, max_len=cfg.seq_len)

        self.has_qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.d_head)
            self.k_norm = RMSNorm(cfg.d_head)

        self.has_lora = cfg.use_attn_lora and n_loops_for_lora > 1
        if self.has_lora:
            self.qk_lora = LoRAAdapter(cfg.d_model, 2 * cfg.d_model, cfg.lora_rank, n_loops_for_lora)

    def forward(self, x, loop_idx=0):
        B, S, D = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        if self.has_lora:
            qk_delta = self.qk_lora(x, loop_idx)
            dq, dk = qk_delta.chunk(2, dim=-1)
            q = q + dq
            k = k + dk

        q = q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        if self.has_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos, sin = self.rope(S, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(out)


def _has_grouped_mm():
    """Check if torch.nn.functional.grouped_mm is available (PyTorch >= 2.8)."""
    return hasattr(F, "grouped_mm") or hasattr(torch, "_grouped_mm")


def grouped_mm_status(device=None, dtype=None):
    """Return a dict describing grouped_mm availability and runtime readiness.

    Intended for one-time logging at the start of training (after the model has
    been moved to its target device) so the active expert-dispatch path is
    recorded in the run log.

    The runtime dispatch checks: (1) API is available, (2) device is CUDA. The
    bf16 requirement is satisfied via autocast in the training loop and is not
    enforced at dispatch time.
    """
    has_api = _has_grouped_mm()
    api_name = (
        "F.grouped_mm" if hasattr(F, "grouped_mm")
        else "torch._grouped_mm" if hasattr(torch, "_grouped_mm")
        else None
    )
    on_cuda = device is not None and str(device).startswith("cuda")
    is_bf16 = dtype is not None and dtype == torch.bfloat16

    will_activate = has_api and on_cuda

    return {
        "has_api": has_api,
        "api_name": api_name,
        "on_cuda": on_cuda,
        "is_bf16": is_bf16,
        "active": will_activate,
    }


def _grouped_mm(mat_a, mat_b, offs):
    """Dispatch to the best available grouped_mm implementation."""
    if hasattr(F, "grouped_mm"):
        return F.grouped_mm(mat_a, mat_b, offs=offs)
    elif hasattr(torch, "_grouped_mm"):
        return torch._grouped_mm(mat_a, mat_b, offs=offs)
    else:
        raise RuntimeError("No grouped_mm implementation available; requires PyTorch >= 2.8.")


class MoELayer(nn.Module):
    """
    Mixture of Experts with grouped GEMM dispatch.
    All expert weights are stacked into single tensors.
    SwiGLU activation: out = w2(silu(w1(x)) * w3(x))

    Uses torch.nn.functional.grouped_mm when available (PyTorch >= 2.8
    on CUDA with bf16), otherwise falls back to a Python loop over experts.
    """
    def __init__(self, cfg: ModelConfig, n_loops_for_routers: int = 1):
        super().__init__()
        self.cfg = cfg
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.d_model = cfg.d_model
        self.d_ff = cfg.d_ff

        # Each expert weight is stored as a separate 2D parameter so that Muon
        # orthogonalises each expert's update independently via Newton-Schulz.
        # The per-expert tensors are stacked into 3D tensors in the forward pass
        # for grouped_mm or batched bmm dispatch.
        self.w1s = nn.ParameterList([
            nn.Parameter(torch.empty(cfg.d_ff, cfg.d_model))
            for _ in range(cfg.n_experts)
        ])
        self.w3s = nn.ParameterList([
            nn.Parameter(torch.empty(cfg.d_ff, cfg.d_model))
            for _ in range(cfg.n_experts)
        ])
        self.w2s = nn.ParameterList([
            nn.Parameter(torch.empty(cfg.d_model, cfg.d_ff))
            for _ in range(cfg.n_experts)
        ])

        # Initialise each per-expert weight slice independently.
        for plist in [self.w1s, self.w3s, self.w2s]:
            for p in plist:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

        self.has_per_loop_routers = cfg.per_loop_routers and n_loops_for_routers > 1
        if self.has_per_loop_routers:
            self.routers = nn.ModuleList([
                nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
                for _ in range(n_loops_for_routers)
            ])
        else:
            self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

        self._use_grouped_mm = _has_grouped_mm()

    def _stack_weights(self):
        """Stack per-expert 2D params into 3D tensors for batched ops."""
        w1 = torch.stack(list(self.w1s))  # (E, d_ff, d_model)
        w3 = torch.stack(list(self.w3s))  # (E, d_ff, d_model)
        w2 = torch.stack(list(self.w2s))  # (E, d_model, d_ff)
        return w1, w3, w2

    def _experts_forward_grouped_mm(self, sorted_x, offs, w1, w3, w2):
        """
        Fast path using torch grouped_mm.
        All tensors must already be bf16 — the caller is responsible for casting.

        grouped_mm computes: slice_i @ mat_b[i]  (no implicit transpose)
        So mat_b must be (E, in_dim, out_dim) and contiguous.

        Weights are (E, out_dim, in_dim) — transposed for grouped_mm.
        """
        w1t = w1.transpose(1, 2).contiguous()
        w3t = w3.transpose(1, 2).contiguous()
        w2t = w2.transpose(1, 2).contiguous()

        h1 = _grouped_mm(sorted_x, w1t, offs)
        h1 = F.silu(h1)
        h3 = _grouped_mm(sorted_x, w3t, offs)
        h = h1 * h3
        return _grouped_mm(h, w2t, offs)

    def _experts_forward_loop_bmm(self, sorted_x, offs):
        """
        Fallback using padded bmm — no Python loop, no graph breaks,
        torch.compile friendly. Works on any device/dtype.

        Pads each expert's token batch to the max count, runs a single
        bmm, then extracts valid results back to the jagged layout.
        """
        NK, D = sorted_x.shape
        E = self.n_experts

        # Compute per-expert start/end from cumulative offs
        ends = offs.long()                                   # (E,)
        starts = torch.zeros_like(ends)
        starts[1:] = ends[:-1]
        counts = ends - starts                               # (E,)
        max_count = counts.max()                             # scalar

        # Build gather indices: (E, max_count)
        arange = torch.arange(max_count, device=sorted_x.device)
        idx = starts.unsqueeze(1) + arange.unsqueeze(0)     # (E, max_count)
        idx = idx.clamp(max=NK - 1)

        # Gather into (E, max_count, D)
        padded_x = sorted_x[idx.reshape(-1)].view(E, max_count, D)

        # Mask for valid positions: (E, max_count)
        mask = arange.unsqueeze(0) < counts.unsqueeze(1)

        # Zero out padding
        padded_x = padded_x * mask.unsqueeze(2).to(padded_x.dtype)

        # Stack the per-expert 2D weights into 3D tensors for batched matmul.
        w1, w3, w2 = self._stack_weights()

        # SwiGLU via batched matmul
        h1 = torch.bmm(padded_x, w1.transpose(1, 2))  # (E, M, d_ff)
        h1 = F.silu(h1)
        h3 = torch.bmm(padded_x, w3.transpose(1, 2))  # (E, M, d_ff)
        h = h1 * h3
        out_padded = torch.bmm(h, w2.transpose(1, 2))  # (E, M, D)

        # Extract valid results back to jagged layout
        # Build flat indices of valid positions in sorted_x
        valid_mask_flat = mask.reshape(-1)                    # (E * max_count,)
        src_flat = out_padded.reshape(-1, D)                  # (E * max_count, D)
        dst_idx_flat = idx.reshape(-1)                        # (E * max_count,)

        # Only write valid (non-padding) positions
        sorted_output = torch.zeros(NK, D, device=sorted_x.device, dtype=sorted_x.dtype)
        valid_src = src_flat[valid_mask_flat]                  # (NK, D) — exactly NK valid
        valid_dst = dst_idx_flat[valid_mask_flat]              # (NK,)
        sorted_output[valid_dst] = valid_src

        return sorted_output

    def forward(self, x, loop_idx=0):
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # (N, D)
        N = x_flat.shape[0]

        if self.has_per_loop_routers:
            logits = self.routers[loop_idx](x_flat)
        else:
            logits = self.router(x_flat)

        topk_vals, topk_idxs = torch.topk(logits, self.top_k, dim=-1)  # (N, K)
        gates = F.softmax(topk_vals, dim=-1)  # (N, K)

        # Flatten top-k: each token appears K times
        flat_expert_ids = topk_idxs.view(-1)     # (N*K,)
        flat_gates = gates.view(-1)              # (N*K,)
        token_idx = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.top_k).reshape(-1)  # (N*K,)
        flat_x = x_flat[token_idx]               # (N*K, D)

        # Sort by expert for contiguous grouped GEMM
        sort_order = flat_expert_ids.argsort(stable=True)
        sorted_x = flat_x[sort_order]            # (N*K, D)
        sorted_gates = flat_gates[sort_order]     # (N*K,)
        sorted_experts = flat_expert_ids[sort_order]
        sorted_token_idx = token_idx[sort_order]

        # Count tokens per expert and find boundaries
        expert_counts = torch.zeros(self.n_experts, dtype=torch.int32, device=x.device)
        expert_counts.scatter_add_(
            0, sorted_experts.int(),
            torch.ones_like(sorted_experts, dtype=torch.int32),
        )
        offs = expert_counts.cumsum(0).to(torch.int32)  # (n_experts,) end-offsets

        # Expert computation
        if self._use_grouped_mm and x.is_cuda:
            # Stack the per-expert 2D weights into 3D tensors and cast to bf16
            # as required by grouped_mm.
            w1, w3, w2 = self._stack_weights()
            orig_dtype = sorted_x.dtype
            sorted_output = self._experts_forward_grouped_mm(
                sorted_x.bfloat16(), offs,
                w1.bfloat16(), w3.bfloat16(), w2.bfloat16(),
            ).to(orig_dtype)
        else:
            sorted_output = self._experts_forward_loop_bmm(sorted_x, offs)

        # Gate and scatter back
        sorted_output = sorted_output * sorted_gates.unsqueeze(1)

        # Unsort and aggregate
        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        output.scatter_add_(0, sorted_token_idx.unsqueeze(1).expand(-1, D), sorted_output)

        return output.view(B, S, D), logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, n_loops_for_group: int = 1):
        super().__init__()
        self.n_loops_for_group = n_loops_for_group
        self.has_per_loop_attn = cfg.per_loop_attn and n_loops_for_group > 1
        self.has_per_loop_norms = n_loops_for_group > 1

        if self.has_per_loop_attn:
            # Independent attention sub-block (with its own pre-norm) per loop
            # iteration; the MoE sub-block remains shared across iterations.
            self.norm1s = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(n_loops_for_group)])
            self.attns = nn.ModuleList([
                MultiHeadAttention(cfg, n_loops_for_lora=1)  # LoRA not needed: separate attention modules
                for _ in range(n_loops_for_group)
            ])
        elif self.has_per_loop_norms:
            self.norm1s = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(n_loops_for_group)])
            self.attn = MultiHeadAttention(cfg, n_loops_for_lora=n_loops_for_group)
        else:
            self.norm1 = RMSNorm(cfg.d_model)
            self.attn = MultiHeadAttention(cfg, n_loops_for_lora=n_loops_for_group)

        if self.has_per_loop_norms:
            self.norm2s = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(n_loops_for_group)])
        else:
            self.norm2 = RMSNorm(cfg.d_model)
        self.moe = MoELayer(cfg, n_loops_for_routers=n_loops_for_group)

    def forward(self, x, loop_idx=0):
        if self.has_per_loop_attn:
            x = x + self.attns[loop_idx](self.norm1s[loop_idx](x), loop_idx=0)
        elif self.has_per_loop_norms:
            x = x + self.attn(self.norm1s[loop_idx](x), loop_idx=loop_idx)
        else:
            x = x + self.attn(self.norm1(x), loop_idx=loop_idx)

        norm2_x = self.norm2s[loop_idx](x) if self.has_per_loop_norms else self.norm2(x)
        moe_out, router_logits = self.moe(norm2_x, loop_idx=loop_idx)
        x = x + moe_out
        return x, router_logits


# ============================================================
# Main model
# ============================================================

class LoopedMoETransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # Build the sequence of unique-block groups from the topology.
        self.groups = nn.ModuleList()
        self.group_loops = []  # list of n_loops per group, indexed by group_idx

        for group_idx, (n_layers, n_loops) in enumerate(cfg.topology):
            group = nn.ModuleList([
                TransformerBlock(cfg, n_loops_for_group=n_loops)
                for _ in range(n_layers)
            ])
            self.groups.append(group)
            self.group_loops.append(n_loops)

        self.norm_out = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        eff_depth = self.cfg.effective_depth
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.xavier_uniform_(p, gain=1.0 / math.sqrt(3 * eff_depth))

    def forward(self, input_ids, targets=None):
        B, S = input_ids.shape
        x = self.drop(self.tok_emb(input_ids))

        all_router_logits = []

        for group_idx, group in enumerate(self.groups):
            n_loops = self.group_loops[group_idx]

            for loop in range(n_loops):
                for layer in group:
                    x, router_logits = layer(x, loop_idx=loop)
                    all_router_logits.append(router_logits)

        logits = self.lm_head(self.norm_out(x))

        loss = None
        aux_loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
            aux_loss = self._load_balance_loss(all_router_logits)

        return logits, loss, aux_loss, all_router_logits

    def _load_balance_loss(self, all_router_logits):
        loss = 0.0
        for logits in all_router_logits:
            probs = F.softmax(logits, dim=-1)
            avg_probs = probs.mean(dim=0)
            assignments = logits.argmax(dim=-1)
            freq = torch.zeros(self.cfg.n_experts, device=logits.device)
            for e in range(self.cfg.n_experts):
                freq[e] = (assignments == e).float().mean()
            loss += (freq * avg_probs).sum() * self.cfg.n_experts
        return loss / len(all_router_logits)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        unique = sum(p.numel() for p in set(self.parameters()))
        return total, unique

    @property
    def layers(self):
        """Flat list of every TransformerBlock across all groups, in execution order.

        Provided for compatibility with utilities that iterate over a flat layer
        list (e.g. analyze_routing in train.py) without knowing about the
        topology grouping.
        """
        all_layers = []
        for group in self.groups:
            all_layers.extend(group)
        return all_layers

    @staticmethod
    def make_variant(variant_name: str, base_cfg_kwargs: dict) -> "LoopedMoETransformer":
        """Factory method for named variants. All have effective depth 32."""
        d = dict(base_cfg_kwargs)
        for old_key in ["n_unique_layers", "n_loops", "n_pre_layers", "n_post_layers"]:
            d.pop(old_key, None)

        # All variants share an effective depth of 32 layers.
        #
        # Sharing strategies for MoE variants:
        #   alltie    : everything shared across loop iterations (FFN + attention + router).
        #   attntie   : per-loop routers; FFN and attention are shared.
        #   lora      : per-loop routers and a per-loop LoRA adapter on attention QK.
        #   experttie : only the expert FFN is shared; attention, routers, and norms are per-loop.
        VARIANT_TABLE = {
            # Baseline: no looping.
            "baseline_32":              dict(topology=[(32, 1)],
                                             per_loop_routers=False, per_loop_attn=False, use_attn_lora=False),

            # MoEUT-style topology: a single group of 2 layers run 16 times, no pre/post.
            "moeut_32_alltie":          dict(topology=[(2, 16)],
                                             per_loop_routers=False, per_loop_attn=False, use_attn_lora=False),
            "moeut_32_attntie":          dict(topology=[(2, 16)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=False),
            "moeut_32_lora":            dict(topology=[(2, 16)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=True),

            # One looped group: 2 + [2 x 14] + 2 = 32.
            "onegroup_32_alltie":       dict(topology=[(2, 1), (2, 14), (2, 1)],
                                             per_loop_routers=False, per_loop_attn=False, use_attn_lora=False),
            "onegroup_32_attntie":       dict(topology=[(2, 1), (2, 14), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=False),
            "onegroup_32_lora":         dict(topology=[(2, 1), (2, 14), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=True),
            "onegroup_32_experttie":       dict(topology=[(2, 1), (2, 14), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=True,  use_attn_lora=False),

            # Two looped groups: 2 + [2 x 7] + [2 x 7] + 2 = 32.
            "twogroup_32_alltie":       dict(topology=[(2, 1), (2, 7), (2, 7), (2, 1)],
                                             per_loop_routers=False, per_loop_attn=False, use_attn_lora=False),
            "twogroup_32_attntie":       dict(topology=[(2, 1), (2, 7), (2, 7), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=False),
            "twogroup_32_lora":         dict(topology=[(2, 1), (2, 7), (2, 7), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=True),
            "twogroup_32_experttie":       dict(topology=[(2, 1), (2, 7), (2, 7), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=True,  use_attn_lora=False),

            # Three looped groups: 2 + [2 x 5] + [2 x 4] + [2 x 5] + 2 = 32.
            "threegroup_32_alltie":     dict(topology=[(2, 1), (2, 5), (2, 4), (2, 5), (2, 1)],
                                             per_loop_routers=False, per_loop_attn=False, use_attn_lora=False),
            "threegroup_32_attntie":     dict(topology=[(2, 1), (2, 5), (2, 4), (2, 5), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=False),
            "threegroup_32_lora":       dict(topology=[(2, 1), (2, 5), (2, 4), (2, 5), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=True),
            "threegroup_32_experttie":     dict(topology=[(2, 1), (2, 5), (2, 4), (2, 5), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=True,  use_attn_lora=False),

            # Four looped groups of size 1: 2 + [1 x 7] x 4 + 2 = 32.
            "fourgroups1_32_alltie":    dict(topology=[(2, 1), (1, 7), (1, 7), (1, 7), (1, 7), (2, 1)],
                                             per_loop_routers=False, per_loop_attn=False, use_attn_lora=False),
            "fourgroups1_32_attntie":    dict(topology=[(2, 1), (1, 7), (1, 7), (1, 7), (1, 7), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=False),
            "fourgroups1_32_lora":      dict(topology=[(2, 1), (1, 7), (1, 7), (1, 7), (1, 7), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=True),
            "fourgroups1_32_experttie":   dict(topology=[(2, 1), (1, 7), (1, 7), (1, 7), (1, 7), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=True,  use_attn_lora=False),

            # Seven looped groups of size 1: 2 + [1 x 4] x 7 + 2 = 32.
            "sevengroups1_32_alltie":   dict(topology=[(2, 1), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (2, 1)],
                                             per_loop_routers=False, per_loop_attn=False, use_attn_lora=False),
            "sevengroups1_32_attntie":   dict(topology=[(2, 1), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=False),
            "sevengroups1_32_lora":     dict(topology=[(2, 1), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=False, use_attn_lora=True),
            "sevengroups1_32_experttie":  dict(topology=[(2, 1), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (2, 1)],
                                             per_loop_routers=True,  per_loop_attn=True,  use_attn_lora=False),

            # Dense baselines (n_experts=1, top_k=1). Same topologies as the MoE
            # variants. d_ff is overridden by the dense config preset (small-dense
            # or 2b-dense) so that per-token active FLOPs match the corresponding
            # MoE variants. Sharing strategies for dense variants:
            #   alltie  : FFN, attention, and norms shared across loops
            #   ffnonly : only FFN shared; attention and norms are per-loop
            # The "attntie" strategy is not applicable to dense models since there
            # is only a single expert and therefore no routing.

            # Dense baseline: no looping.
            "dense_baseline_32":        dict(topology=[(32, 1)], n_experts=1, top_k=1,
                                             per_loop_routers=False, per_loop_attn=False, use_attn_lora=False),

            # Dense two-group: 2 + [2 x 7] + [2 x 7] + 2 = 32.
            "dense_twogroup_32_alltie": dict(topology=[(2, 1), (2, 7), (2, 7), (2, 1)], n_experts=1, top_k=1,
                                             per_loop_routers=False, per_loop_attn=False, use_attn_lora=False),
            "dense_twogroup_32_ffnonly":dict(topology=[(2, 1), (2, 7), (2, 7), (2, 1)], n_experts=1, top_k=1,
                                             per_loop_routers=False, per_loop_attn=True,  use_attn_lora=False),

            # Dense four-group, size 1: 2 + [1 x 7] x 4 + 2 = 32.
            "dense_fourgroups1_32_alltie":  dict(topology=[(2, 1), (1, 7), (1, 7), (1, 7), (1, 7), (2, 1)], n_experts=1, top_k=1,
                                                  per_loop_routers=False, per_loop_attn=False, use_attn_lora=False),
            "dense_fourgroups1_32_ffnonly": dict(topology=[(2, 1), (1, 7), (1, 7), (1, 7), (1, 7), (2, 1)], n_experts=1, top_k=1,
                                                  per_loop_routers=False, per_loop_attn=True,  use_attn_lora=False),

            # Dense seven-group, size 1: 2 + [1 x 4] x 7 + 2 = 32.
            "dense_sevengroups1_32_alltie":  dict(topology=[(2, 1), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (2, 1)], n_experts=1, top_k=1,
                                                   per_loop_routers=False, per_loop_attn=False, use_attn_lora=False),
            "dense_sevengroups1_32_ffnonly": dict(topology=[(2, 1), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (1, 4), (2, 1)], n_experts=1, top_k=1,
                                                   per_loop_routers=False, per_loop_attn=True,  use_attn_lora=False),
        }

        if variant_name not in VARIANT_TABLE:
            raise ValueError(f"Unknown variant: {variant_name}. "
                             f"Available: {sorted(VARIANT_TABLE.keys())}")

        merged = {**d, **VARIANT_TABLE[variant_name]}
        cfg = ModelConfig(**merged)
        return LoopedMoETransformer(cfg)
