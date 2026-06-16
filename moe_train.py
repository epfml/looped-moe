#!/usr/bin/env python3
"""
moe_train.py — MoE pretraining on DCLM-edu with wandb logging + DDP
===============================================================

Three real MoE architectures (OLMoE / Qwen3-MoE / DeepSeekMoE),
full bf16 training scalable across multiple GPUs (if model fits single GPU, with grad checkpointing).

Uses:
  - HuggingFace transformers v5 with grouped_mm (fused expert GEMM)
  - torch.optim.Muon (official PyTorch) or AdamW
  - DCLM-edu dataset via your data.py module
  - wandb logging

┌─────────────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Config                  │ Total    │ Active   │ Adam GB  │ Muon GB  │
├─────────────────────────┼──────────┼──────────┼──────────┼──────────┤
│ A) OLMoE-1B-7B          │  6.8B    │  1.2B    │ ~109     │ ~82      │
│ B) Qwen3-MoE-style      │  6.2B    │  0.6B    │  ~98     │ ~74      │
│ C) DeepSeekMoE-style     │  7.0B    │  1.1B    │ ~112     │ ~84      │
└─────────────────────────┴──────────┴──────────┴──────────┴──────────┘

Launch (1 GPU):
  python moe_train.py --arch olmoe --optimizer muon
  python moe_train.py --arch qwen3moe --optimizer muon --compile
  python moe_train.py --arch deepseek --optimizer adam --no-wandb
  python moe_train.py --arch olmoe --dry-run              # 100 steps

Launch (4 GPUs DDP):
  torchrun --standalone --nproc_per_node=4 moe_train.py --arch deepseek --optimizer muon
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data import get_dataloader, get_eval_batches

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not installed — logging to console only")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Enable TF32 for faster fp32 matmuls on Ampere/Hopper
torch.set_float32_matmul_precision('high')

# ──────────────────────────────────────────────────────────────────────
# 1. Architecture Configs
# ──────────────────────────────────────────────────────────────────────

def get_olmoe_config(scale="regular"):
    """OLMoE-1B-7B (Allen AI, Sep 2024) - arxiv.org/abs/2409.02060"""
    from transformers import OlmoeConfig
    is_small = scale == "small"
    is_tiny = scale == "tiny"
    return OlmoeConfig(
        vocab_size=100277,            
        hidden_size=512 if is_tiny else (1024 if is_small else 2048),
        intermediate_size=256 if is_tiny else (512 if is_small else 1024),
        num_hidden_layers=16,
        num_attention_heads=4 if is_tiny else (8 if is_small else 16),
        num_key_value_heads=4 if is_tiny else (8 if is_small else 16),
        num_experts=64,
        num_experts_per_tok=8,
        router_aux_loss_coef=0.01,
        max_position_embeddings=4096,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        clip_qkv=8.0,                
        tie_word_embeddings=False,
        use_cache=False,
        output_router_logits=True,
    )

def get_qwen3moe_config(scale="regular"):
    """Qwen3-MoE-style (Qwen Team, Apr 2025) - arxiv.org/abs/2505.09388"""
    from transformers import Qwen3MoeConfig
    is_small = scale == "small"
    is_tiny = scale == "tiny"
    return Qwen3MoeConfig(
        vocab_size=100277,
        hidden_size=384 if is_tiny else (768 if is_small else 1536),
        intermediate_size=1024 if is_tiny else (2048 if is_small else 4096),
        moe_intermediate_size=192 if is_tiny else (384 if is_small else 768),
        num_hidden_layers=28,
        num_attention_heads=6 if is_tiny else (12 if is_small else 24),
        num_key_value_heads=1 if is_tiny else (2 if is_small else 4),
        num_experts=60,
        num_experts_per_tok=4,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        # The HF default of 0.001 is calibrated for the bias-based balancer used in
        # production Qwen3-MoE; the HF implementation does not include that mechanism,
        # so we use 0.01 to match the standard auxiliary-loss-only setting (as in OLMoE).
        router_aux_loss_coef=0.01,
        max_position_embeddings=4096,
        rope_theta=10000.0,            
        hidden_act="silu",
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        use_cache=False,
        output_router_logits=True,
    )

def get_deepseek_config(scale="regular"):
    """DeepSeekMoE-style (DeepSeek-AI, 2024)"""
    from transformers import OlmoeConfig
    is_small = scale == "small"
    is_tiny = scale == "tiny"
    return OlmoeConfig(
        vocab_size=100277,
        hidden_size=512 if is_tiny else (1024 if is_small else 2048),
        intermediate_size=256 if is_tiny else (512 if is_small else 1024),
        num_hidden_layers=16,
        num_attention_heads=4 if is_tiny else (8 if is_small else 16),
        num_key_value_heads=4 if is_tiny else (8 if is_small else 16),
        num_experts=64,
        num_experts_per_tok=6,
        # Using 0.01 in place of the HF default of 0.001, which is calibrated for
        # bias-based load balancing not implemented in the HF DeepSeekMoE class.
        router_aux_loss_coef=0.01,
        max_position_embeddings=4096,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        clip_qkv=8.0,                
        tie_word_embeddings=False,
        use_cache=False,
        output_router_logits=True,
    )
    


ARCH_REGISTRY = {
    "olmoe": ("OLMoE-1B-7B", get_olmoe_config),
    "qwen3moe": ("Qwen3-MoE-style", get_qwen3moe_config),
    "deepseek": ("DeepSeekMoE-style", get_deepseek_config),
}

# Train config presets (matching your train.py structure)
TRAIN_CONFIGS = {
    "default": dict(
        batch_size=16, grad_accum=4, seq_len=2048,
        n_steps=20_000, lr=2e-2, min_lr=2e-3,
        warmup_steps=100, weight_decay=0.1, grad_clip=1.0,  # weight decay is set per param group below; this value is unused
        eval_every=500, eval_batches=10, save_every=500,
        log_every=10, compile_model=True,
    ),
}


# ──────────────────────────────────────────────────────────────────────
# 2. Expert Weight Tying
# ──────────────────────────────────────────────────────────────────────

def tie_expert_layers(model, group_size: int, skip_first: int = 2, skip_last: int = 2, master_process: bool = True):
    """
    Tie expert parameters across consecutive layers to save memory.

    Within each group, all layers share the same expert weight tensors
    (gate_up_proj and down_proj). Router weights (gate.weight) and all
    attention/norm params remain independent per layer.

    Gradients accumulate correctly through all tied layers because they
    reference the same nn.Parameter object — autograd sums gradients
    from every use site.

    Args:
        model: HF transformers MoE model (OlmoeForCausalLM or Qwen3MoeForCausalLM)
        group_size: number of consecutive layers that share expert weights
        skip_first: layers at the start left untied (default: 2)
        skip_last: layers at the end left untied (default: 2)

    Returns:
        dict with tying info for logging

    Example (16 layers, group_size=5, skip 2+2):
        layers 0,1:   untied (independent experts)
        layers 2-6:   share experts from layer 2
        layers 7-11:  share experts from layer 7
        layers 12,13: share experts from layer 12
        layers 14,15: untied (independent experts)
    """
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    # Navigate to the layer list — handle both OLMoE and Qwen3MoE structures
    if hasattr(raw, "model") and hasattr(raw.model, "layers"):
        layers = raw.model.layers
    elif hasattr(raw, "layers"):
        layers = raw.layers
    else:
        raise ValueError("Cannot find model layers for expert tying")

    n_layers = len(layers)
    middle_start = skip_first
    middle_end = n_layers - skip_last

    if middle_end <= middle_start:
        if master_process:
            print(f"  Expert tying: nothing to tie (only {n_layers} layers, "
                  f"skipping {skip_first}+{skip_last})")
        return {"tied": False}

    middle_indices = list(range(middle_start, middle_end))

    # Chunk into groups
    groups = []
    for i in range(0, len(middle_indices), group_size):
        groups.append(middle_indices[i:i + group_size])

    # Count params before tying
    params_before = sum(p.numel() for p in raw.parameters())

    # Tie within each group: all layers point to the first layer's expert params
    total_tied = 0
    for group in groups:
        leader_idx = group[0]
        leader = layers[leader_idx]

        for follower_idx in group[1:]:
            follower = layers[follower_idx]
            # Tie expert weights (gate_up_proj, down_proj)
            follower.mlp.experts.gate_up_proj = leader.mlp.experts.gate_up_proj
            follower.mlp.experts.down_proj = leader.mlp.experts.down_proj
            # Router (gate) stays INDEPENDENT — each layer routes differently
            total_tied += 1

    params_after = sum(p.numel() for p in raw.parameters())
    saved = params_before - params_after

    info = {
        "tied": True,
        "group_size": group_size,
        "groups": groups,
        "skip_first": skip_first,
        "skip_last": skip_last,
        "layers_tied": total_tied,
        "params_before": params_before,
        "params_after": params_after,
        "params_saved": saved,
    }

    if master_process:
        print(f"  Expert tying: group_size={group_size}, skip {skip_first}+{skip_last}")
        print(f"  Groups: {groups}")
        print(f"  Layers sharing experts: {total_tied} "
              f"(each reuses leader's gate_up_proj + down_proj)")
        print(f"  Params: {params_before/1e9:.2f}B → {params_after/1e9:.2f}B "
              f"(saved {saved/1e9:.2f}B, {saved/params_before*100:.1f}%)")

    # Verify: check that tied layers actually share the same object
    for group in groups:
        leader = layers[group[0]]
        for idx in group[1:]:
            assert layers[idx].mlp.experts.gate_up_proj is leader.mlp.experts.gate_up_proj, \
                f"Tying failed: layer {idx} gate_up_proj not shared with layer {group[0]}"
            assert layers[idx].mlp.experts.down_proj is leader.mlp.experts.down_proj, \
                f"Tying failed: layer {idx} down_proj not shared with layer {group[0]}"
            # Router must NOT be tied
            assert layers[idx].mlp.gate.weight is not leader.mlp.gate.weight, \
                f"Bug: layer {idx} router is tied with layer {group[0]} — should be independent"
    if master_process:
        print(f"  Verification: all ties correct, routers independent")

    return info


def build_optimizer(model, optimizer_type, lr, weight_decay, args, config, master_process: bool = True):
    """Build optimizer. Muon for 2D hidden params, AdamW for embeddings/head/1D/Routers.

    For 3D expert weight tensors [E, out, in] (from HF transformers v5 grouped_mm),
    we create 2D proxy parameters that share storage with the 3D tensor. Each expert's
    [out, in] slice becomes a separate 2D leaf param passed to Muon. A post-backward
    hook copies the 3D grad into the proxy grads so Muon can step on them.
    """
    if optimizer_type == "adamw":
        standard_params = []
        tied_expert_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            
            is_tied_expert = False
            if p.dim() == 3 and getattr(args, "tie_group_size", 0) > 1:
                try:
                    parts = name.split('.')
                    if "layers" in parts:
                        layer_idx = int(parts[parts.index("layers") + 1])
                        if args.tie_skip_first <= layer_idx < (config.num_hidden_layers - args.tie_skip_last):
                            is_tied_expert = True
                except Exception:
                    pass
                    
            if is_tied_expert: tied_expert_params.append(p)
            else: standard_params.append(p)

        # AdamW WD = 0.01 (Keller Jordan / nanoGPT convention).
        # Override the `weight_decay` arg passed in (which comes from train_config dict).
        adamw_wd = 0.01

        return torch.optim.AdamW([
            {"params": standard_params, "weight_decay": adamw_wd},
            {
                "params": tied_expert_params,
                "is_tied_expert": True,
                # Standard uncompensated weight decay
                "weight_decay": adamw_wd,
            }
        ], lr=lr, weight_decay=adamw_wd, betas=(0.9, 0.95))

    elif optimizer_type == "muon":
        muon_standard = []
        muon_tied_experts = []
        adamw_params = [] 
        expert_proxies = [] 

        raw = model.module if isinstance(model, DDP) else model
        raw = raw._orig_mod if hasattr(raw, '_orig_mod') else raw
        
        embed_head_keywords = {"embed_tokens", "lm_head", "wte", "wpe", "embed", "head"}

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_embed_head = any(kw in name for kw in embed_head_keywords)
            
            # Identify Router
            is_router = "gate" in name and "weight" in name and p.dim() == 2

            if is_embed_head or is_router:
                adamw_params.append(p)
            elif p.dim() == 2:
                muon_standard.append(p)
            elif p.dim() == 3:
                # Isolate ONLY the experts that fall within the explicitly tied layers
                is_tied_expert = False
                if getattr(args, "tie_group_size", 0) > 1:
                    try:
                        parts = name.split('.')
                        if "layers" in parts:
                            layer_idx = int(parts[parts.index("layers") + 1])
                            if args.tie_skip_first <= layer_idx < (config.num_hidden_layers - args.tie_skip_last):
                                is_tied_expert = True
                    except Exception:
                        pass
                        
                E = p.shape[0]
                proxies = []
                for i in range(E):
                    proxy = torch.nn.Parameter(p.data[i], requires_grad=False)
                    proxies.append(proxy)
                    if is_tied_expert:
                        muon_tied_experts.append(proxy)
                    else:
                        muon_standard.append(proxy)
                expert_proxies.append((p, proxies))
            else:
                adamw_params.append(p)

        if master_process:
            n_muon = sum(p.numel() for p in muon_standard + muon_tied_experts)
            n_adamw = sum(p.numel() for p in adamw_params)
            print(f"  Muon params: {n_muon:,} | AdamW params: {n_adamw:,}")

        from torch.optim import Muon as TorchMuon

        # Decoupled weight decay (Loshchilov & Hutter, 2019).
        # Muon body weights use weight_decay=0.1, following common production MoE recipes
        # (Llama, Qwen, DeepSeek). AdamW handles embeddings, output head, routers, and 1D
        # parameters; weight_decay=0.01 follows Keller Jordan's reference Muon recipe.
        muon_wd = 0.1
        adamw_wd = 0.01

        muon_opt = TorchMuon([
            {"params": muon_standard},
            {
                "params": muon_tied_experts,
                "is_tied_expert": True,
                # Standard uncompensated weight decay
                "weight_decay": muon_wd,
            }
        ], lr=lr, momentum=0.95, nesterov=True, weight_decay=muon_wd)

        adamw_opt = torch.optim.AdamW(
            adamw_params,
            lr=lr * 0.1,
            weight_decay=adamw_wd,
            betas=(0.9, 0.95),
        )

        class DualOptimizer:
            def __init__(self, muon, adamw, expert_proxies):
                self.muon = muon
                self.adamw = adamw
                self.expert_proxies = expert_proxies
                self.param_groups = muon.param_groups + adamw.param_groups

            def zero_grad(self, set_to_none=False):
                # Zero the parameters registered to the standard optimisers.
                self.muon.zero_grad(set_to_none=set_to_none)
                self.adamw.zero_grad(set_to_none=set_to_none)

                # The 3D expert tensors are not registered with either optimiser
                # (they are mirrored through 2D proxies); zero their gradients explicitly.
                for param_3d, _ in self.expert_proxies:
                    if param_3d.grad is not None:
                        if set_to_none:
                            param_3d.grad = None
                        else:
                            param_3d.grad.zero_()

            def _sync_expert_grads(self):
                # Copy the per-expert slices of the 3D parameter gradient into the proxy
                # parameters that Muon optimises. The assignment is unconditional: experts
                # that received no tokens still need a (zero) gradient slot so that decoupled
                # weight decay is applied uniformly across all expert weights.
                for param_3d, proxies in self.expert_proxies:
                    if param_3d.grad is None:
                        continue
                    for i, proxy in enumerate(proxies):
                        proxy.grad = param_3d.grad[i]

            def step(self):
                self._sync_expert_grads()
                self.muon.step()
                self.adamw.step()

            def state_dict(self):
                return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

            def load_state_dict(self, state_dict):
                self.muon.load_state_dict(state_dict["muon"])
                self.adamw.load_state_dict(state_dict["adamw"])

        return DualOptimizer(muon_opt, adamw_opt, expert_proxies)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


# ──────────────────────────────────────────────────────────────────────
# 3. LR Schedule
# ──────────────────────────────────────────────────────────────────────

def get_lr(step, warmup_steps, n_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ──────────────────────────────────────────────────────────────────────
# 4. Eval
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, eval_batches, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    use_amp = device.type == "cuda"

    for input_ids, targets in eval_batches:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            # Disable router logits during evaluation so HF skips the aux-loss computation.
            out = model(input_ids=input_ids, labels=input_ids, output_router_logits=False)
        
        total_loss += out.loss.item() * targets.numel()
        total_tokens += targets.numel()

    model.train()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(min(avg_loss, 20))


# ──────────────────────────────────────────────────────────────────────
# 5. Verification
# ──────────────────────────────────────────────────────────────────────

def verify_grouped_mm(model, config, arch, master_process: bool = True):
    impl = getattr(config, "_experts_implementation", "unknown")
    if not master_process:
        return
    print(f"  experts_implementation: {impl}")

    has_gmm = hasattr(torch.nn.functional, "grouped_mm")
    has_gmm_priv = hasattr(torch, "_grouped_mm")
    print(f"  torch.nn.functional.grouped_mm: {has_gmm}")

    try:
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        # HF 5.x: don't use .base_model — it's a property returning the inner
        # OlmoeModel directly, breaking the old `raw.base_model.model.layers`
        # pattern that assumed PEFT-style wrapping.
        if hasattr(raw, "model") and hasattr(raw.model, "layers"):
            layers = raw.model.layers
        elif hasattr(raw, "layers"):
            layers = raw.layers
        else:
            raise AttributeError("Cannot locate layer list on model")
        experts = layers[0].mlp.experts
        print(f"  gate_up_proj: {tuple(experts.gate_up_proj.shape)} "
              f"(3D: {experts.gate_up_proj.ndim == 3})")
    except Exception as e:
        print(f"  Could not inspect expert weights: {e}")

    if impl == "grouped_mm" and (has_gmm or has_gmm_priv):
        print(f"  → Fused grouped GEMM active ✓")
    elif impl == "batched_mm":
        print(f"  → Batched BMM (faster than loop, slower than grouped)")
    else:
        print(f"  → Eager for-loop fallback")


# ──────────────────────────────────────────────────────────────────────
# 5b. Depth-scaled Xavier init
# ──────────────────────────────────────────────────────────────────────

def _reinit_weights_depth_scaled(model, config, master_process: bool = True):
    """Re-initialize weights with depth-scaled Xavier, matching custom model.py.

    HF default is truncated_normal(σ=0.02) which ignores depth.
    Custom model.py uses xavier_uniform with gain = 1/√(3·depth).
    """
    eff_depth = config.num_hidden_layers
    gain = 1.0 / math.sqrt(3 * eff_depth)

    reinit_count = 0
    for name, p in model.named_parameters():
        if p.dim() == 2:
            nn.init.xavier_uniform_(p, gain=gain)
            reinit_count += 1
        elif p.dim() == 3:
            # Stacked expert weights (n_experts, out, in) — init each slice
            for i in range(p.shape[0]):
                nn.init.xavier_uniform_(p[i], gain=gain)
            reinit_count += 1
        # 1D params (norms, biases) keep their default init

    if master_process:
        print(f"  Depth-scaled Xavier init: gain=1/√(3×{eff_depth})={gain:.4f}, "
              f"re-initialized {reinit_count} tensors")


# ──────────────────────────────────────────────────────────────────────
# 6. Training
# ──────────────────────────────────────────────────────────────────────

def train(args):
    # --- DDP SETUP ---
    is_ddp = int(os.environ.get('RANK', -1)) != -1
    if is_ddp:
        dist.init_process_group("nccl")
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = torch.device(args.device) if args.device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            
    # Force disable wandb on all worker processes
    if not master_process:
        os.environ["WANDB_MODE"] = "disabled"
        
    tcfg = dict(TRAIN_CONFIGS.get(args.train_config, TRAIN_CONFIGS["default"]))

    # CLI overrides
    if args.n_steps is not None:
        tcfg["n_steps"] = args.n_steps
    if args.batch_size is not None:
        tcfg["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        tcfg["grad_accum"] = args.grad_accum
    if args.seq_len is not None:
        tcfg["seq_len"] = args.seq_len

    if args.dry_run:
        tcfg["n_steps"] = 100
        tcfg["eval_every"] = 50
        tcfg["log_every"] = 10
        if master_process:
            print("DRY RUN MODE: 100 steps")

    use_amp = device.type == "cuda"
    amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    arch_name, config_fn = ARCH_REGISTRY[args.arch]

    # --- Define unified run name for logs and checkpoints ---
    tie_gs = args.tie_group_size if args.tie_group_size and args.tie_group_size > 1 else 0
    run_name = f"{args.arch}-g{tie_gs}" if tie_gs else f"{args.arch}-notie"
    
    # Append width expansion to run name so checkpoints don't overwrite!
    if getattr(args, "expand_tied_experts", None) is not None:
        run_name += f"-we{args.expand_tied_experts}"
    if getattr(args, "scale", "regular") == "small":
        run_name += "-small"
    elif getattr(args, "scale", "regular") == "tiny":
        run_name += "-tiny"

    if master_process:
        print(f"\n{'='*70}")
        print(f"  MoE TRAINING: {arch_name}")
        print(f"{'='*70}")
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Device: {device} | DDP World Size: {ddp_world_size}")

    # ── Build model ──
    if master_process:
        print(f"\nBuilding model ... (Scale: {args.scale})")
    config = config_fn(scale=getattr(args, "scale", "regular"))
    config._experts_implementation = "grouped_mm"

    if args.arch == "qwen3moe":
        from transformers import Qwen3MoeForCausalLM
        model = Qwen3MoeForCausalLM(config)
    else:
        from transformers import OlmoeForCausalLM
        model = OlmoeForCausalLM(config)

    
    # ─── Router load-balancing + z-loss ──────
    # HF 5.8's OlmoeTopKRouter / Qwen3MoeTopKRouter classes return raw logits as the
    # first element of their output tuple, which propagates unchanged to
    # out.router_logits. We can therefore consume gate_logits directly.

    def heterogeneous_load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask=None):
        import torch.nn.functional as F
        if gate_logits is None or len(gate_logits) == 0:
            return 0.0

        # HF post-multiplies the return value by router_aux_loss_coef before
        # adding it to the language-model loss. Pre-divide the z-loss term so
        # the effective z-loss coefficient stays equal to args.z_loss_coef
        # regardless of router_aux_loss_coef.
        hf_aux_coef = getattr(config, "router_aux_loss_coef", 0.01)
        actual_z_loss_coef = args.z_loss_coef / hf_aux_coef

        aux_loss = 0.0
        valid_layers = 0
        for layer_logits in gate_logits:
            if layer_logits is None:
                continue
            if isinstance(layer_logits, tuple):
                layer_logits = layer_logits[0]
            n_exp = layer_logits.size(-1)
            flat_logits = layer_logits.view(-1, n_exp)

            # Upcast to float32 before softmax/logsumexp to avoid bf16 overflow.
            float_logits = flat_logits.float()
            routing_probs = F.softmax(float_logits, dim=-1).type_as(flat_logits)

            # Router z-loss (Zoph et al., 2022).
            z_loss = torch.mean(torch.logsumexp(float_logits, dim=-1) ** 2) * actual_z_loss_coef

            _, selected = torch.topk(routing_probs, top_k, dim=-1)
            expert_mask = F.one_hot(selected, n_exp)

            # The /top_k is required so frequencies sum to 1 (cf. HF #43688).
            tokens_per_expert = expert_mask.sum(dim=1).float().mean(dim=0) / top_k
            router_prob_per_expert = routing_probs.mean(dim=0)

            # Per-layer load-balancing loss (Shazeer 2017; Fedus 2022).
            layer_aux_loss = (tokens_per_expert * router_prob_per_expert).sum() * n_exp

            aux_loss += layer_aux_loss + z_loss
            valid_layers += 1

        return aux_loss / valid_layers if valid_layers > 0 else 0.0


    # ─── Robust patch injection across HF 5.3+ ────────────────────────────
    def _install_router_loss_patch(fn, master_process):
        """Patch every known location of HF's load_balancing_loss_func.

        Asserts at least one site was patched. If HF moves the symbol again,
        this fails loudly at startup rather than silently letting HF's stock
        implementation run (which would drop the z-loss term).
        """
        targets = []
        candidates = [
            "transformers.models.olmoe.modeling_olmoe",
            "transformers.models.qwen3_moe.modeling_qwen3_moe",
            "transformers.loss.loss_utils",
            "transformers.modeling_utils",
        ]
        for modpath in candidates:
            try:
                mod = __import__(modpath, fromlist=["load_balancing_loss_func"])
            except ImportError:
                continue
            if hasattr(mod, "load_balancing_loss_func"):
                mod.load_balancing_loss_func = fn
                targets.append(modpath)
        assert targets, (
            "load_balancing_loss_func not found in any known HF location. "
            "HF transformers API has changed — patch is dead."
        )
        if master_process:
            print(f"  Patched load_balancing_loss_func in: {targets}")

    _install_router_loss_patch(heterogeneous_load_balancing_loss_func, master_process)
    

    # ── Heterogeneous Expert Transplant (For strict Iso-Parameter matching) ──
    if getattr(args, "expand_tied_experts", None) is not None:
        import copy
        if master_process:
            print(f"\n  [Transplant] Building heterogeneous model: Middle layers expanded to {args.expand_tied_experts} experts...")
        
        # 1. Create a config for the expanded middle layers
        config_expanded = copy.deepcopy(config)
        config_expanded.num_experts = args.expand_tied_experts
        
        # 2. Build a temporary dummy model to generate the larger expert matrices
        if args.arch == "qwen3moe":
            dummy_model = Qwen3MoeForCausalLM(config_expanded)
        else:
            dummy_model = OlmoeForCausalLM(config_expanded)
            
        # 3. Transplant the expanded MLPs into the middle layers of our real model
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        dummy_raw = dummy_model._orig_mod if hasattr(dummy_model, "_orig_mod") else dummy_model
        
        layers = raw.model.layers if hasattr(raw, "model") else raw.layers
        dummy_layers = dummy_raw.model.layers if hasattr(dummy_raw, "model") else dummy_raw.layers
        
        middle_start = args.tie_skip_first
        middle_end = len(layers) - args.tie_skip_last
        
        for i in range(middle_start, middle_end):
            layers[i].mlp = dummy_layers[i].mlp
            
        # 4. Free the dummy model from RAM
        del dummy_model
        if master_process:
            print(f"  [Transplant] Complete! Layers {middle_start} to {middle_end-1} now have {args.expand_tied_experts} experts.")

    # ── Depth-scaled Xavier init (matching custom model.py) ──
    _reinit_weights_depth_scaled(model, config, master_process)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # ── Expert weight tying (before .to(device) to save GPU memory) ──
    tie_info = {"tied": False}
    if args.tie_group_size and args.tie_group_size > 1:
        if master_process:
            print(f"\nExpert weight tying:")
        tie_info = tie_expert_layers(
            model,
            group_size=args.tie_group_size,
            skip_first=args.tie_skip_first,
            skip_last=args.tie_skip_last,
            master_process=master_process,
        )

    model = model.to(device)
    
    # Wrap in DDP
    if is_ddp:
        model = DDP(
            model, 
            device_ids=[ddp_local_rank], 
            find_unused_parameters=False,
            broadcast_buffers=False,       # Transformers don't use BatchNorm, saves memory/time
            gradient_as_bucket_view=True,  # <--- THE VRAM SAVER
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if master_process:
        print(f"  Total params:    {total_params:>14,} ({total_params/1e9:.2f}B)")
        print(f"  Trainable:       {trainable_params:>14,} ({trainable_params/1e9:.2f}B)")
        print(f"  Est. Adam VRAM:  {trainable_params * 16 / 1e9:.1f} GB")
        print(f"  Est. Muon VRAM:  {trainable_params * 12 / 1e9:.1f} GB")
        if device.type == "cuda":
            print(f"  Model loaded:    {torch.cuda.memory_allocated()/1e9:.1f} GB")

        print(f"\nExpert backend:")
    verify_grouped_mm(model, config, args.arch, master_process)

    # ── Optimizer (BEFORE compile, so param ids match) ──
    opt_type = args.optimizer
    
    # If passed via CLI, use it. Otherwise, use the config dictionary.
    if args.lr is not None:
        opt_lr = args.lr
    else:
        opt_lr = tcfg["lr"]

    if master_process:
        print(f"\nOptimizer: {opt_type}, LR: {opt_lr}")
    
    # Passing args and config through to safely parse the exact layer strings
    optimizer = build_optimizer(model, opt_type, opt_lr, tcfg["weight_decay"], args, config, master_process)

    # ── Checkpoint Resumption ──
    start_step = 0
    best_eval_loss = float("inf")
    wandb_run_id = None
    wandb_resume_step = None  # step to resume wandb logging from (overwrites later data)

    # --- Auto-Resume Logic ---
    if args.auto_resume and not args.resume:
        import glob
        import re
        save_dir_path = Path(args.save_dir)
        
        # Look for checkpoints matching this exact run name
        search_pattern = str(save_dir_path / f"{run_name}_step*.pt")
        step_ckpts = glob.glob(search_pattern)
        
        latest_ckpt = None
        max_step = -1
        
        for ckpt_str in step_ckpts:
            # Extract the step number from the filename
            match = re.search(r'_step(\d+)\.pt$', ckpt_str)
            if match:
                s = int(match.group(1))
                if s > max_step:
                    max_step = s
                    latest_ckpt = ckpt_str
                    
        if latest_ckpt:
            args.resume = latest_ckpt
            if master_process:
                print(f"\n  [Auto-Resume] Detected crash/restart. Resuming from latest step: {latest_ckpt}")
        else:
            # Fallback to best if no step checkpoints exist yet
            best_ckpt = save_dir_path / f"{run_name}_best.pt"
            if best_ckpt.exists():
                args.resume = str(best_ckpt)
                if master_process:
                    print(f"\n  [Auto-Resume] No step checkpoints found. Falling back to best: {best_ckpt}")

    # --- Standard Resume Loading ---
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            if master_process:
                print(f"\n  Resuming from checkpoint: {ckpt_path}")
            
            # Load to CPU first to avoid a transient VRAM spike from duplicated tensors
            # during deserialisation; load_state_dict moves tensors to the appropriate
            # device afterwards.
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            raw_model = model.module if is_ddp else model
            raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
            raw_model.load_state_dict(ckpt["model_state_dict"])
            
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                
                # --- Move custom optimizer states (Muon) to the GPU ---
                if hasattr(optimizer, "muon") and hasattr(optimizer, "adamw"):
                    for opt in [optimizer.muon, optimizer.adamw]:
                        for state in opt.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(device)
                else:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
            elif master_process:
                print("  Warning: No optimizer state found in checkpoint.")

            # --- Restore RNG states to prevent Dataloader scrambling ---
            if "rng_state" in ckpt and ckpt["rng_state"] is not None:
                torch.set_rng_state(ckpt["rng_state"])
            if "cuda_rng_state" in ckpt and ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"])

            # The checkpoint saved the step that was JUST COMPLETED (weights already updated).
            # To avoid a double-update on the exact same data, we start at the NEXT step.
            saved_step = ckpt.get("step", -1)
            start_step = saved_step + 1 if saved_step >= 0 else 0
            
            # Safely load best_eval_loss to prevent NoneType crash
            best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
            if best_eval_loss is None:
                best_eval_loss = float("inf")
                
            wandb_run_id = ckpt.get("wandb_run_id")
            wandb_resume_step = saved_step  # used for resume_from to overwrite stale data
            
            if master_process:
                print(f"  → Resumed at step {start_step} (Best loss: {best_eval_loss:.4f})")
        else:
            if master_process:
                print(f"\n  Checkpoint {ckpt_path} not found. Starting fresh.")

    # ── Compile ──
    if tcfg["compile_model"] and args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            if master_process:
                print(f"  torch.compile: enabled")
        except Exception as e:
            if master_process:
                print(f"  torch.compile: failed ({e})")

    # ── Data (your DCLM-edu data module) ──
    if master_process:
        print(f"\nData:")
    train_loader = get_dataloader(
        seq_len=tcfg["seq_len"],
        batch_size=tcfg["batch_size"],
        num_workers=4,
    )
    eval_batches = get_eval_batches(
        seq_len=tcfg["seq_len"],
        batch_size=tcfg["batch_size"],
        n_batches=tcfg["eval_batches"],
    )
    grad_accum = tcfg["grad_accum"]
    eff_batch_tokens = tcfg["batch_size"] * grad_accum * tcfg["seq_len"] * ddp_world_size
    if master_process:
        print(f"  Eval batches: {len(eval_batches)}")
        print(f"  Effective batch: {tcfg['batch_size']}×{grad_accum}×{tcfg['seq_len']}×{ddp_world_size}gpu = {eff_batch_tokens:,} tok/step")

    # ── wandb ──
    run = None
    if HAS_WANDB and not args.no_wandb and master_process:
        print(f"\nInitializing wandb ...", flush=True)
        
        wandb_kwargs = {
            "project": args.wandb_project,
            "group": args.run_group,
            "config": {
                "arch": args.arch,
                "arch_name": arch_name,
                "optimizer": opt_type,
                "lr": opt_lr,
                "z_loss_coef": args.z_loss_coef,
                "tied_lr_divisor": getattr(args, "tied_lr_divisor", 1.0),
                "train_config": tcfg,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "experts_impl": getattr(config, "_experts_implementation", "unknown"),
                "num_experts": config.num_experts,
                "expand_tied_experts": args.expand_tied_experts,
                "num_experts_per_tok": config.num_experts_per_tok,
                "hidden_size": config.hidden_size,
                "num_hidden_layers": config.num_hidden_layers,
                "gradient_checkpointing": args.gradient_checkpointing,
                "compile": args.compile,
                "tie_info": tie_info,
                # Explicit top-level fields for easy filtering in wandb
                "batch_size": tcfg["batch_size"],
                "grad_accum": grad_accum,
                "seq_len": tcfg["seq_len"],
                "ddp_world_size": ddp_world_size,
                "eff_batch_tokens": eff_batch_tokens,
            }
        }
        
        if wandb_run_id and args.resume:
            # Standard append mode (WandB rewind is not working properly or not available)
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "must"
            print(f"  wandb: Resuming run {wandb_run_id} (append mode)", flush=True)
        else:
            wandb_kwargs["name"] = run_name
            print(f"  wandb: {run_name}", flush=True)
            
        run = wandb.init(**wandb_kwargs)

    # ── Training loop ──
    n_steps = tcfg["n_steps"]
    step = start_step  # <--- Updated for resume
    losses = []
    t_start = time.time()
    
    if master_process:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    if master_process:
        print("  Fetching first training batch ...", flush=True)
    train_iter = iter(train_loader)
    
    # ── Fast-forward DataLoader if resuming ──
    if start_step > 0:
        batches_to_skip = start_step * grad_accum
        if master_process:
            print(f"  Fast-forwarding stream by {batches_to_skip} micro-batches (may take a minute)...", flush=True)
        skip_pbar = tqdm(total=batches_to_skip, desc="Fast-forward", ncols=100) if (HAS_TQDM and master_process) else None
        
        for _ in range(batches_to_skip):
            try:
                next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                next(train_iter)
            if skip_pbar: skip_pbar.update(1)
        if skip_pbar: skip_pbar.close()

    _first_batch = next(train_iter)
    if master_process:
        print("  First batch ready. Starting training loop.", flush=True)

    pbar = tqdm(total=n_steps, initial=start_step, desc=run_name, ncols=100) if (HAS_TQDM and master_process) else None

    # Default to 0 so we don't have undefined variables if not using muon
    adamw_lr = 0.0

    while step < n_steps:
    
        # debug output for data loader check (DDP vs not)
        if step == 0:
            sample_input, _ = next(iter(train_loader))
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[rank={rank}] first batch first 10 tokens: {sample_input[0, :10].tolist()}", flush=True)
    
        # LR schedule
        min_lr = opt_lr * (tcfg["min_lr"] / tcfg["lr"])
        lr = get_lr(step, tcfg["warmup_steps"], n_steps, opt_lr, min_lr)
        
        divisor = getattr(args, "tied_lr_divisor", 1.0)
        
        if opt_type == "muon":
            for pg in optimizer.muon.param_groups:
                pg["lr"] = lr / divisor if pg.get("is_tied_expert", False) else lr
                
            # Scale the AdamW max and min LR 
            adamw_lr = get_lr(step, tcfg["warmup_steps"], n_steps, opt_lr * 0.1, min_lr * 0.1)
            
            for pg in optimizer.adamw.param_groups:
                pg["lr"] = adamw_lr / divisor if pg.get("is_tied_expert", False) else adamw_lr
        else:
            # Pure AdamW route
            for pg in optimizer.param_groups:
                pg["lr"] = lr / divisor if pg.get("is_tied_expert", False) else lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_aux = 0.0
        accum_lm_loss = 0.0  # pure CE loss (without aux/z), for cross-run comparison

        for micro_step in range(grad_accum):
            if _first_batch is not None:
                input_ids, targets = _first_batch
                _first_batch = None
            else:
                try:
                    input_ids, targets = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    input_ids, targets = next(train_iter)

            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Skip DDP gradient synchronisation on all but the last micro-step to keep
            # accumulated steps within a single optimiser step communication-free.
            require_backward_grad_sync = (micro_step == grad_accum - 1)
            if is_ddp and not require_backward_grad_sync:
                sync_context = model.no_sync()
            else:
                sync_context = nullcontext()

            with sync_context:
                with amp_ctx():
                    # `targets` from data.py is already shifted; passing labels=input_ids
                    # avoids a second shift inside HF.
                    out = model(input_ids=input_ids, labels=input_ids, output_router_logits=True)
                    loss = out.loss
                    aux_loss = getattr(out, "aux_loss", None)

                    loss_scaled = loss / grad_accum

                loss_scaled.backward()

            accum_loss += out.loss.item() / grad_accum
            if aux_loss is not None and not isinstance(aux_loss, int):
                accum_aux += aux_loss.item() / grad_accum
                # HF returns out.loss = lm_loss + router_aux_loss_coef * out.aux_loss.
                # In our patched load-balancing function, out.aux_loss already contains
                # the router z-loss term. Subtracting (router_aux_loss_coef * out.aux_loss)
                # therefore recovers the pure language-model cross-entropy.
                hf_aux_coef = getattr(config, "router_aux_loss_coef", 0.01)
                aux_contribution = hf_aux_coef * aux_loss.item()
                accum_lm_loss += (out.loss.item() - aux_contribution) / grad_accum
            else:
                accum_lm_loss += out.loss.item() / grad_accum

        # Average the per-rank loss metrics for logging consistency under DDP.
        if is_ddp:
            metrics = torch.tensor([accum_loss, accum_aux, accum_lm_loss], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            accum_loss, accum_aux, accum_lm_loss = metrics.tolist()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()

        losses.append(accum_loss)

        # Logging
        # --- only the master process should print and log to WandB ---
        if master_process and (step % tcfg["log_every"] == 0 or step < 10):
            avg_loss = sum(losses[-tcfg["log_every"]:]) / len(losses[-tcfg["log_every"]:])
            elapsed = time.time() - t_start
            tokens_per_sec = ((step - start_step) + 1) * eff_batch_tokens / elapsed

            # Cross-loop agreement metric: agreement between routing decisions
            # across layers within the same tying group, measured as one minus
            # the average normalised count of distinct top-k expert selections.
            # A value of 1.0 means all layers in a tied group select the same
            # top-k experts; 0.0 means they select disjoint sets.
            routing_diversity = 0.0
            cross_loop_agreement = 1.0
            if tie_info.get("tied") and hasattr(out, "router_logits") and out.router_logits is not None:
                k = getattr(config, "num_experts_per_tok", 1)
                group_diversities = []
                for g in tie_info["groups"]:
                    if len(g) < 2: continue
                    g_logits = [out.router_logits[l] for l in g]
                    E_layer = g_logits[0].size(-1)
                    g_topk = [torch.topk(logits, k, dim=-1).indices for logits in g_logits]
                    all_choices = torch.cat(g_topk, dim=-1)
                    one_hot = torch.zeros(all_choices.size(0), E_layer, device=all_choices.device)
                    one_hot.scatter_(1, all_choices, 1.0)
                    avg_unique = one_hot.sum(dim=1).mean().item()
                    max_possible = min(len(g) * k, E_layer)
                    if max_possible > k:
                        div = (avg_unique - k) / (max_possible - k)
                        group_diversities.append(div)
                if group_diversities:
                    routing_diversity = sum(group_diversities) / len(group_diversities)
                    cross_loop_agreement = 1.0 - routing_diversity

            # === ROUTER METRICS ===
            # Computed only on Rank 0, every N steps, using the final micro-batch.
            router_metrics = {}
            
            # Pull from our hooked modules to bypass HF's corrupted tuple, 
            # or fallback to HF if the hook isn't active.
            working_metrics_logits = getattr(out, "router_logits", None)
                
            if working_metrics_logits is not None:
                with torch.no_grad(): # Ensure no autograd history is tracked
                    max_logit = 0.0
                    total_entropy = 0.0
                    max_fraction = 0.0
                    dead_fraction_sum = 0.0
                    valid_layers = 0
                    
                    for logits in working_metrics_logits:
                        if logits is None: continue
                        
                        # Safely unpack if it's a tuple
                        if isinstance(logits, tuple):
                            logits = logits[0]
                            
                        n_exp = logits.size(-1)
                        
                        # Upcast to float32 for safe metric calculation
                        flat_logits = logits.view(-1, n_exp).float() 
                        
                        # THIS WILL NOW BE THE REAL RAW LOGIT
                        max_logit = max(max_logit, flat_logits.max().item())
                        
                        probs = torch.nn.functional.softmax(flat_logits, dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1).mean().item()
                        total_entropy += entropy
                        
                        # Calculate top-1 fraction for speed
                        top1 = probs.argmax(dim=-1)
                        counts = torch.bincount(top1, minlength=n_exp).float()
                        fractions = counts / counts.sum()
                        
                        max_fraction = max(max_fraction, fractions.max().item())
                        # Fraction of experts in this layer that received no tokens.
                        # Averaged across layers below, this is in [0, 1] and is
                        # comparable across architectures with different n_experts.
                        dead_fraction_sum += (counts == 0.0).sum().item() / n_exp
                        valid_layers += 1
                        
                    if valid_layers > 0:
                        router_metrics = {
                            "router/max_logit": max_logit,
                            "router/routing_entropy": total_entropy / valid_layers,
                            "router/max_expert_fraction": max_fraction,
                            "router/dead_expert_fraction": dead_fraction_sum / valid_layers,
                        }

            # Update terminal strings
            if opt_type == "muon":
                lr_str_pbar = f"m:{lr:.5f}|a:{adamw_lr:.5f}"
                lr_str_print = f"muon={lr:.6f} adamw={adamw_lr:.6f}"
            else:
                lr_str_pbar = f"{lr:.6f}"
                lr_str_print = f"{lr:.6f}"

            agree_str = f" agree={cross_loop_agreement:.2f}" if tie_info.get("tied") else ""

            if pbar:
                pbar.set_postfix(loss=f"{avg_loss:.4f}", tps=f"{tokens_per_sec:.0f}", lr=lr_str_pbar)

            print(f"  [Step {step}] loss={avg_loss:.4f} aux={accum_aux:.4f} "
                  f"tps={tokens_per_sec:.0f} lr={lr_str_print}{agree_str} elapsed={elapsed:.0f}s",
                  flush=True)

            if run:
                log_dict = {
                    "train/loss": avg_loss,
                    "train/lm_loss": accum_lm_loss,  # Pure cross-entropy, no aux/z contribution.
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/aux_loss": accum_aux,
                    "train/tokens_seen": (step + 1) * eff_batch_tokens,
                }
                
                if tie_info.get("tied"):
                    log_dict["router/cross_loop_agreement"] = cross_loop_agreement
                    
                # append router metrics
                log_dict.update(router_metrics)
                
                # Dynamically split the LR metrics for WandB
                if opt_type == "muon":
                    log_dict["train/lr_muon"] = lr
                    log_dict["train/lr_adamw"] = adamw_lr
                else:
                    log_dict["train/lr"] = lr

                if device.type == "cuda":
                    log_dict["system/gpu_mem_gb"] = torch.cuda.memory_allocated() / 1e9
                    log_dict["system/gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    
                wandb.log(log_dict, step=step)

        # Eval
        if step > 0 and step % tcfg["eval_every"] == 0:
            eval_loss, eval_ppl = evaluate(model, eval_batches, device)

            if master_process:
                print(f"\n  [Step {step}] Eval Loss: {eval_loss:.4f} | PPL: {eval_ppl:.2f}")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    ckpt_path = save_dir / f"{run_name}_best.pt"
                    # DDP wraps the model in a .module attribute, so we need to unwrap it to save it cleanly
                    raw_model = model.module if is_ddp else model
                    raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
                    
                    torch.save({
                        "step": step,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        "best_eval_loss": best_eval_loss,
                        "arch": args.arch,
                        "config": config.to_dict(),
                        "expand_tied_experts": args.expand_tied_experts,
                        "wandb_run_id": run.id if run else None,
                    }, ckpt_path)

                if run:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/perplexity": eval_ppl,
                        "eval/best_loss": best_eval_loss,
                    }, step=step)

        # Save periodic checkpoint
        if step > 0 and step % tcfg["save_every"] == 0:
            if master_process:
                ckpt_path = save_dir / f"{run_name}_step{step}.pt"
                raw_model = model.module if is_ddp else model
                raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
                torch.save({
                    "step": step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "best_eval_loss": best_eval_loss,
                    "arch": args.arch,
                    "expand_tied_experts": args.expand_tied_experts,
                    "wandb_run_id": run.id if run else None,
                }, ckpt_path)
                print(f"  → Checkpoint: {ckpt_path}")

        if master_process and pbar:
            pbar.update(1)
        step += 1

    if master_process and pbar:
        pbar.close()
    
    # ── Save Final Checkpoint ──
    if master_process:
        ckpt_path = save_dir / f"{run_name}_step{step}.pt"
        raw_model = model.module if is_ddp else model
        raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
        torch.save({
            "step": step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "best_eval_loss": best_eval_loss, 
            "arch": args.arch,
            "expand_tied_experts": args.expand_tied_experts,
            "wandb_run_id": run.id if run else None,
        }, ckpt_path)
        print(f"\n  → Saved final checkpoint: {ckpt_path}")
        
    # ── Final eval ──
    final_eval_loss, final_ppl = evaluate(model, eval_batches, device)
    elapsed = time.time() - t_start

    results = {
        "arch": args.arch,
        "arch_name": arch_name,
        "optimizer": opt_type,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "final_eval_loss": final_eval_loss,
        "final_eval_ppl": final_ppl,
        "best_eval_loss": best_eval_loss,
        "training_time_sec": elapsed,
        "tokens_per_sec": n_steps * eff_batch_tokens / elapsed,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0,
    }

    if master_process:
        print(f"\n{'='*70}")
        print(f"  Final:  Loss={final_eval_loss:.4f}  PPL={final_ppl:.2f}  "
              f"Best={best_eval_loss:.4f}  Time={elapsed:.0f}s  "
              f"tok/s={results['tokens_per_sec']:,.0f}")
        if device.type == "cuda":
            print(f"  Peak VRAM: {results['peak_vram_gb']:.1f} GB")
        print(f"{'='*70}")

        if run:
            wandb.log({f"final/{k}": v for k, v in results.items() if isinstance(v, (int, float))})
            wandb.finish()

        # Save results
        results_path = save_dir / f"{run_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

    # Crucial DDP Cleanup
    if is_ddp:
        dist.destroy_process_group()

    return results


# ──────────────────────────────────────────────────────────────────────
# 7. CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="MoE pretraining on DCLM-edu + finephrase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arch", default="olmoe", choices=list(ARCH_REGISTRY.keys()))
    p.add_argument("--scale", default="regular", choices=["regular", "small", "tiny"],
                   help="Scale of the model. 'small' reduces width by 2x, 'tiny' reduces by 4x for fast iteration.")

    # Training
    p.add_argument("--train-config", default="default", choices=list(TRAIN_CONFIGS.keys()))
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="100 steps")

    # Optimizer
    p.add_argument("--optimizer", default="muon", choices=["adamw", "muon"])
    p.add_argument("--lr", type=float, default=None,
                   help="Override LR (default: 0.02 muon, 3e-4 adamw)")
    p.add_argument("--z-loss-coef", type=float, default=1e-4, 
                   help="Coefficient for the router Z-loss penalty (default: 1e-4)")

    # System
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--compile", action="store_true", help="torch.compile")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="checkpoints")

    # Expert tying
    p.add_argument("--tie-group-size", type=int, default=4,
                   help="Tie expert weights across N consecutive layers (default: 4). "
                        "Saves ~(N-1)/N of expert memory. Set to 1 to disable.")
    p.add_argument("--tie-skip-first", type=int, default=2,
                   help="Leave first N layers untied (default: 2)")
    p.add_argument("--tie-skip-last", type=int, default=2,
                   help="Leave last N layers untied (default: 2)")
    p.add_argument("--expand-tied-experts", type=int, default=None, 
                   help="Override number of experts ONLY in the tied middle layers (for param-matching)")
    p.add_argument("--tied-lr-divisor", type=float, default=1.0,
                   help="Divide learning rate by this factor for 3D expert parameters (e.g., 4.0 for group_size 4)")
                   
    # wandb
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="moe-pretraining")
    p.add_argument("--run-group", type=str, default="moe-v1")
    
    # resume from checkpoint
    p.add_argument("--resume", type=str, default=None, 
                   help="Path to checkpoint to resume from (e.g., checkpoints/olmoe_best.pt)")
    p.add_argument("--auto-resume", action="store_true", 
                   help="Auto-detect and resume from the latest step checkpoint for this run name")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()
