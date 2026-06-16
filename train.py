"""
train.py — Main training loop for Looped MoE ablation study
=============================================================
Trains all 6 variants sequentially on DCLM-edu+finephrase, logs to wandb,
computes routing diversity metrics, and produces a final comparison table.

Usage:
    python train.py                          # Run all variants
    python train.py --variant full_stack     # Run a single variant
    python train.py --dry-run                # Quick sanity check (100 steps)
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import LoopedMoETransformer, ModelConfig, grouped_mm_status
from data import get_dataloader, get_eval_batches

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not installed — logging to console only")


# ============================================================
# Config
# ============================================================

# ============================================================
# Experiment groups
# ============================================================

# Experiment 1: Fine-grained MoE (32 experts, top_k=8) at depth 32
# Configs: small, 2b
EXP1_DEPTH32 = [
    # Upper bound: no looping
    "baseline_32",
    # MoEUT-style (no pre/post, maximal sharing)
    "moeut_32_alltie",
    "moeut_32_attntie",
    "moeut_32_lora",
    # Single looped core: 2+[2×14]+2
    "onegroup_32_alltie",
    "onegroup_32_attntie",
    "onegroup_32_lora",
    "onegroup_32_experttie",
    # Two looped cores: 2+[2×7]+[2×7]+2
    "twogroup_32_alltie",
    "twogroup_32_attntie",
    "twogroup_32_lora",
    "twogroup_32_experttie",
    # Three looped cores: 2+[2×5]+[2×4]+[2×5]+2
    "threegroup_32_alltie",
    "threegroup_32_attntie",
    "threegroup_32_lora",
    "threegroup_32_experttie",
    # Four groups, size 1: 2+[1×7]×4+2
    "fourgroups1_32_alltie",
    "fourgroups1_32_attntie",
    "fourgroups1_32_lora",
    "fourgroups1_32_experttie",
    # Seven groups, size 1: 2+[1×4]×7+2
    "sevengroups1_32_alltie",
    "sevengroups1_32_attntie",
    "sevengroups1_32_lora",
    "sevengroups1_32_experttie",
]

# Experiment 2: Coarse MoE (8 experts, top_k=2) at depth 32
# Same architecture variants, different expert granularity
# Configs: small-coarse, 2b-coarse
EXP2_COARSE = EXP1_DEPTH32  # same variant list, different config

# Experiment 3: Dense baselines (n_experts=1, top_k=1) at depth 32
# Same topologies, d_ff matched to MoE active FLOPs
# Configs: small-dense, 2b-dense
EXP3_DENSE = [
    "dense_baseline_32",
    "dense_twogroup_32_alltie",
    "dense_twogroup_32_ffnonly",
    "dense_fourgroups1_32_alltie",
    "dense_fourgroups1_32_ffnonly",
    "dense_sevengroups1_32_alltie",
    "dense_sevengroups1_32_ffnonly",
]

# All variants
VARIANTS = sorted(EXP1_DEPTH32 + EXP3_DENSE)

# ============================================================
# Config presets
# ============================================================
# Fine-grained: 32 experts, d_ff/4, top_k=8 (same active params)
# Coarse:       8 experts, d_ff,   top_k=2 (original)

CONFIGS = {
    # --- Fine-grained experts (Experiment 1) ---
    "tiny": dict(
        base_cfg=dict(
            d_model=256, n_heads=4, d_ff=128, n_experts=16, top_k=8,
            seq_len=256, vocab_size=100277, dropout=0.0, lora_rank=16,
        ),
        train_cfg=dict(
            batch_size=128, grad_accum=1, n_steps=2000, lr=3e-4, min_lr=3e-5,
            warmup_steps=50, weight_decay=0.01, grad_clip=1.0,
            aux_loss_coeff=0.01, z_loss_coeff=1e-4, eval_every=200, eval_batches=20,
            save_every=1000, log_every=20, compile_model=True,
        ),
    ),
    "small": dict(
        base_cfg=dict(
            d_model=512, n_heads=8, d_ff=256, n_experts=32, top_k=8,
            seq_len=512, vocab_size=100277, dropout=0.0, lora_rank=32,
        ),
        train_cfg=dict(
            batch_size=64, grad_accum=4, n_steps=10000, lr=3e-4, min_lr=3e-5,
            warmup_steps=50, weight_decay=0.01, grad_clip=1.0,
            aux_loss_coeff=0.01, z_loss_coeff=1e-4, eval_every=500, eval_batches=20,
            save_every=2500, log_every=50, compile_model=True,
        ),
    ),
    "small-coarse": dict(
        base_cfg=dict(
            d_model=512, n_heads=8, d_ff=1024, n_experts=8, top_k=2,
            seq_len=512, vocab_size=100277, dropout=0.0, lora_rank=32,
        ),
        train_cfg=dict(
            batch_size=64, grad_accum=4, n_steps=10000, lr=3e-4, min_lr=3e-5,
            warmup_steps=50, weight_decay=0.01, grad_clip=1.0,
            aux_loss_coeff=0.01, z_loss_coeff=1e-4, eval_every=500, eval_batches=20,
            save_every=2500, log_every=50, compile_model=True,
        ),
    ),
    "2b": dict(
        base_cfg=dict(
            d_model=2048, n_heads=16, d_ff=1376, n_experts=32, top_k=8,
            seq_len=1024, vocab_size=100277, dropout=0.0, lora_rank=64,
        ),
        train_cfg=dict(
            batch_size=8, grad_accum=8, n_steps=10000, lr=3e-4, min_lr=3e-5,
            warmup_steps=100, weight_decay=0.01, grad_clip=1.0,
            aux_loss_coeff=0.01, z_loss_coeff=1e-4, eval_every=500, eval_batches=20,
            save_every=2500, log_every=25, compile_model=True,
        ),
    ),
    "2b-coarse": dict(
        base_cfg=dict(
            d_model=2048, n_heads=16, d_ff=5504, n_experts=8, top_k=2,
            seq_len=1024, vocab_size=100277, dropout=0.0, lora_rank=64,
        ),
        train_cfg=dict(
            batch_size=8, grad_accum=8, n_steps=10000, lr=3e-4, min_lr=3e-5,
            warmup_steps=100, weight_decay=0.01, grad_clip=1.0,
            aux_loss_coeff=0.01, z_loss_coeff=1e-4, eval_every=500, eval_batches=20,
            save_every=2500, log_every=25, compile_model=True,
        ),
    ),
    # --- Dense baselines (Experiment 3) ---
    # d_ff = top_k × d_ff_per_expert to match MoE active FLOPs per token.
    # n_experts/top_k overridden to 1 by dense_ variant dicts.
    "small-dense": dict(
        base_cfg=dict(
            d_model=512, n_heads=8, d_ff=2048, n_experts=1, top_k=1,
            seq_len=512, vocab_size=100277, dropout=0.0, lora_rank=32,
        ),
        train_cfg=dict(
            batch_size=64, grad_accum=4, n_steps=10000, lr=3e-4, min_lr=3e-5,
            warmup_steps=50, weight_decay=0.01, grad_clip=1.0,
            aux_loss_coeff=0.0, z_loss_coeff=0.0, eval_every=500, eval_batches=20,
            save_every=2500, log_every=50, compile_model=True,
        ),
    ),
    "2b-dense": dict(
        base_cfg=dict(
            d_model=2048, n_heads=16, d_ff=11008, n_experts=1, top_k=1,
            seq_len=1024, vocab_size=100277, dropout=0.0, lora_rank=64,
        ),
        train_cfg=dict(
            batch_size=8, grad_accum=8, n_steps=10000, lr=3e-4, min_lr=3e-5,
            warmup_steps=100, weight_decay=0.01, grad_clip=1.0,
            aux_loss_coeff=0.0, z_loss_coeff=0.0, eval_every=500, eval_batches=20,
            save_every=2500, log_every=25, compile_model=True,
        ),
    ),
}

# Per-variant training overrides for 2b/2b-coarse configs (VRAM constrained)
# Keep effective batch constant: bs × grad_accum × seq_len = 65536 tokens/step
VARIANT_TRAIN_OVERRIDES_2B = {
    "baseline_32":                dict(batch_size=1, grad_accum=64),
}

# Per-variant training overrides for small/small-coarse configs
VARIANT_TRAIN_OVERRIDES_SMALL = {
    "baseline_32":            dict(batch_size=32, grad_accum=8),
}

# Active config (overridden by --config flag)
BASE_CFG = CONFIGS["small"]["base_cfg"]
TRAIN_CFG = CONFIGS["small"]["train_cfg"]


# ============================================================
# Training
# ============================================================

def get_lr(step, warmup_steps, n_steps, lr, min_lr):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def build_optimizer(model, optimizer_type, lr, weight_decay, tied_lr_mode="none"):
    """Construct the optimiser.

    Muon optimises 2D hidden weights; AdamW optimises routers, embeddings, the
    output head, and 1D parameters.

    The ``tied_lr_mode`` argument controls the per-step learning-rate scaling
    applied to parameters that live in looped groups (n_loops > 1). The same
    nn.Parameter is invoked n_loops times during the forward pass, so its
    accumulated gradient is approximately n_loops times larger than that of a
    non-shared parameter. This can be compensated by dividing its learning rate.

      ``none``   : no scaling (uniform LR across all parameters).
      ``linear`` : divide LR by n_loops.
      ``sqrt``   : divide LR by sqrt(n_loops).

    Parameter groups with n_loops = 1 are unaffected by all modes. Routers,
    embeddings, and the output head are routed to AdamW and are not scaled,
    since they are not shared across loop iterations in any of the supported
    architectures.
    """
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        )

    elif optimizer_type == "muon":
        adamw_params = []

        raw = model._orig_mod if hasattr(model, '_orig_mod') else model

        embed_and_head_ids = set()
        for p in raw.tok_emb.parameters():
            embed_and_head_ids.add(id(p))
        for p in raw.lm_head.parameters():
            embed_and_head_ids.add(id(p))

        # Map each unique parameter object to the n_loops value of the group it
        # belongs to. Native parameter sharing across loop iterations means the
        # same Parameter is referenced once inside its TransformerBlock, so the
        # block's n_loops_for_group is the multiplier on its gradient at backward.
        param_id_to_n_loops = {}
        for group_idx, group in enumerate(raw.groups):
            n_loops = raw.group_loops[group_idx]
            for p in group.parameters():
                if id(p) not in param_id_to_n_loops:
                    param_id_to_n_loops[id(p)] = n_loops

        # Bucket Muon parameters by n_loops so per-bucket LR and WD scaling can
        # be applied. The seen_ids set prevents double-registration of natively
        # shared parameters that appear under multiple names in named_parameters().
        muon_buckets = {}  # n_loops -> list of params
        seen_ids = set()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))

            is_embed_head = id(p) in embed_and_head_ids
            is_router = "router" in name and "weight" in name and p.dim() == 2

            if is_embed_head or is_router:
                adamw_params.append(p)
            elif p.dim() == 2:
                n_loops = param_id_to_n_loops.get(id(p), 1)
                muon_buckets.setdefault(n_loops, []).append(p)
            else:
                adamw_params.append(p)

        n_muon_total = sum(len(ps) for ps in muon_buckets.values())
        n_muon_numel = sum(p.numel() for ps in muon_buckets.values() for p in ps)
        bucket_summary = {k: len(v) for k, v in sorted(muon_buckets.items())}
        print(f"  Muon params: {n_muon_numel:,} ({n_muon_total} tensors)")
        print(f"    n_loops distribution: {bucket_summary}")
        print(f"  AdamW params: {sum(p.numel() for p in adamw_params):,} "
              f"({len(adamw_params)} tensors)")
        print(f"  tied_lr_mode: {tied_lr_mode}")

        from torch.optim import Muon as TorchMuon

        # Decoupled weight decay (Loshchilov & Hutter, 2019). Muon hidden weights
        # use weight_decay=0.1, following common production MoE recipes (Llama,
        # Qwen, DeepSeek). AdamW handles routers, embeddings, head, and 1D
        # parameters; the AdamW weight decay is left at 0 in this codebase to
        # match the existing small-ablation training runs.
        muon_wd = 0.1
        adamw_wd = 0.0

        # One Muon parameter group per n_loops bucket. Each group is tagged with
        # n_loops so the LR schedule can apply the per-bucket divisor.
        muon_param_groups = [
            {
                "params": params,
                "n_loops": n_loops,
                "weight_decay": muon_wd,
            }
            for n_loops, params in sorted(muon_buckets.items())
        ]

        muon_opt = TorchMuon(
            muon_param_groups,
            lr=lr,
            momentum=0.95,
            nesterov=True,
            weight_decay=muon_wd,
            ns_steps=5,
        )
        adamw_opt = torch.optim.AdamW(
            adamw_params,
            lr=lr * 0.1,
            weight_decay=adamw_wd,
            betas=(0.9, 0.95),
        )

        # Wrapper that exposes a single optimiser interface stepping both Muon
        # and AdamW together.
        class DualOptimizer:
            def __init__(self, muon, adamw, tied_lr_mode):
                self.muon = muon
                self.adamw = adamw
                self.tied_lr_mode = tied_lr_mode
                self.param_groups = muon.param_groups + adamw.param_groups

            def zero_grad(self, set_to_none=False):
                self.muon.zero_grad(set_to_none=set_to_none)
                self.adamw.zero_grad(set_to_none=set_to_none)

            def step(self):
                self.muon.step()
                self.adamw.step()

            def state_dict(self):
                return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

        return DualOptimizer(muon_opt, adamw_opt, tied_lr_mode)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def _tied_lr_divisor(n_loops, mode):
    """Return the learning-rate divisor for a parameter group with the given n_loops.

    Modes:

      ``none``   -> 1.0 (no scaling).
      ``linear`` -> n_loops.
      ``sqrt``   -> sqrt(n_loops).

    Untied groups (n_loops <= 1) always return 1.0 regardless of the mode.
    """
    if n_loops <= 1:
        return 1.0
    if mode == "none":
        return 1.0
    elif mode == "linear":
        return float(n_loops)
    elif mode == "sqrt":
        return math.sqrt(n_loops)
    else:
        raise ValueError(f"Unknown tied_lr_mode: {mode}")


@torch.no_grad()
def evaluate(model, eval_batches, device, cfg):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    use_amp = device.type == "cuda"

    for input_ids, targets in eval_batches:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss, _, _ = model(input_ids, targets)
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    model.train()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(min(avg_loss, 20))


@torch.no_grad()
def analyze_routing(model, eval_batches, device):
    """Measure routing diversity across loops within each looped group."""
    # Get the raw model if compiled
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model

    # Check if any group has loops > 1
    has_loops = any(n > 1 for n in raw.group_loops)
    if not has_loops:
        return {"cross_loop_agreement": 1.0, "avg_routing_entropy": 0.0}

    model.eval()
    all_agreements = []
    all_entropies = []
    use_amp = device.type == "cuda"

    for input_ids, targets in eval_batches[:3]:  # use 3 batches
        input_ids = input_ids.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            x = raw.tok_emb(input_ids)

            for group_idx, group in enumerate(raw.groups):
                n_loops = raw.group_loops[group_idx]

                if n_loops <= 1:
                    # Non-looped group: just run forward, collect entropy
                    for loop in range(n_loops):
                        for layer in group:
                            x, router_logits = layer(x, loop_idx=loop)
                            probs = F.softmax(router_logits, dim=-1)
                            ent = -(probs * (probs + 1e-10).log()).sum(-1).mean()
                            all_entropies.append(ent.item())
                else:
                    # Looped group: run each loop, compare routing decisions
                    per_loop_decisions = []
                    x_input = x  # save input to this group for each loop

                    for loop in range(n_loops):
                        x_loop = x_input
                        loop_decisions = []
                        for layer in group:
                            x_loop, router_logits = layer(x_loop, loop_idx=loop)
                            top1 = router_logits.argmax(dim=-1)
                            loop_decisions.append(top1)
                            probs = F.softmax(router_logits, dim=-1)
                            ent = -(probs * (probs + 1e-10).log()).sum(-1).mean()
                            all_entropies.append(ent.item())
                        per_loop_decisions.append(torch.cat(loop_decisions))

                        if loop == n_loops - 1:
                            x = x_loop  # use last loop's output to continue

                    # Cross-loop agreement
                    for i in range(n_loops):
                        for j in range(i + 1, n_loops):
                            agree = (per_loop_decisions[i] == per_loop_decisions[j]).float().mean().item()
                            all_agreements.append(agree)

    model.train()

    avg_agreement = sum(all_agreements) / len(all_agreements) if all_agreements else 1.0
    avg_entropy = sum(all_entropies) / len(all_entropies) if all_entropies else 0.0

    return {
        "cross_loop_agreement": avg_agreement,
        "avg_routing_entropy": avg_entropy,
    }


def train_variant(variant_name, args, device, save_dir):
    """Train one model variant and return results."""
    print(f"\n{'=' * 70}")
    print(f"  TRAINING VARIANT: {variant_name}")
    print(f"{'=' * 70}")

    # Build model
    model = LoopedMoETransformer.make_variant(variant_name, BASE_CFG)
    total_params, unique_params = model.count_params()
    effective_depth = model.cfg.effective_depth

    print(f"  Unique params:    {unique_params:>12,}")
    print(f"  Total params:     {total_params:>12,}")
    print(f"  Effective depth:  {effective_depth}")
    print(f"  Layers x Loops:   {model.cfg.n_unique_layers} x {model.cfg.n_loops}")
    print(f"  Pre/Post layers:  {model.cfg.n_pre_layers} / {model.cfg.n_post_layers}")
    print(f"  Per-loop routers: {model.cfg.per_loop_routers}")
    print(f"  Attn LoRA:        {model.cfg.use_attn_lora}")
    print(f"  Experts:          {model.cfg.n_experts} x d_ff={model.cfg.d_ff}, top_k={model.cfg.top_k}")
    print(f"  Topology:         {model.cfg.topology}")

    model = model.to(device)

    # Use bf16 autocast on CUDA for speed + grouped_mm compatibility
    use_amp = device.type == "cuda"
    amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    # Log grouped_mm status — the fast path needs: API available + CUDA
    # (bf16 is handled by autocast, not checked at dispatch time)
    gmm = grouped_mm_status(
        device=device,
        dtype=torch.bfloat16 if use_amp else torch.float32,
    )
    if gmm["active"]:
        print(f"  grouped_mm:   ACTIVE ({gmm['api_name']} on {device})")
        if not gmm["is_bf16"]:
            print(f"  WARNING: autocast not bf16 — grouped_mm requires bf16 inputs at runtime")
    else:
        reasons = []
        if not gmm["has_api"]:
            reasons.append(f"API not found (need PyTorch >= 2.8, have {torch.__version__})")
        if not gmm["on_cuda"]:
            reasons.append(f"not on CUDA (device={device})")
        print(f"  grouped_mm:   FALLBACK (padded bmm) — {'; '.join(reasons)}")

    # Optimizer — must be built BEFORE torch.compile so param ids match
    opt_type = args.optimizer if hasattr(args, 'optimizer') else "adamw"
    if args.lr is not None:
        opt_lr = args.lr
    elif opt_type == "muon":
        opt_lr = 0.02
    else:
        opt_lr = TRAIN_CFG["lr"]  # 3e-4 for adamw
    optimizer = build_optimizer(model, opt_type, opt_lr, TRAIN_CFG["weight_decay"],
                                tied_lr_mode=getattr(args, "tied_lr_mode", "none"))
    print(f"  Optimizer: {opt_type}, LR: {opt_lr}")

    # Compile if available and requested (after optimizer creation)
    if TRAIN_CFG["compile_model"] and hasattr(torch, "compile"):
        try:
            # Raise recompile limit: loop_idx causes specialization for each
            # unique value, but the set is bounded (max_loops per group).
            # Default limit is 8; we need at most max_loops.
            import torch._dynamo.config as dynamo_config
            max_loops = max((nl for _, nl in model.cfg.topology), default=1)
            dynamo_config.cache_size_limit = max(max_loops + 2, 16)
            model = torch.compile(model)
            print(f"  torch.compile: enabled (cache_size_limit={dynamo_config.cache_size_limit})")
        except Exception as e:
            print(f"  torch.compile: failed ({e}), using eager")

    # Data
    tcfg = TRAIN_CFG
    train_loader = get_dataloader(
        seq_len=BASE_CFG["seq_len"],
        batch_size=tcfg["batch_size"],
        num_workers=0,
    )
    eval_batches = get_eval_batches(
        seq_len=BASE_CFG["seq_len"],
        batch_size=tcfg["batch_size"],
        n_batches=tcfg["eval_batches"],
    )
    print(f"  Eval batches loaded: {len(eval_batches)}", flush=True)

    # wandb
    run = None
    if HAS_WANDB and not args.no_wandb:
        print("  Initializing wandb ...", flush=True)
        # Name matches job name: {small|2b}-{f|c}-{variant-with-hyphens}
        gran = "c" if "coarse" in args.config else "f"
        size = args.config.split("-")[0]  # small, 2b
        wandb_name = f"{size}-{gran}-{variant_name.replace('_', '-')}"

        run = wandb.init(
            project="looped-moe-ablation",
            name=wandb_name,
            group=args.run_group,
            config={
                "variant": variant_name,
                "config": args.config,
                "model_config": asdict(model.cfg) if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule) else BASE_CFG,
                "train_config": tcfg,
                "unique_params": unique_params,
                "total_params": total_params,
                "effective_depth": effective_depth,
            },
            reinit=True,
        )
        print(f"  wandb: {wandb_name}", flush=True)

    # Training loop
    n_steps = args.n_steps or tcfg["n_steps"]
    step = 0
    losses = []
    best_eval_loss = float("inf")
    t_start = time.time()

    model.train()
    print("  Fetching first training batch ...", flush=True)
    train_iter = iter(train_loader)
    _first_batch = next(train_iter)  # force the first fetch
    print("  First batch ready. Starting training loop.", flush=True)

    pbar = tqdm(total=n_steps, desc=variant_name, ncols=100)

    while step < n_steps:
        # Cosine learning-rate schedule with warmup.
        min_lr = opt_lr * (TRAIN_CFG["min_lr"] / TRAIN_CFG["lr"])
        lr = get_lr(step, tcfg["warmup_steps"], n_steps, opt_lr, min_lr)
        if opt_type == "muon":
            tied_mode = getattr(optimizer, "tied_lr_mode", "none")
            for pg in optimizer.muon.param_groups:
                n_loops = pg.get("n_loops", 1)
                divisor = _tied_lr_divisor(n_loops, tied_mode)
                pg["lr"] = lr / divisor
            # AdamW peaks at opt_lr * 0.1 with min at min_lr * 0.1. AdamW handles
            # routers, embeddings and the output head, none of which are shared
            # across loop iterations, so no divisor is applied here.
            adamw_lr = get_lr(step, tcfg["warmup_steps"], n_steps, opt_lr * 0.1, min_lr * 0.1)
            for pg in optimizer.adamw.param_groups:
                pg["lr"] = adamw_lr
        else:
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # Gradient accumulation
        grad_accum = tcfg.get("grad_accum", 1)
        optimizer.zero_grad(set_to_none=True)
        accum_lm_loss = 0.0
        accum_aux_loss = 0.0
        accum_z_loss = 0.0

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

            with amp_ctx():
                _, lm_loss, aux_loss, all_router_logits = model(input_ids, targets)

                # Router z-loss (Zoph et al., 2022): per-layer mean of squared
                # logsumexp of the router logits, computed in float32, then averaged
                # across layers. The coefficient is applied directly to the returned
                # loss; no further scaling by a HuggingFace router-aux coefficient
                # is required in this codebase.
                z_loss_coeff = tcfg.get("z_loss_coeff", 0.0)
                if z_loss_coeff > 0 and len(all_router_logits) > 0:
                    z_terms = []
                    for rl in all_router_logits:
                        # rl is either [B*S, n_experts] or [B, S, n_experts];
                        # flatten to [N, n_experts] before logsumexp.
                        flat = rl.view(-1, rl.size(-1)).float()
                        z_terms.append(torch.mean(torch.logsumexp(flat, dim=-1) ** 2))
                    z_loss = sum(z_terms) / len(z_terms)
                else:
                    z_loss = torch.zeros((), device=lm_loss.device)

                loss = (lm_loss
                        + tcfg["aux_loss_coeff"] * aux_loss
                        + z_loss_coeff * z_loss) / grad_accum
            loss.backward()

            accum_lm_loss += lm_loss.item() / grad_accum
            accum_aux_loss += aux_loss.item() / grad_accum
            accum_z_loss += z_loss.item() / grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()

        losses.append(accum_lm_loss)

        # Logging
        if step % tcfg["log_every"] == 0 or step < 10:
            avg_loss = sum(losses[-tcfg["log_every"]:]) / len(losses[-tcfg["log_every"]:])
            elapsed = time.time() - t_start
            grad_accum = tcfg.get("grad_accum", 1)
            tokens_per_sec = (step + 1) * tcfg["batch_size"] * grad_accum * BASE_CFG["seq_len"] / elapsed

            pbar.set_postfix(loss=f"{avg_loss:.4f}", tps=f"{tokens_per_sec:.0f}", lr=f"{lr:.6f}")

            # Print to stdout so the cluster scheduler captures progress; tqdm
            # writes to stderr only.
            if step < 10 or step % tcfg["log_every"] == 0:
                z_str = f" z={accum_z_loss:.3f}" if tcfg.get("z_loss_coeff", 0.0) > 0 else ""
                print(f"  [Step {step}] loss={avg_loss:.4f} aux={accum_aux_loss:.3f}{z_str} "
                      f"tps={tokens_per_sec:.0f} lr={lr:.6f} elapsed={elapsed:.0f}s", flush=True)

            if run:
                log_dict = {
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/aux_loss": accum_aux_loss,
                }
                if tcfg.get("z_loss_coeff", 0.0) > 0:
                    log_dict["train/z_loss"] = accum_z_loss
                wandb.log(log_dict, step=step)

        # Eval
        if step > 0 and step % tcfg["eval_every"] == 0:
            eval_loss, eval_ppl = evaluate(model, eval_batches, device, model.cfg)
            routing_metrics = analyze_routing(model, eval_batches, device)

            print(f"\n  [Step {step}] Eval Loss: {eval_loss:.4f} | PPL: {eval_ppl:.2f} | "
                  f"Cross-loop agreement: {routing_metrics['cross_loop_agreement']:.3f}")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                # Save best checkpoint
                ckpt_path = save_dir / f"{variant_name}_best.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "eval_loss": eval_loss,
                    "variant": variant_name,
                }, ckpt_path)

            if run:
                wandb.log({
                    "eval/loss": eval_loss,
                    "eval/perplexity": eval_ppl,
                    "eval/best_loss": best_eval_loss,
                    **{f"routing/{k}": v for k, v in routing_metrics.items()},
                }, step=step)

        pbar.update(1)
        step += 1

    pbar.close()

    # Final eval
    final_eval_loss, final_ppl = evaluate(model, eval_batches, device, model.cfg)
    routing_metrics = analyze_routing(model, eval_batches, device)
    elapsed = time.time() - t_start

    results = {
        "variant": variant_name,
        "unique_params": unique_params,
        "total_params": total_params,
        "effective_depth": effective_depth,
        "n_unique_layers": model.cfg.n_unique_layers,
        "n_loops": model.cfg.n_loops,
        "per_loop_routers": model.cfg.per_loop_routers,
        "use_attn_lora": model.cfg.use_attn_lora,
        "final_eval_loss": final_eval_loss,
        "final_eval_ppl": final_ppl,
        "best_eval_loss": best_eval_loss,
        "param_memory_mb": unique_params * 4 / (1024 * 1024),
        "training_time_sec": elapsed,
        "tokens_per_sec": n_steps * tcfg["batch_size"] * tcfg.get("grad_accum", 1) * BASE_CFG["seq_len"] / elapsed,
        **routing_metrics,
    }

    print(f"\n  Final:  Loss={final_eval_loss:.4f}  PPL={final_ppl:.2f}  "
          f"Memory={results['param_memory_mb']:.1f}MB  Time={elapsed:.0f}s")

    if run:
        wandb.log({f"final/{k}": v for k, v in results.items() if isinstance(v, (int, float))})
        wandb.finish()

    # Free memory
    del model, optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


# ============================================================
# Main
# ============================================================

def main():
    EXPERIMENT_GROUPS = {
        "exp1": EXP1_DEPTH32,       # fine-grained MoE (use with small, 2b)
        "exp2": EXP2_COARSE,        # coarse MoE (use with small-coarse, 2b-coarse)
        "exp3": EXP3_DENSE,         # dense baselines (use with small-dense, 2b-dense)
    }

    parser = argparse.ArgumentParser(description="Looped MoE Ablation Study")
    parser.add_argument("--variant", type=str, default=None, choices=VARIANTS,
                        help="Run a single variant (default: run all)")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=list(EXPERIMENT_GROUPS.keys()),
                        help="Run a predefined experiment group (exp1=fine-grained, exp2=coarse)")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Override number of training steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test with 100 steps")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--run-group", type=str, default="ablation-v1",
                        help="wandb group name")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints and results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    parser.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"],
                        help="Optimizer to use (default: muon)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (default: 0.02 for muon, 3e-4 for adamw)")
    parser.add_argument("--config", type=str, default="small", choices=list(CONFIGS.keys()),
                        help="Model/training size preset (default: small)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override micro batch size")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Override gradient accumulation steps")
    parser.add_argument("--tied-lr-mode", type=str, default="none",
                        choices=["none", "linear", "sqrt"],
                        help="Per-step learning-rate scaling for parameters in looped groups "
                             "(n_loops > 1). 'none' applies a uniform learning rate; 'linear' "
                             "divides by n_loops; 'sqrt' divides by sqrt(n_loops). Parameters "
                             "in untied groups (n_loops = 1) are unaffected.")
    parser.add_argument("--z-loss-coef", type=float, default=None,
                        help="Override the router z-loss coefficient. If unset, the per-config "
                             "value is used (1e-4 for MoE configs, 0 for dense baselines).")
    args = parser.parse_args()

    # Load the chosen base and training config preset.
    global BASE_CFG, TRAIN_CFG
    BASE_CFG = dict(CONFIGS[args.config]["base_cfg"])
    TRAIN_CFG = dict(CONFIGS[args.config]["train_cfg"])
    print(f"Config: {args.config}")
    print(f"  n_experts={BASE_CFG['n_experts']}, d_ff={BASE_CFG['d_ff']}, "
          f"top_k={BASE_CFG['top_k']}, d_model={BASE_CFG['d_model']}, "
          f"seq_len={BASE_CFG['seq_len']}")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Dirs
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        args.n_steps = 100
        TRAIN_CFG["eval_every"] = 50
        TRAIN_CFG["log_every"] = 10
        print("DRY RUN MODE: 100 steps")

    # Apply CLI overrides for batch size, gradient accumulation, and z-loss.
    if args.batch_size is not None:
        TRAIN_CFG["batch_size"] = args.batch_size
        print(f"  CLI override: batch_size={args.batch_size}")
    if args.grad_accum is not None:
        TRAIN_CFG["grad_accum"] = args.grad_accum
        print(f"  CLI override: grad_accum={args.grad_accum}")
    if args.z_loss_coef is not None:
        TRAIN_CFG["z_loss_coeff"] = args.z_loss_coef
        print(f"  CLI override: z_loss_coeff={args.z_loss_coef}")

    # Select the variants to run.
    if args.variant:
        variants_to_run = [args.variant]
    elif args.experiment:
        variants_to_run = EXPERIMENT_GROUPS[args.experiment]
        print(f"Running experiment group '{args.experiment}': {len(variants_to_run)} variants")
    else:
        variants_to_run = VARIANTS
    all_results = []

    for variant in variants_to_run:
        # Apply per-variant training overrides for the active config.
        overrides = {}
        if args.config in ("2b", "2b-coarse"):
            overrides = VARIANT_TRAIN_OVERRIDES_2B.get(variant, {})
        elif args.config in ("small", "small-coarse"):
            overrides = VARIANT_TRAIN_OVERRIDES_SMALL.get(variant, {})
        for k, v in overrides.items():
            TRAIN_CFG[k] = v
            print(f"  Override for {variant} ({args.config}): {k}={v}")

        # CLI flags take precedence over per-variant overrides.
        if args.batch_size is not None:
            TRAIN_CFG["batch_size"] = args.batch_size
        if args.grad_accum is not None:
            TRAIN_CFG["grad_accum"] = args.grad_accum
        if args.z_loss_coef is not None:
            TRAIN_CFG["z_loss_coeff"] = args.z_loss_coef

        result = train_variant(variant, args, device, save_dir)
        all_results.append(result)

        # Save intermediate results
        with open(save_dir / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # ============================================================
    # Final comparison table
    # ============================================================
    if len(all_results) > 1:
        print("\n\n" + "=" * 110)
        print("FINAL RESULTS COMPARISON")
        print("=" * 110)

        header = (f"{'Variant':<25} {'Params':>10} {'Memory':>8} {'Depth':>6} "
                  f"{'Loss':>8} {'PPL':>8} {'tok/s':>10} {'Agreement':>9}")
        print(header)
        print("-" * 110)

        for r in all_results:
            print(f"  {r['variant']:<23} {r['unique_params']:>10,} "
                  f"{r['param_memory_mb']:>7.1f}MB {r['effective_depth']:>6} "
                  f"{r['final_eval_loss']:>8.4f} {r['final_eval_ppl']:>8.2f} "
                  f"{r['tokens_per_sec']:>10,.0f} "
                  f"{r.get('cross_loop_agreement', 1.0):>9.3f}")

        # Key comparisons
        by_name = {r["variant"]: r for r in all_results}

        if "baseline_nonlooped" in by_name and "looped_alltie" in by_name:
            bl = by_name["baseline_nonlooped"]
            ls = by_name["looped_alltie"]
            gap = ls["final_eval_loss"] - bl["final_eval_loss"]
            mem = bl["param_memory_mb"] / ls["param_memory_mb"]
            speed = bl["tokens_per_sec"] / ls["tokens_per_sec"]
            print(f"\n  Compute-matched gap (looped_alltie vs nonlooped): "
                  f"Δloss={gap:+.4f}  mem_savings={mem:.1f}×  speed_ratio={speed:.2f}×")

        if "baseline_nonlooped" in by_name and "full_stack" in by_name:
            bl = by_name["baseline_nonlooped"]
            fs = by_name["full_stack"]
            gap = fs["final_eval_loss"] - bl["final_eval_loss"]
            mem = bl["param_memory_mb"] / fs["param_memory_mb"]
            print(f"  Full stack vs nonlooped:                          "
                  f"Δloss={gap:+.4f}  mem_savings={mem:.1f}×")

        if "looped_alltie" in by_name and "full_stack" in by_name:
            ls = by_name["looped_alltie"]
            fs = by_name["full_stack"]
            imp = ls["final_eval_loss"] - fs["final_eval_loss"]
            overhead = (fs["unique_params"] - ls["unique_params"]) / ls["unique_params"] * 100
            print(f"  Per-loop differentiation (full vs shared):        "
                  f"Δloss={imp:+.4f}  param_overhead={overhead:+.1f}%")

        print(f"\n  Progressive ablation from fully-shared looped:")
        if "looped_alltie" in by_name:
            base = by_name["looped_alltie"]["final_eval_loss"]
            for name in ["looped_alltie", "untied_routers", "routers_attn_lora", "full_stack"]:
                if name in by_name:
                    r = by_name[name]
                    delta = base - r["final_eval_loss"]
                    print(f"    {name:<25} loss={r['final_eval_loss']:.4f}  "
                          f"Δ={delta:+.4f}  params={r['unique_params']:,}")

    print(f"\nResults saved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
