#!/bin/bash
# ===================================================================
# Component-tying ablation — FULL grid
# ~43 runs total, ~260 GPU-h, ~1 day with 6 concurrent
# ===================================================================
#
# Structure:
#   Section A — Fine-grained grid (18 cells, --tied-lr-mode sqrt)
#     baseline + 5 topologies × 3 modes + moeut × 2 modes = 18 runs
#   Section B — Coarse-grained grid (18 cells, --tied-lr-mode sqrt)
#     baseline + 5 topologies × 3 modes + moeut × 2 modes = 18 runs
#   Section C — LR-mode ablation (6 runs)
#     fourgroups1_32 × 3 modes × {none, linear} divisor.
#     The `sqrt` baseline for these is in Section A as loop-4g-{alltie,attntie,experttie}.
#   Section D — Dense baseline reference (1 run)
#     dense_baseline_32 with --config small-dense
#
# Recipe (matches moe_train.py):
#   - z-loss enabled by default (z_loss_coef=1e-4 from per-config dict)
#   - aux-loss coefficient 0.01 for MoE configs, 0.0 for dense
#   - WD compensation for tied groups (weight_decay × n_loops in tied buckets)
#   - Sections A and B use --tied-lr-mode sqrt (LR / sqrt(n_loops) for tied params),
#     matching moe_train.py's hero-run convention.
#   - Section C tests `none` (no scaling) and `linear` (LR / n_loops) for comparison.
# ===================================================================


# ===================================================================
# SECTION A — Fine-grained grid (small config, fine-grained MoE)
# Establishes Findings 1 (topology spread) and 2 (mode dominance).
# All runs use --tied-lr-mode sqrt (default for headline).
# ===================================================================

# --- A.0: MoE baseline (no tying) ---
python csub.py -n loop-baseline-32 -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant baseline_32 --config small --optimizer muon --tied-lr-mode sqrt --run-group looped-vs-tied 2>&1 | tee -a logs/loop-baseline-32.log"

# --- A.1: 1 group, all 3 modes ---
python csub.py -n loop-1g-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant onegroup_32_alltie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-1g-alltie.log"

python csub.py -n loop-1g-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant onegroup_32_attntie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-1g-attntie.log"

python csub.py -n loop-1g-experttie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant onegroup_32_experttie --config small --optimizer muon --tied-lr-mode sqrt --run-group looped-vs-tied 2>&1 | tee -a logs/loop-1g-experttie.log"

# --- A.2: 2 groups, all 3 modes ---
python csub.py -n loop-2g-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant twogroup_32_alltie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-2g-alltie.log"

python csub.py -n loop-2g-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant twogroup_32_attntie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-2g-attntie.log"

python csub.py -n loop-2g-experttie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant twogroup_32_experttie --config small --optimizer muon --tied-lr-mode sqrt --run-group looped-vs-tied 2>&1 | tee -a logs/loop-2g-experttie.log"

# --- A.3: 3 groups, all 3 modes ---
python csub.py -n loop-3g-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant threegroup_32_alltie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-3g-alltie.log"

python csub.py -n loop-3g-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant threegroup_32_attntie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-3g-attntie.log"

python csub.py -n loop-3g-experttie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant threegroup_32_experttie --config small --optimizer muon --tied-lr-mode sqrt --run-group looped-vs-tied 2>&1 | tee -a logs/loop-3g-experttie.log"

# --- A.4: 4 groups, all 3 modes ---
python csub.py -n loop-4g-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_alltie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-4g-alltie.log"

python csub.py -n loop-4g-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_attntie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-4g-attntie.log"

python csub.py -n loop-4g-experttie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_experttie --config small --optimizer muon --tied-lr-mode sqrt --run-group looped-vs-tied 2>&1 | tee -a logs/loop-4g-experttie.log"

# --- A.7: 7 groups, all 3 modes ---
python csub.py -n loop-7g-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant sevengroups1_32_alltie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-7g-alltie.log"

python csub.py -n loop-7g-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant sevengroups1_32_attntie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-7g-attntie.log"

python csub.py -n loop-7g-experttie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant sevengroups1_32_experttie --config small --optimizer muon --tied-lr-mode sqrt --run-group looped-vs-tied 2>&1 | tee -a logs/loop-7g-experttie.log"

# --- A.M: MoEUT, both modes (no experttie variant — moeut has no untied prelude/coda) ---
python csub.py -n loop-moeut-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant moeut_32_alltie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-moeut-alltie.log"

python csub.py -n loop-moeut-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant moeut_32_attntie --config small --optimizer muon --tied-lr-mode sqrt --run-group sharing-modes 2>&1 | tee -a logs/loop-moeut-attntie.log"


# ===================================================================
# SECTION B — Coarse-grained grid (small-coarse config)
# Establishes Finding 3 (granularity-invariance of the ranking).
# All runs use --tied-lr-mode sqrt.
# ===================================================================

# --- B.0: MoE baseline ---
python csub.py -n loop-coarse-baseline -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant baseline_32 --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-baseline.log"

# --- B.1: 1 group ---
python csub.py -n loop-coarse-1g-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant onegroup_32_alltie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-1g-alltie.log"

python csub.py -n loop-coarse-1g-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant onegroup_32_attntie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-1g-attntie.log"

python csub.py -n loop-coarse-1g-experttie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant onegroup_32_experttie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-1g-experttie.log"

# --- B.2: 2 groups ---
python csub.py -n loop-coarse-2g-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant twogroup_32_alltie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-2g-alltie.log"

python csub.py -n loop-coarse-2g-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant twogroup_32_attntie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-2g-attntie.log"

python csub.py -n loop-coarse-2g-experttie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant twogroup_32_experttie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-2g-experttie.log"

# --- B.3: 3 groups ---
python csub.py -n loop-coarse-3g-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant threegroup_32_alltie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-3g-alltie.log"

python csub.py -n loop-coarse-3g-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant threegroup_32_attntie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-3g-attntie.log"

python csub.py -n loop-coarse-3g-experttie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant threegroup_32_experttie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-3g-experttie.log"

# --- B.4: 4 groups ---
python csub.py -n loop-coarse-4g-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_alltie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-4g-alltie.log"

python csub.py -n loop-coarse-4g-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_attntie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-4g-attntie.log"

python csub.py -n loop-coarse-4g-experttie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_experttie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-4g-experttie.log"

# --- B.7: 7 groups ---
python csub.py -n loop-coarse-7g-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant sevengroups1_32_alltie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-7g-alltie.log"

python csub.py -n loop-coarse-7g-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant sevengroups1_32_attntie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-7g-attntie.log"

python csub.py -n loop-coarse-7g-experttie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant sevengroups1_32_experttie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-7g-experttie.log"

# --- B.M: MoEUT, both modes ---
python csub.py -n loop-coarse-moeut-alltie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant moeut_32_alltie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-moeut-alltie.log"

python csub.py -n loop-coarse-moeut-attntie -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant moeut_32_attntie --config small-coarse --optimizer muon --tied-lr-mode sqrt --run-group granularity 2>&1 | tee -a logs/loop-coarse-moeut-attntie.log"


# ===================================================================
# SECTION C — LR-divisor ablation (fine, fourgroups1_32 topology only)
# Tests whether LR scaling for tied params matters for the headline finding.
# Three modes total in the comparison:
#   sqrt   : already in Section A as loop-4g-{alltie,attntie,experttie}
#   none   : no scaling (uniform LR), this section
#   linear : LR / n_loops, this section
# Topology fourgroups1_32 has n_loops=7 in the middle groups, so this is
# where the divisor effect is most visible.
# ===================================================================

# --- C.none: no LR scaling for tied params ---
python csub.py -n loop-4g-alltie-none -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_alltie --config small --optimizer muon --tied-lr-mode none --run-group lr-divisor 2>&1 | tee -a logs/loop-4g-alltie-none.log"

python csub.py -n loop-4g-attntie-none -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_attntie --config small --optimizer muon --tied-lr-mode none --run-group lr-divisor 2>&1 | tee -a logs/loop-4g-attntie-none.log"

python csub.py -n loop-4g-experttie-none -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_experttie --config small --optimizer muon --tied-lr-mode none --run-group lr-divisor 2>&1 | tee -a logs/loop-4g-experttie-none.log"

# --- C.linear: LR / n_loops ---
python csub.py -n loop-4g-alltie-lin -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_alltie --config small --optimizer muon --tied-lr-mode linear --run-group lr-divisor 2>&1 | tee -a logs/loop-4g-alltie-lin.log"

python csub.py -n loop-4g-attntie-lin -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_attntie --config small --optimizer muon --tied-lr-mode linear --run-group lr-divisor 2>&1 | tee -a logs/loop-4g-attntie-lin.log"

python csub.py -n loop-4g-experttie-lin -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant fourgroups1_32_experttie --config small --optimizer muon --tied-lr-mode linear --run-group lr-divisor 2>&1 | tee -a logs/loop-4g-experttie-lin.log"


# ===================================================================
# SECTION D — Dense baseline reference (1 run)
# A truly dense (non-MoE) transformer at the same effective scale, with
# d_ff sized to match MoE active FFN compute (top_k × d_ff_per_expert).
# Uses dense_baseline_32 with --config small-dense.
# Useful as an "is the MoE worth it at all?" reference point and as a
# control for the looping ablation.
#
# Note: --tied-lr-mode is a no-op here because dense_baseline_32 has no
# looped groups (n_loops=1 throughout). Included for log consistency only.
#
# Other dense variants are also registered if you want to extend the
# ablation into the dense regime: dense_twogroup_32_{alltie,ffnonly},
# dense_fourgroups1_32_{alltie,ffnonly}, dense_sevengroups1_32_{alltie,ffnonly}.
# (Dense uses _ffnonly instead of _experttie — no expert pool to single out.)
# ===================================================================

python csub.py -n loop-dense-baseline -g 1 --node-type h100 --train -t 3h --backofflimit 4 \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && python train.py --variant dense_baseline_32 --config small-dense --optimizer muon --tied-lr-mode sqrt --run-group looped-vs-tied 2>&1 | tee -a logs/loop-dense-baseline.log"


# ===================================================================
# Total: 18 fine + 18 coarse + 6 LR-divisor + 1 dense = 43 runs
# ~130 GPU-hours, ~1 day wall clock with 6 concurrent jobs
# ===================================================================
