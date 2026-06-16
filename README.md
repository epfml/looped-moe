# Tying the Loop: Tied Expert Layers in Mixture-of-Experts Language Models

Reference implementation and experiments for **Expert Tying** - sharing expert FFN weights
across consecutive transformer layers while keeping routers, attention, and normalization
layer-specific. Tying experts in groups of `g` reduces expert parameters by g times.

This repository reproduces the experiments in [the paper](https://arxiv.org/abs/2606.16825) across different MoE
architectures (OLMoE, Qwen3-MoE, DeepSeekMoE) and the controlled component ablation.

## Repository contents

| File | Role |
|------|------|
| `train.py` | Component ablation (paper Section 3): vanilla depth-32 transformer, tying modes and topologies. |
| `model.py` | Expert-tying variants: expert-tensor aliasing and LR handling. |
| `moe_train.py` | Main experiments (paper Section 4): production OLMoE / Qwen3-MoE / DeepSeekMoE. |
| `data.py` | Streaming loader for the 75:25 DCLM-edu / FinePhrase mixture. |
| `eval_downstream.py` | 3-shot downstream accuracy via `lm-evaluation-harness`. |
| `submit_ablation_runs.sh` | Launches the full Section 3 ablation grid (43 runs). |
| `QUICKSTART.md` | Training commands for Section 4 architectures and configurations. |

## Installation

```bash
git clone https://github.com/epfml/looped-moe.git
cd looped-moe
pip install -r requirements.txt
```

Ablation study uses plain PyTorch, whereas the main experiment runs use the HuggingFace `transformers` reference implementations of OLMoE, Qwen3-MoE, and DeepSeekMoE; install exactly the pinned versions. 
Training uses [Muon](https://github.com/KellerJordan/Muon) for 2D hidden weights and AdamW for embeddings, output head, norm gains, and routers.

## Reproducing the paper

### Component ablation (Section 3)

The vanilla depth-32 ablation establishing which components can be tied. Each run is ~3 hours
on a single H100. Launch the full grid (fine + coarse granularity, all topologies and tying
modes, LR-divisor and dense controls):

```bash
bash submit_ablation_runs.sh
```

### Main experiments (Section 4)

Production MoEs at `g=1` (untied baseline), `g=2`, and `g=4`, with optional width expansion,
across all three architectures. The exact commands for every configuration are in
[`QUICKSTART.md`](QUICKSTART.md). Core flags of `moe_train.py`:
`--arch {olmoe, qwen3moe, deepseek}`, `--scale {regular, small, tiny}`,
`--tie-group-size` (1 = untied), `--expand-tied-experts N` (experts per tied middle layer),
`--tied-lr-divisor` (√g: 1.0 for g=1, 1.41 for g=2, 2.0 for g=4).

### Downstream evaluation

```bash
python eval_downstream.py --checkpoint <path-to-checkpoint>
```

Reports macro-average 3-shot accuracy on ARC-Easy, ARC-Challenge, HellaSwag, PIQA,
WinoGrande, and OpenBookQA.

## Citation

```bibtex
@article{jaggi2026tying,
  title   = {Tying the Loop: Tied Expert Layers in Mixture-of-Experts Language Models},
  author  = {Martin Jaggi},
  journal = {arXiv preprint arXiv:2606.16825},
  year    = {2026}
}
```
