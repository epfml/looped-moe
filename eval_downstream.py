"""
eval_downstream.py — Batched Downstream Benchmarking for HF MoE Models
======================================================================
Evaluates checkpoints trained via moe_train.py on standard NLP benchmarks 
using EleutherAI's lm-evaluation-harness with H100-optimized batching.

usage
python csub.py -n eval -g 1 --node-type h100 --train -t 1h \
    --command "source looped-moe-experiment/.venv/bin/activate && cd looped-moe-experiment && mkdir -p logs && \
    python eval_downstream.py \
    --ckpt checkpoints/deepseek-g4-we128_step5000.pt \
    --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa \
    --batch-size 64 \
    --num-fewshot 3 \
    --tie-group-size 4 \
 2>&1 | tee -a logs/eval.log"
 
 
 you can limit the number of questions per benchmark for speed if needed (not active now)
    --limit 250 \
"""

import argparse
import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm

# Import the architecture registry and tying function directly from your training script
from moe_train import ARCH_REGISTRY, tie_expert_layers

try:
    import lm_eval
    from lm_eval.api.model import LM
except ImportError:
    raise ImportError("Please install lm-eval: pip install lm-eval")

class HFMoEEvalWrapper(LM):
    def __init__(self, ckpt_path, device="cuda", batch_size=64, tie_group_size=1, tie_skip_first=2, tie_skip_last=2):
        super().__init__()
        # Use internal variables to avoid setting the protected property
        self._device = torch.device(device)
        self._batch_size = batch_size
        self.enc = tiktoken.get_encoding("cl100k_base")
        
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # 1. Auto-detect architecture and scale from checkpoint
        arch = checkpoint.get("arch", "olmoe") 
        print(f"Detected Architecture: {arch}")
        
        scale = "regular"
        if "tiny" in ckpt_path:
            scale = "tiny"
        elif "small" in ckpt_path:
            scale = "small"
        print(f"Detected Scale: {scale}")
        
        _, config_fn = ARCH_REGISTRY[arch]
        config = config_fn(scale=scale)
        
        # Force grouped_mm for maximum speed on the H100
        config._experts_implementation = "grouped_mm"
        
        # 2. Build the correct Hugging Face model
        if arch == "qwen3moe":
            from transformers import Qwen3MoeForCausalLM
            self.model = Qwen3MoeForCausalLM(config)
        else:
            from transformers import OlmoeForCausalLM
            self.model = OlmoeForCausalLM(config)

        # 2.5 Rebuild Heterogeneous Experts if trained with it!
        expand_tied = checkpoint.get("expand_tied_experts", None)
        if expand_tied is not None:
            print(f"Rebuilding heterogeneous architecture: Middle layers expanded to {expand_tied} experts.")
            import copy
            config_expanded = copy.deepcopy(config)
            config_expanded.num_experts = expand_tied
            
            if arch == "qwen3moe":
                dummy_model = Qwen3MoeForCausalLM(config_expanded)
            else:
                dummy_model = OlmoeForCausalLM(config_expanded)
                
            layers = self.model.model.layers if hasattr(self.model, "model") else self.model.layers
            dummy_layers = dummy_model.model.layers if hasattr(dummy_model, "model") else dummy_model.layers
            
            middle_start = tie_skip_first
            middle_end = len(layers) - tie_skip_last
            
            for i in range(middle_start, middle_end):
                layers[i].mlp = dummy_layers[i].mlp
                
            del dummy_model
            
        # 3. Apply expert tying if the model was trained with it
        if tie_group_size > 1:
            tie_expert_layers(
                self.model, 
                group_size=tie_group_size, 
                skip_first=tie_skip_first, 
                skip_last=tie_skip_last, 
                master_process=True
            )
        
        # 4. Clean keys (remove DDP and torch.compile prefixes)
        state_dict = checkpoint["model_state_dict"]
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            if k.startswith("_orig_mod."):
                k = k[10:]
            clean_state_dict[k] = v
            
        self.model.load_state_dict(clean_state_dict)
        self.model.to(self._device)
        self.model.eval()

    # ─── Required properties for the new lm-eval main branch ───
    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_length(self):
        return 4096

    @property
    def eot_token_id(self):
        return 100257 # Match your tokenizer's BOS/EOS token

    def loglikelihood(self, requests):
        results = []
        
        for i in tqdm(range(0, len(requests), self.batch_size), desc="Evaluating Batches"):
            batch_reqs = requests[i:i+self.batch_size]
            
            batch_inputs = []
            batch_conts = []
            bounds = []
            
            for req in batch_reqs:
                context, continuation = req.args
                ctx_tokens = self.enc.encode_ordinary(context)
                cont_tokens = self.enc.encode_ordinary(continuation)
                
                max_len = 4096 # Match max_position_embeddings
                
                # Check if we need to truncate (leaving 1 space for the BOS token)
                if len(ctx_tokens) + len(cont_tokens) + 1 > max_len:
                    keep = max_len - len(cont_tokens) - 1
                    ctx_tokens = ctx_tokens[-keep:]
                
                # Unconditionally prepend the BoD/BOS token
                ctx_tokens = [100257] + ctx_tokens
                input_tokens = ctx_tokens + cont_tokens
                    
                cont_start = len(ctx_tokens) - 1
                cont_end = len(input_tokens) - 1
                
                batch_inputs.append(input_tokens)
                batch_conts.append(cont_tokens)
                bounds.append((cont_start, cont_end))
                
            # Right-pad for causal batching
            max_batch_len = max(len(toks) for toks in batch_inputs)
            padded_inputs = []
            for toks in batch_inputs:
                pad_len = max_batch_len - len(toks)
                padded_inputs.append(toks + [100257] * pad_len)
                
            input_ids = torch.tensor(padded_inputs, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Extract logits from HF CausalLMOutput
                    out = self.model(input_ids=input_ids, output_router_logits=False)
                    logits = out.logits
                    
            for b_idx, (cont_start, cont_end) in enumerate(bounds):
                cont_logits = logits[b_idx, cont_start:cont_end, :]
                logprobs = F.log_softmax(cont_logits, dim=-1)
                
                cont_tensor = torch.tensor(batch_conts[b_idx], dtype=torch.long).to(self.device)
                token_logprobs = torch.gather(logprobs, dim=-1, index=cont_tensor.unsqueeze(-1)).squeeze(-1)
                total_logprob = token_logprobs.sum().item()
                
                is_greedy = (cont_logits.argmax(dim=-1) == cont_tensor).all().item()
                results.append((total_logprob, is_greedy))
                
        return results

    def generate_until(self, requests):
        return [""] * len(requests)

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--tasks", type=str, default="arc_easy,hellaswag,piqa,winogrande,openbookqa")
    parser.add_argument("--num-fewshot", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64, help="Eval batch size")
    parser.add_argument("--limit", type=int, default=None, help="Max examples per task")
    
    # Passing the exact tie settings used during training
    parser.add_argument("--tie-group-size", type=int, default=1)
    parser.add_argument("--tie-skip-first", type=int, default=2)
    parser.add_argument("--tie-skip-last", type=int, default=2)
    
    args = parser.parse_args()

    model_wrapper = HFMoEEvalWrapper(
        ckpt_path=args.ckpt, 
        batch_size=args.batch_size,
        tie_group_size=args.tie_group_size,
        tie_skip_first=args.tie_skip_first,
        tie_skip_last=args.tie_skip_last
    )

    print(f"\nStarting evaluation on tasks: {args.tasks} ({args.num_fewshot}-shot, bs={args.batch_size})")
    results = lm_eval.simple_evaluate(
        model=model_wrapper,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size, 
        limit=args.limit,
    )
    
    print("\n" + "="*80)
    print(lm_eval.utils.make_table(results))
    print("="*80)
    
    # ── Calculate Average Accuracy ──
    acc_values = []
    if "results" in results:
        for task_name, task_metrics in results["results"].items():
            # lm-eval uses different key suffixes depending on the version/task
            acc = task_metrics.get("acc,none") if "acc,none" in task_metrics else task_metrics.get("acc")
            if acc is not None:
                acc_values.append(acc)
                
    if acc_values:
        avg_acc = sum(acc_values) / len(acc_values)
        # Multiply by 100 to display as a percentage for readability
        print(f"\n🚀 Checkpoint: {args.ckpt}  |  Average Accuracy: {avg_acc * 100:.2f}% \n")
    else:
        print(f"\n⚠️ Could not find standard 'acc' metrics to average  |  Checkpoint: {args.ckpt}\n")
