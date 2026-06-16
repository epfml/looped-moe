"""
data.py — Mixed DCLM-edu + FinePhrase streaming data pipeline
=============================================================
Streams from HuggingFace, mixes datasets by ratio, tokenizes, and safely
shards for DDP.

Tokenizer / document-separator handling
---------------------------------------
The dataset supports multiple tokenizers and separator placements via flags
on `MixedStreamDataset` / `get_dataloader` / `get_eval_batches`:

    tokenizer:
        "cl100k"   → tiktoken cl100k_base, vocab 100277. Default; preserves
                     the original behaviour of this file. Used by moe_train.py
                     for from-scratch training.
        "olmoe"    → AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125"),
                     vocab 50304. Used by surgery_moe_train.py for continued
                     pretraining of OLMoE.
        any HF id  → AutoTokenizer.from_pretrained(<that id>). The default
                     separator id will be the tokenizer's `eos_token_id`.

    insert_separator:
        "before"   → emit ONE separator token BEFORE each document. This is the
                     original behaviour of this file (matches moe_train.py runs).
        "after"    → emit ONE separator token AFTER each document. This matches
                     the pretraining recipe of OLMoE (`dolma tokens
                     --tokenizer.eos_token_id 50279`).
        "none"     → no separator between documents (rarely useful).

    separator_id:
        Optional explicit override of the separator token id. Defaults are:
            cl100k → 100257 (<|endoftext|>)
            olmoe  → 50279  (<|endoftext|>, == tok.eos_token_id)
            other  → tokenizer.eos_token_id

Important: "before" and "after" placements are functionally equivalent on a
continuous concatenated stream — both put exactly one separator between every
pair of consecutive documents. The difference is only at the very first/last
document of the stream. What matters for matching pretraining is the choice of
*token id*, not the side of the doc it sits on.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist
import tiktoken
from datasets import load_dataset, interleave_datasets


# Built-in tokenizer presets. Keys map to (loader_kind, default_separator_id).
# loader_kind is "tiktoken:<name>" for tiktoken or "hf:<model_id>" for HF tokenizers.
_TOKENIZER_PRESETS = {
    "cl100k": ("tiktoken:cl100k_base", 100257),  # cl100k <|endoftext|>
    "olmoe":  ("hf:allenai/OLMoE-1B-7B-0125", 50279),  # OLMoE <|endoftext|>
}


def _resolve_tokenizer(tokenizer_spec, separator_id_override):
    """Return (encoder_kind, encoder_obj, separator_id, vocab_size, name).

    `encoder_kind` is one of "tiktoken" or "hf". `encoder_obj` is the actual
    tokenizer (a tiktoken Encoding or an HF PreTrainedTokenizerBase), both of
    which are picklable so DataLoader workers can be spawned safely on any
    start method (fork, spawn, forkserver).
    """
    if tokenizer_spec in _TOKENIZER_PRESETS:
        loader_kind, default_sep_id = _TOKENIZER_PRESETS[tokenizer_spec]
        name = tokenizer_spec
    else:
        # Treat as a generic HF model id.
        loader_kind = f"hf:{tokenizer_spec}"
        default_sep_id = None  # filled in below from tokenizer.eos_token_id
        name = tokenizer_spec

    if loader_kind.startswith("tiktoken:"):
        enc_name = loader_kind.split(":", 1)[1]
        enc = tiktoken.get_encoding(enc_name)
        encoder_kind = "tiktoken"
        encoder_obj = enc
        vocab_size = enc.n_vocab
    elif loader_kind.startswith("hf:"):
        from transformers import AutoTokenizer
        hf_id = loader_kind.split(":", 1)[1]
        tok = AutoTokenizer.from_pretrained(hf_id)
        encoder_kind = "hf"
        encoder_obj = tok
        vocab_size = tok.vocab_size
        if default_sep_id is None:
            if tok.eos_token_id is None:
                raise ValueError(
                    f"Tokenizer '{hf_id}' has no eos_token_id configured. "
                    f"Pass an explicit separator_id=<int> to get_dataloader / "
                    f"get_eval_batches."
                )
            default_sep_id = tok.eos_token_id
    else:
        raise ValueError(f"Unknown tokenizer loader: {loader_kind}")

    sep_id = separator_id_override if separator_id_override is not None else default_sep_id
    return encoder_kind, encoder_obj, sep_id, vocab_size, name


class MixedStreamDataset(IterableDataset):
    """
    Streams DCLM-edu and FinePhrase, mixes them by ratio, tokenizes,
    and packs into fixed-length sequences.

    See module docstring for tokenizer / insert_separator / separator_id semantics.
    """

    # Backwards-compat class attribute. The default separator id for the
    # original (cl100k) tokenizer is the cl100k <|endoftext|> token id.
    # External code that referenced `MixedStreamDataset.BOS_TOKEN` continues
    # to work. New code should use `dataset.separator_id` instead.
    BOS_TOKEN = 100257

    def __init__(
        self,
        seq_len=512,
        split="train",
        max_tokens=None,
        do_shard=True,
        tokenizer="cl100k",
        insert_separator="before",
        separator_id=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.split = split
        self.max_tokens = max_tokens
        self.do_shard = do_shard

        if insert_separator not in ("before", "after", "none"):
            raise ValueError(
                f"insert_separator must be 'before', 'after', or 'none' (got {insert_separator!r})"
            )
        self.insert_separator = insert_separator

        # Resolve the tokenizer and separator id once so workers don't pay the
        # AutoTokenizer download cost per __iter__ call. The stored
        # _encoder_obj is a tiktoken Encoding or HF PreTrainedTokenizerBase —
        # both are picklable across DataLoader worker start methods.
        encoder_kind, encoder_obj, sep_id, vocab_size, tok_name = _resolve_tokenizer(
            tokenizer, separator_id
        )
        self._encoder_kind = encoder_kind
        self._encoder_obj = encoder_obj
        self.separator_id = sep_id
        self.vocab_size = vocab_size
        self.tokenizer_name = tok_name

        # Keep self.enc for full backwards-compat with anything that reached
        # in and used the old attribute name (cl100k path only; for HF it's
        # the AutoTokenizer object).
        self.enc = encoder_obj

        # Sanity: separator id must fit in the tokenizer's vocab.
        if not (0 <= self.separator_id < self.vocab_size):
            raise ValueError(
                f"separator_id={self.separator_id} is out of range for tokenizer "
                f"'{tok_name}' with vocab_size={self.vocab_size}"
            )

    def _encode(self, text):
        """Encode one document to a list of int token ids, no special tokens.
        Method form (not a stored lambda) so the dataset is picklable across
        DataLoader worker start methods (fork, spawn, forkserver alike).
        """
        if self._encoder_kind == "tiktoken":
            return self._encoder_obj.encode_ordinary(text)
        else:
            return self._encoder_obj.encode(text, add_special_tokens=False)

    def _token_generator(self):
        # 1. Load the raw streams
        ds_dclm = load_dataset("HuggingFaceTB/dclm-edu", split=self.split, streaming=True)
        ds_fp = load_dataset("HuggingFaceFW/finephrase", "all", split=self.split, streaming=True)

        # 2. Add Shuffle Buffers (Crucial for mixed streaming)
        # This keeps 10,000 documents in RAM per dataset and yields randomly from them.
        ds_dclm = ds_dclm.shuffle(buffer_size=10_000, seed=42)
        ds_fp = ds_fp.shuffle(buffer_size=10_000, seed=42)

        # 3. Mix the datasets
        # Update these probabilities to match the true original size ratio you want.
        # Example: 75% DCLM-edu, 25% FinePhrase.
        ds_mixed = interleave_datasets(
            [ds_dclm, ds_fp],
            probabilities=[0.75, 0.25],
            seed=42
        )

        # 4. Perfect sharding for DDP and multi-worker DataLoaders
        # NOTE: Sharding must happen AFTER interleaving to prevent DDP hangs.
        if self.do_shard:
            worker_info = torch.utils.data.get_worker_info()
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            global_rank = dist.get_rank() if dist.is_initialized() else 0

            worker_id = worker_info.id if worker_info else 0
            num_workers = worker_info.num_workers if worker_info else 1

            total_shards = world_size * num_workers
            shard_idx = global_rank * num_workers + worker_id

            if total_shards > 1:
                ds_mixed = ds_mixed.shard(num_shards=total_shards, index=shard_idx)

        # 5. Token Yielding Loop
        # The insert_separator flag controls whether the separator token is emitted
        # before, after, or not at all relative to each document's tokens.
        sep_id = self.separator_id
        emit_before = self.insert_separator == "before"
        emit_after = self.insert_separator == "after"

        total_tokens = 0
        for example in ds_mixed:
            text = example.get("text", "")
            if not text or len(text) < 50:
                continue

            if emit_before:
                yield sep_id
                total_tokens += 1
                if self.max_tokens and total_tokens >= self.max_tokens:
                    return

            tokens = self._encode(text)
            for t in tokens:
                yield t
                total_tokens += 1
                if self.max_tokens and total_tokens >= self.max_tokens:
                    return

            if emit_after:
                yield sep_id
                total_tokens += 1
                if self.max_tokens and total_tokens >= self.max_tokens:
                    return

    def __iter__(self):
        """Yield (input_ids, targets) of shape (seq_len,)."""
        buffer = []
        for token in self._token_generator():
            buffer.append(token)
            if len(buffer) == self.seq_len + 1:
                t = torch.tensor(buffer, dtype=torch.long)
                yield t[:-1], t[1:]
                buffer = []


def get_dataloader(
    seq_len=512,
    batch_size=32,
    num_workers=4,
    max_tokens=None,
    tokenizer="cl100k",
    insert_separator="before",
    separator_id=None,
):
    """Create a streaming DataLoader for the Mixed Dataset.

    See module docstring for tokenizer / insert_separator / separator_id semantics.
    Defaults preserve the original behaviour (cl100k + leading <|endoftext|>).
    """
    dataset = MixedStreamDataset(
        seq_len=seq_len,
        max_tokens=max_tokens,
        do_shard=True,
        tokenizer=tokenizer,
        insert_separator=insert_separator,
        separator_id=separator_id,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_eval_batches(
    seq_len=512,
    batch_size=32,
    n_batches=20,
    tokenizer="cl100k",
    insert_separator="before",
    separator_id=None,
):
    """
    Pre-fetch a fixed set of eval batches from the stream.
    do_shard=False ensures every GPU evaluates on the exact same holdout set.

    See module docstring for tokenizer / insert_separator / separator_id semantics.
    The same values must be passed here as in get_dataloader so that training
    and evaluation see the same document-boundary distribution.
    """
    dataset = MixedStreamDataset(
        seq_len=seq_len,
        max_tokens=10_000_000 + n_batches * batch_size * (seq_len + 1),
        do_shard=False,
        tokenizer=tokenizer,
        insert_separator=insert_separator,
        separator_id=separator_id,
    )

    batches = []
    buffer = []
    skip_count = 0
    skip_target = 10_000_000

    for inp, tgt in dataset:
        skip_count += seq_len
        if skip_count < skip_target:
            continue
        buffer.append((inp, tgt))
        if len(buffer) == batch_size:
            input_ids = torch.stack([b[0] for b in buffer])
            targets = torch.stack([b[1] for b in buffer])
            batches.append((input_ids, targets))
            buffer = []
            if len(batches) >= n_batches:
                break

    return batches
