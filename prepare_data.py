import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer


_TOKENIZER = None
EOT_ID = None
USER_PREFIX = None
ASSISTANT_PREFIX = None


def init_worker(tokenizer_path, eot_id):
    global _TOKENIZER, EOT_ID, USER_PREFIX, ASSISTANT_PREFIX

    _TOKENIZER = Tokenizer.from_file(tokenizer_path)
    EOT_ID = eot_id
    USER_PREFIX = _TOKENIZER.encode("User:\n", add_special_tokens=False,).ids
    ASSISTANT_PREFIX = _TOKENIZER.encode("Assistant:\n", add_special_tokens=False,).ids

def encode(text):
    return _TOKENIZER.encode(text, add_special_tokens=False,).ids

def tokenize_example(messages):
    tokens, mask = [], []

    for turn in messages:
        role = turn["role"]
        content = turn["content"].strip()

        if role == "user":
            prefix = USER_PREFIX
            score = False
        elif role == "assistant":
            prefix = ASSISTANT_PREFIX
            score = True
        else:
            continue

        content_ids = encode(content)
        tokens.extend(prefix)
        mask.extend([0] * len(prefix))
        tokens.extend(content_ids)
        mask.extend([int(score)] * len(content_ids))

        tokens.append(EOT_ID)
        mask.append(int(score))

    return tokens, mask


class ShardWriter:
    def __init__(self, out_dir, split, shard_size):
        self.dir = os.path.join(out_dir, split)
        os.makedirs(self.dir, exist_ok=True)

        self.split = split
        self.shard_size = shard_size
        self.shard_idx = 0

        self.tokens = np.empty(shard_size, dtype=np.uint16)
        self.mask = np.empty(shard_size, dtype=np.uint8)

        self.pos = 0
        self.total_tokens = 0
        self.total_loss_tokens = 0

    def add(self, tokens, mask):
        tokens = np.asarray(tokens, dtype=np.uint16)
        mask = np.asarray(mask, dtype=np.uint8)

        i = 0
        n = len(tokens)

        while i < n:
            take = min(self.shard_size - self.pos, n - i)

            self.tokens[self.pos:self.pos + take] = tokens[i:i + take]
            self.mask[self.pos:self.pos + take] = mask[i:i + take]

            self.pos += take
            i += take

            if self.pos == self.shard_size:
                self.flush()

        self.total_tokens += n
        self.total_loss_tokens += int(mask.sum())

    def flush(self):
        if not self.pos:
            return

        prefix = os.path.join(self.dir, f"shard_{self.split}_{self.shard_idx:05d}",)
        np.save(f"{prefix}_tokens.npy", self.tokens[:self.pos],)
        np.save(f"{prefix}_mask.npy", self.mask[:self.pos],)
        print(f"wrote {prefix} " f"({self.pos:,} tokens)")

        self.shard_idx += 1
        self.pos = 0

    def close(self):
        self.flush()


def process_split(dataset, split, out_dir, shard_size, workers, tokenizer_path, eot_id,):
    writer = ShardWriter(out_dir, split, shard_size,)
    t0 = time.time()

    with mp.Pool(workers, initializer=init_worker, initargs=(tokenizer_path, eot_id),) as pool:
        iterator = pool.imap(tokenize_example, dataset["messages"], chunksize=64,)
        for i, (tokens, mask) in enumerate(iterator, 1):
            if tokens:
                writer.add(tokens, mask)

            if i % 5000 == 0:
                dt = time.time() - t0

                print(
                    f"[{split}] "
                    f"{i:,}/{len(dataset):,} conversations | "
                    f"{writer.total_tokens:,} tokens | "
                    f"{i / dt:.0f} conv/s"
                )

    writer.close()
    pct = (100 * writer.total_loss_tokens / max(1, writer.total_tokens))
    print(
        f"[{split}] {writer.total_tokens:,} tokens | "
        f"{writer.total_loss_tokens:,} loss tokens "
        f"({pct:.1f}%) | "
        f"{writer.shard_idx} shards"
    )
    return (writer.total_tokens, writer.total_loss_tokens,)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ricotta-2.0-PRETRAINED",)
    p.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k",)
    p.add_argument("--out", default="data/ultrachat",)
    p.add_argument("--shard_size", type=int, default=50_000_000,)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),)
    p.add_argument("--limit", type=int, default=None,)
    args = p.parse_args()

    tokenizer_path = os.path.join(args.model, "tokenizer.json",)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eot_id = tokenizer.token_to_id("<|endoftext|>")

    if eot_id is None:
        raise ValueError(
            "Tokenizer has no <|endoftext|> token"
        )

    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65536:
        raise ValueError(
            "Vocabulary no longer fits uint16"
        )

    print(f"Tokenizer: {args.model}")
    print(f"Vocab: {vocab_size:,}")
    print(f"EOT: {eot_id}")
    print(f"Workers: {args.workers}")

    train = load_dataset(args.dataset, split="train_sft",)
    val = load_dataset(args.dataset, split="test_sft",)
    if args.limit:
        train = train.select(range(min(args.limit, len(train))))
        val = val.select(range(min(args.limit, len(val))))

    print(f"train: {len(train):,} conversations | " f"val: {len(val):,}")
    train_tokens, train_loss = process_split(train, "train", args.out, args.shard_size, args.workers, tokenizer_path, eot_id,)
    val_tokens, val_loss = process_split(val, "val", args.out, args.shard_size, args.workers, tokenizer_path, eot_id,)

    meta = {
        "dataset": args.dataset,
        "tokenizer": args.model,
        "vocab_size": vocab_size,
        "eot_token_id": eot_id,
        "format": "User:\\n{text}\\nAssistant:\\n{text}<|endoftext|>",
        "train_tokens": train_tokens,
        "train_loss_tokens": train_loss,
        "val_tokens": val_tokens,
        "val_loss_tokens": val_loss,
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "meta.json"), "w",) as f:
        json.dump(meta, f, indent=2)

    print("\nSummary")
    print(f"train tokens:      {train_tokens:,}")
    print(f"train loss tokens: {train_loss:,}")
    print(f"val tokens:        {val_tokens:,}")
    print(f"val loss tokens:   {val_loss:,}")


if __name__ == "__main__":
    main()