import argparse
import glob
import math
import os
import random
import shutil
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ShardLoader:
    def __init__(self, root, split, B, T, shuffle=True, seed=42):
        self.B, self.T, self.shuffle, self.rng = B, T, shuffle, random.Random(seed)
        self.token_paths = sorted(glob.glob(os.path.join(root, split, f"shard_{split}_*_tokens.npy")))
        self.mask_paths = sorted(glob.glob(os.path.join(root, split, f"shard_{split}_*_mask.npy")))
        if not self.token_paths or len(self.token_paths) != len(self.mask_paths):
            raise ValueError(f"Invalid or missing shards in {root}/{split}")
        self.order = list(range(len(self.token_paths)))
        self.epoch, self.order_pos, self.pos = 0, 0, 0
        if shuffle:
            self.rng.shuffle(self.order)
        self._load()

    def _load(self):
        i = self.order[self.order_pos]
        self.tokens = np.load(self.token_paths[i], mmap_mode="r")
        self.mask = np.load(self.mask_paths[i], mmap_mode="r")
        self.pos = 0

    def _advance(self):
        self.order_pos += 1
        if self.order_pos == len(self.order):
            self.order_pos = 0
            self.epoch += 1
            if self.shuffle:
                self.rng.shuffle(self.order)
        self._load()

    def next_batch(self):
        n = self.B * self.T + 1
        while self.pos + n > len(self.tokens):
            self._advance()

        tok = np.asarray(self.tokens[self.pos:self.pos + n], dtype=np.int64)
        mask = np.asarray(self.mask[self.pos:self.pos + n], dtype=np.bool_)
        self.pos += self.B * self.T

        x = torch.from_numpy(tok[:-1].copy()).view(self.B, self.T)
        y = torch.from_numpy(tok[1:].copy()).view(self.B, self.T)
        m = torch.from_numpy(mask[1:].copy()).view(self.B, self.T)
        return x, torch.where(m, y, -100)

    def reset(self):
        self.order = list(range(len(self.token_paths)))
        self.order_pos = self.epoch = self.pos = 0
        self._load()

    @property
    def total_tokens(self):
        if not hasattr(self, "_total_tokens"):
            self._total_tokens = sum(np.load(p, mmap_mode="r").shape[0] for p in self.token_paths)
        return self._total_tokens


def get_lr(step, max_steps, max_lr, warmup_fraction=0.03, decay_fraction=0.20, min_ratio=0.05):
    warmup = max(1, int(max_steps * warmup_fraction))
    decay_start = max(warmup, int(max_steps * (1.0 - decay_fraction)))

    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step < decay_start:
        return max_lr

    ratio = (step - decay_start) / max(1, max_steps - decay_start - 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(max(ratio, 0.0), 1.0)))
    return max_lr * (min_ratio + (1.0 - min_ratio) * coeff)


def make_optimizers(model, muon_lr, adamw_lr, weight_decay):
    muon_params, adamw_params = [], []

    muon_suffixes = (
        "attn.qkv_proj.weight",
        "attn.o_proj.weight",
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
    )

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and any(name.endswith(s) for s in muon_suffixes):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("This training script requires a PyTorch build containing torch.optim.Muon")

    muon = torch.optim.Muon(
        muon_params,
        lr=muon_lr,
        momentum=0.95,
        weight_decay=weight_decay,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
    )

    adamw = torch.optim.AdamW(
        adamw_params,
        lr=adamw_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=weight_decay,
        fused=torch.cuda.is_available(),
    )

    print(f"Muon: {sum(p.numel() for p in muon_params):,} params")
    print(f"AdamW: {sum(p.numel() for p in adamw_params):,} params")
    return muon, adamw


@torch.no_grad()
def validate(model, loader, steps, device):
    model.eval()
    loader.reset()
    loss = 0.0

    for _ in range(steps):
        x, y = loader.next_batch()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss += model(input_ids=x, labels=y).loss.float().item()

    model.train()
    return loss / steps


def save_model(model, tokenizer, out_dir, source_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    for name in ("modeling_ricotta.py", "configuration_ricotta.py"):
        src = os.path.join(source_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/ricotta-2.0-PRETRAINED")
    p.add_argument("--data", default="data/ultrachat")
    p.add_argument("--out", default="runs/ricotta-2.0")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--batch_tokens", type=int, default=262144)
    p.add_argument("--muon_lr", type=float, default=0.005)
    p.add_argument("--adamw_lr", type=float, default=0.00015)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.03)
    p.add_argument("--decay_fraction", type=float, default=0.20)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--val_steps", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_compile", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    device = "cuda"
    if args.batch_tokens % (args.B * args.T):
        raise ValueError("batch_tokens must be divisible by B*T")

    grad_accum = args.batch_tokens // (args.B * args.T)
    train_loader = ShardLoader(args.data, "train", args.B, args.T, True, args.seed)
    val_loader = ShardLoader(args.data, "val", args.B, args.T, False, args.seed)

    target_tokens = int(train_loader.total_tokens * args.epochs)
    max_steps = math.ceil(target_tokens / args.batch_tokens)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    raw_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)

    raw_model.train()
    train_model = raw_model if args.no_compile else torch.compile(raw_model)

    muon, adamw = make_optimizers(
        raw_model,
        args.muon_lr,
        args.adamw_lr,
        args.weight_decay,
    )

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train.log")

    print(
        f"Ricotta-2.0 SFT | params={raw_model.num_parameters():,} | "
        f"tokens={target_tokens:,} | steps={max_steps:,} | "
        f"B={args.B} T={args.T} accum={grad_accum}"
    )

    with open(log_path, "w") as f:
        f.write(
            f"# Ricotta-2.0 SFT | params={raw_model.num_parameters()} | "
            f"steps={max_steps} | batch_tokens={args.batch_tokens}\n"
        )

    cumulative_tokens = 0

    for step in range(max_steps):
        t0 = time.time()

        adamw.zero_grad(set_to_none=True)
        muon.zero_grad(set_to_none=True)

        loss_accum = torch.zeros((), device=device)

        for _ in range(grad_accum):
            x, y = train_loader.next_batch()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = train_model(input_ids=x, labels=y)
                loss = out.loss / grad_accum

            loss_accum += loss.detach()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

        adamw_lr = get_lr(
            step, max_steps, args.adamw_lr,
            args.warmup_fraction, args.decay_fraction, args.min_lr_ratio,
        )
        muon_lr = get_lr(
            step, max_steps, args.muon_lr,
            args.warmup_fraction, args.decay_fraction, args.min_lr_ratio,
        )

        for group in adamw.param_groups:
            group["lr"] = adamw_lr
        for group in muon.param_groups:
            group["lr"] = muon_lr

        adamw.step()
        muon.step()

        cumulative_tokens += args.batch_tokens

        torch.cuda.synchronize()
        dt = time.time() - t0
        tok_sec = args.batch_tokens / dt
        vram = torch.cuda.max_memory_allocated() / 1e9

        n = step + 1
        print(
            f"{n:6d}/{max_steps} train "
            f"epoch={cumulative_tokens / train_loader.total_tokens:.6f} "
            f"loss={loss_accum.item():.6f} "
            f"tokens={cumulative_tokens} "
            f"lr={adamw_lr:.3e} muon_lr={muon_lr:.3e} "
            f"norm={norm:.4f} tok_sec={tok_sec:.0f} "
            f"dt={dt:.3f} vram_gb={vram:.2f}"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{n} train epoch={cumulative_tokens / train_loader.total_tokens:.8f} "
                f"loss={loss_accum.item():.8f} tokens={cumulative_tokens} "
                f"lr={adamw_lr:.8e} muon_lr={muon_lr:.8e} "
                f"norm={norm:.6f} tok_sec={tok_sec:.2f} dt={dt:.6f} "
                f"vram_gb={vram:.3f}\n"
            )

        if n % args.val_every == 0 or n == max_steps:
            val_loss = validate(raw_model, val_loader, args.val_steps, device)
            print(f"{n:6d} val loss={val_loss:.6f} tokens={cumulative_tokens}")

            with open(log_path, "a") as f:
                f.write(
                    f"{n} val loss={val_loss:.8f} "
                    f"tokens={cumulative_tokens}\n"
                )

        if n % args.save_every == 0:
            save_model(
                raw_model,
                tokenizer,
                os.path.join(args.out, f"step_{n:06d}"),
                args.model,
            )

    save_model(raw_model, tokenizer, os.path.join(args.out, "final"), args.model)
    print(f"Training complete: {os.path.join(args.out, 'final')}")


if __name__ == "__main__":
    main()