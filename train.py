import argparse
import glob
import json
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
        self.B, self.T, self.shuffle = B, T, shuffle
        self.rng = random.Random(seed)

        self.token_paths = sorted(glob.glob(os.path.join(root, split, f"shard_{split}_*_tokens.npy")))
        self.mask_paths = sorted(glob.glob(os.path.join(root, split, f"shard_{split}_*_mask.npy")))

        if not self.token_paths:
            raise ValueError(f"No token shards found in {root}/{split}")
        if len(self.token_paths) != len(self.mask_paths):
            raise ValueError(f"Token/mask shard count mismatch in {root}/{split}")

        self.order = list(range(len(self.token_paths)))
        if self.shuffle:
            self.rng.shuffle(self.order)

        self.order_pos = 0
        self.pos = 0
        self.epoch = 0
        self._total_tokens = None
        self._load()

    def _load(self):
        i = self.order[self.order_pos]
        self.tokens = np.load(self.token_paths[i], mmap_mode="r")
        self.mask = np.load(self.mask_paths[i], mmap_mode="r")
        self.pos = 0

    def _advance(self):
        self.order_pos += 1

        if self.order_pos >= len(self.order):
            self.order_pos = 0
            self.epoch += 1
            if self.shuffle:
                self.rng.shuffle(self.order)

        self._load()

    def next_batch(self):
        n = self.B * self.T

        while self.pos + n > len(self.tokens):
            self._advance()

        tok = np.asarray(
            self.tokens[self.pos:self.pos + n],
            dtype=np.int64,
        ).copy()

        mask = np.asarray(
            self.mask[self.pos:self.pos + n],
            dtype=np.bool_,
        ).copy()

        self.pos += n

        x = torch.from_numpy(tok).view(self.B, self.T)
        m = torch.from_numpy(mask).view(self.B, self.T)

        # HF CausalLM convention: labels are aligned with input_ids.
        # RicottaForCausalLM performs the causal shift internally.
        labels = torch.where(
            m,
            x,
            torch.full_like(x, -100),
        )

        return x, labels

    def reset(self):
        self.order = list(range(len(self.token_paths)))
        self.order_pos = 0
        self.pos = 0
        self.epoch = 0
        self._load()

    @property
    def total_tokens(self):
        if self._total_tokens is None:
            self._total_tokens = sum(
                np.load(path, mmap_mode="r").shape[0]
                for path in self.token_paths
            )
        return self._total_tokens

    @property
    def current_shard(self):
        return self.order[self.order_pos]

    @property
    def shard_progress(self):
        return self.pos / len(self.tokens)


def get_lr(step, max_steps, max_lr, warmup_fraction, decay_fraction, min_lr_ratio):
    warmup_steps = max(1, int(max_steps * warmup_fraction))
    decay_steps = max(1, int(max_steps * decay_fraction))
    decay_start = max_steps - decay_steps

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step < decay_start:
        return max_lr

    ratio = (step - decay_start) / max(1, decay_steps - 1)
    ratio = min(max(ratio, 0.0), 1.0)

    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return max_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * coeff)


def make_optimizers(model, muon_lr, adamw_lr, weight_decay):
    muon_suffixes = (
        "attn.qkv_proj.weight",
        "attn.o_proj.weight",
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
    )

    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 2 and any(name.endswith(suffix) for suffix in muon_suffixes):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError(
            "torch.optim.Muon is unavailable in this PyTorch build. "
            "Use the same modern PyTorch environment as Gouda Core v2 pretraining."
        )

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

    print(f"Muon parameters:  {sum(p.numel() for p in muon_params):,}")
    print(f"AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    return muon, adamw


@torch.no_grad()
def validate(model, loader, steps, device):
    model.eval()
    loader.reset()

    total_loss = 0.0

    for _ in range(steps):
        x, labels = loader.next_batch()

        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        valid = (labels != -100).sum().item()
        if valid == 0:
            raise RuntimeError("Validation batch contains no supervised tokens")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=x, labels=labels)

        if not torch.isfinite(out.loss):
            raise RuntimeError("Non-finite validation loss")

        total_loss += out.loss.float().item()

    model.train()
    return total_loss / steps


def save_checkpoint(model, tokenizer, output_dir, source_model, step, cumulative_tokens):
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(output_dir)

    for filename in ("modeling_gruyere.py", "configuration_gruyere.py"):
        src = os.path.join(source_model, filename)
        dst = os.path.join(output_dir, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)

    state = {
        "step": step,
        "cumulative_tokens": cumulative_tokens,
    }

    torch.save(
        state,
        os.path.join(output_dir, "trainer_state.pt"),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="models/ricotta-2.0-PRETRAINED")
    parser.add_argument("--data", default="data/ultrachat")
    parser.add_argument("--out", default="runs/ricotta-2.0/r2")

    parser.add_argument("--epochs", type=float, default=0.25)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--batch_tokens", type=int, default=131072)

    parser.add_argument("--muon_lr", type=float, default=0.5e-3)
    parser.add_argument("--adamw_lr", type=float, default=1.5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--warmup_fraction", type=float, default=0.03)
    parser.add_argument("--decay_fraction", type=float, default=0.20)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)

    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--val_every", type=int, default=150)
    parser.add_argument("--val_steps", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_compile", action="store_true")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This training configuration requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.set_float32_matmul_precision("high")

    device = "cuda"

    micro_batch_tokens = args.B * args.T

    if args.batch_tokens % micro_batch_tokens != 0:
        raise ValueError(
            f"batch_tokens ({args.batch_tokens}) must be divisible by "
            f"B*T ({micro_batch_tokens})"
        )

    grad_accum_steps = args.batch_tokens // micro_batch_tokens

    train_loader = ShardLoader(
        args.data,
        "train",
        args.B,
        args.T,
        shuffle=True,
        seed=args.seed,
    )

    val_loader = ShardLoader(
        args.data,
        "val",
        args.B,
        args.T,
        shuffle=False,
        seed=args.seed,
    )

    target_tokens = int(train_loader.total_tokens * args.epochs)
    max_steps = math.ceil(target_tokens / args.batch_tokens)

    print(f"device: {device}")
    print(f"train tokens: {train_loader.total_tokens:,}")
    print(f"target tokens: {target_tokens:,}")
    print(f"micro batch: {args.B} x {args.T} = {micro_batch_tokens:,} tokens")
    print(f"gradient accumulation: {grad_accum_steps}")
    print(f"effective batch: {args.batch_tokens:,} tokens")
    print(f"optimizer steps: {max_steps:,}")

    print(f"loading {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    raw_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.float32,
    ).to(device)

    raw_model.train()

    print(f"parameters: {raw_model.num_parameters():,}")

    muon, adamw = make_optimizers(
        raw_model,
        args.muon_lr,
        args.adamw_lr,
        args.weight_decay,
    )

    train_model = raw_model

    if not args.no_compile:
        print("compiling model...")
        train_model = torch.compile(raw_model)
        print("torch.compile enabled")

    os.makedirs(args.out, exist_ok=True)

    log_path = os.path.join(args.out, "train.log")

    with open(log_path, "w") as f:
        f.write(
            f"# Gruyere-2.0 SFT | "
            f"params={raw_model.num_parameters()} | "
            f"train_tokens={train_loader.total_tokens} | "
            f"target_tokens={target_tokens} | "
            f"steps={max_steps} | "
            f"batch_tokens={args.batch_tokens}\n"
        )

    cumulative_tokens = 0

    for step in range(max_steps):
        t0 = time.time()

        adamw.zero_grad(set_to_none=True)
        muon.zero_grad(set_to_none=True)

        loss_accum = torch.zeros((), device=device)
        completed_micro_steps = 0

        for micro_step in range(grad_accum_steps):
            x, labels = train_loader.next_batch()

            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            valid_labels = (labels != -100).sum().item()

            if valid_labels == 0:
                continue

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                out = train_model(
                    input_ids=x,
                    labels=labels,
                )

                if not torch.isfinite(out.loss):
                    logits_finite = torch.isfinite(out.logits).all().item()

                    print("\nNON-FINITE LOSS")
                    print(f"step: {step + 1}")
                    print(f"micro-step: {micro_step + 1}/{grad_accum_steps}")
                    print(f"valid labels: {valid_labels:,}")
                    print(f"input range: {x.min().item()}..{x.max().item()}")
                    print(f"logits finite: {logits_finite}")
                    print(f"loss: {out.loss.item()}")

                    raise RuntimeError("Non-finite training loss")

                loss = out.loss / grad_accum_steps

            loss_accum += loss.detach()
            loss.backward()
            completed_micro_steps += 1

        norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(),
            args.grad_clip,
        )

        if not torch.isfinite(norm):
            raise RuntimeError(
                f"Non-finite gradient norm at step {step + 1}: {norm.item()}"
            )

        adamw_lr = get_lr(
            step,
            max_steps,
            args.adamw_lr,
            args.warmup_fraction,
            args.decay_fraction,
            args.min_lr_ratio,
        )

        muon_lr = get_lr(
            step,
            max_steps,
            args.muon_lr,
            args.warmup_fraction,
            args.decay_fraction,
            args.min_lr_ratio,
        )

        for group in adamw.param_groups:
            group["lr"] = adamw_lr

        for group in muon.param_groups:
            group["lr"] = muon_lr

        adamw.step()
        muon.step()

        cumulative_tokens = min(
            (step + 1) * args.batch_tokens,
            target_tokens,
        )

        epoch = cumulative_tokens / train_loader.total_tokens
        shard_progress = train_loader.shard_progress

        torch.cuda.synchronize()

        dt = time.time() - t0
        tokens_per_sec = args.batch_tokens / dt
        vram = torch.cuda.max_memory_allocated() / 1e9

        log_line = (
            f"step {step + 1:6d}/{max_steps:<6d} | "
            f"epoch {epoch:7.4f}/{args.epochs:.4f} | "
            f"shard {train_loader.current_shard:4d} | "
            f"shard_progress {shard_progress * 100:6.2f}% | "
            f"tokens {cumulative_tokens / 1e9:8.4f}B | "
            f"micro {completed_micro_steps}/{grad_accum_steps} | "
            f"loss {loss_accum.item():.6f} | "
            f"lr {adamw_lr:.4e} | "
            f"muon_lr {muon_lr:.4e} | "
            f"norm {norm.item():.4f} | "
            f"tok/sec {tokens_per_sec:8.0f} | "
            f"dt {dt * 1000:7.1f}ms | "
            f"vram {vram:.2f}GB"
        )

        print(log_line)

        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        if (step + 1) % args.val_every == 0 or step + 1 == max_steps:
            val_loss = validate(
                raw_model,
                val_loader,
                args.val_steps,
                device,
            )

            val_line = (
                f"step {step + 1:6d}/{max_steps:<6d} | "
                f"val_loss {val_loss:.6f} | "
                f"tokens {cumulative_tokens / 1e9:8.4f}B"
            )

            print(val_line)

            with open(log_path, "a") as f:
                f.write(val_line + "\n")

        if (step + 1) % args.save_every == 0:
            checkpoint_dir = os.path.join(
                args.out,
                f"step_{step + 1:06d}",
            )

            save_checkpoint(
                raw_model,
                tokenizer,
                checkpoint_dir,
                args.model,
                step + 1,
                cumulative_tokens,
            )

            print(f"saved {checkpoint_dir}")

    final_dir = os.path.join(args.out, "final")

    save_checkpoint(
        raw_model,
        tokenizer,
        final_dir,
        args.model,
        max_steps,
        cumulative_tokens,
    )

    training_info = {
        "base_model": args.model,
        "dataset": args.data,
        "epochs": args.epochs,
        "sequence_length": args.T,
        "micro_batch_size": args.B,
        "batch_tokens": args.batch_tokens,
        "gradient_accumulation_steps": grad_accum_steps,
        "optimizer_steps": max_steps,
        "adamw_lr": args.adamw_lr,
        "muon_lr": args.muon_lr,
        "weight_decay": args.weight_decay,
        "warmup_fraction": args.warmup_fraction,
        "decay_fraction": args.decay_fraction,
        "min_lr_ratio": args.min_lr_ratio,
        "grad_clip": args.grad_clip,
        "tokens_trained": cumulative_tokens,
    }

    with open(
        os.path.join(final_dir, "sft_config.json"),
        "w",
    ) as f:
        json.dump(training_info, f, indent=2)

    print(f"training complete: {final_dir}")


if __name__ == "__main__":
    main()