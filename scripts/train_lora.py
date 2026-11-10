"""LoRA fine-tuning for the TronOS command parser (PyTorch + Transformers + PEFT).

Loss is computed on the assistant (JSON) tokens only; system/user tokens are masked.

Example:
  python scripts/train_lora.py --config configs/lora_D_attn_mlp.json
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import build_messages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(p):
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def encode(tok, text, target, max_len, prompt="compact"):
    msgs = build_messages(text, prompt)
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": target}], tokenize=False)
    # Drop the trailing newline after the end-of-turn token so the model learns to stop at <|im_end|>.
    full = full.rstrip("\n")
    assert full.startswith(prompt), "chat template is not prefix-stable"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert f_ids[: len(p_ids)] == p_ids
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return f_ids[:max_len], labels[:max_len]


def collate(batch, pad_id, fixed_len=None):
    # Static shapes matter on Apple MPS: every new (batch, seq_len) shape triggers kernel
    # re-compilation (observed ~30 s/step with dynamic padding vs ~2 s/step with fixed padding).
    L = fixed_len or max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, : len(a)] = torch.tensor(a)
        lab[i, : len(b)] = torch.tensor(b)
        att[i, : len(a)] = 1
    return ids, att, lab


@torch.no_grad()
def eval_loss(model, data, bs, pad_id, device, fixed_len=None):
    model.eval()
    tot, ntok = 0.0, 0
    for i in range(0, len(data), bs):
        ids, att, lab = collate(data[i : i + bs], pad_id, fixed_len)
        ids, att, lab = ids.to(device), att.to(device), lab.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        sl, ll = logits[:, :-1], lab[:, 1:]
        loss = torch.nn.functional.cross_entropy(sl.reshape(-1, sl.size(-1)), ll.reshape(-1),
                                                 ignore_index=-100, reduction="sum")
        tot += loss.item()
        ntok += (ll != -100).sum().item()
    model.train()
    return tot / ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[cfg.get("dtype", "float32")]
    base = ROOT / cfg["base_model"] if (ROOT / cfg["base_model"]).exists() else cfg.get("base_model_hf", cfg["base_model"])
    tok = AutoTokenizer.from_pretrained(base)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    lcfg = LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                      target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable={trainable:,} total(base)={total_params:,} ({100*trainable/total_params:.3f}%)", flush=True)

    P = cfg.get("prompt", "compact")
    train = [encode(tok, r["text"], r["target"], cfg["max_len"], P) for r in load_jsonl(ROOT / cfg["train_file"])]
    val = [encode(tok, r["text"], r["target"], cfg["max_len"], P) for r in load_jsonl(ROOT / cfg["val_file"])]
    # fixed padding only helps on MPS (avoids kernel recompiles); on CUDA pad per batch
    FIX = cfg["max_len"] if device == "mps" else None
    lens = [len(x[0]) for x in train]
    print(f"train={len(train)} val={len(val)} seq_len mean={sum(lens)/len(lens):.1f} max={max(lens)}", flush=True)

    mb, accum = cfg["micro_batch_size"], cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(train) / (mb * accum))
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(cfg.get("warmup_ratio", 0.05) * total_steps))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"],
                            weight_decay=cfg.get("weight_decay", 0.0), betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else
        0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))

    log = {"config": cfg, "device": device, "torch": torch.__version__, "machine": platform.machine(),
           "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
           if platform.system() == "Darwin" else platform.processor(),
           "trainable_params": trainable, "total_params_base": total_params,
           "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "history": []}
    val0 = eval_loss(model, val, cfg.get("eval_batch", 8), pad_id, device, FIX)
    print(f"initial val_loss={val0:.4f}", flush=True)
    log["val_loss_before"] = val0

    model.train()
    step = 0
    gnorms = []
    val_every = cfg.get("val_every", 25)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    order = list(range(len(train)))
    for ep in range(cfg["epochs"]):
        random.shuffle(order)
        # bucket by length inside chunks to reduce padding
        run_loss, run_tok = 0.0, 0
        for bi in range(0, len(order), mb * accum):
            idxs = order[bi : bi + mb * accum]
            n_tok_total = sum(sum(1 for l in train[i][1] if l != -100) - 1 for i in idxs)
            for mi in range(0, len(idxs), mb):
                ids, att, lab = collate([train[i] for i in idxs[mi : mi + mb]], pad_id, FIX)
                ids, att, lab = ids.to(device), att.to(device), lab.to(device)
                logits = model(input_ids=ids, attention_mask=att).logits.float()
                sl, ll = logits[:, :-1], lab[:, 1:]
                loss_sum = torch.nn.functional.cross_entropy(sl.reshape(-1, sl.size(-1)), ll.reshape(-1),
                                                             ignore_index=-100, reduction="sum")
                (loss_sum / max(1, n_tok_total)).backward()
                run_loss += loss_sum.item()
                run_tok += (ll != -100).sum().item()
            gnorm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0))
            gnorms.append(gnorm)  # pre-clip total L2 norm of LoRA gradients
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.get("log_every", 10) == 0:
                tl = run_loss / run_tok
                el = time.time() - t0; eta = el / step * (total_steps - step)
                print(f"ep {ep+1} step {step}/{total_steps} ({100*step/total_steps:.0f}%) loss={tl:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} elapsed={el/60:.1f}m eta={eta/60:.1f}m", flush=True)
                gw = gnorms[-cfg.get("log_every", 10):]
                log["history"].append({"step": step, "epoch": ep + 1, "train_loss": tl, "lr": sched.get_last_lr()[0],
                                       "grad_norm_mean": sum(gw) / len(gw), "grad_norm_max": max(gw),
                                       "elapsed_s": time.time() - t0})
                print(f"   grad_norm mean={sum(gw)/len(gw):.3f} max={max(gw):.3f}", flush=True)
                run_loss, run_tok = 0.0, 0
            if val_every and step % val_every == 0:
                vls = eval_loss(model, val, cfg.get("eval_batch", 8), pad_id, device, FIX)
                log["history"].append({"step": step, "epoch": ep + 1, "val_loss_step": vls, "elapsed_s": time.time() - t0})
                print(f"   val_loss@{step}={vls:.4f}", flush=True)
        vl = eval_loss(model, val, cfg.get("eval_batch", 8), pad_id, device, FIX)
        print(f"== epoch {ep+1} val_loss={vl:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
        log["history"].append({"step": step, "epoch": ep + 1, "val_loss": vl, "elapsed_s": time.time() - t0})
    train_time = time.time() - t0
    final_train_loss = eval_loss(model, train, cfg.get("eval_batch", 8), pad_id, device, FIX)
    final_val_loss = log["history"][-1]["val_loss"]

    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    adapter_file = out / "adapter" / "adapter_model.safetensors"
    log.update({"peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else None,
                "grad_norm_overall_max": max(gnorms) if gnorms else None,
                "train_time_s": train_time, "final_train_loss_full_pass": final_train_loss,
                "final_train_loss_last_window": next(h["train_loss"] for h in reversed(log["history"]) if "train_loss" in h),
                "final_val_loss": final_val_loss, "adapter_bytes": adapter_file.stat().st_size})
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: log[k] for k in ("trainable_params", "train_time_s", "final_train_loss_full_pass",
                                          "final_val_loss", "adapter_bytes")}, indent=2))


if __name__ == "__main__":
    main()
