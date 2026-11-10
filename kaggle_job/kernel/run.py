# TronOS / Granite-4.0-H-350M LoRA experiments A-D on Kaggle GPU (2 GPUs in parallel).
import json, os, shutil, subprocess, sys, time
os.environ["PYTHONUNBUFFERED"] = "1"
# Kaggle's input mount layout varies (/kaggle/input/<slug> or deeper) -> search for our scripts folder
SRC = next(r for r, d, f in os.walk("/kaggle/input") if os.path.exists(os.path.join(r, "scripts", "train_lora.py")))
print("dataset found at", SRC, flush=True)
DST = "/kaggle/working/tronx"
shutil.copytree(SRC, DST, dirs_exist_ok=True)
os.chdir(DST); os.makedirs("logs", exist_ok=True); os.makedirs("results", exist_ok=True)
# T4 has 15 GB; Mamba2 PyTorch fallback needs ~6-7 GB per 512-token sample -> micro-batch 1, accumulate 16 (effective 16)
for f in os.listdir("configs"):
    c = json.load(open(f"configs/{f}")); c.update(micro_batch_size=1, grad_accum=16, eval_batch=2)
    json.dump(c, open(f"configs/{f}", "w"), indent=2)
sh = lambda c: subprocess.run(c, shell=True)
sh("nvidia-smi > logs/nvidia_smi.txt; pip list 2>/dev/null | grep -iE '^(torch|transformers|peft|accelerate) ' > logs/versions_before.txt")
sh("pip install -q 'transformers==5.22.2' 'peft==0.21.2' accelerate > logs/pip.log 2>&1")
# peft 0.21 rejects Kaggle's preinstalled torchao 0.10 (needs >0.16); we do not use torchao -> remove it
sh("pip uninstall -y torchao >> logs/pip.log 2>&1")
sh("python -c \"import peft; from peft import LoraConfig; print('peft ok')\" >> logs/pip.log 2>&1")
sh("pip list 2>/dev/null | grep -iE '^(torch|transformers|peft|accelerate) ' > logs/versions.txt; cat logs/versions.txt")
# warm HF cache once so the two GPU jobs do not race on the download
sh("python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer as T; T.from_pretrained('ibm-granite/granite-4.0-h-350m'); AutoModelForCausalLM.from_pretrained('ibm-granite/granite-4.0-h-350m')\" > logs/download.log 2>&1")

def chain(exps):
    cmds = []
    for e in exps:
        cmds.append(f"python -u scripts/train_lora.py --config configs/{e}.json 2>&1 | sed -u 's/^/[{e}] /' | tee logs/{e}_train.log")
        for sp in ("val", "test"):
            cmds.append(f"python -u scripts/eval_hf.py --model ibm-granite/granite-4.0-h-350m --adapter runs/granite/{e}/adapter "
                        f"--split {sp} --prompt schema --data-dir data/v1.2 --out-dir results/eval --name granite_v12_lora{e}_{sp} "
                        f"--batch 16 2>&1 | sed -u 's/^/[{e} eval {sp}] /' | tee logs/{e}_eval_{sp}.log")
    return " ; ".join(cmds)

gpu_mon = subprocess.Popen("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw "
                           "--format=csv,noheader -l 30 | sed -u 's/^/[GPU] /' | tee logs/gpu_usage.csv", shell=True)
t0 = time.time()
# run 2 of experiments: C and D finished in run 1 (kernel v4); A and B were cut off by the 2 h limit -> rerun only A, B
procs = [subprocess.Popen(chain(["A_qv"]), shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}),
         subprocess.Popen(chain(["B_attn"]), shell=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"})]
for p in procs:
    p.wait()
gpu_mon.terminate()
summary = {"wall_s": time.time() - t0}
for e in ("A_qv", "B_attn", "C_mlp", "D_attn_mlp"):
    tl = f"runs/granite/{e}/train_log.json"
    if os.path.exists(tl):
        L = json.load(open(tl)); summary[e] = {k: L.get(k) for k in ("trainable_params", "train_time_s", "final_train_loss_full_pass", "final_val_loss", "adapter_bytes")}
    for sp in ("val", "test"):
        m = f"results/eval/granite_v12_lora{e}_{sp}/metrics.json"
        if os.path.exists(m):
            M = json.load(open(m)); summary.setdefault(e, {})[sp] = {k: M.get(k) for k in ("intent_accuracy", "field_accuracy", "exact_match", "json_validity", "unsupported_recall", "target_loss", "gen_wall_s")}
json.dump(summary, open("results/summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
# keep outputs small: drop the HF cache copy, keep adapters/logs/results
shutil.rmtree("data", ignore_errors=True)
