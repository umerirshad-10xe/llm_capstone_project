import csv
import copy
import os
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe in worker processes
import matplotlib.pyplot as plt
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Imports from your project (adjust paths if needed)
# ──────────────────────────────────────────────────────────────────────────────
import config as cfg
from part1_data import load_text, build_tokenizer, make_splits, get_batch
from part4_models import MHA_ModernLanguageModel, GQA_ModernLanguageModel

# ==========================================
# Part 4D: Experiments + learning summary
# ==========================================
# Goal:
# - Keep cfg.n_embd fixed
# - Try 20 meaningful head/layer/lr combinations across MHA and GQA
# - Run experiments in parallel across CPU cores using ProcessPoolExecutor
# - Compare MHA vs GQA (and different n_kv_head values for GQA)
# - Record validation loss / accuracy and write summary tables
#
# Parallelism strategy:
#   Each configuration is an independent training run with its own model copy,
#   so there are NO shared mutable states between workers.  We launch one
#   worker process per config and collect results as they finish.
#   Within each worker, PyTorch's DataLoader / intra-op parallelism is also
#   enabled via torch.set_num_threads().
#
# Constraints that must hold for every config:
#   cfg.n_embd % n_head == 0
#   n_head % n_kv_head == 0   (GQA only)
# ──────────────────────────────────────────────────────────────────────────────

# ── How many worker processes to use ─────────────────────────────────────────
# Defaults to all physical cores. Override with the MAX_WORKERS env variable.
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", os.cpu_count() or 4))

# ── Intra-op threads per worker ───────────────────────────────────────────────
# Each worker gets a fair slice of available threads so they do not fight.
THREADS_PER_WORKER = max(1, (os.cpu_count() or 4) // MAX_WORKERS)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  20 MEANINGFUL CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════
#
# Design rationale:
#   MHA configs  (10): sweep head count (2/4/8/16), layer depth (2/4/6/8),
#                      and three learning rates (1e-2, 3e-2, 1e-1).
#   GQA configs  (10): match the MHA head/layer grid but vary n_kv_head
#                      (1, 2, 4) to study different KV-sharing ratios.
#
#   All constraints are pre-verified:
#       cfg.n_embd (128) % n_head == 0  ✓
#       n_head % n_kv_head == 0          ✓ (GQA only)
#
#   The spread across heads/layers/lr is intentional so that the resulting
#   plots reveal meaningful trends rather than clustered noise.
# ──────────────────────────────────────────────────────────────────────────────
CONFIGS = [
    # ── MHA configs ───────────────────────────────────────────────────────────
    # Vary head count at fixed depth to see attention-head scaling
    {"model": "MHA", "n_head":  2, "n_layer": 4, "lr": 1e-2, "n_kv_head": None},   # 1  few heads, moderate lr
    {"model": "MHA", "n_head":  4, "n_layer": 4, "lr": 1e-2, "n_kv_head": None},   # 2  base MHA
    {"model": "MHA", "n_head":  8, "n_layer": 4, "lr": 1e-2, "n_kv_head": None},   # 3  more heads
    {"model": "MHA", "n_head": 16, "n_layer": 4, "lr": 1e-2, "n_kv_head": None},   # 4  many heads, same depth
    # Vary depth at fixed heads
    {"model": "MHA", "n_head":  4, "n_layer": 2, "lr": 3e-2, "n_kv_head": None},   # 5  shallow
    {"model": "MHA", "n_head":  4, "n_layer": 6, "lr": 3e-2, "n_kv_head": None},   # 6  medium-deep
    {"model": "MHA", "n_head":  4, "n_layer": 8, "lr": 3e-2, "n_kv_head": None},   # 7  deep
    # Higher lr to probe convergence speed
    {"model": "MHA", "n_head":  8, "n_layer": 6, "lr": 1e-1, "n_kv_head": None},   # 8  large lr
    {"model": "MHA", "n_head":  8, "n_layer": 6, "lr": 3e-2, "n_kv_head": None},   # 9  medium lr
    {"model": "MHA", "n_head":  8, "n_layer": 8, "lr": 1e-2, "n_kv_head": None},   # 10 deep + conservative lr

    # ── GQA configs ───────────────────────────────────────────────────────────
    # n_kv_head = n_head/2  (mild sharing, MHA-like behaviour)
    {"model": "GQA", "n_head":  4, "n_layer": 4, "lr": 1e-2, "n_kv_head": 2},      # 11
    {"model": "GQA", "n_head":  8, "n_layer": 4, "lr": 1e-2, "n_kv_head": 4},      # 12
    {"model": "GQA", "n_head": 16, "n_layer": 4, "lr": 1e-2, "n_kv_head": 8},      # 13
    # n_kv_head = n_head/4  (moderate sharing)
    {"model": "GQA", "n_head":  8, "n_layer": 4, "lr": 3e-2, "n_kv_head": 2},      # 14
    {"model": "GQA", "n_head":  8, "n_layer": 6, "lr": 3e-2, "n_kv_head": 2},      # 15
    {"model": "GQA", "n_head": 16, "n_layer": 6, "lr": 3e-2, "n_kv_head": 4},      # 16
    # n_kv_head = 1  (maximum sharing — MQA-like)
    {"model": "GQA", "n_head":  4, "n_layer": 6, "lr": 1e-1, "n_kv_head": 1},      # 17 aggressive sharing
    {"model": "GQA", "n_head":  8, "n_layer": 6, "lr": 1e-1, "n_kv_head": 1},      # 18
    # Deep GQA with different KV ratios
    {"model": "GQA", "n_head":  8, "n_layer": 8, "lr": 1e-2, "n_kv_head": 4},      # 19
    {"model": "GQA", "n_head":  8, "n_layer": 8, "lr": 1e-2, "n_kv_head": 2},      # 20 deep + maximum sharing
]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  HELPER FUNCTIONS  (same interface as original, no shared state)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_split_metrics(model, train_data, val_data, split, eval_iters=50):
    model.eval()
    losses      = torch.zeros(eval_iters)
    accuracies  = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(split, train_data, val_data)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
        if logits.dim() == 2:
            logits = logits.view(Y.size(0), Y.size(1), -1)
        preds = logits.argmax(dim=-1)
        accuracies[k] = (preds == Y).float().mean().item()
    model.train()
    return losses.mean().item(), accuracies.mean().item()


def format_config_label(model_type, n_head, n_layer, lr, n_kv_head):
    if model_type == "GQA":
        return f"GQA_h{n_head}_kv{n_kv_head}_l{n_layer}_lr{lr:g}"
    return f"MHA_h{n_head}_l{n_layer}_lr{lr:g}"


def train_for_steps(model, train_data, val_data, steps, lr,
                    eval_every, eval_iters, config_label):
    """Train model for `steps` steps; returns list of metric dicts."""
    opt     = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []

    for step in range(1, steps + 1):
        xb, yb = get_batch("train", train_data, val_data)
        _, train_loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        train_loss.backward()
        opt.step()

        if step % eval_every == 0 or step == 1 or step == steps:
            tl, ta = evaluate_split_metrics(model, train_data, val_data, "train", eval_iters)
            vl, va = evaluate_split_metrics(model, train_data, val_data, "val",   eval_iters)
            history.append({
                "config":     config_label,
                "step":       step,
                "train_loss": tl,
                "val_loss":   vl,
                "train_acc":  ta,
                "val_acc":    va,
            })
            print(
                f"[{config_label}] step {step:4d}/{steps} | "
                f"train_loss={tl:.4f} val_loss={vl:.4f} | "
                f"train_acc={ta:.4f} val_acc={va:.4f}"
            )

    return history


# ══════════════════════════════════════════════════════════════════════════════
# 3.  WORKER  — runs in its own process, fully isolated
# ══════════════════════════════════════════════════════════════════════════════

def run_config_worker(cfg_item, steps, eval_every, eval_iters):
    """
    Called inside a subprocess via ProcessPoolExecutor.
    Each call is completely independent: it builds its own model, its own
    optimizer, and reads data from tensors passed by value (fork-safe).

    Returns: (config_label, row_dict, history_list)
    """
    # ── Limit intra-op threads so workers do not fight over cores ─────────────
    torch.set_num_threads(THREADS_PER_WORKER)

    # ── Per-worker cfg overrides (we use a local namespace to stay thread-safe)
    local_cfg        = types.SimpleNamespace(**vars(cfg))
    local_cfg.n_head  = cfg_item["n_head"]
    local_cfg.n_layer = cfg_item["n_layer"]
    if cfg_item["model"] == "GQA":
        local_cfg.n_kv_head = cfg_item["n_kv_head"]

    # ── Data (re-built per worker; cheap for character-level tokenisation) ─────
    text = load_text()
    vocab_size, _, _, encode, _ = build_tokenizer(text)
    train_data, val_data = make_splits(text, encode)

    # ── Build model ───────────────────────────────────────────────────────────
    model_type    = cfg_item["model"]
    n_head        = cfg_item["n_head"]
    n_layer       = cfg_item["n_layer"]
    lr            = cfg_item["lr"]
    n_kv_head     = cfg_item["n_kv_head"]
    config_label  = format_config_label(model_type, n_head, n_layer, lr, n_kv_head)

    # Temporarily patch the global cfg so model constructors see the right values
    cfg.n_head  = n_head
    cfg.n_layer = n_layer
    if model_type == "GQA":
        cfg.n_kv_head = n_kv_head

    device = torch.device("cpu")  # CPU workers; change to "cuda" if GPUs are available
    model  = (
        GQA_ModernLanguageModel(vocab_size)
        if model_type == "GQA"
        else MHA_ModernLanguageModel(vocab_size)
    ).to(device)

    model_size_m = sum(p.numel() for p in model.parameters()) / 1e6
    t0           = time.perf_counter()

    history = train_for_steps(
        model, train_data, val_data,
        steps, lr, eval_every, eval_iters, config_label,
    )

    elapsed = time.perf_counter() - t0
    final   = history[-1]

    row = {
        "steps":        steps,
        "config":       config_label,
        "model":        model_type,
        "n_head":       n_head,
        "n_layer":      n_layer,
        "lr":           lr,
        "n_kv_head":    n_kv_head if n_kv_head is not None else "—",
        "model_size_m": round(model_size_m, 4),
        "train_loss":   final["train_loss"],
        "val_loss":     final["val_loss"],
        "train_acc":    final["train_acc"],
        "val_acc":      final["val_acc"],
        "elapsed_s":    round(elapsed, 1),
    }
    print(f"  ✓  {config_label}  finished in {elapsed:.1f}s | val_acc={final['val_acc']:.4f}")
    return config_label, row, history


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_histories(histories, out_path, title, metric_key, y_label):
    if not histories:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    for label, points in histories.items():
        steps  = [p["step"]       for p in points]
        values = [p[metric_key]   for p in points]
        ax.plot(steps, values, linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_size_vs_accuracy(rows, out_path):
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    sizes   = [r["model_size_m"] for r in rows]
    accs    = [r["val_acc"]      for r in rows]
    labels  = [r["config"]       for r in rows]
    colors  = ["tab:blue" if r["model"] == "MHA" else "tab:orange" for r in rows]
    ax.scatter(sizes, accs, c=colors, alpha=0.85, s=80)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (sizes[i], accs[i]), fontsize=6.5, alpha=0.9,
                    xytext=(4, 3), textcoords="offset points")
    # Legend patches
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color="tab:blue",   label="MHA"),
        mpatches.Patch(color="tab:orange", label="GQA"),
    ])
    ax.set_title("Validation Accuracy vs Model Size (M params)")
    ax.set_xlabel("Model Size (M params)")
    ax.set_ylabel("Validation Accuracy")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mha_vs_gqa_bar(results, out_path):
    """Side-by-side bar chart: average val_acc per model type."""
    mha_accs = [r["val_acc"] for r in results if r["model"] == "MHA"]
    gqa_accs = [r["val_acc"] for r in results if r["model"] == "GQA"]
    fig, ax  = plt.subplots(figsize=(7, 5))
    ax.bar(["MHA", "GQA"],
           [sum(mha_accs)/len(mha_accs) if mha_accs else 0,
            sum(gqa_accs)/len(gqa_accs) if gqa_accs else 0],
           color=["tab:blue", "tab:orange"], width=0.4)
    ax.set_ylabel("Mean Validation Accuracy")
    ax.set_title("MHA vs GQA — Average Validation Accuracy (all 20 configs)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Write only the fields declared in fieldnames (ignore extras)
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    steps      = 2000
    eval_every = 250
    eval_iters = 40
    top_k      = 5
    top10_k    = 10

    root_dir  = Path("experiment_runs")
    step_dir  = root_dir / f"steps_{steps}"
    step_dir.mkdir(parents=True, exist_ok=True)
    log_path  = step_dir / "run.log"

    # ── Validate configs before launching workers ──────────────────────────────
    valid_configs = []
    for c in CONFIGS:
        if cfg.n_embd % c["n_head"] != 0:
            print(f"  SKIP {c} — n_embd ({cfg.n_embd}) not divisible by n_head ({c['n_head']})")
            continue
        if c["model"] == "GQA" and (c["n_kv_head"] is None or c["n_head"] % c["n_kv_head"] != 0):
            print(f"  SKIP {c} — n_head ({c['n_head']}) not divisible by n_kv_head ({c['n_kv_head']})")
            continue
        valid_configs.append(c)

    print(f"\n{'='*70}")
    print(f"  Part 4D — Parallel experiment sweep")
    print(f"  Configs : {len(valid_configs)} / {len(CONFIGS)}")
    print(f"  Workers : {MAX_WORKERS}  (threads/worker: {THREADS_PER_WORKER})")
    print(f"  Steps   : {steps}")
    print(f"{'='*70}\n")

    results      = {}   # config_label -> row_dict
    all_histories= {}   # config_label -> list[metric_dict]

    t_total = time.perf_counter()

    # ── Parallel execution ─────────────────────────────────────────────────────
    # We use spawn-safe ProcessPoolExecutor.  Each worker is independent.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_config_worker, cfg_item, steps, eval_every, eval_iters): cfg_item
            for cfg_item in valid_configs
        }
        for future in as_completed(futures):
            cfg_item = futures[future]
            try:
                label, row, history = future.result()
                results[label]       = row
                all_histories[label] = history
            except Exception as exc:
                lbl = format_config_label(
                    cfg_item["model"], cfg_item["n_head"], cfg_item["n_layer"],
                    cfg_item["lr"], cfg_item["n_kv_head"]
                )
                print(f"  ✗  {lbl} raised an exception: {exc}")

    elapsed_total = time.perf_counter() - t_total
    print(f"\nAll configs finished in {elapsed_total:.1f}s total.")

    # ── Flatten and sort ───────────────────────────────────────────────────────
    all_rows  = list(results.values())
    sorted_rows = sorted(all_rows, key=lambda r: r["val_acc"], reverse=True)
    top5      = sorted_rows[:top_k]
    top10     = sorted_rows[:top10_k]

    # ── CSV outputs ───────────────────────────────────────────────────────────
    csv_fields = [
        "steps", "config", "model", "n_head", "n_layer", "lr", "n_kv_head",
        "model_size_m", "train_loss", "val_loss", "train_acc", "val_acc", "elapsed_s",
    ]
    write_csv(step_dir / "experiments.csv",         csv_fields, sorted_rows)
    write_csv(root_dir  / "all_results_flat.csv",   csv_fields, sorted_rows)
    write_csv(root_dir  / "all_step_top5.csv",      csv_fields, top5)

    metric_rows = []
    for hist in all_histories.values():
        metric_rows.extend(hist)
    write_csv(
        step_dir / "experiments_metrics.csv",
        ["config", "step", "train_loss", "val_loss", "train_acc", "val_acc"],
        metric_rows,
    )

    # ── Log file ──────────────────────────────────────────────────────────────
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Part 4D experiment run\n")
        log.write(f"steps={steps}, eval_every={eval_every}, eval_iters={eval_iters}, "
                  f"configs={len(valid_configs)}, workers={MAX_WORKERS}\n\n")
        for row in sorted_rows:
            log.write(
                f"[{row['config']}] val_acc={row['val_acc']:.4f} val_loss={row['val_loss']:.4f} "
                f"train_acc={row['train_acc']:.4f} train_loss={row['train_loss']:.4f} "
                f"size={row['model_size_m']:.3f}M elapsed={row['elapsed_s']}s\n"
            )

    # ── Plots ─────────────────────────────────────────────────────────────────
    top5_labels    = {r["config"] for r in top5}
    top5_histories = {k: v for k, v in all_histories.items() if k in top5_labels}

    plot_metric_histories(top5_histories, step_dir / "top5_training_loss.png",
                          f"Top-5 configs ({steps} steps): Training Loss",      "train_loss", "Train Loss")
    plot_metric_histories(top5_histories, step_dir / "top5_validation_loss.png",
                          f"Top-5 configs ({steps} steps): Validation Loss",    "val_loss",   "Validation Loss")
    plot_metric_histories(top5_histories, step_dir / "top5_training_accuracy.png",
                          f"Top-5 configs ({steps} steps): Training Accuracy",  "train_acc",  "Train Accuracy")
    plot_metric_histories(top5_histories, step_dir / "top5_validation_accuracy.png",
                          f"Top-5 configs ({steps} steps): Validation Accuracy","val_acc",    "Validation Accuracy")

    plot_size_vs_accuracy(top10,     step_dir / "top10_size_vs_val_accuracy.png")
    plot_size_vs_accuracy(sorted_rows, root_dir / "global_top10_size_vs_val_accuracy.png")
    plot_mha_vs_gqa_bar(sorted_rows,   root_dir / "mha_vs_gqa_avg_accuracy.png")

    # ── Markdown summary ──────────────────────────────────────────────────────
    with (step_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write(f"# Experiment Summary (steps={steps})\n\n")
        f.write(f"> Total wall time: {elapsed_total:.1f}s across {len(valid_configs)} configs "
                f"({MAX_WORKERS} parallel workers)\n\n")
        f.write("## All 20 configs ranked by validation accuracy\n\n")
        f.write("| rank | config | model | n_head | n_layer | lr | n_kv_head | size(M) | val_acc | val_loss | train_acc | train_loss | time(s) |\n")
        f.write("|------|--------|-------|--------|---------|----|-----------|---------|---------|----------|-----------|------------|--------|\n")
        for rank, row in enumerate(sorted_rows, 1):
            f.write(
                f"| {rank} | {row['config']} | {row['model']} | {row['n_head']} | {row['n_layer']} | "
                f"{row['lr']:g} | {row['n_kv_head']} | {row['model_size_m']:.3f} | "
                f"{row['val_acc']:.4f} | {row['val_loss']:.4f} | {row['train_acc']:.4f} | "
                f"{row['train_loss']:.4f} | {row['elapsed_s']} |\n"
            )
        f.write("\n## Generated plots\n")
        for p in [
            "top5_training_loss.png", "top5_validation_loss.png",
            "top5_training_accuracy.png", "top5_validation_accuracy.png",
            "top10_size_vs_val_accuracy.png",
        ]:
            f.write(f"- `{p}`\n")

    with (root_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Global Summary (2000-step run, 20 configs)\n\n")
        f.write(f"> Wall time: {elapsed_total:.1f}s | Workers: {MAX_WORKERS}\n\n")
        f.write("## Top-10 by validation accuracy\n\n")
        f.write("| rank | config | model | size(M) | val_acc | val_loss |\n")
        f.write("|------|--------|-------|---------|---------|----------|\n")
        for rank, row in enumerate(top10, 1):
            f.write(
                f"| {rank} | {row['config']} | {row['model']} | "
                f"{row['model_size_m']:.3f} | {row['val_acc']:.4f} | {row['val_loss']:.4f} |\n"
            )
        f.write("\n## Generated plots\n")
        for p in ["global_top10_size_vs_val_accuracy.png", "mha_vs_gqa_avg_accuracy.png"]:
            f.write(f"- `{p}`\n")

    print(f"\nDone. Outputs under: {root_dir}")
    print(f"Best config: {sorted_rows[0]['config']}  val_acc={sorted_rows[0]['val_acc']:.4f}")


if __name__ == "__main__":
    main()