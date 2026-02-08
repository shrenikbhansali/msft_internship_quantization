#!/usr/bin/env python3
import json
import os
import re
import glob
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("COLAB_RESULTS_ROOT", "/home/heck2/sbhansali8/msftquant/colab_results/content/results")
OUT_DIR = os.environ.get("FIGS_OUT_DIR", "/home/heck2/sbhansali8/msftquant/figs")
REPORT_PATH = os.environ.get("REPORT_PATH", "/home/heck2/sbhansali8/msftquant/report/report.md")

EX1_DIR = os.path.join(ROOT, "ex1")
EX2_DIR = os.path.join(ROOT, "ex2")

RUN_RE = re.compile(r"_bs(\d+)_sb(\d+)_cc(\w+)")
DATASET_N = 5153


def load_runs(root_dir):
    rows = []
    if not os.path.isdir(root_dir):
        return rows
    for d in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, d)
        if not os.path.isdir(path) or d == "default":
            continue
        m = RUN_RE.search(d)
        if not m:
            continue
        bs = int(m.group(1))
        sb = int(m.group(2))
        cc = m.group(3)
        metrics_files = sorted(glob.glob(os.path.join(path, "metrics*.json")))
        if not metrics_files:
            continue
        data = json.load(open(metrics_files[-1]))
        res = data["results"]["lambada_openai"]
        eff = (
            data.get("n-samples", {})
            .get("lambada_openai", {})
            .get("effective")
        ) or DATASET_N
        acc = res.get("acc,none")
        ppl = res.get("perplexity,none")
        time_s = float(data.get("total_evaluation_time_seconds"))
        ex_per_s = eff / time_s if time_s else None
        rows.append({
            "run_id": d,
            "block_size": bs,
            "scale_bits": sb,
            "custom_cuda": cc,
            "acc": acc,
            "ppl": ppl,
            "time_s": time_s,
            "n_effective": eff,
            "examples_per_s": ex_per_s,
        })
    return rows


def load_runs_from_report(table_title):
    if not os.path.isfile(REPORT_PATH):
        return []
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    rows = []
    in_table = False
    for line in lines:
        if table_title in line:
            in_table = True
            continue
        if in_table:
            if not line.strip() or not line.lstrip().startswith("|"):
                break
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if not parts or parts[0].startswith("---") or parts[0] == "block_size":
                continue
            if len(parts) < 8:
                continue
            try:
                bs = int(parts[0])
                sb = int(parts[1])
                cc = parts[2]
                acc = float(parts[3].split("\u00b1")[0].strip())
                ppl = float(parts[4].split("\u00b1")[0].strip())
                time_s = float(parts[5])
                ex_per_s = float(parts[6])
                run_id = parts[7]
            except ValueError:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "block_size": bs,
                    "scale_bits": sb,
                    "custom_cuda": cc,
                    "acc": acc,
                    "ppl": ppl,
                    "time_s": time_s,
                    "examples_per_s": ex_per_s,
                }
            )
    return rows


def plot_metric(rows, metric_key, title, ylabel, out_path):
    # Organize by scale_bits and custom_cuda
    series = defaultdict(list)
    for r in rows:
        key = (r["scale_bits"], r["custom_cuda"])
        series[key].append(r)

    plt.figure(figsize=(6, 4))
    for (sb, cc), items in sorted(series.items()):
        items = sorted(items, key=lambda x: x["block_size"])
        xs = [i["block_size"] for i in items]
        ys = [i[metric_key] for i in items]
        label = f"sb={sb}, cc={cc}"
        plt.plot(xs, ys, marker="o", label=label)

    plt.title(title)
    plt.xlabel("block_size")
    plt.ylabel(ylabel)
    plt.xticks(sorted(set([r["block_size"] for r in rows])))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ex1 = load_runs(EX1_DIR)
    if not ex1:
        ex1 = load_runs_from_report("Table 2.1: Exercise 1 sweep results")
    ex2 = load_runs(EX2_DIR)
    if not ex2:
        ex2 = load_runs_from_report("Table 2.2: Exercise 2 sweep results")

    if ex1:
        plot_metric(
            ex1,
            metric_key="acc",
            title="Ex1 Accuracy vs Block Size",
            ylabel="accuracy",
            out_path=os.path.join(OUT_DIR, "fig4_ex1_acc_vs_blocksize.png"),
        )
        plot_metric(
            ex1,
            metric_key="examples_per_s",
            title="Ex1 Speed vs Block Size",
            ylabel="examples/sec",
            out_path=os.path.join(OUT_DIR, "fig5_ex1_speed_vs_blocksize.png"),
        )

    if ex2:
        plot_metric(
            ex2,
            metric_key="acc",
            title="Ex2 Accuracy vs Block Size",
            ylabel="accuracy",
            out_path=os.path.join(OUT_DIR, "fig6_ex2_acc_vs_blocksize.png"),
        )
        plot_metric(
            ex2,
            metric_key="examples_per_s",
            title="Ex2 Speed vs Block Size",
            ylabel="examples/sec",
            out_path=os.path.join(OUT_DIR, "fig7_ex2_speed_vs_blocksize.png"),
        )


if __name__ == "__main__":
    main()
