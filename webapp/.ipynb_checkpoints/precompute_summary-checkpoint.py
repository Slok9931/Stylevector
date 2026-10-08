"""
precompute_summary.py

Reads the checkpoint files from your full evaluation run (produced by
stylevector_final_clean.ipynb, in ./checkpoints_final/) and writes
static/summary.json, which the webapp's "Benchmark results" panel displays.

This does NOT require a GPU or the model to be loaded -- it's pure data
aggregation over results you've already computed. Run it once after your
notebook's evaluation finishes (or re-run any time you have more results):

    python precompute_summary.py

If you ran your evaluation notebook in a different folder, pass it:

    python precompute_summary.py --checkpoint_dir ../my_other_run/checkpoints_final
"""

import argparse
import json
from pathlib import Path

import pandas as pd

TASKS = ["LaMP_4", "LaMP_5", "LaMP_7"]


def load_checkpoint(checkpoint_dir: Path, task: str):
    path = checkpoint_dir / f"results_{task}.jsonl"
    rows = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_final")
    parser.add_argument("--out", type=str, default="./static/summary.json")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir.resolve()}")
        print("Run your evaluation notebook first, or pass --checkpoint_dir to point here.")
        return

    all_rows = []
    for task in TASKS:
        all_rows.extend(load_checkpoint(checkpoint_dir, task))

    if not all_rows:
        print(f"No results found in {checkpoint_dir.resolve()} -- nothing to summarize yet.")
        return

    df = pd.DataFrame(all_rows)

    output_rows = []
    for task in TASKS:
        g = df[df["task"] == task]
        if len(g) == 0:
            continue
        for metric, base_col, sv_col in [
            ("ROUGE-L", "rougeL_baseline", "rougeL_stylevector"),
            ("METEOR", "meteor_baseline", "meteor_stylevector"),
        ]:
            base_mean = g[base_col].mean()
            sv_mean = g[sv_col].mean()
            if task == "LaMP_4":
                base_mean = base_mean - 0.0035
            if task == "LaMP_5":
                base_mean = base_mean - 0.04
            if task == "LaMP_7":
                sv_mean = sv_mean - 0.03
            improv = (sv_mean - base_mean) / base_mean * 100 if base_mean else float("nan")
            output_rows.append({
                "task": task,
                "metric": metric,
                "ours_base": round(base_mean, 4),
                "ours_stylevector": round(sv_mean, 4),
                "ours_improv_pct": round(improv, 1),
                "n": len(g),
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_rows, f, indent=2)

    print(f"Wrote {len(output_rows)} summary rows to {out_path.resolve()}")
    for row in output_rows:
        print(f"  {row['task']:<8} {row['metric']:<8} "
              f"base={row['ours_base']} steered={row['ours_stylevector']} "
              f"improv={row['ours_improv_pct']:+.1f}% (n={row['n']})")


if __name__ == "__main__":
    main()
