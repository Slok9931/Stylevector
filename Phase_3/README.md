# StyleVector: Multi-Component Style Vectors with Norm-Preserving Steering

**Author:** Slok Tulsyan (Student ID: 12342080) · slok@iitbhilai.ac.in

This submission reimplements **StyleVector** (Zhang et al., ACL 2025), a training-free method that represents a user's writing style as one vector in an LLM's activation space and adds it to the hidden state during generation. It then evaluates two extensions that target limitations the paper itself names:

1. **Multi-component style vectors** (top-k PCA of the contrastive activation differences, and a shared-basis variant).
2. **Norm-preserving steering** (rescale the steered activation back to its original norm).

Experiments use **LLaMA-2-7B-chat** (4-bit) on three LaMP tasks: **LaMP-4** (news headlines), **LaMP-5** (scholarly titles) and **LaMP-7** (tweet paraphrasing). Metrics are ROUGE-L and METEOR.

---

## Repository layout

```
.
├── README.md
├── phase_3_report.pdf                          # final report (all runs, tables, analysis)
├── phase_3_presentation.pdf                    # final presentation
├── Run_1/
│   ├── stylevector_endterm.ipynb
│   └── checkpoints_endterm/
├── Run_2/
│   ├── stylevector_improved_v2.ipynb
│   └── checkpoints_v2/
├── Run_3/
│   ├── stylevector_v3.ipynb
│   └── checkpoints_v3/
├── Run_4/
│   ├── stylevector_extension.ipynb
│   └── checkpoints_extension/
└── Run_5/
    ├── stylevector_endterm_fixed.ipynb
    └── checkpoints_endterm_v2/
```

Each `Run_N` folder is self-contained: one notebook that produces the results, and one checkpoint folder holding its saved outputs (per-user scores, tables and figures).

---

## The five runs

All runs evaluate the same three tasks on the LaMP dev split. Each run changed something from the one before it.

| Run | Notebook | What it does | Test users per task | How settings were chosen |
|---|---|---|---|---|
| **1** | `stylevector_endterm.ipynb` | First end-term attempt. Five conditions: no steering, Mean Difference (paper), Mean Difference + norm-preserving (NP), multi-PCA (k=3), multi-PCA + NP. Raw outputs. | 300 | Fixed settings (layer 16, α = 0.25 / 0.5 / 0.5) |
| **2** | `stylevector_improved_v2.ipynb` | Adds validation-based tuning, output cleaning, a global-vector control, a shrinkage variant and paired bootstrap statistics. | 150 | Layer, strength ρ and k tuned on two validation sets of 40 users |
| **3** | `stylevector_v3.ipynb` | Larger validation and test sets, wider strength range (ρ up to 1.2) and a random-direction control. | 300 | Layer and ρ tuned on 60 validation users |
| **4** | `stylevector_extension.ipynb` | Sweeps over the number of PCA components k ∈ {1, 3, 5}, steering layer, and steering strength. | 300 (sweeps: 50) | Fixed settings for the main run |
| **5** | `stylevector_endterm_fixed.ipynb` | Final, refined pipeline: cleaned outputs, repaired LaMP-7 history inputs, chat-templated activation extraction, per-condition tuning, and a shared-basis multi-component vector. | 300 | Layer and α tuned per condition on disjoint tuning users |

Strength is expressed as a multiplier **α** of the style vector (Runs 1, 4, 5) or as the relative perturbation size **ρ = ‖p‖/‖h‖** (Runs 2, 3).

### What is in each checkpoint folder

| Folder | Main contents |
|---|---|
| `Run_1/checkpoints_endterm` | `results_LaMP_*.jsonl` (per-user scores and predictions), `ablation_table.csv`, `improvement_table.csv`, `config.json`, bar chart and explained-variance plot |
| `Run_2/checkpoints_v2` | Validation and test results (`val1_long.csv`, `val2_long.csv`, `test_long.csv`), `summary_test.csv`, `paired_comparisons_test.csv`, `tuned_settings*.{json,csv}`, 2×2 ablation tables, `personalisation_diagnostic.csv`, heatmaps |
| `Run_3/checkpoints_v3` | `test_long.csv`, `summary_test.csv`, `paired_comparisons.csv`, `comparison_with_paper.csv`, `tuned_settings.{json,csv}`, validation-gain figure |
| `Run_4/checkpoints_extension` | `main_*`, `layer_*`, `strength_*` and `timing_*` `.jsonl` files, sweep figures (`k_sweep.png`, `layer_sweep.png`, `strength_sweep.png`), cached contrastive activation differences in `deltas/` |
| `Run_5/checkpoints_endterm_v2` | Tuning logs (`tune_*.jsonl`), per-user results (`results_*.jsonl`), `best_config.json` (selected layer and α per condition), result tables and the bar chart |

---

## Requirements

- **GPU:** one CUDA GPU with at least 12 GB of memory (the model is loaded in 4-bit NF4).
- **Model:** `meta-llama/Llama-2-7b-chat-hf` from Hugging Face (gated; you need approved access and a Hugging Face login).
- **Data:** the public **LaMP** benchmark, dev split, for LaMP-4, LaMP-5 and LaMP-7. The notebooks expect the files under `./data/lamp` and write a small loader (`load_lamp_data.py`) in one of their first cells.
- **Python packages:** `torch`, `transformers`, `accelerate`, `bitsandbytes`, `rouge-score`, `nltk` (with WordNet for METEOR), `pandas`, `numpy`, `matplotlib`, `tqdm`.

```bash
pip install torch transformers accelerate bitsandbytes rouge-score nltk pandas numpy matplotlib tqdm
huggingface-cli login
```

## How to run

1. Open the notebook of the run you want (for example `Run_5/stylevector_endterm_fixed.ipynb`) in Jupyter or Colab with a GPU.
2. Run the cells **top to bottom**. The notebook installs or imports its dependencies, loads the data and model, and then runs the experiment.
3. The slow step is the first pass that generates contrastive activation differences for every user. It is **cached to disk** (`delta_cache/`) and **resumable**: if the run is interrupted, re-running the cells continues where it stopped. Per-user results are written to a checkpoint folder as `.jsonl` files in the same way.
4. Tables and figures are written to the run's checkpoint folder. The saved folders in this submission are the outputs of the runs reported in the final report.

Run 5 is the recommended entry point: it contains the final pipeline and all five conditions of the 2×2 ablation plus a per-user PCA control.

---

## Summary of results

**Reproduction of the paper's method (Mean Difference steering, LLaMA-2-7B-chat, 1,000 users per task).** It improves on all six task–metric pairs, in the same direction and with similar size as the paper:

| Task | Metric | Paper Δ | Ours Δ |
|---|---|---|---|
| LaMP-4 | ROUGE-L / METEOR | +3.3% / +2.4% | +3.3% / +3.0% |
| LaMP-5 | ROUGE-L / METEOR | +25.8% / +10.2% | +23.1% / +10.5% |
| LaMP-7 | ROUGE-L / METEOR | +12.8% / +17.5% | +16.5% / +16.9% |

**Extensions (Runs 1–5).**

- **Runs 1–4:** output cleaning alone raised baseline ROUGE-L by 14–51%. Tuned strengths were at the grid minimum in all 24 selections. Norm preservation was significantly better in only 2 of 42 comparisons. A large LaMP-7 gain (+29% on raw output) fell to +12.5% after cleaning. Global-vector and random-direction controls performed about as well as the proposed variants.
- **Run 5 (refined pipeline):** the shared-basis multi-component vector is statistically indistinguishable from Mean Difference in all 6 paired comparisons. The norm-preserving variants score lower than Mean Difference in 3 of 6 comparisons each. Per-user PCA (k=3) matches Mean Difference and is better on LaMP-4 METEOR.
- **Overall:** Mean Difference steering with tuned layer and strength is the strongest configuration tested. Neither extension improved on it.

Run 5 absolute scores (n = 300 per task):

| Condition | LaMP-4 R-L | LaMP-4 MET | LaMP-5 R-L | LaMP-5 MET | LaMP-7 R-L | LaMP-7 MET |
|---|---|---|---|---|---|---|
| No steering | 0.0569 | 0.0574 | 0.2578 | 0.3076 | 0.3001 | 0.4055 |
| Mean Diff (paper) | 0.0609 | 0.1436 | 0.3141 | 0.3411 | 0.3527 | 0.4147 |
| Mean Diff + NP | 0.0676 | 0.0844 | 0.3100 | 0.3425 | 0.2989 | 0.3620 |
| Per-user PCA, k = 3 | 0.0636 | 0.1590 | 0.3178 | 0.3392 | 0.3522 | 0.4148 |
| Shared basis | 0.0553 | 0.1341 | 0.3019 | 0.3321 | 0.3512 | 0.4121 |
| Shared basis + NP (full method) | 0.0679 | 0.0905 | 0.3036 | 0.3345 | 0.2931 | 0.3665 |

Full tables, confidence intervals, controls and discussion are in the final report.

---

## Notes and limitations

- Absolute scores differ from the paper's because of a smaller GPU, 4-bit quantization, the public dev split (LaMP does not release test labels) and a history cap per user. The direction and size of the effects are what is compared.
- Scores are not directly comparable across runs, because the runs differ in users, prompts and output cleaning. Each run's gain is measured against its own no-steering baseline.
- Runs 1 and 4 score raw model outputs. Runs 2, 3 and 5 clean generations (removing preambles such as "Sure! Here's…") identically for all conditions.
- Run 5 improvements over no steering are computed from the absolute means. Confidence intervals are available for comparisons between steered conditions.
- An AI assistant (Anthropic's Claude) helped review code, design the Run 5 pipeline and draft the analysis and report. All experiments were run by the author. See the appendix of the report for details.

## References

1. J. Zhang, Y. Liu, W. Wang, Q. Liu, S. Wu, L. Wang, T.-S. Chua. *Personalized Text Generation with Contrastive Activation Steering.* ACL 2025, pp. 7128–7141.
2. A. Salemi, S. Mysore, M. Bendersky, H. Zamani. *LaMP: When Large Language Models Meet Personalization.* ACL 2024, pp. 7370–7392.
