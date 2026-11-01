# StyleVector: Multi-Component Style Vectors with Norm-Preserving Steering

**Author:** Slok Tulsyan · Student ID 12342080 · slok@iitbhilai.ac.in

This submission reimplements **StyleVector** (Zhang et al., ACL 2025), a training-free method that represents a user's writing style as one vector in an LLM's activation space and adds it to the hidden state during generation. It then evaluates two extensions that target limitations the paper names:

1. **Multi-component style vectors:** top-k PCA of the contrastive activation differences, and a shared-basis variant.
2. **Norm-preserving (NP) steering:** rescale the steered activation back to its original norm.

Everything uses **LLaMA-2-7B-chat** (4-bit) on three LaMP tasks: **LaMP-4** (news headlines), **LaMP-5** (scholarly titles) and **LaMP-7** (tweet paraphrasing), with ROUGE-L and METEOR. The full write-up is in the final report.

---

## Repository layout

```
.
├── Final_Report.pdf
├── ML_Project_Paper.pdf
├── Phase_1
│   └── 12342080_ML_Project_Proposal.pdf
├── Phase_2
│   ├── ablation_report.pdf
│   ├── checkpoints_final
│   │   ├── best_config.json
│   │   ├── full_results.csv
│   │   ├── results_LaMP_4.jsonl
│   │   ├── results_LaMP_5.jsonl
│   │   └── results_LaMP_7.jsonl
│   ├── comparison_chart.png
│   ├── comparison_table.csv
│   ├── phase_2_presentation.pdf
│   ├── phase_2_report.pdf
│   ├── README.md
│   ├── requirements.txt
│   ├── stylevector_phase_2.ipynb
│   └── webapp
│       ├── __pycache__
│       ├── app.py
│       ├── download_lamp.sh
│       ├── lamp_loader.py
│       ├── precompute_summary.py
│       ├── README.md
│       ├── requirements.txt
│       ├── static
│       ├── templates
│       └── webapp_cache
├── Phase_3
│   ├── phase_3_presentation.pdf
│   ├── phase_3_report.pdf
│   ├── README.md
│   ├── Run_1
│   │   ├── checkpoints_endterm
│   │   └── stylevector_endterm.ipynb
│   ├── Run_2
│   │   ├── checkpoints_v2
│   │   └── stylevector_improved_v2.ipynb
│   ├── Run_3
│   │   ├── checkpoints_v3
│   │   └── stylevector_v3.ipynb
│   ├── Run_4
│   │   ├── checkpoints_extension
│   │   └── stylevector_extension.ipynb
│   └── Run_5
│       ├── checkpoints_endterm_v2
│       └── stylevector_endterm_fixed.ipynb
└── README.md
```

Each `Run_N` folder holds one notebook and one checkpoint folder with that run's saved outputs (per-user scores, tables, figures). The reproduction of the paper's method and the live demonstration web app (Flask) are described in the report (Sections IV and V-A); they are not part of the `Run_N` folders.

---

## How the report maps to the project

| Report section | Content |
|---|---|
| I. Introduction | Motivation, project criterion (reproduce when code is unavailable, plus an improvement), team, contributions |
| II. Related Work | Personalization (RAG, PEFT, LaMP), activation steering, multi-direction methods |
| III. Proposed Methodology | StyleVector, per-user PCA, shared-basis projection, norm-preserving steering, controls, algorithm |
| IV. Experimental Setup | Data, model and hardware, metrics, bootstrap statistics, the five runs, demo system |
| V. Results and Analysis | Reproduction (n = 1,000), 2×2 grid across runs, Run 5 results and paired comparisons |
| VI. Ablation Study | Reproduction ablations (strength, sample size, prompts, history cap) and extension ablations (k, layer, strength, controls, output cleaning) |
| VII–IX | Discussion, limitations, conclusion and future work |
| Appendix A | AI prompts given to the LLM |

---

## The five extension runs

All runs use the same three tasks and the LaMP dev split. Each run changed something from the one before.

| Run | Notebook | What it does | Test users / task | How settings were chosen |
|---|---|---|---|---|
| **1** | `stylevector_endterm.ipynb` | Five conditions: no steering, Mean Difference (paper), Mean Difference + NP, multi-PCA (k=3), multi-PCA + NP. Raw outputs. | 300 | Fixed (layer 16, α = 0.25 / 0.5 / 0.5) |
| **2** | `stylevector_improved_v2.ipynb` | Adds validation tuning, output cleaning, global-vector and shrinkage controls, paired bootstrap. | 150 | Layer, ρ, k tuned on two validation sets of 40 users |
| **3** | `stylevector_v3.ipynb` | Larger validation/test sets, wider strength grid (ρ up to 1.2), random-direction control. | 300 | Layer and ρ tuned on 60 validation users |
| **4** | `stylevector_extension.ipynb` | Sweeps over k ∈ {1, 3, 5}, layer and strength. | 300 (sweeps: 50) | Fixed for the main run |
| **5** | `stylevector_endterm_fixed.ipynb` | Final pipeline: cleaned outputs, repaired LaMP-7 history inputs, chat-templated activations, per-condition tuning, shared-basis multi-component vector. | 300 | Layer and α tuned per condition on disjoint tuning users |

Strength is a multiple **α** of the style vector (Runs 1, 4, 5) or the relative size **ρ = ‖p‖/‖h‖** (Runs 2, 3).

### Checkpoint folder contents

| Folder | Main contents |
|---|---|
| `Run_1/checkpoints_endterm` | `results_LaMP_*.jsonl`, `ablation_table.csv`, `improvement_table.csv`, `config.json`, bar chart and explained-variance plot |
| `Run_2/checkpoints_v2` | `val1_long.csv`, `val2_long.csv`, `test_long.csv`, `summary_test.csv`, `paired_comparisons_test.csv`, `tuned_settings*`, 2×2 tables, `personalisation_diagnostic.csv`, heatmaps |
| `Run_3/checkpoints_v3` | `test_long.csv`, `summary_test.csv`, `paired_comparisons.csv`, `comparison_with_paper.csv`, `tuned_settings*`, validation-gain figure |
| `Run_4/checkpoints_extension` | `main_*`, `layer_*`, `strength_*`, `timing_*` `.jsonl`, sweep figures, cached activation differences in `deltas/` |
| `Run_5/checkpoints_endterm_v2` | `tune_*.jsonl`, `results_*.jsonl`, `best_config.json`, result tables, bar chart |

---

## Requirements

- **GPU:** one CUDA GPU with at least 12 GB of memory (the model is loaded in 4-bit NF4).
- **Model:** `meta-llama/Llama-2-7b-chat-hf` from Hugging Face (gated; requires approved access and `huggingface-cli login`).
- **Data:** the public **LaMP** benchmark (https://lamp-benchmark.github.io/), dev split, LaMP-4/5/7, placed under `./data/lamp`. The notebooks write a small loader (`load_lamp_data.py`) in one of their first cells.
- **Packages:** `torch`, `transformers`, `accelerate`, `bitsandbytes`, `rouge-score`, `nltk` (with WordNet for METEOR), `pandas`, `numpy`, `matplotlib`, `tqdm`.

```bash
pip install torch transformers accelerate bitsandbytes rouge-score nltk pandas numpy matplotlib tqdm
huggingface-cli login
```

## How to run

1. Open a run's notebook (for example `Run_5/stylevector_endterm_fixed.ipynb`) in Jupyter or Colab with a GPU.
2. Run the cells **top to bottom**.
3. The slow step is the first pass that computes contrastive activation differences for each user. It is **cached to disk** (`delta_cache/`) and **resumable**. Per-user results are saved as `.jsonl` files in the checkpoint folder, so an interrupted run continues where it stopped.
4. Tables and figures are written to the run's checkpoint folder. The saved folders in this submission are the outputs reported in the final report.

**Run 5 is the recommended entry point**: it contains the final pipeline and all conditions of the 2×2 ablation plus a per-user PCA control.

## Rebuilding the report

```bash
cd Final_Report_LaTeX_Source
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

It uses the supplied IEEEtran template (`IEEEtran.cls`, `IEEEtran.bst`).

---

## Summary of results

**Reproduction of the paper's method** (Mean Difference steering, n = 1,000 users per task). All six task–metric pairs improve, in the paper's direction and with similar size:

| Task | Metric | Paper Δ | Ours Δ |
|---|---|---|---|
| LaMP-4 | ROUGE-L / METEOR | +3.2% / +2.5% | +3.3% / +3.0% |
| LaMP-5 | ROUGE-L / METEOR | +25.8% / +10.2% | +23.1% / +10.5% |
| LaMP-7 | ROUGE-L / METEOR | +12.8% / +17.5% | +16.5% / +16.9% |

**Extensions (Runs 1–5).**

- **Runs 1–4:** output cleaning alone raised baseline ROUGE-L by 14–51%. The tuned strength was at the grid minimum in all 24 selections. Norm preservation was significantly better in 2 of 42 comparisons. A large LaMP-7 gain in Run 4 (+29% on raw output) fell to +12.5% after cleaning. Global-vector and random-direction controls performed about as well as the proposed variants.
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

Confidence intervals, controls, sweeps and discussion are in the final report.

---

## Notes and limitations

- Absolute scores differ from the paper's because of a smaller GPU, 4-bit quantization, the public dev split (LaMP releases no test labels) and a per-user history cap. The direction and relative size of the effects are what is compared.
- Scores are not directly comparable across runs (different users, prompts and cleaning). Each run's gain is measured against its own no-steering baseline.
- Runs 1 and 4 score raw model outputs. Runs 2, 3 and 5 clean generations (removing preambles such as "Sure! Here's…") identically for all conditions.
- Run 5 improvements over no steering are computed from absolute means. Confidence intervals exist for comparisons between steered conditions.
- Some settings (model identifier, and for some runs the history cap and PCA k) are not stored in the archives and are marked "not recorded" in the report.
- An AI assistant (Anthropic's Claude) helped with code review, the Run 5 pipeline design, analysis and drafting. All experiments were run by the author. The prompts are listed in Appendix A of the report.

## References

1. J. Zhang, Y. Liu, W. Wang, Q. Liu, S. Wu, L. Wang, T.-S. Chua. *Personalized Text Generation with Contrastive Activation Steering.* ACL 2025, pp. 7128–7141.
2. A. Salemi, S. Mysore, M. Bendersky, H. Zamani. *LaMP: When Large Language Models Meet Personalization.* ACL 2024, pp. 7370–7392.
