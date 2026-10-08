# StyleVector Demo Webapp

A live demo: pick a person from the LaMP dataset, write a prompt (or use
their actual held-out test query), and see the base (non-personalized)
generation next to the StyleVector-steered generation, with analytics about
their style vector and how it compares to the full benchmark run.

## Important: this uses the exact validated logic from your notebook

The model-loading, chat-template formatting, activation extraction, Mean
Difference style vector, and steering hook code in `app.py` are copied
directly from `stylevector_final_clean.ipynb` — not reimplemented. This was
deliberate: it was built and syntax-checked without GPU access, so reusing
already-proven code minimizes the risk of new bugs in the parts that
couldn't be tested end-to-end before being handed off to you.

**What I could verify without a GPU:** Python syntax (`py_compile`),
undefined-name/import errors (`pyflakes`), that every HTML element ID
referenced in `app.js` actually exists in `index.html`, and that the JSON
schema `precompute_summary.py` produces matches what the frontend expects to
render. **What I could not verify:** actual model behavior, generation
quality, or real end-to-end request/response timing, since that requires
your GPU. If something doesn't work exactly as expected on first run, it's
most likely a small thing (a typo, a path issue) rather than a structural
problem — let me know the error and I can fix it quickly.

## Setup (on your lab PC, same environment as your notebook)

```bash
cd ~/stylevector_project
mkdir -p webapp && cd webapp
# copy these files here: app.py, lamp_loader.py, download_lamp.sh,
# requirements.txt, precompute_summary.py, static/, templates/

source ~/stylevector_env/bin/activate
pip install -q -r requirements.txt

# Reuse your existing LaMP data if you already have it downloaded elsewhere
# in stylevector_project -- either copy it in, or symlink:
ln -s ~/stylevector_project/data ./data
# (or, if you don't have it yet: bash download_lamp.sh ./data/lamp)
```

## Populate the benchmark summary panel (optional but recommended)

If you've already run `stylevector_final_clean.ipynb` and have results in
`~/stylevector_project/checkpoints_final/`, generate the summary the webapp
displays:

```bash
python precompute_summary.py --checkpoint_dir ~/stylevector_project/checkpoints_final
```

This writes `static/summary.json`. If you skip this step, the webapp still
works fully — the "Benchmark results" panel just shows a placeholder message
instead of the aggregate table.

## Run it

Same pattern as your Jupyter workflow — inside `tmux` so it survives your
laptop sleeping:

```bash
tmux new -s webapp
source ~/stylevector_env/bin/activate
cd ~/stylevector_project/webapp
python app.py
```

Wait for `Model loaded.` to print (a couple of minutes, same as your
notebook's model-loading cell). Then detach: `Ctrl+b` then `d`.

From your Mac, tunnel port 5000 (in addition to, or instead of, your
existing 8888 tunnel for Jupyter):

```bash
ssh -L 5000:localhost:5000 user@10.10.0.247
```

Open in your browser:

```
http://localhost:5000
```

## Using the demo

1. **Pick a task** (News Headline / Scholarly Title / Tweet Paraphrasing)
   and **a person** from the dropdown — this loads a snippet of their real
   writing history.
2. **"Write my own prompt"** mode: type your own article/abstract/tweet
   text. You'll see base vs. steered output side by side, plus which of
   that person's past writings most shaped their style vector. No ground
   truth exists for a made-up prompt, so no ROUGE-L/METEOR is shown here —
   this mode is for qualitative, visual demonstration.
3. **"Use this person's held-out test query"** mode: uses that exact
   person's real test-set input from LaMP. Because a reference answer
   exists for this one, you'll also see live ROUGE-L/METEOR for both
   outputs — a genuine, on-the-spot quantitative comparison.
4. The right-hand panel shows your full evaluation run's aggregate numbers
   (same table as your notebook's Part 6), for context alongside the live
   single-example demo.

## Performance notes

- First selection of a given person computes their style vector (multiple
  generation calls under the hood, same cost as in your notebook) — this
  takes a bit. It's cached afterward (both in memory and to disk in
  `webapp_cache/`), so revisiting the same person later in the same demo
  session, or in a future session, is instant.
- Each "Generate" click does two full generations (base + steered) — expect
  roughly the same per-call latency you saw in your notebook's timing tests
  on this hardware.
- If you're demoing live to your professor, it's worth pre-selecting a few
  people beforehand (just open their dropdown entry once) so their style
  vectors are already cached before the actual presentation.

## Files

- `app.py` — Flask backend, model loading, all StyleVector logic, API routes
- `lamp_loader.py` — LaMP data parsing (same logic as the notebook)
- `download_lamp.sh` — fetches LaMP data if you don't already have it
- `precompute_summary.py` — turns your notebook's checkpoint results into
  the JSON the benchmark panel displays
- `templates/index.html`, `static/style.css`, `static/app.js` — the frontend
- `webapp_cache/` — created automatically; stores computed style vectors
  per person so they don't need recomputing across sessions
