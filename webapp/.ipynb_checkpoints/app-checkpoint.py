"""
app.py — StyleVector demo webapp backend.

Serves a single-page UI where you pick a person from the LaMP dataset, write
a prompt, and see the base (non-personalized) generation next to the
StyleVector-steered generation, plus analytics about that person's style
vector and how it was built.

All model-interaction logic here (chat template formatting, activation
extraction, Mean Difference style vector, steering hook) is copied directly
from the validated reproduction notebook -- not reimplemented from scratch --
to minimize the risk of new bugs in a component that could not be tested on
GPU before being handed off.

Run with:
    python app.py
Then tunnel port 5000 from your Mac the same way you've been tunneling
Jupyter's 8888, and open http://localhost:5000 in your browser.
"""

import json
import os
import random
import re
import time
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rouge_score import rouge_scorer

from lamp_loader import (
    TASK_FRIENDLY_NAMES,
    TASK_PROMPT_TEMPLATES,
    load_task_split,
)

# ---------------------------------------------------------------------------
# Config -- mirrors the final clean reproduction notebook exactly
# ---------------------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DATA_DIR = Path("../data/lamp")
CACHE_DIR = Path("./webapp_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Where your full evaluation notebook's checkpoint results live. Default
# assumes you're running this app from ~/stylevector_project/webapp/ and
# your notebook results are at ~/stylevector_project/checkpoints_final/.
# Override with: EVAL_CHECKPOINT_DIR=/some/other/path python app.py
EVAL_CHECKPOINT_DIR = Path(os.environ.get("EVAL_CHECKPOINT_DIR", "../checkpoints_final"))

TASKS = ["LaMP_4", "LaMP_5", "LaMP_7"]

MAX_HISTORY_PER_USER_BY_TASK = {
    "LaMP_4": 60,
    "LaMP_5": 60,
    "LaMP_7": 15,
}
MAX_GEN_TOKENS_PER_TASK = {
    "LaMP_4": 40,
    "LaMP_5": 40,
    "LaMP_7": 40,
}
# Hardcoded from the earlier validated grid search -- same as the final
# clean reproduction notebook.
BEST_CONFIG = {
    "LaMP_4": {"layer": 16, "alpha": 0.25},
    "LaMP_5": {"layer": 16, "alpha": 0.5},
    "LaMP_7": {"layer": 16, "alpha": 0.5},
}

# Must match the notebook's shuffle seed -- this is what makes user_ids in
# the dropdown line up with user_ids in your checkpoint results, so real
# improvement scores get attached to the correct people.
RNG_SEED = 42

# How many users to show in the dropdown per task (sorted best-improvement
# first once we have scores), and how many of the top performers to star.
DEMO_POOL_SIZE = 40
STAR_TOP_N = 5

# ---------------------------------------------------------------------------
# NLTK METEOR, with the same fallback used throughout the project
# ---------------------------------------------------------------------------
NLTK_OK = True
try:
    from nltk import word_tokenize
    from nltk.translate.meteor_score import meteor_score as _real_meteor
    _ = word_tokenize("test")
    _ = _real_meteor([["test"]], ["test"])
except Exception:
    NLTK_OK = False


def _simple_tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _simple_meteor_proxy(pred_tokens, gold_tokens):
    if not pred_tokens or not gold_tokens:
        return 0.0
    matches = sum(1 for t in pred_tokens if t in gold_tokens)
    precision = matches / len(pred_tokens)
    recall = matches / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def score_pair(pred, gold):
    rouge_l = scorer.score(gold, pred)["rougeL"].fmeasure
    if NLTK_OK:
        meteor = _real_meteor([word_tokenize(gold)], word_tokenize(pred))
    else:
        meteor = _simple_meteor_proxy(_simple_tokenize(pred), _simple_tokenize(gold))
    return rouge_l, meteor


# ---------------------------------------------------------------------------
# Model loading (4-bit, same hardware-driven choice as the notebook)
# ---------------------------------------------------------------------------
print("Loading tokenizer + model... (this can take a couple of minutes)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quant_config,
)
model.eval()
model.generation_config.max_length = 1024
N_LAYERS = len(model.model.layers)
print(f"Model loaded. Layers: {N_LAYERS} | "
      f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# --- NEW: load the aggregate benchmark numbers, so we can star any user
# whose individual improvement meets or exceeds what we found overall for
# that task (not just an arbitrary top-N) ---
def load_task_aggregate_improvement():
    """Reads static/summary.json (built by precompute_summary.py) and
    returns {task: aggregate_rougeL_improv_pct}. Empty dict if unavailable."""
    summary_path = Path("./static/summary.json")
    result = {}
    if not summary_path.exists():
        return result
    with open(summary_path) as f:
        rows = json.load(f)
    for row in rows:
        if row.get("metric") == "ROUGE-L":
            result[row["task"]] = row["ours_improv_pct"]
    return result


TASK_AGGREGATE_IMPROV = load_task_aggregate_improvement()
if TASK_AGGREGATE_IMPROV:
    print(f"Loaded aggregate benchmark improvement per task (star threshold): {TASK_AGGREGATE_IMPROV}")
else:
    print("No summary.json found -- star-marking will fall back to top-N by improvement. "
          "Run precompute_summary.py to enable threshold-based starring.")

# ---------------------------------------------------------------------------
# Core StyleVector functions -- copied from the validated notebook
# ---------------------------------------------------------------------------
def build_input_ids(tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    try:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return inputs["input_ids"].to(device)
    except Exception:
        formatted = f"<s>[INST] {prompt} [/INST]"
        return tokenizer(formatted, return_tensors="pt").input_ids.to(device)


def generate_text(prompt, max_new_tokens=40, do_sample=False):
    input_ids = build_input_ids(tokenizer, prompt, model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            do_sample=do_sample, pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = out[0][input_ids.shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def get_last_token_activation(text, layer_idx, max_len=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
    captured = {}

    def hook(module, inp, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hs[:, -1, :].detach().float().cpu()

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return captured["hidden"].squeeze(0)


def compute_user_deltas_detailed(history, task, layer_idx, max_history, max_gen_tokens):
    """Same as compute_user_deltas in the notebook, but also returns the
    (x_i, y_i) pair alongside each delta, so the app can later rank history
    items by how much they align with the final style vector."""
    prompt_fn = TASK_PROMPT_TEMPLATES[task]
    history = history[:max_history]
    items = []
    for x_i, y_i in history:
        if not x_i or not y_i:
            continue
        y_hat_i = generate_text(prompt_fn(x_i), max_new_tokens=max_gen_tokens)
        a_p = get_last_token_activation(f"{x_i} {y_i}", layer_idx)
        a_n = get_last_token_activation(f"{x_i} {y_hat_i}", layer_idx)
        delta = a_p - a_n
        items.append({"x_i": x_i, "y_i": y_i, "delta": delta})
    return items


def make_steering_hook(direction, alpha):
    direction = direction.clone()

    def hook(module, inp, output):
        hs = output[0] if isinstance(output, tuple) else output
        vec = (alpha * direction).to(hs.dtype).to(hs.device)
        hs = hs.clone()
        hs[:, -1, :] = hs[:, -1, :] + vec
        return (hs,) + output[1:] if isinstance(output, tuple) else hs

    return hook


def generate_with_steering(prompt, layer_idx, direction, alpha, max_new_tokens=40):
    input_ids = build_input_ids(tokenizer, prompt, model.device)
    handle = model.model.layers[layer_idx].register_forward_hook(make_steering_hook(direction, alpha))
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    gen_tokens = out[0][input_ids.shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Data: load per-user benchmark results from your full evaluation run, then
# build the demo pool from users we actually have scores for -- sorted by
# ROUGE-L improvement, best first, so top performers are easy to find and
# can be starred.
# ---------------------------------------------------------------------------
def load_full_shuffled_pool(task):
    """Loads the FULL dev-split pool for a task and shuffles with the same
    seed the notebook used, so user_ids here line up with user_ids in the
    notebook's checkpoint results."""
    pool = load_task_split(
        data_dir=DATA_DIR, task=task, split="dev",
        max_users=None,
        max_history_per_user=MAX_HISTORY_PER_USER_BY_TASK[task],
        min_history_len=3,
    )
    ids = list(pool.keys())
    random.Random(RNG_SEED).shuffle(ids)
    return pool, ids


def load_eval_results(task):
    """Reads per-user baseline/steered ROUGE-L from your notebook's
    checkpoint file, if available. Returns {} if not found -- the app still
    works fully without this, just without star-marking."""
    path = EVAL_CHECKPOINT_DIR / f"results_{task}.jsonl"
    results = {}
    if not path.exists():
        return results
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            base = row.get("rougeL_baseline")
            sv = row.get("rougeL_stylevector")
            if base is None or sv is None or base == 0:
                continue
            results[row["user_id"]] = {
                "rougeL_baseline": round(base, 4),
                "rougeL_stylevector": round(sv, 4),
                "improvement_pct": round((sv - base) / base * 100, 1),
            }
    return results


print("Loading user pools + benchmark results for the demo...")
user_pools = {}
eval_results_by_task = {}

for task in TASKS:
    full_pool, shuffled_ids = load_full_shuffled_pool(task)
    eval_results = load_eval_results(task)
    eval_results_by_task[task] = eval_results

    if eval_results:
        scored_ids = [uid for uid in shuffled_ids if uid in eval_results]
        scored_ids.sort(key=lambda uid: eval_results[uid]["improvement_pct"], reverse=True)
        if scored_ids:
            demo_ids = scored_ids[:DEMO_POOL_SIZE]
        else:
            print(f"  [warn] {task}: checkpoint results found but no user_id overlap with the "
                  f"shuffled pool (RNG_SEED mismatch?). Falling back to an arbitrary subset; "
                  f"star-marking will be unavailable for this task.")
            demo_ids = shuffled_ids[:DEMO_POOL_SIZE]
    else:
        print(f"  [warn] {task}: no evaluation results found at {EVAL_CHECKPOINT_DIR.resolve()} -- "
              f"star-marking disabled. Run your evaluation notebook first, or set "
              f"EVAL_CHECKPOINT_DIR to point at existing results.")
        demo_ids = shuffled_ids[:DEMO_POOL_SIZE]

    user_pools[task] = {uid: full_pool[uid] for uid in demo_ids}
    n_scored = sum(1 for uid in demo_ids if uid in eval_results)
    print(f"  {task}: {len(user_pools[task])} users in demo dropdown ({n_scored} have benchmark scores)")


# ---------------------------------------------------------------------------
# Style vector cache: computed lazily per (task, user_id), kept in memory
# and persisted to disk so repeated demo runs don't recompute.
# ---------------------------------------------------------------------------
_style_vector_cache = {}


def _cache_file(task, user_id):
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(user_id))
    return CACHE_DIR / f"{task}__{safe_id}.pt"


def get_or_build_style_vector(task, user_id):
    key = (task, user_id)
    if key in _style_vector_cache:
        return _style_vector_cache[key]

    disk_path = _cache_file(task, user_id)
    if disk_path.exists():
        data = torch.load(disk_path)
        _style_vector_cache[key] = data
        return data

    user = user_pools[task][user_id]
    layer_idx = BEST_CONFIG[task]["layer"]
    max_history = MAX_HISTORY_PER_USER_BY_TASK[task]
    max_gen_tokens = MAX_GEN_TOKENS_PER_TASK[task]

    items = compute_user_deltas_detailed(user["history"], task, layer_idx, max_history, max_gen_tokens)
    if not items:
        raise ValueError(f"No usable history for {task}/{user_id}")

    deltas = torch.stack([it["delta"] for it in items])
    mean_vec = deltas.mean(dim=0)

    # Rank each history item by cosine similarity to the final style vector --
    # this is the "which of this person's past writings most shaped their
    # style vector" analytic, shown in the UI's influences panel.
    sims = []
    for it in items:
        cos_sim = torch.nn.functional.cosine_similarity(
            it["delta"].unsqueeze(0), mean_vec.unsqueeze(0)
        ).item()
        sims.append({"x_i": it["x_i"], "y_i": it["y_i"], "similarity": round(cos_sim, 4)})
    sims.sort(key=lambda d: d["similarity"], reverse=True)

    data = {
        "vector": mean_vec,
        "norm": mean_vec.norm().item(),
        "n_history_used": len(items),
        "top_influences": sims[:3],
    }
    torch.save(data, disk_path)
    _style_vector_cache[key] = data
    return data


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tasks")
def api_tasks():
    return jsonify([
        {"id": t, "name": TASK_FRIENDLY_NAMES[t]} for t in TASKS
    ])


@app.route("/api/users/<task>")
def api_users(task):
    if task not in user_pools:
        return jsonify({"error": f"Unknown task {task}"}), 404

    eval_results = eval_results_by_task.get(task, {})
    out = []
    for uid, u in user_pools[task].items():
        preview = u["history"][0][1] if u["history"] else ""
        scores = eval_results.get(uid)
        out.append({
            "user_id": uid,
            "preview": preview[:120],
            "history_size": len(u["history"]),
            "held_out_query": u["query_input"],   # <-- NEW: exposes the actual test query for the "benchmark" mode preview
            "improvement_pct": scores["improvement_pct"] if scores else None,
            "rougeL_baseline": scores["rougeL_baseline"] if scores else None,
            "rougeL_stylevector": scores["rougeL_stylevector"] if scores else None,
            "starred": False,
        })

    # Star every user whose individual improvement meets or exceeds this
    # task's own aggregate improvement (from your full evaluation run) --
    # i.e. "at least as good as what we reported overall", not an
    # arbitrary top-N. Falls back to top-5 if we don't have an aggregate
    # number to compare against (e.g. summary.json not generated yet).
    threshold = TASK_AGGREGATE_IMPROV.get(task)
    if threshold is not None:
        for entry in out:
            if entry["improvement_pct"] is not None and entry["improvement_pct"] >= threshold:
                entry["starred"] = True
    else:
        starred_count = 0
        for entry in out:
            if entry["improvement_pct"] is not None and starred_count < STAR_TOP_N:
                entry["starred"] = True
                starred_count += 1

    return jsonify(out)


@app.route("/api/style_vector", methods=["POST"])
def api_style_vector():
    body = request.get_json(force=True)
    task = body["task"]
    user_id = body["user_id"]

    t0 = time.time()
    data = get_or_build_style_vector(task, user_id)
    elapsed = time.time() - t0

    return jsonify({
        "layer": BEST_CONFIG[task]["layer"],
        "alpha": BEST_CONFIG[task]["alpha"],
        "vector_norm": round(data["norm"], 3),
        "n_history_used": data["n_history_used"],
        "top_influences": data["top_influences"],
        "build_time_seconds": round(elapsed, 2),
        "was_cached": elapsed < 0.5,  # rough heuristic for display purposes
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    body = request.get_json(force=True)
    task = body["task"]
    user_id = body["user_id"]
    mode = body.get("mode", "free")  # "free" or "benchmark"
    custom_text = body.get("prompt", "")

    user = user_pools[task][user_id]
    prompt_fn = TASK_PROMPT_TEMPLATES[task]
    max_gen_tokens = MAX_GEN_TOKENS_PER_TASK[task]
    layer_idx = BEST_CONFIG[task]["layer"]
    alpha = BEST_CONFIG[task]["alpha"]

    if mode == "benchmark":
        raw_text = None
        prompt = user["query_input"]  # already the paper's rendered template
        gold = user["query_gold"]
    else:
        raw_text = custom_text.strip()
        if not raw_text:
            return jsonify({"error": "Please enter some text."}), 400
        prompt = prompt_fn(raw_text)
        gold = None

    style_data = get_or_build_style_vector(task, user_id)
    vec = style_data["vector"]

    t0 = time.time()
    baseline_output = generate_text(prompt, max_new_tokens=max_gen_tokens)
    t1 = time.time()
    steered_output = generate_with_steering(prompt, layer_idx, vec, alpha, max_new_tokens=max_gen_tokens)
    t2 = time.time()

    # Style alignment: cosine similarity between each output's own activation
    # and the person's style vector. Unlike ROUGE-L/METEOR, this needs no
    # ground-truth reference, so it's available in BOTH modes -- including
    # free-form prompts, where there's nothing to score against otherwise.
    base_act = get_last_token_activation(f"{prompt} {baseline_output}", layer_idx)
    steer_act = get_last_token_activation(f"{prompt} {steered_output}", layer_idx)
    base_alignment = round(
        torch.nn.functional.cosine_similarity(base_act.unsqueeze(0), vec.unsqueeze(0)).item(), 4
    )
    steer_alignment = round(
        torch.nn.functional.cosine_similarity(steer_act.unsqueeze(0), vec.unsqueeze(0)).item(), 4
    )

    response = {
        "task": task,
        "user_id": user_id,
        "mode": mode,
        "prompt_used": prompt,
        "baseline_output": baseline_output,
        "steered_output": steered_output,
        "baseline_time_seconds": round(t1 - t0, 2),
        "steered_time_seconds": round(t2 - t1, 2),
        "layer": layer_idx,
        "alpha": alpha,
        "gold": gold,
        "scores": None,
        "style_alignment": {"baseline": base_alignment, "steered": steer_alignment},
    }

    if gold is not None:
        r_base, m_base = score_pair(baseline_output, gold)
        r_sv, m_sv = score_pair(steered_output, gold)
        response["scores"] = {
            "baseline": {"rougeL": round(r_base, 4), "meteor": round(m_base, 4)},
            "steered": {"rougeL": round(r_sv, 4), "meteor": round(m_sv, 4)},
        }

    return jsonify(response)


@app.route("/api/summary")
def api_summary():
    """
    Serves the aggregate paper-vs-ours comparison table, precomputed from
    your full evaluation run (see precompute_summary.py). Returns null if
    that file hasn't been generated yet, so the frontend can show a
    friendly placeholder instead of erroring.
    """
    summary_path = Path("./static/summary.json")
    if not summary_path.exists():
        return jsonify(None)
    with open(summary_path) as f:
        return jsonify(json.load(f))

@app.route("/midterm-presentation")
def presentation():
    """Self-contained slide deck for presenting the project. The Live Demo
    slide embeds this same app's `/` route in an iframe, so the deck and
    the working demo stay one integrated experience -- no separate tab or
    context switch needed during the presentation."""
    return render_template("presentation.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)