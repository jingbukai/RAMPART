# Benchmark evaluation

These are project evaluation wrappers for external benchmarks. They are more detailed than the minimal red/blue core. Obtain benchmark data and upstream source separately; retain their citations and terms. All commands below run from the **release root**. Replace model paths, model IDs, and endpoints with your services.

```bash
python -m pip install -r Blue_team/benchmarks/requirements.txt
```

Local generation additionally needs a compatible PyTorch/vLLM installation. Install that in a separate CUDA environment if needed. These files retain existing evaluation logic, with portable defaults and a deferred CBT constants import; they are not a new validation of every historical experiment.

## CounselBench-Adv

Upstream: [CounselBench](https://github.com/llm-eval-mental-health/CounselBench). The wrapper obtains the adversarial dataset and one-shot judge examples from their public Hugging Face sources. No dataset or human-annotation files are bundled.

Prepare the examples once:

```bash
python Blue_team/benchmarks/counselbench_adv_eval.py prepare-examples --output data/counselbench/judge_examples.json
```

Generate three responses per prompt. Omit `--lora-path` for an unadapted baseline. Use a new output directory for each model/adapter:

```bash
python Blue_team/benchmarks/counselbench_adv_eval.py generate \
  --model-path /path/to/base-model --lora-path /path/to/adapter \
  --num-responses 3 --temperature 0.7 --top-p 0.8 --top-k 20 --seed 42 \
  --output outputs/counselbench/model_responses.jsonl
```

Add `--limit-prompts 2` for a small execution check; remove it for a full evaluation. Downloads are enabled by default; use `--no-download` with an existing cache for offline data loading. `generate-transformers` is available for models incompatible with the vLLM path. Specify architecture-appropriate stop tokens with repeated `--stop-token-id` options.

Judge the saved responses using an OpenAI-compatible service. Export `JUDGE_API_KEY`, `JUDGE_MODEL`, and `JUDGE_BASE_URL` before running:

```bash
python Blue_team/benchmarks/counselbench_adv_eval.py judge \
  --input outputs/counselbench/model_responses.jsonl \
  --output outputs/counselbench/judge_a/judgments.jsonl \
  --examples-file data/counselbench/judge_examples.json \
  --judge-model "$JUDGE_MODEL" --api-base "$JUDGE_BASE_URL" \
  --api-key-env JUDGE_API_KEY --concurrency 2
```

Repeat with another judge and a **different judgment directory** if the protocol calls for two independent judges. The six target failure categories are apathetic, assumptions, symptoms, judgmental, medication, and therapy. Read the generated summary for counts, target failure rates, and failures; do not treat an interrupted or partial run as a complete score. Any mean across judges must use the same generated responses and complete comparable item sets.

Generation and judging save records separately. Rerun with unchanged settings and output files to resume; avoid `--overwrite` unless intentionally replacing a run. The script also checks configuration fingerprints to prevent mixing incompatible saved results.

## CBT-Bench Levels I and II

Upstream: [code](https://github.com/mianzhang/CBT-Bench), [data](https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench). Clone these separately; neither is redistributed in this release:

```bash
git clone https://github.com/mianzhang/CBT-Bench.git third_party/CBT-Bench
git clone https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench data/CBT-Bench
export CBT_OFFICIAL_DIR="$PWD/third_party/CBT-Bench"
```

The wrapper imports upstream `constants.py` for the original prompts and label order. Preserve the upstream Git revisions in your experiment notes. Check that the dataset JSON files contain actual JSON rather than Git LFS pointer text.

```bash
python Blue_team/benchmarks/cbt_eval.py validate --data-dir data/CBT-Bench
python Blue_team/benchmarks/cbt_eval.py run \
  --data-dir data/CBT-Bench --tasks all \
  --model "$COUNSELOR_MODEL" --base-url "$COUNSELOR_BASE_URL" \
  --api-key-env COUNSELOR_API_KEY --workers 2 \
  --output-dir outputs/cbt12 --run-name model_b1
```

`--tasks level1`, `level2`, or `qa,cd,pc,fc` selects tasks. Add `--limit 2` for an execution check. `run-local --model-path ... --lora-path ...` uses vLLM instead of a served endpoint. `score` recomputes metrics from saved predictions; `compare` compares two summaries. Use the corresponding subcommand's `--help` for file formats and options.

Level I QA gold answers are hidden in the public data. The wrapper exports predictions for the [official scoring tool](https://huggingface.co/spaces/Psychotherapy-LLM/CBT-QA), and leaves local accuracy unset rather than inventing a score. Level II reports multi-label metrics, including macro-F1, with the benchmark's label order. Official-compatible, strict, and lenient parsing metrics are kept distinct.

## CBT-Bench Level III

This workflow compares model replies to human references, using exercise-specific criteria. Use the same data checkout as above:

```bash
python Blue_team/benchmarks/cbt_dp_level3_eval.py validate --data-dir data/CBT-Bench
python Blue_team/benchmarks/cbt_dp_level3_eval.py generate-local \
  --data-dir data/CBT-Bench --model-path /path/to/base-model --lora-path /path/to/adapter \
  --temperature 0.7 --top-p 1 --max-new-tokens 256 --seed 42 \
  --output outputs/cbt3/generations.jsonl
python Blue_team/benchmarks/cbt_dp_level3_eval.py judge \
  --input outputs/cbt3/generations.jsonl --output outputs/cbt3/judge/judgments.jsonl \
  --judge-model "$JUDGE_MODEL" --api-base "$JUDGE_BASE_URL" \
  --api-key-env JUDGE_API_KEY --orders both --concurrency 2
```

The judge evaluates both response orders to reduce order bias. Four criteria plus an overall rating use scores from −2 to +2; signs are normalized so positive values favor the model over the human reference. Win/tie/loss counts use the averaged overall score per completed exercise. They do not compare B₀ directly against B₁. Cases missing a required order are excluded from completed cases and the summary reports incomplete coverage.

Rebuild summaries without API calls:

```bash
python Blue_team/benchmarks/cbt_dp_level3_eval.py summarize \
  --input outputs/cbt3/generations.jsonl --judgments outputs/cbt3/judge/judgments.jsonl \
  --output-dir outputs/cbt3/summary --orders both
```

Use `--limit` consistently for small generation/judging checks. Keep each judge configuration in a separate directory and retain raw judgments alongside summaries.

## VERA-MH

Upstream: [SpringCare/VERA-MH](https://github.com/SpringCare/VERA-MH). This release delegates persona simulation, rubric navigation, and scoring to the upstream framework rather than substituting a simplified score. Clone it and follow its environment/provider setup:

```bash
git clone https://github.com/SpringCare/VERA-MH.git third_party/VERA-MH
```

After configuring upstream-supported models and installing its dependencies, inspect the actual pipeline options:

```bash
python Blue_team/benchmarks/vera_mh.py --repo third_party/VERA-MH -- --help
```

For a small run, set model IDs recognized by that checkout and use:

```bash
python Blue_team/benchmarks/vera_mh.py --repo third_party/VERA-MH -- \
  --user-agent "$VERA_USER_MODEL" --provider-agent "$VERA_PROVIDER_MODEL" \
  --judge-model "$VERA_JUDGE_MODEL" --runs 1 --turns 4 --max-personas 2
```

Use `--python /path/to/vera-environment/bin/python` to launch its dedicated environment. `--entry generate` and `--entry judge` forward arguments to the upstream standalone stages. Custom model endpoints must be registered using the upstream provider interface; arbitrary identifiers from `config.json` are not automatically supported.

Record the upstream commit, persona selection, user model, counselor snapshot, judge, and turn convention for every evaluation. Use its saved transcripts and scoring pipeline for final metrics. The small example above is an execution check, not the RAMPART paper's evaluation profile.
