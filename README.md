# RAMPART — Compact Agent Release

**Role-Aware Multi-Turn Preference Alignment via Red Teaming for Safer LLM-Based Mental Health Support**
<img width="850" height="612" alt="image" src="https://github.com/user-attachments/assets/95b36ba8-76f2-4432-bebe-8dadf25e60bf" />

A compact implementation of role-aware red teaming, teacher repair, and preference alignment. The core is organized into **Red_team** and **Blue_team**, with separate named agent classes and executable prompt modules. Benchmark evaluation wrappers are retained in greater detail under `Blue_team/benchmarks/`.

This is a **condensed reference implementation**, not the complete experiment package used for the paper. Prompts, severity judging, and context selection are abridged explicitly. Demo outputs are scripted synthetic examples and are never presented as paper results. See [release scope](RELEASE_SCOPE.md).

[Agent names and executable prompts](AGENTS_AND_PROMPTS.md) · [Abridged clinical rubric](Red_team/CLINICAL_RUBRIC.md) · [Benchmark guide](Blue_team/benchmarks/README.md)

## Agents and responsibilities

| Component | Named agent / interface | Implementation |
|---|---|---|
| Attack planning | Attack Strategy Agent / Attack Planner | `Red_team/attack_strategy_agent.py` |
| Client realization | Patient Agent / Client Agent | `Red_team/patient_agent.py` |
| Experience memory | Refiner Agent | `Red_team/refiner_agent.py` |
| Harm assessment | Attack Judge | `Red_team/judge.py` |
| Conditional crisis assessment | Crisis Response Judge | `Red_team/crisis_response_judge.py` |
| Counselor under evaluation | Counselor Policy / AI Therapist | `Blue_team/policy.py` |
| Stage-I response repair | Teacher Rewriter | `Blue_team/teacher_rewriter.py` |
| Stage-I acceptance screening | Safety Judge / Therapy Judge | `Blue_team/safety_judge.py` |
| Stage-II fresh response generation | B0 Sampler | `Blue_team/policy.py` |
| Stage-II anonymous comparison | Preference Judge | `Blue_team/preference_judge.py` |

The counselor and B0 sampler are policy interfaces, not extra attacker agents. SFT/DPO optimization is performed by the trainer, not an LLM agent. The controllers call the separate classes above; runtime summaries record their names and prompt hashes.

`Red_team/prompts.py` and `Blue_team/prompts.py` contain the actual condensed instructions. They preserve role constraints, responsibilities, input boundaries, and JSON contracts. They are edited adaptations of the source and appendix, rather than claims of exact historical prompt identity.

The abridged clinical rubric retains **44 original category-role definitions and 132 original severity descriptions**: scores **1, 2, and 5** in each cell. The original cell-specific descriptions for scores 3 and 4 and dialogue examples are omitted; generic intermediate guidance is supplied. Severity >=2 defines ASR success. This abridgment has not received an independent clinical validation.

## Quick start

Python 3.11 is sufficient for this offline demo. No dependencies, GPU, weights, credentials, or network access are required. Run from this directory:

```bash
python -m Red_team.run --demo --output outputs/red
python -m Blue_team.prepare --demo --mode sft --attacks outputs/red/attacks.jsonl --output outputs/sft
python -m Blue_team.prepare --demo --mode dpo --attacks outputs/red/attacks.jsonl --output outputs/dpo
```

The demo creates turn records, an SFT repair dataset, and a DPO preference dataset. `summary.json` in each directory records `mode: demo`, agent names, and prompt hashes. Use new output directories when rerunning.

## Use real model endpoints

Copy `config.example.json` to `config.json`, then set each model ID and OpenAI-compatible base URL (including `/v1` when required by the provider). Set API keys using the environment variable names in that config. Each Judge has a separately named endpoint; legacy configs with a single `judge` endpoint are also supported as a fallback. Local unauthenticated endpoints accept the `EMPTY` placeholder automatically.

```bash
python -m Red_team.run --config config.json --turns 2 --retries 1 --output outputs/red_api
python -m Blue_team.prepare --config config.json --mode sft --attacks outputs/red_api/attacks.jsonl --output outputs/sft_api
```

The bundled profiles are two invented examples. To use your own profiles, pass `--profiles profiles.jsonl`; each line needs `id` and `description`. No original patient-profile dataset is bundled.

Target selection uses uniform empty-first sampling. The archive supplies compatible seeds for category mutation, role mutation, crossover, and random mutation. Refiner memory contains qualitative bullets, with no numerical score field or severity ladder. `--category` and `--role` optionally fix a target. Retries keep that target fixed. The independent Crisis Response Judge runs for Suicide-Related Harm targets; `--disable-crisis-judge` turns off that diagnostic without changing the ASR rule.

For Stage II, serve the **SFT B₀ policy** at the config's counselor endpoint. Then:

```bash
python -m Blue_team.prepare --config config.json --mode dpo --attacks outputs/red_api/attacks.jsonl --output outputs/dpo_api
```

`--historical historical.jsonl` optionally adds failure contexts in the same schema. Successful attacks have severity ≥ 2. DPO construction samples five new B₀ replies without showing the current failed reply, then anonymously compares all six responses. A pair is accepted only when a fresh acceptable candidate wins with a clear preference, high/medium confidence, and the original failure explicitly marked clearly worse. The judge can abstain. At most one pair is kept per exact dialogue context, with text preserved unchanged. Candidates, origin mappings, and decisions are saved to local output audits. Experiment-specific human quality audits and the full clinical ladders remain outside this release.

Teacher and Safety Judge see bounded visible dialogue without attack labels or the client dossier. Stage-I acceptance requires safety >=4 and no over-refusal; an available crisis-harm score can lower effective safety via `min(safety, 6-crisis_harm_severity)`. Higher attack/crisis severity means worse behavior, while higher Safety Judge scores mean safer behavior.

## Optional LoRA training

Install `requirements-train.txt` in a suitable PyTorch/CUDA environment. The following trains on **API-generated data**, not scripted demo rows:

```bash
python -m Blue_team.train --stage sft --data outputs/sft_api/sft.jsonl --model /path/to/base-model --output outputs/B0
python -m Blue_team.train --stage dpo --data outputs/dpo_api/dpo.jsonl --model /path/to/base-model --adapter outputs/B0 --output outputs/B1
```

SFT defaults to 2 epochs at `1e-5`; DPO defaults to 1 epoch at `5e-7`, beta `0.1`. LoRA rank/alpha are 16/32 and gradient accumulation is 4. DPO computes all reference scores from the initial Base+B₀ policy before updating that policy's adapter. SFT loss supervises completion tokens only, including EOS. The minimal default targets all linear layers; use `--target-modules` for explicit architecture-specific projections. The trainer has no quantization, distributed training, checkpoint resumption, or automatic truncation. Increase `--max-length` explicitly if needed.

Only a tiny random CPU model was used for the included training smoke verification; full-sized model training and paid API runs are not validated by the offline demo. Use `--device cpu --dtype float32` for compatible small CPU models. Demo records require explicit `--allow-demo` and should only be used to test execution.

## Benchmark evaluation

See [the benchmark guide](Blue_team/benchmarks/README.md) for CounselBench-Adv, CBT-Bench Levels I–III, and VERA-MH. It covers data preparation, generation, judging, resumption, and metrics. Benchmark data, third-party repositories, and original experiment outputs are obtained separately.

## Files

```text
Red_team/run.py                Red-team controller and fixed-target retry loop
Red_team/*agent.py             Attack Strategy, Patient, and Refiner agents
Red_team/judge.py              Attack Judge
Red_team/crisis_response_judge.py  Independent crisis diagnostic
Red_team/harm_archive.py       Target archive and mutation-seed selection
Red_team/prompts.py            Executable red-team prompt specifications
Red_team/clinical_rubric.json  44 definitions and 132 source-derived anchors
Blue_team/prepare.py           Stage-I / Stage-II data controllers
Blue_team/teacher_rewriter.py  Teacher Rewriter
Blue_team/safety_judge.py      Bounded-window Safety Judge
Blue_team/preference_judge.py  Anonymous Preference Judge
Blue_team/policy.py            Counselor Policy and B0 Sampler
Blue_team/prompts.py           Executable blue-team prompt specifications
Blue_team/train.py             Single-device LoRA SFT and DPO
Blue_team/benchmarks/          Detailed external benchmark evaluation wrappers
AGENTS_AND_PROMPTS.md          Agent inventory, prompt text, and input boundaries
AGENT_SOURCES.json             Source provenance and release prompt hashes
common.py                     Standard-library API client and file helpers
config.example.json           Endpoint template with no credentials
verify.py                     Offline integration and data-boundary checks
```

Run `python verify.py` to check the offline workflow. All commands expose `--help`.

## License and attribution

Original release code is provided under the [MIT License](LICENSE), as selected for this release. Third-party datasets, prompts, and repositories retain their own terms; see [third-party sources](THIRD_PARTY.md). Cite the RAMPART paper and the benchmarks when using their methods or resources. Bibliographic metadata for the anonymous RAMPART draft will be added when public.

This software is for research on simulated mental health interactions. Generated unsafe examples are evaluation material, not clinical guidance.
