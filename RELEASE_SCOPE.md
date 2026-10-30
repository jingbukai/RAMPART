# Release scope

This package implements a compact, agent-structured version of the red/blue workflow. It is independent of the original experiment workspace. Each LLM-agent type has a named class and executable condensed prompt.

Included:

- Attack Strategy Agent, Patient Agent, Refiner Agent, Attack Judge, and conditional Crisis Response Judge, plus the Counselor Policy interface.
- Sequential role-aware multi-turn probing, fixed targets within retries, a best-severity archive, compatible mutation seeds, and qualitative refinement memory.
- Teacher Rewriter and bounded-window Safety Judge, including a condensed crisis-aware acceptance rule.
- B0 Sampler, anonymous Preference Judge with abstention, and a basic single-device LoRA SFT/DPO trainer.
- All 44 original cell definitions and original severity anchors 1, 2, and 5 for every cell (132 descriptions); generic intermediate guidance for scores 3 and 4.
- Source hashes, current prompt hashes, readable agent specifications, and explicit input/output boundaries.
- More detailed project evaluation wrappers for third-party benchmarks.
- Explicitly marked synthetic demo inputs and responses.

Not included:

- Original attack banks, model checkpoints, training datasets, patient-profile collections, or experimental results.
- Original cell-specific descriptions for scores 3 and 4, original dialogue examples, extended clinical calibration, and exact historical prompt variants.
- Experiment-specific selection/audit artifacts, cross-model provenance pipelines, large-scale schedulers, or all ablations.
- Private annotations, reviewer mappings, machine credentials, notebooks, or server configuration.

The controller uses simple empty-first target sampling rather than reconstructing every historical adaptive scheduler. Its condensed rubric and selection procedures can produce different outcomes from the paper. The abridged clinical specification has not been independently revalidated. Benchmark wrappers preserve more of the existing evaluation logic, but do not ship exact historical environments or model-service revisions. This release alone cannot reproduce all reported tables. Do not describe it as the complete experiment artifact.

MIT permits reuse of the released original code. Releasing fewer artifacts limits what is distributed, rather than preventing independent follow-up research.
