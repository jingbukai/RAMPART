# Third-party sources

The repository's MIT license applies to original release code and modifications. It does not relicense third-party datasets, prompt specifications, models, or separately downloaded repositories.

| Resource | Upstream source | Release treatment |
|---|---|---|
| CounselBench / CounselBench-Adv | [Official repository](https://github.com/llm-eval-mental-health/CounselBench), [adversarial data](https://huggingface.co/datasets/izi-ano/CounselBench-Adv), [human annotations](https://huggingface.co/datasets/izi-ano/CounselBench-Adv-human-annotation) | Evaluation wrapper retained from the research workspace; data and one-shot examples are obtained separately. Task descriptions/protocol derive from the benchmark. |
| CBT-Bench | [Official repository](https://github.com/mianzhang/CBT-Bench), [dataset](https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench) | Evaluation wrappers retained; upstream constants, data, and human references are obtained separately. Prompt/protocol specifications derive from the benchmark. |
| VERA-MH | [Official repository](https://github.com/SpringCare/VERA-MH) | Only a launcher is included; the framework, rubrics, and personas are installed separately. |

Consult upstream terms before downloading, redistributing, or using those resources. No ownership of upstream benchmark material is claimed. Original evaluation-wrapper file hashes and the corresponding adapted release hashes are recorded in `Blue_team/benchmarks/SOURCES.json`.

License reference: [MIT License text](https://opensource.org/license/mit).
