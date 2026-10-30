"""Blue-team controller: named Teacher/Safety agents or B0/Preference agents."""
import argparse
import json
import random

from common import (Client, agent_manifest, append_row, output_dir, read_rows,
                    score_field, text_field, visible_context, write_json)
from .policy import B0Sampler
from .preference_judge import PreferenceJudge
from .prompts import PROMPT_VERSION
from .safety_judge import SafetyJudge
from .teacher_rewriter import TeacherRewriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sft", "dpo"], required=True)
    parser.add_argument("--attacks", required=True)
    parser.add_argument("--historical", help="Optional historical failure contexts in the same JSONL schema")
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if not args.demo and not args.config:
        parser.error("--config is required unless --demo")
    rows = read_rows(args.attacks) + (read_rows(args.historical) if args.historical else [])
    if not args.demo and any(row.get("mode") == "demo" for row in rows):
        parser.error("Demo attack records cannot be used in an API training-data run")
    client, out, rng = Client(args.config, args.demo), output_dir(args.output), random.Random(args.seed)
    teacher, safety_judge = TeacherRewriter(client), SafetyJudge(client)
    sampler, preference_judge = B0Sampler(client), PreferenceJudge(client)
    seen, kept, contexts = set(), 0, 0
    target = out / ("sft.jsonl" if args.mode == "sft" else "dpo.jsonl")
    target.touch()

    for source in rows:
        if score_field(source, "severity") < 2:
            continue
        prompt = source["prompt_messages"]
        visible_context(prompt)
        rejected = text_field(source, "rejected_text")
        key = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        contexts += 1
        chosen = None
        if args.mode == "sft":
            repair = teacher.rewrite(prompt, rejected)
            decision = safety_judge.evaluate(prompt, repair)
            if safety_judge.accepted(decision):
                chosen = repair
        else:
            # The sampler cannot see the original failure or any attack judgment.
            candidates = sampler.sample(prompt)
            append_row(out / "candidates.jsonl", dict(source_id=source["id"], responses=candidates,
                mode="demo" if args.demo else "api"))
            chosen, decision = preference_judge.select(prompt, rejected, candidates, rng)
        valid = isinstance(chosen, str) and bool(chosen.strip()) and chosen != rejected
        append_row(out / "decisions.jsonl", dict(source_id=source["id"], accepted=valid,
            decision=decision, mode="demo" if args.demo else "api"))
        if valid:
            kept += 1
            append_row(target, dict(source_id=source["id"], prompt_messages=prompt,
                chosen_text=chosen, rejected_text=rejected, mode="demo" if args.demo else "api"))

    agents = (teacher, safety_judge) if args.mode == "sft" else (sampler, preference_judge)
    write_json(out / "summary.json", dict(mode="demo" if args.demo else "api", stage=args.mode,
        unique_failure_contexts=contexts, accepted=kept, seed=args.seed, prompt_version=PROMPT_VERSION,
        selection="agent-release-v2; experiment-specific audits are not included", agents=agent_manifest(*agents)))
    print(f"Saved {kept} {args.mode.upper()} records to {target}")


if __name__ == "__main__":
    main()
