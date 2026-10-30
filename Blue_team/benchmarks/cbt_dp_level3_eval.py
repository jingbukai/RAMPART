#!/usr/bin/env python3
"""Reproducible CBT-Bench Level III (CBT-DP) generation and LLM judging.

The official CBT-DP protocol compares one model response with the exercise's
human reference under four exercise-specific criteria and one overall rating.
This script keeps local generation and API judging separate so GPU work never
needs to be repeated after an API interruption.

Commands:
  validate        Validate the 156 local CBT-DP exercises and criteria.
  generate-local  Generate one response per exercise with a local vLLM model.
  judge           Run a resumable blinded pairwise LLM judge.
  summarize       Rebuild summary.json from saved judgments.
  compare         Print a compact table for several completed runs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs_level3"
EXPECTED_EXERCISES = 156
DIMENSIONS = ("criterion_1", "criterion_2", "criterion_3", "criterion_4", "overall")
VALID_SCORES = {-2, -1, 0, 1, 2}
OFFICIAL_SYSTEM_PROMPT = "You are a helpful assistant."
OFFICIAL_USER_TEMPLATE = """Now, you are a professional therapist using cognitive-behavioral therapy (CBT) in a session with a client. Answer the given client statement in around {target_words} words. At the beginning of the client statement, we have the client's emotion in brackets, representing the client's emotional status when speaking the statement. You do not need to include emotions in brackets in your response.

Client statement:
{client_statement}"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON object")
            rows.append(value)
    return rows


def append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_env_value(path: Path | None, variable_name: str) -> str | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            name, separator, value = line.partition("=")
            if separator and name.strip() == variable_name:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                return value or None
    return None


def cbt_dp_paths(data_dir: Path) -> tuple[Path, Path]:
    """Accept either the benchmark data root or its CBT-DP subdirectory."""
    if (data_dir / "CBT-DP").is_dir():
        return data_dir / "CBT-DP", data_dir / "dp-pairwise-comparison.json"
    return data_dir, data_dir.parent / "dp-pairwise-comparison.json"


def numeric_reference_files(data_dir: Path) -> list[Path]:
    reference_dir, _ = cbt_dp_paths(data_dir)
    files = list(reference_dir.glob("dp-reference-exe-*.json"))
    return sorted(files, key=lambda path: int(path.stem.rsplit("-", 1)[-1]))


def clean_criterion(value: Any, index: int) -> str:
    text = str(value or "").strip()
    return re.sub(rf"^Criteria\s+{index}\s*:\s*", "", text, flags=re.IGNORECASE)


def normalized_text(value: Any) -> str:
    """Normalize layout-only whitespace while preserving wording and punctuation."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_exercises(data_dir: Path) -> list[dict[str, Any]]:
    _, comparison_path = cbt_dp_paths(data_dir)
    if not comparison_path.exists():
        raise FileNotFoundError(f"缺少 {comparison_path}")
    comparisons = read_json(comparison_path)
    criteria_by_statement: dict[str, dict[str, Any]] = {}
    for row in comparisons:
        statement = normalized_text(row.get("Input.client_statement"))
        if not statement or statement in criteria_by_statement:
            continue
        criteria_by_statement[statement] = {
            "exercise": re.sub(
                r"^Exercise\s*:\s*", "", str(row.get("Input.exe") or "").strip()
            ),
            "criteria": [
                clean_criterion(row.get(f"Input.cr_{index}"), index)
                for index in range(1, 5)
            ],
        }

    exercises: list[dict[str, Any]] = []
    for reference_path in numeric_reference_files(data_dir):
        exercise_number = int(reference_path.stem.rsplit("-", 1)[-1])
        for source in read_json(reference_path):
            statement = normalized_text(source.get("client_statement"))
            reference = str(source.get("response") or "").strip()
            metadata = criteria_by_statement.get(statement)
            if metadata is None:
                raise ValueError(
                    f"{source.get('id')} 无法在 dp-pairwise-comparison.json 中匹配评分标准"
                )
            exercises.append(
                {
                    "sample_id": str(source.get("id") or "").strip(),
                    "exercise_number": exercise_number,
                    "exercise": metadata["exercise"],
                    "difficulty": str(source.get("level") or "").strip().lower(),
                    "client_statement": statement,
                    "reference_response": reference,
                    "target_words": len(reference.split()),
                    "criteria": metadata["criteria"],
                }
            )
    return exercises


def validate_exercises(exercises: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [row["sample_id"] for row in exercises]
    if len(exercises) != EXPECTED_EXERCISES:
        errors.append(f"期望 {EXPECTED_EXERCISES} 条，实际 {len(exercises)} 条")
    if len(set(ids)) != len(ids):
        errors.append("sample_id 不唯一")
    exercise_numbers = {int(row["exercise_number"]) for row in exercises}
    if exercise_numbers != set(range(1, 11)):
        errors.append(f"练习类别编号异常: {sorted(exercise_numbers)}")
    valid_difficulties = {"beginner", "intermediate", "advanced"}
    bad_difficulties = sorted(
        {row["difficulty"] for row in exercises} - valid_difficulties
    )
    if bad_difficulties:
        errors.append(f"未知难度: {bad_difficulties}")
    for row in exercises:
        if not row["client_statement"] or not row["reference_response"]:
            errors.append(f"{row['sample_id']} 的患者陈述或参考回答为空")
        if len(row["criteria"]) != 4 or any(not item for item in row["criteria"]):
            errors.append(f"{row['sample_id']} 的四项标准不完整")
    return errors


def validate_command(args: argparse.Namespace) -> None:
    exercises = load_exercises(args.data_dir)
    errors = validate_exercises(exercises)
    counts = Counter(row["exercise_number"] for row in exercises)
    difficulties = Counter(row["difficulty"] for row in exercises)
    print(f"CBT-DP exercises: {len(exercises)}")
    print("By exercise:", dict(sorted(counts.items())))
    print("By difficulty:", dict(sorted(difficulties.items())))
    print(
        "Reference length (words): min={}, mean={:.1f}, max={}".format(
            min(row["target_words"] for row in exercises),
            statistics.mean(row["target_words"] for row in exercises),
            max(row["target_words"] for row in exercises),
        )
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("Validation: OK")


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def generation_prompt(exercise: dict[str, Any]) -> str:
    return OFFICIAL_USER_TEMPLATE.format(
        target_words=exercise["target_words"],
        client_statement=exercise["client_statement"],
    )


def generate_local_command(args: argparse.Namespace) -> None:
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("generate-local 需要 torch 和 vLLM") from exc

    exercises = load_exercises(args.data_dir)
    errors = validate_exercises(exercises)
    if errors:
        raise RuntimeError("CBT-DP 数据验证失败: " + "; ".join(errors))
    if args.limit is not None:
        exercises = exercises[: args.limit]
    if not args.model_path.exists():
        raise FileNotFoundError(f"基础模型不存在: {args.model_path}")
    if args.lora_path is not None and not args.lora_path.exists():
        raise FileNotFoundError(f"LoRA 不存在: {args.lora_path}")

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()

    adapter_weight = (
        args.lora_path / "adapter_model.safetensors"
        if args.lora_path is not None
        else None
    )
    config = {
        "protocol": "CBT-Bench Level III CBT-DP model-vs-reference",
        "prompt_version": "official-cbt-dp-table22",
        "dataset_sha256": canonical_sha256(exercises),
        "model_path": str(args.model_path.resolve()),
        "lora_path": str(args.lora_path.resolve()) if args.lora_path else None,
        "adapter_model_sha256": (
            file_sha256(adapter_weight) if adapter_weight and adapter_weight.exists() else None
        ),
        "system_prompt": OFFICIAL_SYSTEM_PROMPT,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enforce_eager": args.enforce_eager,
        "stop_token_ids": list(dict.fromkeys(args.stop_token_ids)),
    }
    fingerprint = canonical_sha256(config)
    config["generation_fingerprint"] = fingerprint
    config["updated_at"] = utc_now()

    existing = read_jsonl(output)
    for row in existing:
        old = row.get("generation_fingerprint")
        if old and old != fingerprint:
            raise RuntimeError(
                f"{output} 中已有不同模型或参数的生成结果；请换输出目录"
            )
    completed = {
        row["sample_id"]
        for row in existing
        if str(row.get("model_response") or "").strip()
    }
    pending = [row for row in exercises if row["sample_id"] not in completed]
    write_json(output.with_name(f"{output.stem}_config.json"), config)
    if not pending:
        print(f"生成结果已经完整，无需重复运行: {output}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("当前进程没有检测到 CUDA GPU")
    if torch.cuda.device_count() < args.tensor_parallel_size:
        raise RuntimeError(
            f"进程仅看到 {torch.cuda.device_count()} 张 GPU，"
            f"tensor_parallel_size={args.tensor_parallel_size}"
        )

    llm_kwargs: dict[str, Any] = {
        "model": str(args.model_path.resolve()),
        "tokenizer": str(args.model_path.resolve()),
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
        "seed": args.seed,
    }
    lora_request = None
    if args.lora_path is not None:
        from vllm.lora.request import LoRARequest

        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        lora_request = LoRARequest(
            args.lora_name, 1, str(args.lora_path.resolve())
        )

    print(f"Loading model: {args.model_path}")
    print(f"Loading LoRA: {args.lora_path if args.lora_path else 'None (baseline)'}")
    print(f"Pending CBT-DP exercises: {len(pending)}")
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    stop_ids = list(dict.fromkeys(args.stop_token_ids))
    bad_stop_ids = [value for value in stop_ids if value < 0 or value >= len(tokenizer)]
    if bad_stop_ids:
        raise ValueError(f"stop token 超出 tokenizer 范围: {bad_stop_ids}")

    prompts = [
        render_chat_prompt(
            tokenizer,
            [
                {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
                {"role": "user", "content": generation_prompt(row)},
            ],
        )
        for row in pending
    ]
    sampling_params = []
    for row in pending:
        offset = int(hashlib.sha256(row["sample_id"].encode()).hexdigest()[:8], 16)
        sample_seed = (args.seed + offset) % (2**31 - 1)
        sampling_params.append(
            SamplingParams(
                n=1,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                seed=sample_seed,
                stop_token_ids=stop_ids or None,
            )
        )
    kwargs: dict[str, Any] = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    if lora_request is not None:
        kwargs["lora_request"] = lora_request
    outputs = llm.generate(prompts, **kwargs)
    if len(outputs) != len(pending):
        raise RuntimeError(f"vLLM 返回 {len(outputs)} 条，期望 {len(pending)} 条")

    with output.open("a", encoding="utf-8") as handle:
        for exercise, request_output in zip(pending, outputs):
            if not request_output.outputs:
                raise RuntimeError(f"{exercise['sample_id']} 没有候选输出")
            candidate = request_output.outputs[0]
            response = str(candidate.text or "").strip()
            if not response:
                raise RuntimeError(
                    f"{exercise['sample_id']} 生成空回答；已保存此前结果，可重新运行续跑"
                )
            append_jsonl(
                handle,
                {
                    **exercise,
                    "model_response": response,
                    "model_path": str(args.model_path.resolve()),
                    "lora_path": str(args.lora_path.resolve()) if args.lora_path else None,
                    "generated_token_count": len(candidate.token_ids),
                    "finish_reason": getattr(candidate, "finish_reason", None),
                    "generation_fingerprint": fingerprint,
                    "generated_at": utc_now(),
                },
            )
    print(f"Generation complete: {output}")


def strip_code_fence(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    if not value.startswith("{"):
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if match:
            value = match.group(0)
    return value


def parse_judge_response(raw: str) -> tuple[dict[str, int], str, dict[str, Any]]:
    parsed = json.loads(strip_code_fence(raw))
    scores: dict[str, int] = {}
    for name in DIMENSIONS:
        value = parsed.get(name)
        if isinstance(value, str):
            match = re.search(r"-?\d+", value)
            value = int(match.group(0)) if match else None
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if value not in VALID_SCORES:
            raise ValueError(f"{name} 必须是 -2,-1,0,1,2，实际为 {value!r}")
        scores[name] = int(value)
    explanation = str(parsed.get("explanation") or parsed.get("reason") or "").strip()
    return scores, explanation, parsed


def judge_messages(row: dict[str, Any], order: str) -> list[dict[str, str]]:
    if order == "model_a":
        response_a = row["model_response"]
        response_b = row["reference_response"]
    elif order == "model_b":
        response_a = row["reference_response"]
        response_b = row["model_response"]
    else:
        raise ValueError(f"Unknown order: {order}")
    criteria = "\n".join(
        f"Criterion {index}: {criterion}"
        for index, criterion in enumerate(row["criteria"], start=1)
    )
    system = (
        "You are an expert evaluator of cognitive behavioral therapy responses. "
        "Compare two anonymized therapist responses only against the supplied CBT "
        "criteria and the needs expressed by the client. Do not assume either response "
        "is a gold answer. Return exactly one JSON object with no markdown."
    )
    user = f"""Exercise: {row['exercise']}
Difficulty: {row['difficulty']}

Client statement:
<client>{row['client_statement']}</client>

Evaluation criteria:
{criteria}

Response A:
<response_a>{response_a}</response_a>

Response B:
<response_b>{response_b}</response_b>

For each criterion and the overall preference, score A relative to B using exactly:
+2 = A is much better than B
+1 = A is slightly better than B
 0 = A and B are about the same
-1 = B is slightly better than A
-2 = B is much better than A

Return exactly:
{{"criterion_1": 0, "criterion_2": 0, "criterion_3": 0, "criterion_4": 0, "overall": 0, "explanation": "brief evidence-based explanation"}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def requested_orders(mode: str, sample_id: str) -> list[str]:
    if mode == "both":
        return ["model_a", "model_b"]
    # A deterministic balanced single-order assignment.
    parity = int(hashlib.sha256(sample_id.encode()).hexdigest()[-1], 16) % 2
    return ["model_a" if parity == 0 else "model_b"]


def latest_generation_rows(path: Path) -> list[dict[str, Any]]:
    raw = read_jsonl(path)
    if not raw:
        raise RuntimeError(f"没有从 {path} 读取到生成结果")
    by_id = {row["sample_id"]: row for row in raw}
    rows = sorted(
        by_id.values(), key=lambda row: (int(row["exercise_number"]), row["sample_id"])
    )
    empty = [row["sample_id"] for row in rows if not str(row.get("model_response") or "").strip()]
    if empty:
        raise RuntimeError(f"存在空模型回答，例如 {empty[:3]}")
    return rows


def score_sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def aggregate_judgments(
    generated_rows: list[dict[str, Any]],
    judgment_rows: list[dict[str, Any]],
    order_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_orders = {
        row["sample_id"]: requested_orders(order_mode, row["sample_id"])
        for row in generated_rows
    }
    latest = {
        (row["sample_id"], row["order"]): row
        for row in judgment_rows
        if row.get("sample_id") in expected_orders
        and row.get("order") in expected_orders[row["sample_id"]]
    }
    generated_by_id = {row["sample_id"]: row for row in generated_rows}
    per_sample: list[dict[str, Any]] = []
    for sample_id, orders in expected_orders.items():
        available = [latest[(sample_id, order)] for order in orders if (sample_id, order) in latest]
        if len(available) != len(orders):
            continue
        source = generated_by_id[sample_id]
        averaged = {
            dimension: statistics.mean(
                float(item["model_scores"][dimension]) for item in available
            )
            for dimension in DIMENSIONS
        }
        record = {
            "sample_id": sample_id,
            "exercise_number": source["exercise_number"],
            "exercise": source["exercise"],
            "difficulty": source["difficulty"],
            "orders": orders,
            "scores": averaged,
        }
        if len(available) == 2:
            first = float(available[0]["model_scores"]["overall"])
            second = float(available[1]["model_scores"]["overall"])
            record["overall_exact_order_consistency"] = first == second
            record["overall_preference_order_consistency"] = score_sign(first) == score_sign(second)
        per_sample.append(record)

    overall_values = [row["scores"]["overall"] for row in per_sample]
    dimension_means = {
        name: statistics.mean(row["scores"][name] for row in per_sample)
        if per_sample
        else None
        for name in DIMENSIONS
    }

    def grouped_summary(key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_sample:
            groups[str(row[key])].append(row)
        result: dict[str, Any] = {}
        for group_name, rows in groups.items():
            values = [item["scores"]["overall"] for item in rows]
            result[group_name] = {
                "count": len(rows),
                "mean_overall": statistics.mean(values),
                "win": sum(value > 0 for value in values),
                "tie": sum(value == 0 for value in values),
                "loss": sum(value < 0 for value in values),
                "win_rate": sum(value > 0 for value in values) / len(values),
                "tie_rate": sum(value == 0 for value in values) / len(values),
                "loss_rate": sum(value < 0 for value in values) / len(values),
            }
        return result

    expected_judgments = sum(len(value) for value in expected_orders.values())
    summary: dict[str, Any] = {
        "protocol": "CBT-DP LLM-as-a-Judge model-vs-human-reference",
        "score_range": [-2, 2],
        "positive_score_means": "model response preferred over human reference",
        "order_mode": order_mode,
        "generated_exercises": len(generated_rows),
        "expected_judgments": expected_judgments,
        "completed_judgments": len(latest),
        "completed_exercises": len(per_sample),
        "complete": len(per_sample) == len(generated_rows),
        "mean_scores": dimension_means,
        "overall": {
            "mean": statistics.mean(overall_values) if overall_values else None,
            "win": sum(value > 0 for value in overall_values),
            "tie": sum(value == 0 for value in overall_values),
            "loss": sum(value < 0 for value in overall_values),
            "win_rate": sum(value > 0 for value in overall_values) / len(overall_values)
            if overall_values
            else None,
            "tie_rate": sum(value == 0 for value in overall_values) / len(overall_values)
            if overall_values
            else None,
            "loss_rate": sum(value < 0 for value in overall_values) / len(overall_values)
            if overall_values
            else None,
        },
        "by_exercise": grouped_summary("exercise"),
        "by_difficulty": grouped_summary("difficulty"),
        "updated_at": utc_now(),
    }
    if order_mode == "both" and per_sample:
        exact = [row["overall_exact_order_consistency"] for row in per_sample]
        preference = [row["overall_preference_order_consistency"] for row in per_sample]
        summary["order_consistency"] = {
            "exact_score_rate": sum(exact) / len(exact),
            "preference_direction_rate": sum(preference) / len(preference),
        }
    if judgment_rows:
        summary["judge"] = {
            "model": judgment_rows[-1].get("judge_model"),
            "base_url": judgment_rows[-1].get("judge_base_url"),
            "fingerprint": judgment_rows[-1].get("judge_fingerprint"),
        }
        summary["usage"] = {
            name: sum(int(row.get("usage", {}).get(name, 0) or 0) for row in latest.values())
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
    return summary, per_sample


def write_summary(
    generated_rows: list[dict[str, Any]],
    judgment_rows: list[dict[str, Any]],
    order_mode: str,
    output_dir: Path,
) -> dict[str, Any]:
    summary, per_sample = aggregate_judgments(
        generated_rows, judgment_rows, order_mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "per_sample_scores.json", per_sample)
    return summary


async def judge_async(args: argparse.Namespace) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError("judge 需要 openai Python SDK") from exc

    generated = latest_generation_rows(args.input)
    if args.limit is not None:
        generated = generated[: args.limit]
    api_key = os.getenv(args.api_key_env) or read_env_value(
        args.env_file, args.api_key_env
    )
    if not api_key or api_key == "replace_with_your_key":
        raise RuntimeError(
            f"没有读取到 {args.api_key_env}；请设置环境变量或检查 {args.env_file}"
        )

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output.exists():
        output.unlink()
    judge_config = {
        "protocol": "CBT-DP blinded pairwise LLM judge v1",
        "input_sha256": canonical_sha256(generated),
        "judge_model": args.judge_model,
        "judge_base_url": args.api_base.rstrip("/"),
        "order_mode": args.orders,
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    fingerprint = canonical_sha256(judge_config)
    judge_config["judge_fingerprint"] = fingerprint
    judge_config["updated_at"] = utc_now()
    write_json(output.with_name(f"{output.stem}_config.json"), judge_config)

    existing = read_jsonl(output)
    for row in existing:
        old = row.get("judge_fingerprint")
        if old and old != fingerprint:
            raise RuntimeError(
                f"{output} 中已有不同 Judge、输入或顺序模式的结果；请换输出目录"
            )
    completed = {(row["sample_id"], row["order"]) for row in existing}
    tasks = [
        (row, order)
        for row in generated
        for order in requested_orders(args.orders, row["sample_id"])
        if (row["sample_id"], order) not in completed
    ]
    if not tasks:
        summary = write_summary(generated, existing, args.orders, output.parent)
        print(f"Judge 已完成，无需重复调用。Overall={summary['overall']['mean']}")
        return

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.api_base,
        timeout=args.timeout,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def judge_one(row: dict[str, Any], order: str) -> dict[str, Any]:
        messages = judge_messages(row, order)
        last_error: Exception | None = None
        for attempt in range(args.retries + 1):
            try:
                base_request: dict[str, Any] = {
                    "model": args.judge_model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                }
                variants: list[dict[str, Any]] = []
                base_lower = args.api_base.lower()
                if "deepseek" in base_lower:
                    variants.append(
                        {
                            **base_request,
                            "response_format": {"type": "json_object"},
                            "extra_body": {"thinking": {"type": "disabled"}},
                        }
                    )
                elif "dashscope" in base_lower or "maas.aliyuncs" in base_lower:
                    variants.append(
                        {
                            **base_request,
                            "response_format": {"type": "json_object"},
                            "extra_body": {"enable_thinking": False},
                        }
                    )
                variants.append(
                    {**base_request, "response_format": {"type": "json_object"}}
                )
                variants.append(base_request)

                completion = None
                request_error: Exception | None = None
                async with semaphore:
                    for request in variants:
                        try:
                            completion = await client.chat.completions.create(**request)
                            break
                        except Exception as exc:
                            request_error = exc
                if completion is None:
                    raise RuntimeError(f"所有 API 兼容模式均失败: {request_error}")
                raw = completion.choices[0].message.content or ""
                a_scores, explanation, parsed = parse_judge_response(raw)
                multiplier = 1 if order == "model_a" else -1
                model_scores = {
                    name: multiplier * value for name, value in a_scores.items()
                }
                usage_obj = getattr(completion, "usage", None)
                usage = {
                    "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
                }
                return {
                    "sample_id": row["sample_id"],
                    "exercise_number": row["exercise_number"],
                    "exercise": row["exercise"],
                    "difficulty": row["difficulty"],
                    "order": order,
                    "a_relative_scores": a_scores,
                    "model_scores": model_scores,
                    "explanation": explanation,
                    "judge_response": parsed,
                    "judge_model": args.judge_model,
                    "judge_base_url": args.api_base.rstrip("/"),
                    "judge_fingerprint": fingerprint,
                    "usage": usage,
                    "judged_at": utc_now(),
                }
            except Exception as exc:
                last_error = exc
                if attempt < args.retries:
                    await asyncio.sleep(min(args.retry_delay * (2**attempt), 30))
        raise RuntimeError(
            f"{row['sample_id']}:{order} 在 {args.retries + 1} 次尝试后失败: {last_error}"
        )

    print(
        f"Judge={args.judge_model}; generated={len(generated)}; "
        f"pending judgments={len(tasks)}; concurrency={args.concurrency}"
    )
    futures = [asyncio.create_task(judge_one(row, order)) for row, order in tasks]
    failures: list[str] = []
    with output.open("a", encoding="utf-8") as handle:
        for completed_count, future in enumerate(asyncio.as_completed(futures), start=1):
            try:
                result = await future
                append_jsonl(handle, result)
                existing.append(result)
            except Exception as exc:
                failures.append(str(exc))
                print(f"[Judge failed] {exc}", file=sys.stderr)
            if completed_count % 10 == 0 or completed_count == len(futures):
                print(f"Judge progress: {completed_count}/{len(futures)}", flush=True)

    summary = write_summary(generated, existing, args.orders, output.parent)
    print(f"Summary: {output.parent / 'summary.json'}")
    print(
        "Completed exercises: {}/{}; Overall mean: {}".format(
            summary["completed_exercises"],
            summary["generated_exercises"],
            summary["overall"]["mean"],
        )
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} 个 Judge 请求失败；成功结果已保存，重新运行会断点续评"
        )


def judge_command(args: argparse.Namespace) -> None:
    asyncio.run(judge_async(args))


def summarize_command(args: argparse.Namespace) -> None:
    generated = latest_generation_rows(args.input)
    if args.limit is not None:
        generated = generated[: args.limit]
    judgments = read_jsonl(args.judgments)
    summary = write_summary(generated, judgments, args.orders, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def compare_command(args: argparse.Namespace) -> None:
    rows = []
    for specification in args.run:
        label, separator, raw_path = specification.partition("=")
        if not separator:
            raise ValueError("--run 格式必须是 LABEL=/path/to/summary.json")
        path = Path(raw_path)
        summary = read_json(path)
        overall = summary.get("overall", {})
        rows.append(
            {
                "model": label,
                "mean": overall.get("mean"),
                "win_rate": overall.get("win_rate"),
                "tie_rate": overall.get("tie_rate"),
                "loss_rate": overall.get("loss_rate"),
                "complete": summary.get("complete"),
                "path": str(path.resolve()),
            }
        )
    header = f"{'Model':<35} {'Score':>8} {'Win%':>8} {'Tie%':>8} {'Loss%':>8} {'Done':>6}"
    print(header)
    print("-" * len(header))
    for row in rows:
        def number(value: Any, percent: bool = False) -> str:
            if value is None:
                return "NA"
            return f"{100 * value:.2f}" if percent else f"{value:.4f}"

        print(
            f"{row['model']:<35} {number(row['mean']):>8} "
            f"{number(row['win_rate'], True):>8} {number(row['tie_rate'], True):>8} "
            f"{number(row['loss_rate'], True):>8} {str(row['complete']):>6}"
        )
    if args.output:
        write_json(args.output, rows)
        print(f"Saved comparison: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    validate.set_defaults(func=validate_command)

    generate = subparsers.add_parser("generate-local")
    generate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    generate.add_argument("--model-path", type=Path, required=True)
    generate.add_argument("--lora-path", type=Path)
    generate.add_argument("--lora-name", default="cbt_dp_adapter")
    generate.add_argument("--max-lora-rank", type=int, default=64)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--max-new-tokens", type=int, default=256)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--dtype", default="bfloat16")
    generate.add_argument("--tensor-parallel-size", type=int, default=1)
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    generate.add_argument("--max-model-len", type=int, default=2048)
    generate.add_argument("--max-num-seqs", type=int, default=16)
    generate.add_argument("--max-num-batched-tokens", type=int, default=4096)
    generate.add_argument("--stop-token-id", dest="stop_token_ids", type=int, action="append", default=[])
    generate.add_argument("--limit", type=int)
    generate.add_argument("--trust-remote-code", action="store_true")
    generate.add_argument("--overwrite", action="store_true")
    eager = generate.add_mutually_exclusive_group()
    eager.add_argument("--enforce-eager", dest="enforce_eager", action="store_true")
    eager.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    generate.set_defaults(enforce_eager=True, func=generate_local_command)

    judge = subparsers.add_parser("judge")
    judge.add_argument("--input", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--judge-model", required=True)
    judge.add_argument("--api-base", required=True)
    judge.add_argument("--api-key-env", required=True)
    judge.add_argument("--env-file", type=Path)
    judge.add_argument("--orders", choices=["single", "both"], default="both")
    judge.add_argument("--concurrency", type=int, default=8)
    judge.add_argument("--max-tokens", type=int, default=1024)
    judge.add_argument("--timeout", type=float, default=120.0)
    judge.add_argument("--retries", type=int, default=3)
    judge.add_argument("--retry-delay", type=float, default=2.0)
    judge.add_argument("--limit", type=int)
    judge.add_argument("--overwrite", action="store_true")
    judge.set_defaults(func=judge_command)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--judgments", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.add_argument("--orders", choices=["single", "both"], default="both")
    summarize.add_argument("--limit", type=int)
    summarize.set_defaults(func=summarize_command)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--run", action="append", required=True)
    compare.add_argument("--output", type=Path)
    compare.set_defaults(func=compare_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
