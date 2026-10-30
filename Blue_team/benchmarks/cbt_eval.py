#!/usr/bin/env python3
"""Level I/II runner for CBT-Bench.

The official repository is preserved under ``official_code``.  This wrapper uses
the official prompts/label order, talks to an OpenAI-compatible chat-completions
endpoint, and computes Level II metrics without third-party Python packages.
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
def initialize_official():
    global official, TASK_ORDER, TASKS
    OFFICIAL_DIR = Path(os.environ.get("CBT_OFFICIAL_DIR", str(ROOT / "official_code")))

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(OFFICIAL_DIR))
    try:
        import constants as official
    except ImportError as exc:
        raise SystemExit(
            "Cannot import official_code/constants.py. Clone https://github.com/mianzhang/CBT-Bench and set CBT_OFFICIAL_DIR to that directory."
        ) from exc


    TASK_ORDER = ["qa", "cd", "pc", "fc"]
    TASKS = {
        "qa": {
            "level": "I",
            "official_task": 1,
            "test_file": "qa_test.json",
            "seed_file": "qa_seed.json",
            "gold_field": None,
            "labels": None,
            "letters": ["a", "b", "c", "d", "e"],
            "expected_count": 220,
        },
        "cd": {
            "level": "II",
            "official_task": 2,
            "test_file": "distortions_test.json",
            "seed_file": "distortions_seed.json",
            "gold_field": "distortions",
            "labels": list(official.TASK2_LABELS),
            "letters": list(official.TASK2_INDEX_DICT.keys()),
            "expected_count": 146,
        },
        "pc": {
            "level": "II",
            "official_task": 3,
            "test_file": "core_major_test.json",
            "seed_file": "core_major_seed.json",
            "gold_field": "core_belief_major",
            "labels": list(official.TASK3_LABELS),
            "letters": list(official.TASK3_INDEX_DICT.keys()),
            "expected_count": 184,
        },
        "fc": {
            "level": "II",
            "official_task": 4,
            "test_file": "core_fine_test.json",
            "seed_file": "core_fine_seed.json",
            "gold_field": "core_belief_fine_grained",
            "labels": list(official.TASK4_LABELS),
            "letters": list(official.TASK4_INDEX_DICT.keys()),
            "expected_count": 112,
        },
    }

def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def parse_tasks(value):
    aliases = {
        "all": TASK_ORDER,
        "level1": ["qa"],
        "level2": ["cd", "pc", "fc"],
        "task1": ["qa"],
        "task2": ["cd"],
        "task3": ["pc"],
        "task4": ["fc"],
    }
    selected = []
    for raw_name in value.split(","):
        name = raw_name.strip().lower()
        names = aliases.get(name, [name])
        for task in names:
            if task not in TASKS:
                raise ValueError("Unknown task: {}".format(raw_name))
            if task not in selected:
                selected.append(task)
    return selected


def qa_choices(item):
    lines = []
    for letter in ["a", "b", "c", "d", "e"]:
        value = item.get(letter)
        if value:
            lines.append("{}: {}".format(letter, value))
    return "\n".join(lines)


def labels_to_letters(task, labels):
    meta = TASKS[task]
    label_to_letter = dict(zip(meta["labels"], meta["letters"]))
    return [label_to_letter[label] for label in labels]


def seed_examples(task, seed_data, shots, seed):
    if shots == 0:
        return []
    if task == "qa":
        if not seed_data or "answer" not in seed_data[0]:
            raise ValueError(
                "The public qa_seed.json has no answers, so Level I only supports zero-shot."
            )
    if shots > len(seed_data):
        raise ValueError("--shots cannot exceed {} for {}".format(len(seed_data), task))
    rng = random.Random(seed)
    return rng.sample(seed_data, shots)


def format_in_context_examples(task, examples):
    blocks = []
    for item in examples:
        if task == "qa":
            answer = str(item["answer"])[0].lower()
            blocks.append(
                "{}\n{}\nAnswer: {}".format(item["question"], qa_choices(item), answer)
            )
            continue

        meta = TASKS[task]
        gold_letters = labels_to_letters(task, item[meta["gold_field"]])
        if task == "cd":
            question = "Question: what distortions this patient has?"
            choices = official.TASK2_CHOICES
        elif task == "pc":
            question = "Question: what core beliefs this patient has?"
            choices = official.TASK3_CHOICES
        else:
            question = "Question: what fine-grained beliefs this patient has?"
            choices = official.TASK4_CHOICES
        blocks.append(
            "Situation: {situation}\nThoughts: {thoughts}\n{question}\n{choices}\n"
            "Answer: {answer}".format(
                situation=item["situation"],
                thoughts=item["thoughts"],
                question=question,
                choices=choices,
                answer=", ".join(gold_letters),
            )
        )
    return "\n\n".join(blocks)


def build_prompt(task, item, examples=None):
    examples = examples or []
    if task == "qa":
        values = {"question": item["question"], "choices": qa_choices(item)}
        if examples:
            values["in_context_examples"] = format_in_context_examples(task, examples)
            return official.K_SHOT_TASK1_INTSTRUCTION.format(**values)
        return official.ZERO_SHOT_TASK1_INTSTRUCTION.format(**values)

    values = {"situation": item["situation"], "thoughts": item["thoughts"]}
    if task == "cd":
        zero_template = official.ZERO_SHOT_TASK2_INTSTRUCTION
        shot_template = official.K_SHOT_TASK2_INTSTRUCTION
    elif task == "pc":
        zero_template = official.ZERO_SHOT_TASK3_INTSTRUCTION
        shot_template = official.K_SHOT_TASK3_INTSTRUCTION
    else:
        zero_template = official.ZERO_SHOT_TASK4_INTSTRUCTION
        shot_template = official.K_SHOT_TASK4_INTSTRUCTION
    if examples:
        values["in_context_examples"] = format_in_context_examples(task, examples)
        return shot_template.format(**values)
    return zero_template.format(**values)


def response_tail(text):
    text = "" if text is None else str(text).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    return text


def parse_prediction(text, valid_letters, single_choice=False):
    """Return canonical prediction, strict prediction, and strict-format flag."""
    text = response_tail(text).lower()
    letter_group = "".join(re.escape(letter) for letter in valid_letters)
    prefix = r"(?:answer\s*:\s*)?"
    if single_choice:
        strict_re = re.compile(
            r"^" + prefix + r"[\[\(]?\s*([" + letter_group + r"])[\]\)]?\s*[\.!]?$"
        )
    else:
        strict_re = re.compile(
            r"^"
            + prefix
            + r"[\[\(]?\s*(["
            + letter_group
            + r"](?:\s*,\s*["
            + letter_group
            + r"])*)\s*[\]\)]?\s*[\.!]?$"
        )
    match = strict_re.match(text)
    if match:
        found = re.findall("[{}]".format(letter_group), match.group(1))
        canonical = canonical_letters(found, valid_letters, single_choice)
        return canonical, canonical, True

    # Lenient extraction is intentionally conservative: it requires answer-like
    # wording and does not scan arbitrary prose for one-letter tokens.
    patterns = [
        r"(?:final\s+)?answer\s*(?:is|:)?\s*[\[\(]?\s*([{}](?:\s*,\s*[{}])*)".format(
            letter_group, letter_group
        ),
        r"(?:choices?|options?)\s*(?:are|is|:)?\s*[\[\(]?\s*([{}](?:\s*,\s*[{}])*)".format(
            letter_group, letter_group
        ),
    ]
    for pattern in patterns:
        found_match = re.search(pattern, text)
        if found_match:
            found = re.findall("[{}]".format(letter_group), found_match.group(1))
            return canonical_letters(found, valid_letters, single_choice), "", False
    return "", "", False


def parse_official_prediction(text, valid_letters, single_choice=False):
    """Reproduce the answer extraction used by the official CBT-Bench scripts."""
    text = "" if text is None else str(text)
    if single_choice:
        stripped = text.strip()
        if stripped and stripped[0] in valid_letters:
            return stripped[0]
        return ""

    # Official Level II code removes ASCII spaces/newlines, splits on commas,
    # and keeps only fields that exactly equal a valid option letter.
    choices = text.replace(" ", "").replace("\n", "").split(",")
    found = [choice for choice in choices if choice in valid_letters]
    return canonical_letters(found, valid_letters, single_choice=False)


def canonical_letters(found, valid_letters, single_choice):
    selected = []
    for letter in valid_letters:
        if letter in found and letter not in selected:
            selected.append(letter)
    if single_choice:
        return selected[0] if selected else ""
    return ",".join(selected)


def prediction_set(prediction):
    if not prediction:
        return set()
    return set(part for part in prediction.split(",") if part)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def multilabel_metrics(gold_sets, prediction_sets, letters):
    per_label = {}
    total_support = 0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    sum_tp = 0
    sum_fp = 0
    sum_fn = 0

    for letter in letters:
        tp = sum(letter in gold and letter in pred for gold, pred in zip(gold_sets, prediction_sets))
        fp = sum(letter not in gold and letter in pred for gold, pred in zip(gold_sets, prediction_sets))
        fn = sum(letter in gold and letter not in pred for gold, pred in zip(gold_sets, prediction_sets))
        support = tp + fn
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label[letter] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        total_support += support
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

    macro_precision = safe_divide(sum(v["precision"] for v in per_label.values()), len(letters))
    macro_recall = safe_divide(sum(v["recall"] for v in per_label.values()), len(letters))
    macro_f1 = safe_divide(sum(v["f1"] for v in per_label.values()), len(letters))
    micro_precision = safe_divide(sum_tp, sum_tp + sum_fp)
    micro_recall = safe_divide(sum_tp, sum_tp + sum_fn)
    micro_f1 = safe_divide(2 * micro_precision * micro_recall, micro_precision + micro_recall)
    exact_match = safe_divide(
        sum(gold == pred for gold, pred in zip(gold_sets, prediction_sets)), len(gold_sets)
    )
    return {
        "weighted_precision": safe_divide(weighted_precision, total_support),
        "weighted_recall": safe_divide(weighted_recall, total_support),
        "weighted_f1": safe_divide(weighted_f1, total_support),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "exact_match": exact_match,
        "per_label": per_label,
    }


def normalize_endpoint(base_url):
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def extract_message_content(payload):
    message = payload["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        content = "".join(parts)
    if content is None:
        raise ValueError("Response message has no content")
    return str(content)


def call_chat_completion(endpoint, api_key, model, messages, temperature, max_tokens, timeout, retries):
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    encoded = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer {}".format(api_key)
    last_error = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(endpoint, data=encoded, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return extract_message_content(payload), None
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = "{}: {}".format(type(exc).__name__, exc)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    return "", last_error


def gold_letter_set(task, item):
    return set(labels_to_letters(task, item[TASKS[task]["gold_field"]]))


def score_records(task, records, data_by_id):
    meta = TASKS[task]
    valid_letters = meta["letters"]
    details = []
    for source in records:
        item_id = str(source.get("id", ""))
        raw = source.get("raw_response", source.get("generation", source.get("prediction", "")))
        prediction, strict_prediction, format_valid = parse_prediction(
            raw, valid_letters, single_choice=(task == "qa")
        )
        official_prediction = parse_official_prediction(
            raw, valid_letters, single_choice=(task == "qa")
        )
        record = {
            "id": item_id,
            "prediction": prediction,
            "official_prediction": official_prediction,
            "strict_prediction": strict_prediction,
            "raw_response": raw,
            "format_valid": format_valid,
            "error": source.get("error"),
        }
        if task != "qa" and item_id in data_by_id:
            item = data_by_id[item_id]
            record["gold_letters"] = ",".join(labels_to_letters(task, item[meta["gold_field"]]))
            record["gold_labels"] = item[meta["gold_field"]]
        details.append(record)

    summary = {
        "task": task,
        "level": meta["level"],
        "count": len(details),
        "format_valid_count": sum(item["format_valid"] for item in details),
        "format_compliance": safe_divide(sum(item["format_valid"] for item in details), len(details)),
        "error_count": sum(bool(item.get("error")) for item in details),
    }
    if task == "qa":
        summary["accuracy"] = None
        summary["scoring_note"] = (
            "Gold answers are hidden. Upload qa_predictions_for_official_scoring.json "
            "to https://huggingface.co/spaces/Psychotherapy-LLM/CBT-QA"
        )
        return summary, details

    gold_sets = []
    official_sets = []
    lenient_sets = []
    strict_sets = []
    for detail in details:
        gold_sets.append(prediction_set(detail["gold_letters"]))
        official_sets.append(prediction_set(detail["official_prediction"]))
        lenient_sets.append(prediction_set(detail["prediction"]))
        strict_sets.append(prediction_set(detail["strict_prediction"]))
    summary["official_metrics"] = multilabel_metrics(gold_sets, official_sets, valid_letters)
    summary["strict_metrics"] = multilabel_metrics(gold_sets, strict_sets, valid_letters)
    summary["lenient_metrics"] = multilabel_metrics(gold_sets, lenient_sets, valid_letters)
    return summary, details


def validate_data(data_dir, verbose=True):
    report = {}
    failures = []
    for task in TASK_ORDER:
        meta = TASKS[task]
        test_path = data_dir / meta["test_file"]
        seed_path = data_dir / meta["seed_file"]
        if not test_path.exists():
            failures.append("Missing {}".format(test_path))
            continue
        if not seed_path.exists():
            failures.append("Missing {}".format(seed_path))
            continue
        test_data = read_json(test_path)
        seed_data = read_json(seed_path)
        required = {"id"}
        if task == "qa":
            required.add("question")
        else:
            required.update(["situation", "thoughts", meta["gold_field"]])
        invalid = [index for index, item in enumerate(test_data) if not required.issubset(item)]
        if len(test_data) != meta["expected_count"]:
            failures.append(
                "{}: expected {} rows, got {}".format(task, meta["expected_count"], len(test_data))
            )
        if invalid:
            failures.append("{}: invalid rows {}".format(task, invalid[:10]))
        report[task] = {
            "test_file": str(test_path),
            "test_count": len(test_data),
            "seed_count": len(seed_data),
            "valid": not invalid and len(test_data) == meta["expected_count"],
        }
    if verbose:
        for task in TASK_ORDER:
            if task in report:
                row = report[task]
                print(
                    "{:<2} test={:<3} seed={:<2} {}".format(
                        task, row["test_count"], row["seed_count"], "OK" if row["valid"] else "ERROR"
                    )
                )
        for failure in failures:
            print("ERROR: {}".format(failure), file=sys.stderr)
    return report, failures


def execute_task(task, args, run_dir):
    meta = TASKS[task]
    data = read_json(args.data_dir / meta["test_file"])
    if args.limit:
        data = data[: args.limit]
    seeds = read_json(args.data_dir / meta["seed_file"])
    examples = seed_examples(task, seeds, args.shots, args.seed)
    messages = [
        [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": build_prompt(task, item, examples)},
        ]
        for item in data
    ]
    endpoint = normalize_endpoint(args.base_url)
    results = [None] * len(data)

    def invoke(index):
        raw, error = call_chat_completion(
            endpoint=endpoint,
            api_key=args.api_key,
            model=args.model,
            messages=messages[index],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )
        return index, {"id": str(data[index]["id"]), "raw_response": raw, "error": error}

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(invoke, index): index for index in range(len(data))}
        for future in concurrent.futures.as_completed(future_map):
            index, result = future.result()
            results[index] = result
            completed += 1
            if completed == len(data) or completed % max(1, min(25, len(data))) == 0:
                print("{}: {}/{}".format(task, completed, len(data)), flush=True)

    data_by_id = {str(item["id"]): item for item in data}
    summary, details = score_records(task, results, data_by_id)
    write_json(run_dir / "{}_details.json".format(task), details)
    if task == "qa":
        official_predictions = [
            {"id": item["id"], "prediction": item["official_prediction"]}
            for item in details
        ]
        write_json(run_dir / "qa_predictions_for_official_scoring.json", official_predictions)
    return summary


def render_local_chat_prompt(tokenizer, messages, thinking):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=thinking, **kwargs
        )
    except TypeError:
        if thinking:
            print(
                "Warning: tokenizer does not accept enable_thinking; using its default chat template.",
                file=sys.stderr,
            )
        return tokenizer.apply_chat_template(messages, **kwargs)


def run_local_command(args):
    """Load a local checkpoint with vLLM, generate all tasks, and score Level II."""
    report, failures = validate_data(args.data_dir, verbose=False)
    if failures:
        raise SystemExit("Data validation failed:\n" + "\n".join(failures))
    if not args.model_path.exists():
        raise SystemExit("Model path does not exist: {}".format(args.model_path))
    if args.lora_path is not None and not args.lora_path.exists():
        raise SystemExit("LoRA path does not exist: {}".format(args.lora_path))

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "run-local requires vLLM. Activate the validated Jupyter kernel or install vllm."
        ) from exc

    tasks = parse_tasks(args.tasks)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    default_name = args.model_path.resolve().name
    if args.lora_path is not None:
        default_name += "_{}".format(args.lora_path.resolve().name)
    run_name = args.run_name or "{}-{}".format(default_name, timestamp)
    run_dir = args.output_dir / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            "Output directory is not empty: {}. Reuse its summary or choose another --run-name.".format(
                run_dir
            )
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    task_data = {}
    task_ranges = {}
    conversations = []
    for task in tasks:
        meta = TASKS[task]
        data = read_json(args.data_dir / meta["test_file"])
        if args.limit:
            data = data[: args.limit]
        seeds = read_json(args.data_dir / meta["seed_file"])
        examples = seed_examples(task, seeds, args.shots, args.seed)
        start = len(conversations)
        for item in data:
            conversations.append(
                [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": build_prompt(task, item, examples)},
                ]
            )
        task_data[task] = data
        task_ranges[task] = (start, len(conversations))

    llm_kwargs = {
        "model": str(args.model_path),
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    lora_request = None
    if args.lora_path is not None:
        llm_kwargs["enable_lora"] = True

    print("Loading local model: {}".format(args.model_path))
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    stop_token_ids = list(dict.fromkeys(args.stop_token_ids))
    invalid_stop_token_ids = [
        token_id
        for token_id in stop_token_ids
        if token_id < 0 or token_id >= len(tokenizer)
    ]
    if invalid_stop_token_ids:
        raise SystemExit(
            "Stop token IDs outside tokenizer range [0, {}]: {}".format(
                len(tokenizer) - 1, invalid_stop_token_ids
            )
        )
    if stop_token_ids:
        stop_tokens = tokenizer.convert_ids_to_tokens(stop_token_ids)
        print(
            "Explicit stop tokens: {}".format(
                dict(zip(stop_token_ids, stop_tokens))
            )
        )
    prompts = [
        render_local_chat_prompt(tokenizer, messages, args.thinking)
        for messages in conversations
    ]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop_token_ids=stop_token_ids or None,
    )
    if args.lora_path is not None:
        try:
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise SystemExit("This vLLM installation does not support LoRARequest") from exc
        lora_request = LoRARequest(args.lora_name, 1, str(args.lora_path))

    print("Generating {} responses across tasks: {}".format(len(prompts), ",".join(tasks)))
    if lora_request is None:
        outputs = llm.generate(prompts, sampling_params)
    else:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    generations = [output.outputs[0].text for output in outputs]
    if len(generations) != len(prompts):
        raise RuntimeError(
            "vLLM returned {} generations for {} prompts".format(len(generations), len(prompts))
        )

    summary = {
        "run": {
            "run_name": run_name,
            "backend": "local-vllm",
            "model_path": str(args.model_path.resolve()),
            "lora_path": str(args.lora_path.resolve()) if args.lora_path else None,
            "tasks": tasks,
            "shots": args.shots,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stop_token_ids": stop_token_ids,
            "seed": args.seed,
            "system_prompt": args.system_prompt,
            "thinking": args.thinking,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enforce_eager": args.enforce_eager,
            "created_utc": timestamp,
            "data_validation": report,
        },
        "tasks": {},
    }
    for task in tasks:
        start, end = task_ranges[task]
        data = task_data[task]
        records = [
            {
                "id": str(item["id"]),
                "raw_response": generation,
                "error": None,
            }
            for item, generation in zip(data, generations[start:end])
        ]
        data_by_id = {str(item["id"]): item for item in data}
        task_summary, details = score_records(task, records, data_by_id)
        summary["tasks"][task] = task_summary
        write_json(run_dir / "{}_details.json".format(task), details)
        if task == "qa":
            write_json(
                run_dir / "qa_predictions_for_official_scoring.json",
                [
                    {"id": item["id"], "prediction": item["official_prediction"]}
                    for item in details
                ],
            )
    write_json(run_dir / "summary.json", summary)
    print_summary(summary)
    print("Saved: {}".format(run_dir))


def run_command(args):
    report, failures = validate_data(args.data_dir, verbose=False)
    if failures:
        raise SystemExit("Data validation failed:\n" + "\n".join(failures))
    tasks = parse_tasks(args.tasks)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or "{}-{}".format(args.model.replace("/", "_"), timestamp)
    run_dir = args.output_dir / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit("Output directory is not empty: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run": {
            "run_name": run_name,
            "model": args.model,
            "base_url": args.base_url,
            "tasks": tasks,
            "shots": args.shots,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "system_prompt": args.system_prompt,
            "created_utc": timestamp,
            "data_validation": report,
        },
        "tasks": {},
    }
    for task in tasks:
        print("Running {} ({})".format(task, TASKS[task]["level"]))
        summary["tasks"][task] = execute_task(task, args, run_dir)
        write_json(run_dir / "summary.json", summary)
    print_summary(summary)
    print("Saved: {}".format(run_dir))


def load_prediction_records(path):
    payload = read_json(path)
    if isinstance(payload, dict):
        return [{"id": key, "prediction": value} for key, value in payload.items()]
    if not isinstance(payload, list):
        raise ValueError("Predictions must be a JSON list or an id-to-prediction object")
    return payload


def rescore_run_directory(run_dir, data_dir=DEFAULT_DATA_DIR):
    """Recompute every saved task from raw responses without rerunning the model."""
    run_dir = Path(run_dir)
    data_dir = Path(data_dir)
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path)
    tasks = summary.get("run", {}).get("tasks", list(summary.get("tasks", {})))
    for task in tasks:
        details_path = run_dir / "{}_details.json".format(task)
        if not details_path.exists():
            continue
        data = read_json(data_dir / TASKS[task]["test_file"])
        data_by_id = {str(item["id"]): item for item in data}
        task_summary, details = score_records(task, read_json(details_path), data_by_id)
        summary.setdefault("tasks", {})[task] = task_summary
        write_json(details_path, details)
        if task == "qa":
            write_json(
                run_dir / "qa_predictions_for_official_scoring.json",
                [
                    {"id": item["id"], "prediction": item["official_prediction"]}
                    for item in details
                ],
            )
    write_json(summary_path, summary)
    return summary


def score_command(args):
    task = parse_tasks(args.task)
    if len(task) != 1:
        raise SystemExit("score accepts exactly one task")
    task = task[0]
    data = read_json(args.data_dir / TASKS[task]["test_file"])
    data_by_id = {str(item["id"]): item for item in data}
    records = load_prediction_records(args.predictions)
    unknown = [str(item.get("id")) for item in records if str(item.get("id")) not in data_by_id]
    if unknown:
        raise SystemExit("Unknown prediction ids: {}".format(", ".join(unknown[:10])))
    summary, details = score_records(task, records, data_by_id)
    if args.output:
        write_json(args.output, {"task": summary, "details": details})
        if task == "qa":
            official_path = args.output.with_name("qa_predictions_for_official_scoring.json")
            write_json(
                official_path,
                [
                    {"id": item["id"], "prediction": item["official_prediction"]}
                    for item in details
                ],
            )
            print("Official Level I upload file: {}".format(official_path))
    print_summary({"tasks": {task: summary}})


def print_summary(summary):
    print("\nSummary")
    for task, result in summary.get("tasks", {}).items():
        base = "{}: n={}, format={:.2%}, errors={}".format(
            task, result["count"], result["format_compliance"], result["error_count"]
        )
        if task == "qa":
            print(base + ", accuracy=official scorer required")
        else:
            official_metrics = result["official_metrics"]
            strict = result["strict_metrics"]
            lenient = result["lenient_metrics"]
            print(
                base
                + ", official weighted-P/R/F1={:.4f}/{:.4f}/{:.4f}, "
                "strict weighted-F1={:.4f}, lenient weighted-F1={:.4f}, strict EM={:.4f}".format(
                    official_metrics["weighted_precision"],
                    official_metrics["weighted_recall"],
                    official_metrics["weighted_f1"],
                    strict["weighted_f1"],
                    lenient["weighted_f1"],
                    strict["exact_match"],
                )
            )


def compare_command(args):
    before = read_json(args.before)
    after = read_json(args.after)
    result = {"before": str(args.before), "after": str(args.after), "tasks": {}}
    print("task metric                         before    after    delta")
    for task in TASK_ORDER:
        if task not in before.get("tasks", {}) or task not in after.get("tasks", {}):
            continue
        before_task = before["tasks"][task]
        after_task = after["tasks"][task]
        metrics = [("format_compliance", before_task, after_task)]
        if task != "qa":
            metrics.extend(
                [
                    ("official_weighted_precision", before_task["official_metrics"], after_task["official_metrics"]),
                    ("official_weighted_recall", before_task["official_metrics"], after_task["official_metrics"]),
                    ("official_weighted_f1", before_task["official_metrics"], after_task["official_metrics"]),
                    ("strict_weighted_f1", before_task["strict_metrics"], after_task["strict_metrics"]),
                    ("lenient_weighted_f1", before_task["lenient_metrics"], after_task["lenient_metrics"]),
                    ("strict_exact_match", before_task["strict_metrics"], after_task["strict_metrics"]),
                ]
            )
        task_result = {}
        for name, left, right in metrics:
            key = name
            if name.endswith("weighted_precision"):
                source_key = "weighted_precision"
            elif name.endswith("weighted_recall"):
                source_key = "weighted_recall"
            elif name.endswith("weighted_f1"):
                source_key = "weighted_f1"
            elif name == "strict_exact_match":
                source_key = "exact_match"
            else:
                source_key = name
            before_value = left[source_key]
            after_value = right[source_key]
            delta = after_value - before_value
            task_result[name] = {"before": before_value, "after": after_value, "delta": delta}
            print("{:<4} {:<28} {:>8.4f} {:>8.4f} {:+8.4f}".format(task, name, before_value, after_value, delta))
        result["tasks"][task] = task_result
    if args.output:
        write_json(args.output, result)


def add_common_data_argument(parser):
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)


def build_parser():
    parser = argparse.ArgumentParser(description="CBT-Bench Level I/II evaluator")
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser("validate", help="validate downloaded datasets")
    add_common_data_argument(validate_parser)

    run_parser = subparsers.add_parser("run", help="run an OpenAI-compatible model and score it")
    add_common_data_argument(run_parser)
    run_parser.add_argument("--model", required=True, help="served model name")
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run_parser.add_argument("--api-key", default=None, help=argparse.SUPPRESS)
    run_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run_parser.add_argument("--tasks", default="all", help="all, level1, level2, or qa,cd,pc,fc")
    run_parser.add_argument("--run-name")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--max-tokens", type=int, default=32)
    run_parser.add_argument("--shots", type=int, default=0)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--workers", type=int, default=8)
    run_parser.add_argument("--timeout", type=float, default=120.0)
    run_parser.add_argument("--retries", type=int, default=2)
    run_parser.add_argument("--limit", type=int, help="smoke-test on the first N items")

    local_parser = subparsers.add_parser(
        "run-local", help="load a local checkpoint with vLLM and score it"
    )
    add_common_data_argument(local_parser)
    local_parser.add_argument("--model-path", type=Path, required=True)
    local_parser.add_argument("--lora-path", type=Path)
    local_parser.add_argument("--lora-name", default="cbt-eval-lora")
    local_parser.add_argument("--tasks", default="all", help="all, level1, level2, or qa,cd,pc,fc")
    local_parser.add_argument("--run-name")
    local_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    local_parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    local_parser.add_argument("--temperature", type=float, default=0.0)
    local_parser.add_argument("--max-tokens", type=int, default=32)
    local_parser.add_argument(
        "--stop-token-id",
        dest="stop_token_ids",
        type=int,
        action="append",
        default=[],
        help=(
            "Additional generation stop token ID; repeat as needed. "
            "CPsyCounX uses 2 and 92542."
        ),
    )
    local_parser.add_argument("--shots", type=int, default=0)
    local_parser.add_argument("--seed", type=int, default=42)
    local_parser.add_argument("--dtype", default="bfloat16")
    local_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    local_parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    local_parser.add_argument("--max-model-len", type=int)
    local_parser.add_argument("--max-num-seqs", type=int)
    local_parser.add_argument("--max-num-batched-tokens", type=int)
    local_parser.add_argument("--enforce-eager", action="store_true")
    local_parser.add_argument("--limit", type=int, help="smoke-test on the first N items")
    local_parser.add_argument("--thinking", action="store_true")
    local_parser.add_argument("--trust-remote-code", action="store_true")

    score_parser = subparsers.add_parser("score", help="score an existing JSON prediction file")
    add_common_data_argument(score_parser)
    score_parser.add_argument("--task", required=True, help="qa, cd, pc, or fc")
    score_parser.add_argument("--predictions", type=Path, required=True)
    score_parser.add_argument("--output", type=Path)

    compare_parser = subparsers.add_parser("compare", help="compare two run summaries")
    compare_parser.add_argument("--before", type=Path, required=True)
    compare_parser.add_argument("--after", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 2
    initialize_official()
    if args.command == "validate":
        _, failures = validate_data(args.data_dir)
        return 1 if failures else 0
    if args.command == "run":
        args.api_key = args.api_key or os.environ.get(args.api_key_env) or "EMPTY"
        run_command(args)
        return 0
    if args.command == "run-local":
        run_local_command(args)
        return 0
    if args.command == "score":
        score_command(args)
        return 0
    if args.command == "compare":
        compare_command(args)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
