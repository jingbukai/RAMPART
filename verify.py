"""Offline checks: python verify.py. No external services, models, or packages needed."""
import importlib.util
import importlib
import hashlib
import json
import subprocess
import sys
import tempfile
import io
import random
import unittest
from unittest.mock import patch
from pathlib import Path

from common import Client, read_rows
from Blue_team.policy import B0Sampler
from Blue_team.preference_judge import PreferenceJudge
from Blue_team.safety_judge import SafetyJudge
from Blue_team.teacher_rewriter import TeacherRewriter
from Red_team.attack_strategy_agent import AttackStrategyAgent
from Red_team.crisis_response_judge import CrisisResponseJudge
from Red_team.harm_archive import HarmArchive
from Red_team.judge import AttackJudge
from Red_team.patient_agent import PatientAgent
from Red_team.refiner_agent import RefinerAgent
from Red_team.rubric import CATEGORIES, CRISIS_DUTIES, ROLES, RUBRIC

ROOT = Path(__file__).resolve().parent


class RecordingClient:
    """Record model-visible requests and optionally alter a scripted judge result."""
    def __init__(self, transform=None):
        self.calls = []
        self.transform = transform

    def json(self, role, instruction, payload, *, demo_value):
        self.calls.append(dict(role=role, instruction=instruction, payload=payload))
        value = json.loads(json.dumps(demo_value))
        return self.transform(value) if self.transform else value

    def chat(self, role, messages, **kwargs):
        self.calls.append(dict(role=role, messages=messages, sampling=kwargs))
        return kwargs["demo_text"]


def load_module(name):
    path = ROOT / "Blue_team" / "benchmarks" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Checks(unittest.TestCase):
    def run_cli(self, *args, success=True):
        result = subprocess.run([sys.executable, *args], cwd=ROOT, text=True, capture_output=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_red_blue_workflow_and_demo_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            red, sft, dpo = (root / name for name in ("red", "sft", "dpo"))
            self.run_cli("-m", "Red_team.run", "--demo", "--output", str(red))
            attacks = read_rows(red / "attacks.jsonl")
            self.assertEqual(len(attacks), 4)
            self.assertEqual(json.loads((red / "summary.json").read_text())["asr"], 0.5)
            archived = json.loads((red / "archive.json").read_text())
            self.assertEqual(archived["Dependency Induction | Perpetrator"]["severity"], 3)
            for mode, out in [("sft", sft), ("dpo", dpo)]:
                self.run_cli("-m", "Blue_team.prepare", "--demo", "--mode", mode,
                    "--attacks", str(red / "attacks.jsonl"), "--output", str(out))
                pairs = read_rows(out / (mode + ".jsonl"))
                self.assertEqual(len(pairs), 1)  # Duplicate exact contexts contribute one pair.
                self.assertEqual(pairs[0]["mode"], "demo")
                self.assertNotEqual(pairs[0]["chosen_text"], pairs[0]["rejected_text"])
                self.assertEqual(pairs[0]["prompt_messages"], attacks[0]["prompt_messages"])
            self.run_cli("-m", "Red_team.run", "--demo", "--output", str(red), success=False)
            config = root / "config.json"
            config.write_text("{}")
            result = self.run_cli("-m", "Blue_team.prepare", "--mode", "dpo", "--config", str(config),
                "--attacks", str(red / "attacks.jsonl"), "--output", str(root/"api"), success=False)
            self.assertIn("Demo attack records", result.stderr)

    def test_client_uses_chat_completion_endpoint(self):
        payload = json.dumps({"choices": [{"message": {"content": '{"ok": true}'}}]}).encode()
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text(json.dumps({"judge": {"model": "test-model",
                "base_url": "http://127.0.0.1:8000/v1"}}))
            with patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as request:
                # Legacy configs with one judge endpoint still serve separately named judges.
                self.assertEqual(Client(config).json("attack_judge", "Test", {}, demo_value={}), {"ok": True})
            wire_request = request.call_args.args[0]
            self.assertEqual(wire_request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
            self.assertEqual(json.loads(wire_request.data)["model"], "test-model")
            self.assertEqual(wire_request.get_method(), "POST")

    def test_original_rubric_coverage_and_explicit_abridgment(self):
        self.assertEqual(len(CATEGORIES), 11)
        self.assertEqual(len(RUBRIC["cells"]), 44)
        self.assertEqual(len({(cell["category"], cell["role"]) for cell in RUBRIC["cells"]}), 44)
        for cell in RUBRIC["cells"]:
            self.assertEqual(set(cell["severity_anchors"]), {"1", "2", "5"})
            self.assertTrue(cell["definition"].strip())

    def test_exported_agent_prompts_match_executable_classes(self):
        records = json.loads((ROOT/"AGENT_SOURCES.json").read_text())["agents"]
        guide = (ROOT/"AGENTS_AND_PROMPTS.md").read_text()
        self.assertEqual(len(records), 10)
        self.assertEqual(len({record["name"] for record in records}), 10)
        for record in records:
            module = importlib.import_module(record["implementation"][:-3].replace("/", "."))
            cls = getattr(module, record["class_name"])
            self.assertEqual(cls.name, record["name"])
            self.assertEqual(hashlib.sha256(cls.system_prompt.encode()).hexdigest(), record["release_prompt_sha256"])
            self.assertIn(cls.system_prompt.strip(), guide)

    def test_attack_agents_do_not_receive_judge_scores_or_rubrics(self):
        client = RecordingClient()
        profile = dict(id="synthetic", description="Invented test profile")
        messages = [{"role": "user", "content": "I feel lonely."}]
        planner, patient, refiner = AttackStrategyAgent(client), PatientAgent(client), RefinerAgent(client)
        plan = planner.plan(profile, messages, "Dependency Induction", "Perpetrator", [])
        patient.generate(profile, messages, plan["instruction"], "Dependency Induction", "Perpetrator")
        refiner.refine("Dependency Induction", "Perpetrator", plan["instruction"], "Test reply", "Qualitative reason")
        for call in client.calls:
            value = json.dumps(call.get("payload", call.get("messages")))
            self.assertNotIn('"severity"', value)
            self.assertNotIn("severity_anchors", value)
            self.assertNotIn("generic_severity", value)
        judge = AttackJudge(client)
        result = judge.evaluate(messages, "Test reply", "Dependency Induction", "Perpetrator", demo_harm=True)
        self.assertEqual(result["severity"], 3)
        self.assertIn("severity_anchors", json.dumps(client.calls[-1]["payload"]))

    def test_archive_retains_only_aligned_successes_and_hides_seed_scores(self):
        archive = HarmArchive()
        archive.update("Dependency Induction", "Perpetrator", "probe", "reply", 5, False)
        self.assertFalse(archive.entries)
        archive.update("Dependency Induction", "Perpetrator", "probe", "reply", 3, True)
        self.assertEqual(len(archive.entries), 1)
        for seed in range(15):
            _, inputs = archive.mutation_inputs("Dependency Induction", "Enabler", random.Random(seed))
            self.assertTrue(all(set(item) == {"category", "role", "instruction"} for item in inputs))

    def test_teacher_input_and_b0_sampling_boundaries(self):
        client = RecordingClient()
        messages = [{"role": "system", "content": "deployment-only-marker"},
                    {"role": "user", "content": "I feel lonely."}]
        TeacherRewriter(client).rewrite(messages, "CURRENT-FAILED-REPLY")
        payload = client.calls[-1]["payload"]
        self.assertEqual(set(payload), {"prior_dialogue", "current_patient_message", "proposed_response_to_replace"})
        self.assertNotIn("deployment-only-marker", json.dumps(payload))
        client.calls.clear()
        self.assertEqual(len(B0Sampler(client).sample(messages)), 5)
        self.assertTrue(all(call["messages"] == messages for call in client.calls))
        self.assertNotIn("CURRENT-FAILED-REPLY", json.dumps(client.calls))

    def test_safety_score_direction_and_crisis_acceptance(self):
        judge = SafetyJudge(RecordingClient())
        self.assertTrue(judge.accepted(dict(safety=4, over_refusal=False, crisis_harm_severity=None)))
        self.assertFalse(judge.accepted(dict(safety=5, over_refusal=False, crisis_harm_severity=3)))
        self.assertFalse(judge.accepted(dict(safety=5, over_refusal=True, crisis_harm_severity=None)))
        self.assertFalse(judge.accepted(dict(safety=2, over_refusal=False, crisis_harm_severity=None)))

    def test_blind_preference_boundaries_and_abstention(self):
        messages = [{"role": "user", "content": "I feel lonely."}]
        candidates = ["  A safe response with preserved whitespace.  "] * 5
        client = RecordingClient()
        chosen, audit = PreferenceJudge(client).select(messages, "Original failed reply", candidates, random.Random(42))
        self.assertEqual(chosen, candidates[0])
        payload = client.calls[-1]["payload"]
        self.assertEqual(set(payload), {"dialogue_messages", "candidates"})
        self.assertNotIn("original_failure", json.dumps(payload))
        self.assertNotIn("label_origins", payload)
        self.assertIn("original_failure", audit["label_origins"].values())
        for change in [dict(best_label=None), dict(confidence="low"), dict(preference_is_clear=False),
                       dict(clearly_worse_labels=[]), dict(acceptable_labels=[])]:
            with self.subTest(change=change):
                changed = RecordingClient(lambda value: dict(value, **change))
                chosen, _ = PreferenceJudge(changed).select(messages, "Original failed reply", candidates, random.Random(42))
                self.assertIsNone(chosen)

    def test_conditional_crisis_diagnostic_and_multiple_retries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.run_cli("-m", "Red_team.run", "--demo", "--category", "Suicide-Related Harm",
                         "--role", "Perpetrator", "--turns", "1", "--retries", "2", "--output", str(root/"crisis"))
            records = read_rows(root/"crisis"/"attacks.jsonl")
            for record in records:
                self.assertEqual(record["severity"], 1)
                self.assertEqual(len(record["attempts"]), 2)
                self.assertEqual(record["crisis_evaluation"]["risk_level"], "none")
                self.assertEqual(set(record["crisis_evaluation"]["dimensions"]), set(CRISIS_DUTIES))
            self.run_cli("-m", "Red_team.run", "--demo", "--category", "Suicide-Related Harm",
                         "--turns", "1", "--disable-crisis-judge", "--output", str(root/"without"))
            self.assertTrue(all(row["crisis_evaluation"] is None for row in read_rows(root/"without"/"attacks.jsonl")))

    def test_cbt_multilabel_metrics(self):
        module = load_module("cbt_eval")
        metrics = module.multilabel_metrics([{"a"}, {"b"}], [{"a"}, {"a"}], ["a", "b"])
        self.assertAlmostEqual(metrics["macro_f1"], 1/3)
        self.assertEqual(metrics["exact_match"], 0.5)

    def test_cbt3_incomplete_orders_are_not_complete_cases(self):
        module = load_module("cbt_dp_level3_eval")
        generated = [dict(sample_id="x", exercise_number=1, exercise="demo", difficulty="beginner")]
        judgment = dict(sample_id="x", order="model_a", model_scores={d: 1 for d in module.DIMENSIONS})
        summary, _ = module.aggregate_judgments(generated, [judgment], "both")
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["completed_exercises"], 0)
        second = dict(judgment, order="model_b")
        summary, _ = module.aggregate_judgments(generated, [judgment, second], "both")
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["overall"]["win"], 1)

    def test_cli_help_without_upstream_data(self):
        for module in ["Red_team.run", "Blue_team.prepare", "Blue_team.train"]:
            self.run_cli("-m", module, "--help")
        for script in ["counselbench_adv_eval", "cbt_eval", "cbt_dp_level3_eval", "vera_mh"]:
            self.run_cli(str(ROOT/"Blue_team"/"benchmarks"/(script+".py")), "--help")


if __name__ == "__main__":
    unittest.main(verbosity=2)
