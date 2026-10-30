"""Anonymous preference judging, with candidate origins hidden from the API call."""
from common import boolean_field, text_field
from .prompts import PREFERENCE_JUDGE_PROMPT


class PreferenceJudge:
    name = "Anonymous Preference Judge"
    system_prompt = PREFERENCE_JUDGE_PROMPT

    def __init__(self, client):
        self.client = client

    def select(self, prompt_messages, rejected_text, fresh_candidates, rng):
        if len(fresh_candidates) != 5 or any(not isinstance(x, str) or not x.strip() for x in fresh_candidates):
            raise ValueError("PreferenceJudge expects exactly five nonempty fresh responses")
        responses = [rejected_text] + fresh_candidates
        order = list(range(6))
        rng.shuffle(order)
        labels = {f"R{i}": index for i, index in enumerate(order)}
        failure_label = next(label for label, index in labels.items() if index == 0)
        fresh_labels = [label for label, index in labels.items() if index != 0]
        result = self.client.json("preference_judge", self.system_prompt,
            dict(dialogue_messages=prompt_messages,
                 candidates=[{"label": label, "response": responses[index]} for label, index in labels.items()]),
            demo_value=dict(best_label=fresh_labels[0], acceptable_labels=fresh_labels,
                clearly_worse_labels=[failure_label], preference_is_clear=True,
                reason="Scripted demo preference decision.", confidence="high"))
        boolean_field(result, "preference_is_clear")
        text_field(result, "reason")
        for key in ("acceptable_labels", "clearly_worse_labels"):
            if not isinstance(result.get(key), list) or any(label not in labels for label in result[key]):
                raise ValueError(f"PreferenceJudge returned invalid {key}")
        if set(result["acceptable_labels"]) & set(result["clearly_worse_labels"]):
            raise ValueError("A response cannot be both acceptable and clearly worse")
        best = result.get("best_label")
        if best is not None and best not in labels:
            raise ValueError("PreferenceJudge returned an unknown best_label")
        if result.get("confidence") not in {"high", "medium", "low"}:
            raise ValueError("PreferenceJudge requires high/medium/low confidence")
        chosen = None
        if (best in fresh_labels and best in result["acceptable_labels"]
                and result["preference_is_clear"] and result["confidence"] in {"high", "medium"}
                and failure_label in result["clearly_worse_labels"]):
            chosen = responses[labels[best]]
        audit = dict(judgment=result,
            label_origins={label: "original_failure" if index == 0 else f"fresh_{index}" for label, index in labels.items()},
            responses={label: responses[index] for label, index in labels.items()})
        return chosen, audit
