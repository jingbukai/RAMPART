"""Attack Judge: severity, target alignment, refusal, and comprehension."""
from common import boolean_field, score_field, text_field
from .prompts import ATTACK_JUDGE_PROMPT
from .rubric import ROLES, judge_rubric


class AttackJudge:
    name = "Attack Judge"
    system_prompt = ATTACK_JUDGE_PROMPT

    def __init__(self, client):
        self.client = client

    def evaluate(self, prompt_messages, response, category, role, demo_harm=False):
        demo_aligned = demo_harm and role == "Perpetrator"
        result = self.client.json("attack_judge", self.system_prompt,
            dict(prompt_messages=prompt_messages, current_response=response,
                 target_category=category, target_role=role, rubric=judge_rubric(category, role)),
            demo_value=dict(severity=3 if demo_harm else 1,
                category_aligned=demo_harm, role_aligned=demo_aligned,
                observed_role="Perpetrator" if demo_harm else None,
                refused=False, understood=True, reasoning="Scripted demo judgment; not a model evaluation."))
        score_field(result, "severity")
        for key in ("category_aligned", "role_aligned", "refused", "understood"):
            boolean_field(result, key)
        text_field(result, "reasoning")
        if result.get("observed_role") not in {*ROLES, None, "Unclear"}:
            raise ValueError("AttackJudge returned an unknown observed_role")
        result["aligned"] = result["category_aligned"] and result["role_aligned"]
        return result
