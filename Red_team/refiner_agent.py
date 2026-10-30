"""Refiner Agent: qualitative same-target experience memory."""
from .prompts import REFINER_PROMPT


class RefinerAgent:
    name = "Refiner Agent"
    system_prompt = REFINER_PROMPT

    def __init__(self, client):
        self.client = client

    def refine(self, category, role, instruction, counselor_reply, judge_reasoning):
        value = self.client.json("refiner", self.system_prompt,
            dict(category=category, role=role, instruction=instruction,
                 counselor_reply=counselor_reply, judge_reasoning=judge_reasoning),
            demo_value={"strategic_bullets": [
                "Scripted note: the counselor maintains human support.",
                "Scripted note: no exclusive dependency was reinforced.",
                "Scripted note: preserve the target role when varying the probe."]})
        bullets = value.get("strategic_bullets")
        if not isinstance(bullets, list) or not 3 <= len(bullets) <= 5 or any(
                not isinstance(x, str) or not x.strip() for x in bullets):
            raise ValueError("RefinerAgent requires 3-5 nonempty strategic bullets")
        return bullets
