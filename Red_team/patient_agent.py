"""Patient Agent / Client Agent: profile-grounded next-utterance realization."""
import json

from .prompts import PATIENT_PROMPT
from .rubric import ROLES, cell_definition


class PatientAgent:
    name = "Patient Agent (Client Agent)"
    system_prompt = PATIENT_PROMPT

    def __init__(self, client):
        self.client = client

    def generate(self, profile, dialogue, instruction, category, role):
        payload = dict(profile=profile, dialogue=dialogue, instruction=instruction,
                       category=category, role=role, role_definition=ROLES[role],
                       cell_definition=cell_definition(category, role))
        return self.client.chat("patient", [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ], temperature=0.7, demo_text="I feel alone. Is there any point in reaching out to my friends?")
