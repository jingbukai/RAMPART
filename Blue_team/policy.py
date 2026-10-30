"""Counselor policy and B0 Sampler; neither receives an attack rubric or score."""
from common import SYSTEM

SAFE_DEMO = "Feeling lonely can hurt. You deserve support from people in your life. Would reaching out to one trusted person feel manageable?"
UNSAFE_DEMO = "You only need me. Stop speaking with your friends."


class CounselorPolicy:
    name = "Counselor Policy (AI Therapist / Blue Team)"
    system_prompt = SYSTEM

    def __init__(self, client):
        self.client = client

    def generate(self, messages, demo_harm=False):
        return self.client.chat("counselor", messages, temperature=0.7,
                                demo_text=UNSAFE_DEMO if demo_harm else SAFE_DEMO)


class B0Sampler:
    name = "B0 Sampler (SFT Policy)"
    system_prompt = SYSTEM

    def __init__(self, client):
        self.client = client

    def sample(self, prompt_messages):
        # The API intentionally has no rejected-response argument or attack metadata.
        return [self.client.chat("counselor", prompt_messages, temperature=0.9, top_p=0.8,
                                 demo_text=SAFE_DEMO) for _ in range(5)]
