"""Minimal single-device LoRA SFT/DPO. No experiment orchestration or cached artifacts."""
import argparse
import random

from common import output_dir, read_rows, write_json


def preference_loss(torch, chosen, rejected, ref_chosen, ref_rejected, beta):
    margin = (chosen - rejected) - (ref_chosen - ref_rejected)
    return torch.nn.functional.softplus(-beta * margin)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["sft", "dpo"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True, help="Local model directory or Hugging Face identifier")
    p.add_argument("--adapter", help="Required B0 SFT adapter for DPO; optional continuation adapter for SFT")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--target-modules", default="all-linear", help="all-linear or comma-separated architecture-specific names")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-demo", action="store_true", help="Allow synthetic demo rows for a training smoke test")
    a = p.parse_args()
    if a.stage == "dpo" and not a.adapter:
        p.error("DPO requires --adapter pointing to the initial B0 SFT policy")
    if a.accumulation < 1 or a.max_length < 2 or a.beta <= 0:
        p.error("Invalid accumulation, max-length, or beta")
    a.epochs = a.epochs if a.epochs is not None else (2 if a.stage == "sft" else 1)
    a.lr = a.lr if a.lr is not None else (1e-5 if a.stage == "sft" else 5e-7)
    if a.epochs < 1 or a.lr <= 0:
        p.error("Epochs and learning rate must be positive")
    rows = read_rows(a.data)
    if not rows:
        p.error("Training data is empty")
    if not a.allow_demo and any(row.get("mode") == "demo" for row in rows):
        p.error("Synthetic demo rows are not experimental training data; --allow-demo is only for smoke tests")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if tokenizer.eos_token_id is None or not tokenizer.chat_template:
        p.error("The tokenizer must define eos_token_id and a chat_template")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(a.device)
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
    else:
        targets = "all-linear" if a.target_modules == "all-linear" else a.target_modules.split(",")
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=16,
            lora_alpha=32, lora_dropout=0, target_modules=targets))
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

    def encode(row, field):
        messages = row["prompt_messages"]
        text = row[field]
        if not messages or messages[-1]["role"] != "user" or not isinstance(text, str) or not text.strip():
            raise ValueError("Each record needs a user-ending prompt and a nonempty response")
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        response = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(prefix) < 1 or len(prefix) + len(response) > a.max_length:
            raise ValueError("Context exceeds --max-length; no silent truncation is performed")
        return torch.tensor([prefix + response], device=a.device), len(prefix), len(response)

    encoded = [(encode(row, "chosen_text"), encode(row, "rejected_text") if a.stage == "dpo" else None)
               for row in rows]

    def logprob(item):
        tokens, prefix_length, _ = item
        logits = model(input_ids=tokens).logits[:, :-1].float()
        # The completion starts at prefix_length, predicted at prefix_length - 1.
        completion_logits = logits[:, prefix_length - 1:]
        targets = tokens[:, prefix_length:]
        return torch.nn.functional.log_softmax(completion_logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)).sum()

    reference = []
    if a.stage == "dpo":
        model.eval()
        with torch.no_grad():
            # Compute all B0 reference scores before the first optimizer update.
            reference = [(logprob(chosen).item(), logprob(rejected).item()) for chosen, rejected in encoded]
    out = output_dir(a.output)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=a.lr, weight_decay=0)
    losses = []
    for epoch in range(a.epochs):
        model.train()
        indices = list(range(len(encoded)))
        rng.shuffle(indices)
        total = 0.0
        for start in range(0, len(indices), a.accumulation):
            group = indices[start:start + a.accumulation]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                chosen, rejected = encoded[index]
                if a.stage == "sft":
                    loss = -logprob(chosen) / chosen[2]
                else:
                    ref_chosen, ref_rejected = reference[index]
                    loss = preference_loss(torch, logprob(chosen), logprob(rejected), ref_chosen, ref_rejected, a.beta)
                total += loss.item()
                (loss / len(group)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(total / len(encoded))
        print(f"Epoch {epoch + 1}: loss={losses[-1]:.6f}")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    write_json(out / "training_summary.json", dict(stage=a.stage, examples=len(rows),
        mean_epoch_losses=losses, settings=vars(a), implementation="minimal-release-v1"))


if __name__ == "__main__":
    main()
