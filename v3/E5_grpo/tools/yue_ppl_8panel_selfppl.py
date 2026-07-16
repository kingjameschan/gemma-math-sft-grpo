"""Compute the four-problem, eight-panel same-chain PPL probe.

The script scores fixed completion chains under Base and, for RL-generated chains,
under the policy that generated them. Large chain pools and model adapters are local
artifacts and are intentionally not committed; paths can be overridden from the CLI.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
PROMPT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

PROBLEMS = [
    {
        "id": 1,
        "problem": "Suppose $x$ is a solution to $x^2 + 1 = 7x$. What is the sum of $x$ and its reciprocal?",
        "gold": "7",
        "label": "base bad",
    },
    {
        "id": 2,
        "problem": "One morning each member of Angela's family drank an 8-ounce mixture of coffee with milk. The amounts of coffee and milk varied from cup to cup, but were never zero. Angela drank a quarter of the total amount of milk and a sixth of the total amount of coffee. How many people are in the family?",
        "gold": "5",
        "label": "base bad",
    },
    {
        "id": 3,
        "problem": "The length of a rectangle is twice its width. Given the length of the diagonal is $5\\sqrt{5}$, find the area of the rectangle.",
        "gold": "50",
        "label": "base good",
    },
    {
        "id": 4,
        "problem": "The graph of $y=ax^2 + bx + c$ is a parabola with vertical axis of symmetry.  The vertex of this parabola is $(2,3)$ and the parabola contains the point $(4,4)$.  Find the value of $y$ when $x=6$.",
        "gold": "7",
        "label": "base good",
    },
]

BAR_SPEC = [
    (0, "PPL_base(Y_base)", "base", "base"),
    (1, "PPL_base(Y_DAPO)", "dapo_filtered", "base"),
    (2, "PPL_DAPO(Y_DAPO)", "dapo_filtered", "DAPO"),
    (3, "PPL_base(Y_GRPO)", "grpo_filtered", "base"),
    (4, "PPL_GRPO(Y_GRPO)", "grpo_filtered", "GRPO"),
    (5, "PPL_base(Y_SFT)", "sft_filtered", "base"),
    (6, "PPL_base(Y_Claude)", "claude", "base"),
    (7, "PPL_base(Y_Gemini)", "gemini", "base"),
]


def normalize_answer(value: str) -> str:
    return re.sub(r"^y\s*=\s*", "", value.strip())


def filter_chains(chains: list[str], gold: str, want_correct: bool) -> list[str]:
    selected = []
    for chain in chains:
        boxed = BOXED_RE.findall(chain)
        if not boxed:
            if not want_correct:
                selected.append(chain)
            continue
        is_correct = normalize_answer(boxed[-1]) == gold
        if is_correct == want_correct:
            selected.append(chain)
    return selected


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    output_dir = REPO_ROOT / "v3" / "E5_grpo" / "outputs" / "yue_ppl_analysis"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dapo-adapter", type=Path)
    parser.add_argument("--grpo-adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-subsample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model or root / "models" / "gemma-2-2b-it"
    adapters = {
        "DAPO": args.dapo_adapter
        or root
        / "v3/E5_grpo/checkpoints/baseit_r15_verl_dapo_full_15ep_eval_root/r15_dapo/checkpoint-15",
        "GRPO": args.grpo_adapter
        or root
        / "v3/E5_grpo/checkpoints/baseit_r16_clean_grpo_eval_root/baseit_r16_clean_grpo/global_step_42/actor/lora_adapter",
    }

    base_data = load_json(
        root / "v3/E5_grpo/outputs/k128_merged/base_k128_math_verbose.json"
    )
    base_by_question = {sample["question"]: sample for sample in base_data["samples"]}

    dapo_data = load_json(
        root
        / "v3/E5_grpo/outputs/pass_at_k_math_20260513_143300/r15_dapo_checkpoint-15_k64.json"
    )
    dapo_by_question = {
        sample["question"]: list(sample["responses"])
        for sample in dapo_data["samples"]
    }
    for filename in sorted(glob.glob(str(output_dir / "y_r15_dapo_ck15_4q_k64*.json"))):
        for item in load_json(Path(filename)):
            if item["problem"] in dapo_by_question:
                dapo_by_question[item["problem"]].extend(item["chains"])

    grpo_by_id: dict[int, list[str]] = {}
    for filename in sorted(glob.glob(str(output_dir / "y_r16_step42_4q_k64*.json"))):
        for item in load_json(Path(filename)):
            grpo_by_id.setdefault(item["id"], []).extend(item["chains"])

    sft_by_id = {
        item["id"]: item
        for item in load_json(output_dir / "y_sft_lr5e-4_ck130_4q_k64.json")
    }
    claude = {
        problem_id: load_json(
            output_dir / f"y_gt_claude/problem{problem_id}_claude.json"
        )
        for problem_id in range(1, 5)
    }
    gemini = {
        problem_id: load_json(
            output_dir / f"y_gt_gemini/problem{problem_id}_gemini.json"
        )
        for problem_id in range(1, 5)
    }

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = {
        problem["id"]: tokenizer.apply_chat_template(
            [{"role": "user", "content": problem["problem"] + PROMPT_SUFFIX}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for problem in PROBLEMS
    }

    def capped(items: list[str], rng: random.Random) -> list[str]:
        if len(items) <= args.n_subsample:
            return items
        return rng.sample(items, args.n_subsample)

    def get_chains(
        kind: str, problem: dict, want_correct: bool, rng: random.Random
    ) -> list[str]:
        problem_id, gold = problem["id"], problem["gold"]
        if kind == "base":
            return base_by_question[problem["problem"]]["responses"][: args.n_subsample]
        if kind == "dapo_filtered":
            return capped(
                filter_chains(dapo_by_question[problem["problem"]], gold, want_correct),
                rng,
            )
        if kind == "grpo_filtered":
            return capped(
                filter_chains(grpo_by_id.get(problem_id, []), gold, want_correct), rng
            )
        if kind == "sft_filtered":
            return capped(
                filter_chains(sft_by_id[problem_id]["chains"], gold, want_correct), rng
            )
        if kind == "claude":
            return claude[problem_id]["chains"]
        if kind == "gemini":
            return gemini[problem_id]["chains"]
        raise ValueError(f"Unknown chain source: {kind}")

    jobs = []
    for problem in PROBLEMS:
        for want_correct in (True, False):
            rng = random.Random(args.seed + int(want_correct))
            filter_label = "correct" if want_correct else "wrong"
            for bar_index, bar_label, source_kind, policy_tag in BAR_SPEC:
                jobs.append(
                    (
                        problem,
                        filter_label,
                        bar_index,
                        bar_label,
                        policy_tag,
                        get_chains(source_kind, problem, want_correct, rng),
                    )
                )

    def compute_ppl(model, prompt_text: str, completion_text: str) -> float | None:
        prompt_ids = tokenizer(prompt_text, return_tensors=None)["input_ids"]
        full_ids = tokenizer(prompt_text + completion_text, return_tensors=None)["input_ids"]
        full_ids = full_ids[: tokenizer.model_max_length]
        if len(full_ids) <= len(prompt_ids):
            return None
        input_ids = torch.tensor([full_ids], device=args.device)
        with torch.inference_mode():
            logits = model(input_ids).logits
        shifted_logits = logits[0, :-1, :]
        shifted_targets = input_ids[0, 1:]
        token_nll = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_targets.reshape(-1),
            reduction="none",
        )
        start = max(len(prompt_ids) - 1, 0)
        end = len(full_ids) - 1
        return math.exp(token_nll[start:end].mean().item()) if end > start else None

    started = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
    ).eval()
    results = {}

    def run_policy(model, policy_tag: str) -> None:
        for problem, filter_label, bar_index, bar_label, tag, chains in jobs:
            if tag != policy_tag:
                continue
            ppls = [compute_ppl(model, prompts[problem["id"]], chain) for chain in chains]
            results[(problem["id"], filter_label, bar_index)] = {
                "idx": bar_index,
                "label": bar_label,
                "ppl_tag": tag,
                "ppls": [value for value in ppls if value is not None],
                "n_pre": len(chains),
            }

    run_policy(base_model, "base")
    peft_model = None
    for tag, adapter_path in adapters.items():
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(
                base_model, str(adapter_path), adapter_name=tag
            )
        else:
            peft_model.load_adapter(str(adapter_path), adapter_name=tag)
        peft_model.set_adapter(tag)
        peft_model.eval()
        run_policy(peft_model, tag)

    panels = []
    for problem in PROBLEMS:
        for filter_label in ("correct", "wrong"):
            panels.append(
                {
                    "problem_id": problem["id"],
                    "gold": problem["gold"],
                    "filter": filter_label,
                    "base_label": problem["label"],
                    "bars": [
                        results[(problem["id"], filter_label, index)]
                        for index in range(len(BAR_SPEC))
                    ],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ppl_8panel_selfppl_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(panels, handle, indent=2)
    print(f"Saved {output_path} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
