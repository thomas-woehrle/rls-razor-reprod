"""Tool-use formatting, the correctness check, and the GRPO reward.

The check is the one ``eval_tooluse.py`` scores the test set with, so the
reward GRPO optimises is the metric reported for both methods.
"""

from __future__ import annotations

import json
import re
from collections import Counter


def format_for_sft(row):
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "completion": [
            {"role": "assistant", "content": "\n".join(row["golden_response"])},
        ],
    }


def format_for_grpo(row):
    # golden_answer is a dataset column, so GRPOTrainer passes it to the reward.
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "golden_answer": row["golden_answer"],
    }


def tooluse_is_correct(response, golden_answer) -> bool:
    predicted_actions = re.findall(r"Action:\s*(\w+)", response)
    action_input_blocks = re.findall(r"Action Input:\s*({.*?})", response, re.DOTALL)
    predicted_inputs: dict = {}
    for block in action_input_blocks:
        try:
            predicted_inputs.update(json.loads(block))
        except json.JSONDecodeError:
            continue

    golden_actions = [item["Action"] for item in golden_answer]
    golden_inputs: dict = {}
    for item in golden_answer:
        try:
            golden_inputs.update(json.loads(item["Action_Input"]))
        except json.JSONDecodeError:
            continue

    return (
        Counter(predicted_actions) == Counter(golden_actions)
        and predicted_inputs == golden_inputs
    )


def tooluse_reward(completions, golden_answer, **kwargs) -> list[float]:
    """Binary success reward. Completions are conversational: one assistant message."""
    return [
        float(tooluse_is_correct(completion[0]["content"], answer))
        for completion, answer in zip(completions, golden_answer, strict=True)
    ]
