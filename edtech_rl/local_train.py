from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import random
import statistics
from typing import Sequence

from .domain import (
    LessonOutcome,
    TeachingAction,
    action_space,
    evaluate_lesson,
    render_lesson,
    sample_weak_profile,
)


@dataclass
class LocalTrainConfig:
    steps: int = 300
    batch_size: int = 28
    eval_students: int = 120
    test_size: int = 24
    learning_rate: float = 0.12
    temperature: float = 1.0
    token_penalty: float = 0.06
    seed: int = 7
    output_dir: str = "runs/local"
    print_every: int = 10


def parse_args() -> LocalTrainConfig:
    parser = argparse.ArgumentParser(description="Train a token-efficient teacher with local RL simulation.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=28)
    parser.add_argument("--eval-students", type=int, default=120)
    parser.add_argument("--test-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--token-penalty", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="runs/local")
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()
    return LocalTrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_students=args.eval_students,
        test_size=args.test_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        token_penalty=args.token_penalty,
        seed=args.seed,
        output_dir=args.output_dir,
        print_every=args.print_every,
    )


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    scaled = [value / max(temperature, 1e-6) for value in logits]
    max_value = max(scaled)
    exp_values = [math.exp(value - max_value) for value in scaled]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def categorical_sample(probs: Sequence[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for idx, value in enumerate(probs):
        cumulative += value
        if threshold <= cumulative:
            return idx
    return len(probs) - 1


def entropy(probs: Sequence[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in probs)


def evaluate_action_on_profiles(
    action: TeachingAction,
    profile_pool: Sequence,
    token_penalty: float,
    test_size: int,
) -> LessonOutcome:
    lesson = render_lesson(action)
    outcomes = [evaluate_lesson(profile, lesson, token_penalty, test_size=test_size) for profile in profile_pool]
    return LessonOutcome(
        baseline=statistics.fmean(item.baseline for item in outcomes),
        post_lesson=statistics.fmean(item.post_lesson for item in outcomes),
        improvement=statistics.fmean(item.improvement for item in outcomes),
        tokens=round(statistics.fmean(item.tokens for item in outcomes)),
        reward=statistics.fmean(item.reward for item in outcomes),
    )


def precompute_reference_best(
    actions: Sequence[TeachingAction],
    seed: int,
    eval_students: int,
    token_penalty: float,
    test_size: int,
) -> tuple[int, LessonOutcome]:
    rng = random.Random(seed)
    profiles = [sample_weak_profile(rng) for _ in range(eval_students)]
    best_idx = 0
    best_outcome = evaluate_action_on_profiles(actions[0], profiles, token_penalty, test_size)
    for idx in range(1, len(actions)):
        outcome = evaluate_action_on_profiles(actions[idx], profiles, token_penalty, test_size)
        if outcome.reward > best_outcome.reward:
            best_idx = idx
            best_outcome = outcome
    return best_idx, best_outcome


def run_local_training(config: LocalTrainConfig) -> None:
    rng = random.Random(config.seed)
    actions = action_space()
    logits = [0.0 for _ in actions]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    csv_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"

    best_reference_idx, best_reference = precompute_reference_best(
        actions=actions,
        seed=config.seed + 1_001,
        eval_students=config.eval_students,
        token_penalty=config.token_penalty,
        test_size=config.test_size,
    )

    eval_rng = random.Random(config.seed + 99)
    eval_profiles = [sample_weak_profile(eval_rng) for _ in range(config.eval_students)]

    csv_fields = [
        "step",
        "train_reward",
        "train_improvement",
        "train_baseline",
        "train_post",
        "train_tokens",
        "policy_entropy",
        "greedy_reward",
        "greedy_improvement",
        "greedy_tokens",
        "best_reference_reward",
        "best_reference_improvement",
        "best_reference_tokens",
        "best_reference_action",
        "greedy_action",
    ]

    with csv_path.open("w", newline="") as csv_file, metrics_path.open("w") as metrics_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for step in range(1, config.steps + 1):
            probs = softmax(logits, config.temperature)
            batch_records: list[tuple[int, LessonOutcome]] = []

            for _ in range(config.batch_size):
                action_idx = categorical_sample(probs, rng)
                profile = sample_weak_profile(rng)
                lesson = render_lesson(actions[action_idx])
                outcome = evaluate_lesson(
                    profile=profile,
                    lesson_text=lesson,
                    token_penalty=config.token_penalty,
                    test_size=config.test_size,
                )
                batch_records.append((action_idx, outcome))

            reward_mean = statistics.fmean(record[1].reward for record in batch_records)
            for action_idx, outcome in batch_records:
                advantage = outcome.reward - reward_mean
                for i, prob in enumerate(probs):
                    grad = (1.0 if i == action_idx else 0.0) - prob
                    logits[i] += config.learning_rate * advantage * grad

            greedy_idx = max(range(len(logits)), key=logits.__getitem__)
            greedy_outcome = evaluate_action_on_profiles(
                actions[greedy_idx],
                eval_profiles,
                token_penalty=config.token_penalty,
                test_size=config.test_size,
            )

            row = {
                "step": step,
                "train_reward": reward_mean,
                "train_improvement": statistics.fmean(record[1].improvement for record in batch_records),
                "train_baseline": statistics.fmean(record[1].baseline for record in batch_records),
                "train_post": statistics.fmean(record[1].post_lesson for record in batch_records),
                "train_tokens": statistics.fmean(record[1].tokens for record in batch_records),
                "policy_entropy": entropy(probs),
                "greedy_reward": greedy_outcome.reward,
                "greedy_improvement": greedy_outcome.improvement,
                "greedy_tokens": greedy_outcome.tokens,
                "best_reference_reward": best_reference.reward,
                "best_reference_improvement": best_reference.improvement,
                "best_reference_tokens": best_reference.tokens,
                "best_reference_action": actions[best_reference_idx].short_name(),
                "greedy_action": actions[greedy_idx].short_name(),
            }
            writer.writerow(row)
            metrics_file.write(json.dumps(row) + "\n")

            if step % config.print_every == 0 or step == 1:
                print(
                    f"step={step:04d} "
                    f"train_reward={row['train_reward']:+.3f} "
                    f"improve={row['train_improvement']:+.3f} "
                    f"tokens={row['train_tokens']:.1f} "
                    f"greedy_reward={row['greedy_reward']:+.3f} "
                    f"best_ref={row['best_reference_reward']:+.3f}"
                )

    final_probs = softmax(logits, config.temperature)
    ranked = sorted(range(len(actions)), key=lambda idx: final_probs[idx], reverse=True)[:10]
    top_actions = [
        {
            "action": actions[idx].short_name(),
            "probability": final_probs[idx],
        }
        for idx in ranked
    ]
    summary = {
        "config": asdict(config),
        "best_reference_action": actions[best_reference_idx].short_name(),
        "best_reference_reward": best_reference.reward,
        "top_actions": top_actions,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote metrics to {metrics_path}")
    print(f"wrote summary to {summary_path}")


def main() -> None:
    config = parse_args()
    run_local_training(config)


if __name__ == "__main__":
    main()
