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
from .topics import DEFAULT_TOPIC_ID, get_topic


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
    topics: tuple[str, ...] = (DEFAULT_TOPIC_ID,)
    init_skill_max: float = 0.28
    weakest_skill_max: float = 0.12


def _split_topics(text: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in text.split(",") if name.strip())
    if not names:
        return (DEFAULT_TOPIC_ID,)
    for topic_id in names:
        _ = get_topic(topic_id)
    return names


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
    parser.add_argument("--topics", type=str, default=DEFAULT_TOPIC_ID)
    parser.add_argument("--init-skill-max", type=float, default=0.28)
    parser.add_argument("--weakest-skill-max", type=float, default=0.12)
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
        topics=_split_topics(args.topics),
        init_skill_max=args.init_skill_max,
        weakest_skill_max=args.weakest_skill_max,
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
    topic_id: str,
) -> LessonOutcome:
    lesson = render_lesson(action, topic_id=topic_id)
    outcomes = [
        evaluate_lesson(profile, lesson, token_penalty, test_size=test_size, topic_id=topic_id)
        for profile in profile_pool
    ]
    return LessonOutcome(
        baseline=statistics.fmean(item.baseline for item in outcomes),
        post_lesson=statistics.fmean(item.post_lesson for item in outcomes),
        improvement=statistics.fmean(item.improvement for item in outcomes),
        tokens=round(statistics.fmean(item.tokens for item in outcomes)),
        prompt_leak_hits=round(statistics.fmean(item.prompt_leak_hits for item in outcomes), 3),
        reward=statistics.fmean(item.reward for item in outcomes),
    )


def precompute_reference_best(
    actions: Sequence[TeachingAction],
    seed: int,
    eval_students: int,
    token_penalty: float,
    test_size: int,
    topic_id: str,
    init_skill_max: float,
    weakest_skill_max: float,
) -> tuple[int, LessonOutcome]:
    rng = random.Random(seed)
    profiles = [
        sample_weak_profile(
            rng,
            topic_id=topic_id,
            skill_max=init_skill_max,
            weakest_max=weakest_skill_max,
        )
        for _ in range(eval_students)
    ]
    best_idx = 0
    best_outcome = evaluate_action_on_profiles(actions[0], profiles, token_penalty, test_size, topic_id=topic_id)
    for idx in range(1, len(actions)):
        outcome = evaluate_action_on_profiles(actions[idx], profiles, token_penalty, test_size, topic_id=topic_id)
        if outcome.reward > best_outcome.reward:
            best_idx = idx
            best_outcome = outcome
    return best_idx, best_outcome


def _probability_breakdown(
    topic_id: str,
    actions: Sequence[TeachingAction],
    probs: Sequence[float],
    top_k: int = 5,
) -> dict:
    topic = get_topic(topic_id)
    focus = {skill: 0.0 for skill in topic.skills}
    style = {"rule": 0.0, "worked_example": 0.0, "mistake_fix": 0.0}
    budget = {"36": 0.0, "60": 0.0, "92": 0.0}
    quiz = {"quiz": 0.0, "noquiz": 0.0}
    for action, prob in zip(actions, probs):
        focus[action.focus_skill] += prob
        style[action.style] += prob
        budget[str(action.token_budget)] += prob
        quiz["quiz" if action.include_quiz else "noquiz"] += prob
    ranked = sorted(range(len(actions)), key=lambda idx: probs[idx], reverse=True)[:top_k]
    top_actions = [
        {
            "action": actions[idx].short_name(),
            "probability": probs[idx],
            "lesson": render_lesson(actions[idx], topic_id=topic_id),
        }
        for idx in ranked
    ]
    return {
        "focus": focus,
        "style": style,
        "budget": budget,
        "quiz": quiz,
        "top_actions": top_actions,
    }


def run_local_training(config: LocalTrainConfig) -> None:
    rng = random.Random(config.seed)
    topic_ids = config.topics
    action_map = {topic_id: action_space(topic_id=topic_id) for topic_id in topic_ids}
    logits_map = {topic_id: [0.0 for _ in action_map[topic_id]] for topic_id in topic_ids}

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    csv_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"
    strategy_path = output_dir / "strategy.jsonl"

    reference_best_idx: dict[str, int] = {}
    reference_best: dict[str, LessonOutcome] = {}
    eval_profiles: dict[str, list] = {}
    for idx, topic_id in enumerate(topic_ids):
        reference_best_idx[topic_id], reference_best[topic_id] = precompute_reference_best(
            actions=action_map[topic_id],
            seed=config.seed + 1_001 + idx,
            eval_students=config.eval_students,
            token_penalty=config.token_penalty,
            test_size=config.test_size,
            topic_id=topic_id,
            init_skill_max=config.init_skill_max,
            weakest_skill_max=config.weakest_skill_max,
        )
        eval_rng = random.Random(config.seed + 99 + idx)
        eval_profiles[topic_id] = [
            sample_weak_profile(
                eval_rng,
                topic_id=topic_id,
                skill_max=config.init_skill_max,
                weakest_max=config.weakest_skill_max,
            )
            for _ in range(config.eval_students)
        ]

    csv_fields = [
        "step",
        "topics",
        "train_reward",
        "train_improvement",
        "train_baseline",
        "train_post",
        "train_tokens",
        "train_prompt_leaks",
        "policy_entropy",
        "eval_reward",
        "eval_improvement",
        "eval_tokens",
        "eval_prompt_leaks",
        "best_reference_reward",
        "greedy_actions_json",
        "best_reference_actions_json",
    ]

    with (
        csv_path.open("w", newline="") as csv_file,
        metrics_path.open("w") as metrics_file,
        strategy_path.open("w") as strategy_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for step in range(1, config.steps + 1):
            probs_map = {
                topic_id: softmax(logits_map[topic_id], config.temperature)
                for topic_id in topic_ids
            }
            per_topic_batch: dict[str, list[tuple[int, LessonOutcome]]] = {topic_id: [] for topic_id in topic_ids}

            for _ in range(config.batch_size):
                topic_id = rng.choice(topic_ids)
                probs = probs_map[topic_id]
                actions = action_map[topic_id]
                action_idx = categorical_sample(probs, rng)
                profile = sample_weak_profile(
                    rng,
                    topic_id=topic_id,
                    skill_max=config.init_skill_max,
                    weakest_max=config.weakest_skill_max,
                )
                lesson = render_lesson(actions[action_idx], topic_id=topic_id)
                outcome = evaluate_lesson(
                    profile=profile,
                    lesson_text=lesson,
                    token_penalty=config.token_penalty,
                    test_size=config.test_size,
                    topic_id=topic_id,
                )
                per_topic_batch[topic_id].append((action_idx, outcome))

            all_outcomes: list[LessonOutcome] = []
            for topic_id in topic_ids:
                records = per_topic_batch[topic_id]
                if not records:
                    continue
                probs = probs_map[topic_id]
                reward_mean = statistics.fmean(record[1].reward for record in records)
                for action_idx, outcome in records:
                    advantage = outcome.reward - reward_mean
                    for i, prob in enumerate(probs):
                        grad = (1.0 if i == action_idx else 0.0) - prob
                        logits_map[topic_id][i] += config.learning_rate * advantage * grad
                all_outcomes.extend(outcome for _, outcome in records)

            greedy_actions: dict[str, str] = {}
            greedy_metrics: dict[str, dict] = {}
            best_reference_actions: dict[str, str] = {}
            best_reference_rewards: list[float] = []
            eval_rewards: list[float] = []
            eval_improvements: list[float] = []
            eval_tokens: list[float] = []
            eval_leaks: list[float] = []
            strategy_row = {"step": step, "topics": list(topic_ids), "per_topic": {}}
            entropy_values = []

            for topic_id in topic_ids:
                actions = action_map[topic_id]
                probs = softmax(logits_map[topic_id], config.temperature)
                entropy_values.append(entropy(probs))

                greedy_idx = max(range(len(logits_map[topic_id])), key=logits_map[topic_id].__getitem__)
                greedy_action = actions[greedy_idx]
                greedy_outcome = evaluate_action_on_profiles(
                    action=greedy_action,
                    profile_pool=eval_profiles[topic_id],
                    token_penalty=config.token_penalty,
                    test_size=config.test_size,
                    topic_id=topic_id,
                )
                greedy_actions[topic_id] = greedy_action.short_name()
                greedy_metrics[topic_id] = {
                    "reward": greedy_outcome.reward,
                    "improvement": greedy_outcome.improvement,
                    "tokens": greedy_outcome.tokens,
                    "prompt_leaks": greedy_outcome.prompt_leak_hits,
                }
                eval_rewards.append(greedy_outcome.reward)
                eval_improvements.append(greedy_outcome.improvement)
                eval_tokens.append(greedy_outcome.tokens)
                eval_leaks.append(greedy_outcome.prompt_leak_hits)

                best_action = action_map[topic_id][reference_best_idx[topic_id]]
                best_reference_actions[topic_id] = best_action.short_name()
                best_reference_rewards.append(reference_best[topic_id].reward)

                strategy_row["per_topic"][topic_id] = {
                    "greedy_action": greedy_action.short_name(),
                    "breakdown": _probability_breakdown(topic_id, actions, probs),
                    "eval": greedy_metrics[topic_id],
                    "reference_best_action": best_action.short_name(),
                    "reference_best_reward": reference_best[topic_id].reward,
                }

            if not all_outcomes:
                raise RuntimeError("No training outcomes were generated. Increase --batch-size.")

            row = {
                "step": step,
                "topics": ",".join(topic_ids),
                "train_reward": statistics.fmean(item.reward for item in all_outcomes),
                "train_improvement": statistics.fmean(item.improvement for item in all_outcomes),
                "train_baseline": statistics.fmean(item.baseline for item in all_outcomes),
                "train_post": statistics.fmean(item.post_lesson for item in all_outcomes),
                "train_tokens": statistics.fmean(item.tokens for item in all_outcomes),
                "train_prompt_leaks": statistics.fmean(item.prompt_leak_hits for item in all_outcomes),
                "policy_entropy": statistics.fmean(entropy_values),
                "eval_reward": statistics.fmean(eval_rewards),
                "eval_improvement": statistics.fmean(eval_improvements),
                "eval_tokens": statistics.fmean(eval_tokens),
                "eval_prompt_leaks": statistics.fmean(eval_leaks),
                "best_reference_reward": statistics.fmean(best_reference_rewards),
                "greedy_actions_json": json.dumps(greedy_actions, sort_keys=True),
                "best_reference_actions_json": json.dumps(best_reference_actions, sort_keys=True),
            }
            writer.writerow(row)
            metrics_file.write(json.dumps(row) + "\n")
            strategy_file.write(json.dumps(strategy_row) + "\n")

            if step % config.print_every == 0 or step == 1:
                top_preview = " | ".join(
                    f"{topic_id}:{greedy_actions[topic_id]}"
                    for topic_id in topic_ids
                )
                print(
                    f"step={step:04d} "
                    f"train_reward={row['train_reward']:+.3f} "
                    f"eval_reward={row['eval_reward']:+.3f} "
                    f"improve={row['eval_improvement']:+.3f} "
                    f"tokens={row['eval_tokens']:.1f} "
                    f"leaks={row['eval_prompt_leaks']:.3f} "
                    f"entropy={row['policy_entropy']:.3f} "
                    f"top={top_preview}"
                )

    summary = {
        "config": asdict(config),
        "topics": list(topic_ids),
        "reference_best": {
            topic_id: {
                "action": action_map[topic_id][reference_best_idx[topic_id]].short_name(),
                "reward": reference_best[topic_id].reward,
                "improvement": reference_best[topic_id].improvement,
                "tokens": reference_best[topic_id].tokens,
                "prompt_leaks": reference_best[topic_id].prompt_leak_hits,
            }
            for topic_id in topic_ids
        },
        "final_policies": {},
    }
    for topic_id in topic_ids:
        probs = softmax(logits_map[topic_id], config.temperature)
        ranked = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)[:8]
        summary["final_policies"][topic_id] = [
            {
                "action": action_map[topic_id][idx].short_name(),
                "probability": probs[idx],
                "lesson": render_lesson(action_map[topic_id][idx], topic_id=topic_id),
            }
            for idx in ranked
        ]
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote metrics to {metrics_path}")
    print(f"wrote strategy log to {strategy_path}")
    print(f"wrote summary to {summary_path}")


def main() -> None:
    config = parse_args()
    run_local_training(config)


if __name__ == "__main__":
    main()
