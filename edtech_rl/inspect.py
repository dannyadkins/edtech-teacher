from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import Any

from .domain import (
    StudentModel,
    action_space,
    evaluate_lesson,
    extract_lesson_features,
    parse_action_name,
    render_lesson,
    sample_weak_profile,
)
from .topics import DEFAULT_TOPIC_ID, get_topic, list_topics


def cmd_list_topics(_: argparse.Namespace) -> None:
    for topic in list_topics():
        print(f"{topic.topic_id}: {topic.display_name}")
        print(f"  objective: {topic.objective}")
        print(f"  skills: {', '.join(topic.skills)}")


def _sample_profile(seed: int, topic_id: str, init_skill_max: float, weakest_skill_max: float):
    rng = random.Random(seed)
    return sample_weak_profile(
        rng,
        topic_id=topic_id,
        skill_max=init_skill_max,
        weakest_max=weakest_skill_max,
    )


def cmd_inspect_student(args: argparse.Namespace) -> None:
    topic = get_topic(args.topic)
    profile = _sample_profile(args.seed, args.topic, args.init_skill_max, args.weakest_skill_max)
    actions = action_space(args.topic)

    if args.action:
        action = parse_action_name(args.action)
    else:
        action = actions[0]

    lesson = render_lesson(action, topic_id=args.topic)
    outcome = evaluate_lesson(
        profile=profile,
        lesson_text=lesson,
        token_penalty=args.token_penalty,
        test_size=args.test_size,
        topic_id=args.topic,
    )
    pre_items = topic.make_test_items(profile.seed, size=min(args.preview_items, args.test_size))
    post_items = topic.make_test_items(profile.seed + 8_191, size=min(args.preview_items, args.test_size))
    student = StudentModel.from_profile(profile)
    deltas = student.learn_from_lesson(lesson, topic)
    features = extract_lesson_features(lesson, topic)

    print(f"topic={topic.topic_id} ({topic.display_name})")
    print(f"seed={args.seed}")
    print(f"selected_action={action.short_name()}")
    print(f"lesson_tokens={outcome.tokens}")
    print(f"baseline={outcome.baseline:.4f}")
    print(f"post={outcome.post_lesson:.4f}")
    print(f"improvement={outcome.improvement:+.4f}")
    print(f"prompt_leak_hits={outcome.prompt_leak_hits}")
    print(f"reward={outcome.reward:+.4f}")
    print("")
    print("initial_skills:")
    for skill, value in sorted(profile.skills.items(), key=lambda item: item[1]):
        print(f"  {skill}: {value:.3f}")
    print("skill_deltas_after_lesson:")
    for skill, delta in sorted(deltas.items(), key=lambda item: item[1], reverse=True):
        print(f"  {skill}: {delta:+.4f}")
    print("detected_features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    print("")
    print("lesson_text:")
    print(lesson)
    print("")
    print("sample_pretest_items (hidden answers shown here only for debugging):")
    for item in pre_items:
        print(f"  Q: {item.prompt}")
        print(f"  A: {item.answer}")
    print("")
    print("sample_posttest_items (used for reward; should not be in lesson):")
    for item in post_items:
        print(f"  Q: {item.prompt}")
        print(f"  A: {item.answer}")


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def cmd_baseline_check(args: argparse.Namespace) -> None:
    topic = get_topic(args.topic)
    rng = random.Random(args.seed)
    baselines: list[float] = []
    for _ in range(args.students):
        profile = sample_weak_profile(
            rng,
            topic_id=args.topic,
            skill_max=args.init_skill_max,
            weakest_max=args.weakest_skill_max,
        )
        student = StudentModel.from_profile(profile)
        test_items = topic.make_test_items(profile.seed, size=args.test_size)
        baselines.append(student.expected_score(test_items, topic))

    baselines_sorted = sorted(baselines)
    mean_value = sum(baselines) / max(1, len(baselines))
    print(f"topic={args.topic}")
    print(f"students={args.students}")
    print(f"mean_baseline={mean_value:.4f}")
    print(f"min_baseline={baselines_sorted[0]:.4f}")
    print(f"p25_baseline={_percentile(baselines_sorted, 25):.4f}")
    print(f"p50_baseline={_percentile(baselines_sorted, 50):.4f}")
    print(f"p75_baseline={_percentile(baselines_sorted, 75):.4f}")
    print(f"max_baseline={baselines_sorted[-1]:.4f}")
    if args.max_mean_baseline is not None:
        passed = mean_value <= args.max_mean_baseline
        print(f"check_mean<={args.max_mean_baseline:.4f}: {'PASS' if passed else 'FAIL'}")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _select_indices(length: int, points: int) -> list[int]:
    if length <= points:
        return list(range(length))
    indexes = [0]
    for i in range(1, points - 1):
        indexes.append(round(i * (length - 1) / (points - 1)))
    indexes.append(length - 1)
    unique_sorted = sorted(set(indexes))
    return unique_sorted


def _best_key(metric_map: dict[str, float]) -> tuple[str, float]:
    best_name = ""
    best_value = float("-inf")
    for name, value in metric_map.items():
        if value > best_value:
            best_name = name
            best_value = value
    return best_name, best_value


def cmd_summarize_run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    metrics_csv = run_dir / "metrics.csv"
    strategy_jsonl = run_dir / "strategy.jsonl"
    if not metrics_csv.exists():
        raise SystemExit(f"Missing metrics file: {metrics_csv}")
    rows = _read_csv_rows(metrics_csv)
    if not rows:
        raise SystemExit(f"No metrics in: {metrics_csv}")

    first = rows[0]
    last = rows[-1]
    train_delta = float(last["train_reward"]) - float(first["train_reward"])
    eval_delta = float(last["eval_reward"]) - float(first["eval_reward"])
    print(f"run_dir={run_dir}")
    print(f"steps={len(rows)}")
    print(f"topics={last.get('topics', '')}")
    print(
        "train_reward: "
        f"{float(first['train_reward']):+.3f} -> {float(last['train_reward']):+.3f} "
        f"(delta={train_delta:+.3f})"
    )
    print(
        "eval_reward: "
        f"{float(first['eval_reward']):+.3f} -> {float(last['eval_reward']):+.3f} "
        f"(delta={eval_delta:+.3f})"
    )
    print(
        "eval_improvement: "
        f"{float(first['eval_improvement']):+.4f} -> {float(last['eval_improvement']):+.4f}"
    )
    print(
        "eval_tokens: "
        f"{float(first['eval_tokens']):.2f} -> {float(last['eval_tokens']):.2f}"
    )
    print(
        "eval_prompt_leaks: "
        f"{float(first['eval_prompt_leaks']):.3f} -> {float(last['eval_prompt_leaks']):.3f}"
    )

    if not strategy_jsonl.exists():
        print(f"strategy log not found at {strategy_jsonl}")
        return
    strategy_rows = _read_jsonl(strategy_jsonl)
    if not strategy_rows:
        print("strategy log is empty")
        return

    print("")
    print("strategy_checkpoints:")
    checkpoints = _select_indices(len(strategy_rows), max(3, args.points))
    for idx in checkpoints:
        row = strategy_rows[idx]
        step = row["step"]
        print(f"  step={step}")
        for topic_id, data in row["per_topic"].items():
            greedy = data["greedy_action"]
            breakdown = data["breakdown"]
            focus_name, focus_prob = _best_key(breakdown["focus"])
            style_name, style_prob = _best_key(breakdown["style"])
            budget_name, budget_prob = _best_key(breakdown["budget"])
            quiz_name, quiz_prob = _best_key(breakdown["quiz"])
            print(
                f"    {topic_id}: greedy={greedy} "
                f"focus={focus_name}({focus_prob:.2f}) "
                f"style={style_name}({style_prob:.2f}) "
                f"budget={budget_name}({budget_prob:.2f}) "
                f"quiz={quiz_name}({quiz_prob:.2f})"
            )

    print("")
    print("strategy_drift_first_to_last:")
    first_strategy = strategy_rows[0]["per_topic"]
    last_strategy = strategy_rows[-1]["per_topic"]
    for topic_id in sorted(last_strategy):
        first_breakdown = first_strategy[topic_id]["breakdown"]
        last_breakdown = last_strategy[topic_id]["breakdown"]
        focus_before, _ = _best_key(first_breakdown["focus"])
        focus_after, _ = _best_key(last_breakdown["focus"])
        style_before, _ = _best_key(first_breakdown["style"])
        style_after, _ = _best_key(last_breakdown["style"])
        budget_before, _ = _best_key(first_breakdown["budget"])
        budget_after, _ = _best_key(last_breakdown["budget"])
        quiz_before, _ = _best_key(first_breakdown["quiz"])
        quiz_after, _ = _best_key(last_breakdown["quiz"])
        print(
            f"  {topic_id}: "
            f"focus {focus_before}->{focus_after}, "
            f"style {style_before}->{style_after}, "
            f"budget {budget_before}->{budget_after}, "
            f"quiz {quiz_before}->{quiz_after}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect teacher-student RL runs and test leakage.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-topics", help="List available built-in topics.")
    list_parser.set_defaults(func=cmd_list_topics)

    inspect_parser = subparsers.add_parser(
        "inspect-student",
        help="Show one student profile, lesson, tests, and reward breakdown.",
    )
    inspect_parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC_ID)
    inspect_parser.add_argument("--seed", type=int, default=123)
    inspect_parser.add_argument("--action", type=str, default="")
    inspect_parser.add_argument("--test-size", type=int, default=24)
    inspect_parser.add_argument("--preview-items", type=int, default=5)
    inspect_parser.add_argument("--token-penalty", type=float, default=0.06)
    inspect_parser.add_argument("--init-skill-max", type=float, default=0.28)
    inspect_parser.add_argument("--weakest-skill-max", type=float, default=0.12)
    inspect_parser.set_defaults(func=cmd_inspect_student)

    baseline_parser = subparsers.add_parser(
        "baseline-check",
        help="Measure baseline (pre-lesson) knowledge across sampled students.",
    )
    baseline_parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC_ID)
    baseline_parser.add_argument("--students", type=int, default=200)
    baseline_parser.add_argument("--test-size", type=int, default=24)
    baseline_parser.add_argument("--seed", type=int, default=123)
    baseline_parser.add_argument("--init-skill-max", type=float, default=0.12)
    baseline_parser.add_argument("--weakest-skill-max", type=float, default=0.05)
    baseline_parser.add_argument("--max-mean-baseline", type=float, default=None)
    baseline_parser.set_defaults(func=cmd_baseline_check)

    summary_parser = subparsers.add_parser(
        "summarize-run",
        help="Summarize reward progress and strategy evolution from a run directory.",
    )
    summary_parser.add_argument("--run-dir", type=str, required=True)
    summary_parser.add_argument("--points", type=int, default=5)
    summary_parser.set_defaults(func=cmd_summarize_run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
