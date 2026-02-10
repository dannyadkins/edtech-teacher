from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import statistics
import time

from .domain import build_teacher_prompt, evaluate_lesson, sample_weak_profile
from .topics import DEFAULT_TOPIC_ID, get_topic


@dataclass
class TinkerTrainConfig:
    base_model: str = "Qwen/Qwen3-8B"
    base_url: str | None = None
    steps: int = 80
    batch_size: int = 8
    group_size: int = 4
    max_tokens: int = 96
    temperature: float = 0.8
    learning_rate: float = 4e-5
    rank: int = 32
    ttl_seconds: int = 86_400
    token_penalty: float = 0.06
    test_size: int = 24
    seed: int = 17
    output_dir: str = "runs/tinker"
    topic_id: str = DEFAULT_TOPIC_ID
    init_skill_max: float = 0.28
    weakest_skill_max: float = 0.12
    sample_log_per_step: int = 3
    api_timeout_sec: float = 120.0


def parse_args() -> TinkerTrainConfig:
    parser = argparse.ArgumentParser(description="Train a token-efficient teacher with the Tinker API.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--ttl-seconds", type=int, default=86_400)
    parser.add_argument("--token-penalty", type=float, default=0.06)
    parser.add_argument("--test-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=str, default="runs/tinker")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC_ID)
    parser.add_argument("--init-skill-max", type=float, default=0.28)
    parser.add_argument("--weakest-skill-max", type=float, default=0.12)
    parser.add_argument("--sample-log-per-step", type=int, default=3)
    parser.add_argument("--api-timeout-sec", type=float, default=120.0)
    args = parser.parse_args()
    _ = get_topic(args.topic)
    return TinkerTrainConfig(
        base_model=args.base_model,
        base_url=args.base_url,
        steps=args.steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        rank=args.rank,
        ttl_seconds=args.ttl_seconds,
        token_penalty=args.token_penalty,
        test_size=args.test_size,
        seed=args.seed,
        output_dir=args.output_dir,
        topic_id=args.topic,
        init_skill_max=args.init_skill_max,
        weakest_skill_max=args.weakest_skill_max,
        sample_log_per_step=args.sample_log_per_step,
        api_timeout_sec=args.api_timeout_sec,
    )


def _encode(tokenizer, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _decode(tokenizer, tokens: list[int]) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        try:
            return str(backend.decode(tokens))
        except Exception:
            pass
    try:
        pieces = tokenizer.convert_ids_to_tokens(tokens)
        return str(tokenizer.convert_tokens_to_string(pieces))
    except Exception:
        return ""


def _sequence_list(result) -> list:
    sequences = getattr(result, "sequences", None)
    if sequences is None:
        sequences = getattr(result, "samples", None)
    if sequences is None:
        raise RuntimeError("Tinker sample response had no `sequences` or `samples` attribute.")
    return list(sequences)


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def _wait_for_future_result(future, timeout_sec: float, poll_sec: float = 0.25):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if future.done():
            return future.result()
        time.sleep(poll_sec)
    raise TimeoutError(f"future did not complete within {timeout_sec:.1f}s")


def run_tinker_training(config: TinkerTrainConfig) -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _load_env_file(".env")
    try:
        import tinker
        from tinker import types
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install Tinker first with `pip install -e '.[tinker]'`."
        ) from exc

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    csv_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"
    sample_log_path = output_dir / "sample_texts.jsonl"

    service_kwargs = {}
    if config.base_url:
        service_kwargs["base_url"] = config.base_url

    print("init: creating service client")
    service_client = tinker.ServiceClient(**service_kwargs)
    print("init: creating LoRA training client")
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model,
        rank=config.rank,
    )
    print("init: loading tokenizer")
    tokenizer = training_client.get_tokenizer()
    print("init: tokenizer ready")

    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    rng = random.Random(config.seed)
    start = time.time()

    csv_fields = [
        "step",
        "mean_reward",
        "mean_improvement",
        "mean_tokens",
        "mean_prompt_leaks",
        "datums",
        "groups",
        "elapsed_sec",
    ]

    with (
        csv_path.open("w", newline="") as csv_file,
        metrics_path.open("w") as jsonl_file,
        sample_log_path.open("w") as sample_log_file,
    ):
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for step in range(1, config.steps + 1):
            print(f"step={step:04d} stage=save_weights_and_create_sampling_client")
            try:
                sampling_client = training_client.save_weights_and_get_sampling_client(
                    name=f"step-{step:04d}",
                )
            except Exception as exc:
                print(
                    f"step={step:04d} save_weights_and_get_sampling_client timeout/error: {exc}"
                )
                continue

            group_data = []
            futures = []
            for _ in range(config.batch_size):
                profile = sample_weak_profile(
                    rng,
                    topic_id=config.topic_id,
                    skill_max=config.init_skill_max,
                    weakest_max=config.weakest_skill_max,
                )
                prompt_text = build_teacher_prompt(
                    profile,
                    token_budget=config.max_tokens,
                    topic_id=config.topic_id,
                )
                prompt_tokens = _encode(tokenizer, prompt_text)
                if len(prompt_tokens) < 2:
                    continue
                prompt = types.ModelInput.from_ints(prompt_tokens)
                future = sampling_client.sample(
                    prompt=prompt,
                    num_samples=config.group_size,
                    sampling_params=sampling_params,
                )
                futures.append(future)
                group_data.append((profile, prompt_tokens))

            print(f"step={step:04d} stage=collect_samples futures={len(futures)}")
            datums = []
            group_rewards = []
            group_improvements = []
            group_tokens = []
            group_leaks = []
            step_samples: list[dict] = []

            for future, (profile, prompt_tokens) in zip(futures, group_data):
                try:
                    result = _wait_for_future_result(
                        future,
                        timeout_sec=config.api_timeout_sec,
                    )
                except Exception as exc:
                    print(f"step={step:04d} sample timeout/error: {exc}")
                    continue
                sequences = _sequence_list(result)
                rollout_data = []
                rollout_rewards = []

                for sequence in sequences:
                    sampled_tokens = list(getattr(sequence, "tokens", []))
                    sampled_logprobs = list(getattr(sequence, "logprobs", []))
                    if not sampled_tokens or not sampled_logprobs:
                        continue
                    if len(sampled_logprobs) == len(sampled_tokens) - 1:
                        sampled_logprobs = [sampled_logprobs[0]] + sampled_logprobs
                    if len(sampled_logprobs) != len(sampled_tokens):
                        usable = min(len(sampled_tokens), len(sampled_logprobs))
                        if usable <= 0:
                            continue
                        sampled_tokens = sampled_tokens[:usable]
                        sampled_logprobs = sampled_logprobs[:usable]
                    sampled_logprobs = [float(value) for value in sampled_logprobs]
                    lesson_text = _decode(tokenizer, sampled_tokens)
                    outcome = evaluate_lesson(
                        profile=profile,
                        lesson_text=lesson_text,
                        token_penalty=config.token_penalty,
                        test_size=config.test_size,
                        topic_id=config.topic_id,
                    )
                    step_samples.append(
                        {
                            "topic": config.topic_id,
                            "weakest_skill": profile.weakest_skill,
                            "reward": outcome.reward,
                            "improvement": outcome.improvement,
                            "tokens": outcome.tokens,
                            "prompt_leaks": outcome.prompt_leak_hits,
                            "lesson_text": lesson_text,
                        }
                    )
                    rollout_data.append((sampled_tokens, sampled_logprobs, outcome))
                    rollout_rewards.append(outcome.reward)

                if not rollout_data:
                    continue

                mean_group_reward = statistics.fmean(rollout_rewards)
                group_rewards.append(mean_group_reward)
                group_improvements.append(statistics.fmean(item[2].improvement for item in rollout_data))
                group_tokens.append(statistics.fmean(item[2].tokens for item in rollout_data))
                group_leaks.append(statistics.fmean(item[2].prompt_leak_hits for item in rollout_data))

                for sampled_tokens, sampled_logprobs, outcome in rollout_data:
                    advantage = outcome.reward - mean_group_reward
                    if abs(advantage) < 1e-8:
                        continue
                    observation_length = len(prompt_tokens) - 1
                    model_input_tokens = prompt_tokens + sampled_tokens[:-1]
                    target_tokens = [0] * observation_length + sampled_tokens
                    padded_logprobs = [0.0] * observation_length + sampled_logprobs
                    padded_advantages = [0.0] * observation_length + [advantage] * len(sampled_tokens)
                    expected_len = observation_length + len(sampled_tokens)
                    if (
                        len(model_input_tokens) != expected_len
                        or len(target_tokens) != expected_len
                        or len(padded_logprobs) != expected_len
                        or len(padded_advantages) != expected_len
                    ):
                        continue
                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(model_input_tokens),
                        loss_fn_inputs={
                            "target_tokens": types.TensorData(
                                data=target_tokens,
                                dtype="int64",
                                shape=[len(target_tokens)],
                            ),
                            "logprobs": types.TensorData(
                                data=padded_logprobs,
                                dtype="float32",
                                shape=[len(padded_logprobs)],
                            ),
                            "advantages": types.TensorData(
                                data=padded_advantages,
                                dtype="float32",
                                shape=[len(padded_advantages)],
                            ),
                        },
                    )
                    datums.append(datum)

            if datums:
                try:
                    _ = training_client.forward_backward(
                        data=datums,
                        loss_fn="importance_sampling",
                    ).result(timeout=config.api_timeout_sec)
                    _ = training_client.optim_step(adam_params=adam_params).result(
                        timeout=config.api_timeout_sec
                    )
                except Exception as exc:
                    print(f"step={step:04d} train_step timeout/error: {exc}")

            row = {
                "step": step,
                "mean_reward": statistics.fmean(group_rewards) if group_rewards else 0.0,
                "mean_improvement": statistics.fmean(group_improvements) if group_improvements else 0.0,
                "mean_tokens": statistics.fmean(group_tokens) if group_tokens else 0.0,
                "mean_prompt_leaks": statistics.fmean(group_leaks) if group_leaks else 0.0,
                "datums": len(datums),
                "groups": len(group_rewards),
                "elapsed_sec": time.time() - start,
            }
            writer.writerow(row)
            jsonl_file.write(json.dumps(row) + "\n")
            sorted_samples = sorted(step_samples, key=lambda item: item["reward"], reverse=True)
            sample_log_file.write(
                json.dumps(
                    {
                        "step": step,
                        "top_samples": sorted_samples[: max(1, config.sample_log_per_step)],
                        "worst_sample": sorted_samples[-1] if sorted_samples else None,
                    }
                )
                + "\n"
            )

            print(
                f"step={step:04d} "
                f"reward={row['mean_reward']:+.3f} "
                f"improve={row['mean_improvement']:+.3f} "
                f"tokens={row['mean_tokens']:.1f} "
                f"leaks={row['mean_prompt_leaks']:.3f} "
                f"datums={row['datums']}"
            )

    final_path = training_client.save_weights_for_sampler(
        name="final",
        ttl_seconds=config.ttl_seconds,
    ).result(timeout=config.api_timeout_sec).path
    summary = {
        "final_model_path": final_path,
        "config": config.__dict__,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote metrics to {metrics_path}")
    print(f"wrote sample texts to {sample_log_path}")
    print(f"wrote summary to {summary_path}")
    print(f"final sampler path: {final_path}")


def main() -> None:
    config = parse_args()
    run_tinker_training(config)


if __name__ == "__main__":
    main()
