# Edtech Teacher RL (Token-Efficient)

This repository contains a teacher-student RL sandbox for edtech:

- `Teacher` is optimized to teach in as few tokens as possible.
- `Student` starts weak on decimal-to-binary conversion and gets tested before/after the lesson.
- Reward is based on learning gain over baseline minus token cost.

Reward used in both trainers:

```
reward = 100 * (post_test_score - baseline_score) - token_penalty * lesson_tokens
```

## What is included

- Local runnable trainer: `edtech_rl/local_train.py`
  - No external API needed.
  - Uses policy gradient over a discrete teaching action space.
  - Logs progress to `runs/local/metrics.csv` and `runs/local/metrics.jsonl`.
- Tinker trainer scaffold: `edtech_rl/tinker_train.py`
  - Uses Tinker `sample(...)` rollouts and `forward_backward(..., loss_fn="importance_sampling")`.
  - Same reward and student evaluator as local mode.
  - Logs to `runs/tinker/`.

## Quickstart (local mode)

```bash
python3 -m edtech_rl.local_train --steps 200 --batch-size 32
```

Example output fields:

- `train_reward`: mean reward in current training batch
- `train_improvement`: mean post-vs-baseline improvement
- `greedy_reward`: reward of current best action on a fixed eval cohort
- `best_reference_reward`: best known single action on that cohort

The gap between `greedy_reward` and `best_reference_reward` makes progress easy to see over time.

## Tinker mode

Install:

```bash
pip install -e ".[tinker]"
```

Set your key:

```bash
export TINKER_API_KEY=...
```

Run:

```bash
python3 -m edtech_rl.tinker_train \
  --base-model Qwen/Qwen3-8B \
  --steps 80 \
  --batch-size 8 \
  --group-size 4
```

Notes:

- The Tinker script recreates a sampling client from current LoRA weights each step.
- Rollout advantages are centered per prompt group.
- Token-efficiency is explicit in the reward via `--token-penalty`.
