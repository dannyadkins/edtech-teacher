# Edtech Teacher RL (Token-Efficient)

Teacher-student RL sandbox where:

- `Teacher` tries to teach with minimal tokens.
- `Student` starts weak, takes a baseline test, receives a lesson, then takes a new post-test.
- Reward favors learning gain and penalizes verbosity and test leakage.

Reward:

```
reward = 100 * (post_score - baseline_score) - token_penalty * lesson_tokens - 5 * prompt_leak_hits
```

`prompt_leak_hits` counts exact post-test prompt matches that appear in the lesson text.

## Built-in Topics

- `decimal_to_binary`
- `fraction_addition`
- `subject_verb_agreement`

List topics:

```bash
python3 -m edtech_rl.inspect list-topics
```

## How Student Learning Works

The student is a weak simulator with topic subskills (0 to 1 mastery). A lesson improves subskills only if it includes relevant features (topic-specific phrases, examples, quiz/self-check), with efficiency reduced if the lesson is too long. Test performance is expected probability over hidden generated test items.

Key anti-cheat mechanics:

- Baseline and post-test use different hidden test sets (different seeds).
- Teacher prompt states it does not know hidden tests.
- Exact test prompt leakage is penalized in reward.

## Make It Very Clear (Commands)

1) Inspect one student and one lesson in detail:

```bash
python3 -m edtech_rl.inspect inspect-student \
  --topic decimal_to_binary \
  --seed 42 \
  --action place_values:rule:60:noquiz \
  --test-size 20 \
  --preview-items 5
```

This prints:

- Initial skill levels
- Lesson text and detected teaching features
- Pre/post hidden sample questions (debug view)
- Baseline, post, improvement, leakage hits, reward

2) Verify the student starts weak (before training):

```bash
python3 -m edtech_rl.inspect baseline-check \
  --topic decimal_to_binary \
  --students 300 \
  --init-skill-max 0.12 \
  --weakest-skill-max 0.05 \
  --max-mean-baseline 0.08
```

3) Train on one topic with strongly weak students:

```bash
python3 -m edtech_rl.local_train \
  --topics decimal_to_binary \
  --steps 300 \
  --batch-size 32 \
  --init-skill-max 0.12 \
  --weakest-skill-max 0.05 \
  --output-dir runs/local-binary
```

4) Train across many topics:

```bash
python3 -m edtech_rl.local_train \
  --topics decimal_to_binary,fraction_addition,subject_verb_agreement \
  --steps 500 \
  --batch-size 48 \
  --init-skill-max 0.14 \
  --weakest-skill-max 0.05 \
  --output-dir runs/local-multitopic
```

5) See teacher strategy evolution over time:

```bash
python3 -m edtech_rl.inspect summarize-run --run-dir runs/local-multitopic --points 6
```

This reads:

- `metrics.csv` for reward/improvement trends
- `strategy.jsonl` for policy breakdown per topic:
  - focus-skill mass
  - style mass (`rule`, `worked_example`, `mistake_fix`)
  - token-budget mass (`36`, `60`, `92`)
  - quiz/noquiz mass
  - top actions at checkpoints

## Output Files Per Local Run

- `metrics.csv`: scalar metrics each step
- `metrics.jsonl`: same metrics in JSONL
- `strategy.jsonl`: detailed per-topic policy snapshots each step
- `summary.json`: final top strategies and config

## Tinker Training

Install:

```bash
pip install -e ".[tinker]"
```

Run on one topic:

```bash
python3 -m edtech_rl.tinker_train \
  --topic fraction_addition \
  --base-model Qwen/Qwen3-8B \
  --steps 80 \
  --batch-size 8 \
  --group-size 4 \
  --init-skill-max 0.12 \
  --weakest-skill-max 0.05 \
  --output-dir runs/tinker-fractions
```

Tinker mode logs `mean_reward`, `mean_improvement`, `mean_tokens`, and `mean_prompt_leaks` to `runs/tinker-*/metrics.csv`.
The script also auto-loads key/value pairs from `.env` in the project root.
