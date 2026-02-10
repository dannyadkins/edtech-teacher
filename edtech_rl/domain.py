from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Iterable

from .topics import DEFAULT_TOPIC_ID, TestItem, TopicSpec, get_topic


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def approximate_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


@dataclass(frozen=True)
class StudentProfile:
    topic_id: str
    skills: dict[str, float]
    distractibility: float
    seed: int

    @property
    def weakest_skill(self) -> str:
        return min(self.skills, key=self.skills.get)

    def to_prompt_text(self) -> str:
        topic = get_topic(self.topic_id)
        ordered = sorted(self.skills.items(), key=lambda item: item[1])
        parts = [f"{topic.skill_prompt_names[name]}={level:.2f}" for name, level in ordered]
        return ", ".join(parts)


def sample_weak_profile(
    rng: random.Random,
    topic_id: str = DEFAULT_TOPIC_ID,
    skill_min: float = 0.02,
    skill_max: float = 0.28,
    weakest_max: float = 0.12,
) -> StudentProfile:
    topic = get_topic(topic_id)
    low = min(skill_min, skill_max)
    high = max(skill_min, skill_max)
    skills = {skill: rng.uniform(low, high) for skill in topic.skills}
    weakest = min(skills, key=skills.get)
    skills[weakest] = rng.uniform(0.0, max(0.0, weakest_max))
    distractibility = rng.uniform(0.03, 0.35)
    seed = rng.randrange(1, 10_000_000)
    return StudentProfile(topic_id=topic_id, skills=skills, distractibility=distractibility, seed=seed)


@dataclass
class StudentModel:
    skills: dict[str, float]
    distractibility: float

    @classmethod
    def from_profile(cls, profile: StudentProfile) -> "StudentModel":
        return cls(skills=dict(profile.skills), distractibility=profile.distractibility)

    def _expected_correct_probability(self, test_item: TestItem, topic: TopicSpec) -> float:
        weight_sum = sum(topic.skill_weights.values())
        weighted_mastery = sum(
            topic.skill_weights[skill] * self.skills.get(skill, 0.0)
            for skill in topic.skills
        ) / max(weight_sum, 1e-8)
        probability = (
            0.02
            + 0.96 * weighted_mastery
            - 0.19 * test_item.difficulty
            - 0.12 * self.distractibility
        )
        return clamp(probability, 0.01, 0.995)

    def expected_score(self, test_items: Iterable[TestItem], topic: TopicSpec) -> float:
        items = list(test_items)
        if not items:
            return 0.0
        expected = [self._expected_correct_probability(item, topic) for item in items]
        return sum(expected) / len(expected)

    def learn_from_lesson(self, lesson_text: str, topic: TopicSpec) -> dict[str, float]:
        features = extract_lesson_features(lesson_text, topic)
        token_count = approximate_token_count(lesson_text)

        efficiency = 1.0
        if token_count < 20:
            efficiency *= 0.78
        if token_count > 70:
            overload = (token_count - 70) / 140.0
            efficiency *= max(0.35, 1.0 - overload - 0.25 * self.distractibility)

        base_gain = 0.17
        gains = {skill: 0.0 for skill in topic.skills}
        for skill in topic.skills:
            if features.get(skill, False):
                gains[skill] += base_gain
        if features.get("example", False):
            for skill in topic.skills:
                gains[skill] += 0.02
        if features.get("quiz", False):
            if "self_check" in gains:
                gains["self_check"] += 0.06
            else:
                gains[topic.skills[0]] += 0.06

        deltas: dict[str, float] = {}
        for skill in topic.skills:
            current = self.skills.get(skill, 0.0)
            delta = efficiency * gains[skill] * (1.0 - current)
            updated = clamp(current + delta, 0.0, 0.995)
            deltas[skill] = updated - current
            self.skills[skill] = updated
        return deltas


def extract_lesson_features(text: str, topic: TopicSpec) -> dict[str, bool]:
    lower = text.lower()
    detected: dict[str, bool] = {}
    for feature, phrases in topic.feature_phrases.items():
        detected[feature] = any(phrase.lower() in lower for phrase in phrases)
    for skill in topic.skills:
        detected.setdefault(skill, False)
    detected.setdefault("example", False)
    detected.setdefault("quiz", False)
    return detected


@dataclass(frozen=True)
class LessonOutcome:
    baseline: float
    post_lesson: float
    improvement: float
    tokens: int
    prompt_leak_hits: float
    reward: float


def evaluate_lesson(
    profile: StudentProfile,
    lesson_text: str,
    token_penalty: float,
    test_size: int = 24,
    seed_offset: int = 0,
    topic_id: str | None = None,
) -> LessonOutcome:
    topic = get_topic(topic_id or profile.topic_id)
    student = StudentModel.from_profile(profile)
    pre_items = topic.make_test_items(profile.seed + seed_offset, size=test_size)
    post_items = topic.make_test_items(profile.seed + seed_offset + 8_191, size=test_size)

    baseline = student.expected_score(pre_items, topic)
    student.learn_from_lesson(lesson_text, topic)
    post = student.expected_score(post_items, topic)

    tokens = approximate_token_count(lesson_text)
    improvement = post - baseline
    prompt_leak_hits = topic.prompt_leak_hits(lesson_text, post_items)
    reward = 100.0 * improvement - token_penalty * tokens - 5.0 * prompt_leak_hits
    return LessonOutcome(
        baseline=baseline,
        post_lesson=post,
        improvement=improvement,
        tokens=tokens,
        prompt_leak_hits=prompt_leak_hits,
        reward=reward,
    )


@dataclass(frozen=True)
class TeachingAction:
    focus_skill: str
    style: str
    token_budget: int
    include_quiz: bool

    def short_name(self) -> str:
        quiz_part = "quiz" if self.include_quiz else "noquiz"
        return f"{self.focus_skill}:{self.style}:{self.token_budget}:{quiz_part}"


def parse_action_name(action_name: str) -> TeachingAction:
    parts = action_name.strip().split(":")
    if len(parts) != 4:
        raise ValueError(
            "Invalid action format. Expected '<focus_skill>:<style>:<token_budget>:<quiz|noquiz>'."
        )
    focus_skill, style, token_budget_text, quiz_text = parts
    token_budget = int(token_budget_text)
    if quiz_text not in ("quiz", "noquiz"):
        raise ValueError("Action quiz suffix must be 'quiz' or 'noquiz'.")
    return TeachingAction(
        focus_skill=focus_skill,
        style=style,
        token_budget=token_budget,
        include_quiz=(quiz_text == "quiz"),
    )


def action_space(topic_id: str = DEFAULT_TOPIC_ID) -> list[TeachingAction]:
    topic = get_topic(topic_id)
    actions: list[TeachingAction] = []
    for focus_skill in topic.skills:
        for style in ("rule", "worked_example", "mistake_fix"):
            for token_budget in (36, 60, 92):
                for include_quiz in (False, True):
                    actions.append(
                        TeachingAction(
                            focus_skill=focus_skill,
                            style=style,
                            token_budget=token_budget,
                            include_quiz=include_quiz,
                        )
                    )
    return actions


def _trim_to_budget(text: str, token_budget: int) -> str:
    words = text.split()
    if len(words) <= token_budget:
        return text
    return " ".join(words[:token_budget])


def render_lesson(action: TeachingAction, topic_id: str = DEFAULT_TOPIC_ID) -> str:
    topic = get_topic(topic_id)
    parts: list[str] = []
    parts.append(f"{topic.display_name} lesson.")
    parts.append(f"Goal: {topic.objective}")
    parts.extend(topic.focus_snippets[action.focus_skill])
    parts.extend(topic.style_snippets[action.style])
    if action.include_quiz:
        parts.extend(topic.quiz_snippets)
    lesson_text = " ".join(parts)
    return _trim_to_budget(lesson_text, action.token_budget)


def build_teacher_prompt(profile: StudentProfile, token_budget: int, topic_id: str | None = None) -> str:
    topic = get_topic(topic_id or profile.topic_id)
    return (
        "You are a patient school teacher.\n"
        f"Topic: {topic.display_name}.\n"
        f"Learning objective: {topic.objective}\n"
        "The student is weak on these subskills (0 to 1 mastery): "
        f"{profile.to_prompt_text()}.\n"
        "Teach the weakest subskill first.\n"
        f"Limit to at most {token_budget} tokens.\n"
        "You do not know the hidden test questions.\n"
        "Do not try to list specific test answers.\n"
        "Output only the lesson text. Do not repeat or mention these instructions."
    )
