from __future__ import annotations

from dataclasses import dataclass
import random
import re
from typing import Iterable

SKILLS = (
    "place_values",
    "division_remainders",
    "reverse_order",
    "self_check",
)

SKILL_PROMPT_NAMES = {
    "place_values": "powers of two / place values",
    "division_remainders": "divide-by-2 with remainders",
    "reverse_order": "reading remainders in reverse order",
    "self_check": "checking answers and spotting mistakes",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def approximate_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


@dataclass(frozen=True)
class StudentProfile:
    skills: dict[str, float]
    distractibility: float
    seed: int

    @property
    def weakest_skill(self) -> str:
        return min(self.skills, key=self.skills.get)

    def to_prompt_text(self) -> str:
        ordered = sorted(self.skills.items(), key=lambda item: item[1])
        parts = [f"{SKILL_PROMPT_NAMES[name]}={level:.2f}" for name, level in ordered]
        return ", ".join(parts)


def sample_weak_profile(rng: random.Random) -> StudentProfile:
    skills = {skill: rng.uniform(0.02, 0.28) for skill in SKILLS}
    weakest = min(skills, key=skills.get)
    skills[weakest] = rng.uniform(0.0, 0.12)
    distractibility = rng.uniform(0.03, 0.35)
    seed = rng.randrange(1, 10_000_000)
    return StudentProfile(skills=skills, distractibility=distractibility, seed=seed)


def make_test_numbers(seed: int, size: int, max_decimal: int = 63) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(3, max_decimal) for _ in range(size)]


@dataclass
class StudentModel:
    skills: dict[str, float]
    distractibility: float

    @classmethod
    def from_profile(cls, profile: StudentProfile) -> "StudentModel":
        return cls(skills=dict(profile.skills), distractibility=profile.distractibility)

    def _difficulty(self, decimal_value: int) -> float:
        return clamp((decimal_value.bit_length() - 2) / 5.0, 0.0, 1.0)

    def _expected_correct_probability(self, decimal_value: int) -> float:
        weighted_mastery = (
            0.23 * self.skills["place_values"]
            + 0.34 * self.skills["division_remainders"]
            + 0.29 * self.skills["reverse_order"]
            + 0.14 * self.skills["self_check"]
        )
        probability = (
            0.02
            + 0.95 * weighted_mastery
            - 0.18 * self._difficulty(decimal_value)
            - 0.12 * self.distractibility
        )
        return clamp(probability, 0.01, 0.995)

    def expected_score(self, test_numbers: Iterable[int]) -> float:
        numbers = list(test_numbers)
        if not numbers:
            return 0.0
        expected = [self._expected_correct_probability(value) for value in numbers]
        return sum(expected) / len(expected)

    def learn_from_lesson(self, lesson_text: str) -> None:
        features = extract_lesson_features(lesson_text)
        token_count = approximate_token_count(lesson_text)
        efficiency = 1.0
        if token_count < 20:
            efficiency *= 0.78
        if token_count > 70:
            overload = (token_count - 70) / 140.0
            efficiency *= max(0.35, 1.0 - overload - 0.25 * self.distractibility)

        gains = {skill: 0.0 for skill in SKILLS}
        if features["place_values"]:
            gains["place_values"] += 0.14
        if features["division_remainders"]:
            gains["division_remainders"] += 0.16
        if features["reverse_order"]:
            gains["reverse_order"] += 0.16
        if features["self_check"]:
            gains["self_check"] += 0.12
        if features["example"]:
            for skill in SKILLS:
                gains[skill] += 0.02
        if features["quiz"]:
            gains["self_check"] += 0.05

        for skill in SKILLS:
            current = self.skills[skill]
            delta = efficiency * gains[skill] * (1.0 - current)
            self.skills[skill] = clamp(current + delta, 0.0, 0.995)


def extract_lesson_features(text: str) -> dict[str, bool]:
    lower = text.lower()
    return {
        "place_values": any(
            phrase in lower
            for phrase in ("powers of two", "place value", "1, 2, 4, 8", "2, 4, 8, 16")
        ),
        "division_remainders": any(
            phrase in lower
            for phrase in ("divide by 2", "divide by two", "remainder", "quotient")
        ),
        "reverse_order": any(
            phrase in lower
            for phrase in ("reverse order", "read backward", "from bottom to top", "reverse the remainders")
        ),
        "self_check": any(
            phrase in lower
            for phrase in ("check your answer", "verify", "common mistake", "convert back")
        ),
        "example": "example" in lower or "->" in lower or "13:" in lower,
        "quiz": any(phrase in lower for phrase in ("your turn", "try this", "quick quiz", "question:")),
    }


@dataclass(frozen=True)
class LessonOutcome:
    baseline: float
    post_lesson: float
    improvement: float
    tokens: int
    reward: float


def evaluate_lesson(
    profile: StudentProfile,
    lesson_text: str,
    token_penalty: float,
    test_size: int = 24,
    seed_offset: int = 0,
) -> LessonOutcome:
    student = StudentModel.from_profile(profile)
    test_numbers = make_test_numbers(profile.seed + seed_offset, size=test_size)
    baseline = student.expected_score(test_numbers)
    student.learn_from_lesson(lesson_text)
    post = student.expected_score(test_numbers)
    tokens = approximate_token_count(lesson_text)
    improvement = post - baseline
    reward = 100.0 * improvement - token_penalty * tokens
    return LessonOutcome(
        baseline=baseline,
        post_lesson=post,
        improvement=improvement,
        tokens=tokens,
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


def action_space() -> list[TeachingAction]:
    actions: list[TeachingAction] = []
    for focus_skill in SKILLS:
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


def render_lesson(action: TeachingAction) -> str:
    parts: list[str] = []
    parts.append("Binary conversion lesson.")
    parts.append("Goal: convert decimal numbers into binary.")

    if action.focus_skill == "place_values":
        parts.append("Use place values based on powers of two: 1, 2, 4, 8, 16, 32.")
        parts.append("A 1 means that place value is included; a 0 means it is excluded.")
    elif action.focus_skill == "division_remainders":
        parts.append("Method: repeatedly divide by 2 and record each remainder.")
        parts.append("Keep dividing the quotient until you reach 0.")
    elif action.focus_skill == "reverse_order":
        parts.append("After collecting remainders, read them in reverse order.")
        parts.append("Reading from bottom to top gives the correct binary answer.")
    elif action.focus_skill == "self_check":
        parts.append("Check your answer by converting binary back into decimal.")
        parts.append("Common mistake: forgetting to verify often hides small errors.")

    if action.style == "rule":
        parts.append("Rule summary: divide by 2, track remainder, reverse order, verify.")
    elif action.style == "worked_example":
        parts.append("Example: 13 -> divide by 2 gives remainders 1, 0, 1, 1; reverse gives 1101.")
        parts.append("Example check: 1101 = 8 + 4 + 1 = 13.")
    elif action.style == "mistake_fix":
        parts.append("Common mistake: writing remainders in the same order instead of reverse order.")
        parts.append("Common mistake: stopping before quotient reaches 0.")

    if action.include_quiz:
        parts.append("Your turn: convert 10 to binary, then check your answer.")
        parts.append("Quick quiz: what binary equals decimal 19?")

    lesson_text = " ".join(parts)
    return _trim_to_budget(lesson_text, action.token_budget)


def build_teacher_prompt(profile: StudentProfile, token_budget: int) -> str:
    return (
        "You are a patient school teacher.\n"
        "Teach the student how to convert decimal numbers to binary.\n"
        "The student is weak on these subskills (0 to 1 mastery): "
        f"{profile.to_prompt_text()}.\n"
        "Write a concise lesson that targets the weakest subskill first.\n"
        f"Limit to at most {token_budget} tokens.\n"
        "Include one concrete example and one quick self-check question."
    )
