from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass(frozen=True)
class TestItem:
    prompt: str
    answer: str
    difficulty: float


@dataclass(frozen=True)
class TopicSpec:
    topic_id: str
    display_name: str
    objective: str
    skills: tuple[str, ...]
    skill_prompt_names: dict[str, str]
    skill_weights: dict[str, float]
    focus_snippets: dict[str, tuple[str, ...]]
    style_snippets: dict[str, tuple[str, ...]]
    quiz_snippets: tuple[str, ...]
    feature_phrases: dict[str, tuple[str, ...]]

    def make_test_items(self, seed: int, size: int) -> list[TestItem]:
        rng = random.Random(seed)
        maker = _TEST_MAKERS[self.topic_id]
        return [maker(rng) for _ in range(size)]

    def prompt_leak_hits(self, lesson_text: str, test_items: list[TestItem]) -> int:
        normalized_lesson = _normalize(lesson_text)
        hits = 0
        for item in test_items:
            if _normalize(item.prompt) in normalized_lesson:
                hits += 1
        return hits


def _make_binary_item(rng: random.Random) -> TestItem:
    value = rng.randint(3, 95)
    prompt = f"Convert decimal {value} to binary."
    answer = format(value, "b")
    difficulty = _clamp((value.bit_length() - 2) / 6.0, 0.0, 1.0)
    return TestItem(prompt=prompt, answer=answer, difficulty=difficulty)


def _make_fraction_item(rng: random.Random) -> TestItem:
    left_den = rng.choice((2, 3, 4, 5, 6, 8, 10, 12))
    right_den = rng.choice((2, 3, 4, 5, 6, 8, 10, 12))
    left_num = rng.randint(1, left_den - 1)
    right_num = rng.randint(1, right_den - 1)
    numer = left_num * right_den + right_num * left_den
    denom = left_den * right_den
    gcd = math.gcd(numer, denom)
    simple_num = numer // gcd
    simple_den = denom // gcd
    prompt = f"Add {left_num}/{left_den} + {right_num}/{right_den}. Give simplest fraction."
    answer = f"{simple_num}/{simple_den}"
    base = math.log2(denom) / 8.5
    simplify_bonus = 0.12 if gcd > 1 else 0.0
    difficulty = _clamp(0.1 + base + simplify_bonus, 0.0, 1.0)
    return TestItem(prompt=prompt, answer=answer, difficulty=difficulty)


_SINGULAR_SUBJECTS = (
    "The teacher",
    "The student",
    "The notebook",
    "The committee",
    "The robot",
)
_PLURAL_SUBJECTS = (
    "The teachers",
    "The students",
    "The notebooks",
    "The committees",
    "The robots",
)
_INTERRUPTING_PHRASES = (
    "with the extra notes",
    "along with the assistant",
    "as well as the examples",
    "near the whiteboard",
)
_VERB_PAIRS = (
    ("is", "are"),
    ("has", "have"),
    ("was", "were"),
)


def _make_sva_item(rng: random.Random) -> TestItem:
    is_singular = rng.random() < 0.5
    subject = rng.choice(_SINGULAR_SUBJECTS if is_singular else _PLURAL_SUBJECTS)
    verb_singular, verb_plural = rng.choice(_VERB_PAIRS)
    phrase = ""
    phrase_bonus = 0.0
    if rng.random() < 0.65:
        phrase = ", " + rng.choice(_INTERRUPTING_PHRASES) + ","
        phrase_bonus = 0.28
    answer = verb_singular if is_singular else verb_plural
    prompt = (
        f"Choose the correct verb ({verb_singular}/{verb_plural}): "
        f"{subject}{phrase} ___ ready for class."
    )
    pair_bonus = 0.07 if verb_singular == "has" else 0.0
    difficulty = _clamp(0.18 + phrase_bonus + pair_bonus, 0.0, 1.0)
    return TestItem(prompt=prompt, answer=answer, difficulty=difficulty)


TOPICS: dict[str, TopicSpec] = {
    "decimal_to_binary": TopicSpec(
        topic_id="decimal_to_binary",
        display_name="Decimal to Binary Conversion",
        objective="Convert decimal numbers into correct binary representations.",
        skills=("place_values", "division_remainders", "reverse_order", "self_check"),
        skill_prompt_names={
            "place_values": "powers of two / place values",
            "division_remainders": "divide-by-2 with remainders",
            "reverse_order": "reading remainders in reverse order",
            "self_check": "checking answers and spotting mistakes",
        },
        skill_weights={
            "place_values": 0.23,
            "division_remainders": 0.34,
            "reverse_order": 0.29,
            "self_check": 0.14,
        },
        focus_snippets={
            "place_values": (
                "Use powers of two place values: 1, 2, 4, 8, 16, 32.",
                "A 1 includes that place value; a 0 excludes it.",
            ),
            "division_remainders": (
                "Repeatedly divide by 2 and write each remainder.",
                "Continue until the quotient reaches 0.",
            ),
            "reverse_order": (
                "After collecting remainders, read them in reverse order.",
                "Bottom-to-top order gives the binary answer.",
            ),
            "self_check": (
                "Check by converting binary back into decimal.",
                "Common mistake: skipping verification hides small errors.",
            ),
        },
        style_snippets={
            "rule": (
                "Rule summary: divide by 2, track remainder, reverse order, verify.",
            ),
            "worked_example": (
                "Example: 13 -> remainders 1, 0, 1, 1; reverse gives 1101.",
                "Check: 1101 = 8 + 4 + 1 = 13.",
            ),
            "mistake_fix": (
                "Common mistake: writing remainders in forward order.",
                "Common mistake: stopping before quotient reaches 0.",
            ),
        },
        quiz_snippets=(
            "Your turn: convert 10 to binary, then check your answer.",
            "Quick quiz: what binary equals decimal 19?",
        ),
        feature_phrases={
            "place_values": ("powers of two", "place value", "1, 2, 4, 8"),
            "division_remainders": ("divide by 2", "remainder", "quotient"),
            "reverse_order": ("reverse order", "bottom-to-top", "read backward"),
            "self_check": ("check your answer", "verify", "common mistake", "convert back"),
            "example": ("example", "->", "13"),
            "quiz": ("your turn", "quick quiz", "question:", "try this"),
        },
    ),
    "fraction_addition": TopicSpec(
        topic_id="fraction_addition",
        display_name="Fraction Addition",
        objective="Add fractions accurately and simplify the final result.",
        skills=("common_denominator", "convert_numerators", "simplify_result", "self_check"),
        skill_prompt_names={
            "common_denominator": "finding a common denominator",
            "convert_numerators": "rewriting equivalent numerators",
            "simplify_result": "simplifying the final fraction",
            "self_check": "checking with estimation or reverse checks",
        },
        skill_weights={
            "common_denominator": 0.30,
            "convert_numerators": 0.25,
            "simplify_result": 0.30,
            "self_check": 0.15,
        },
        focus_snippets={
            "common_denominator": (
                "Find a common denominator before adding fractions.",
                "The LCD keeps the conversion minimal and cleaner.",
            ),
            "convert_numerators": (
                "Rewrite each fraction as an equivalent fraction with the common denominator.",
                "Multiply numerator and denominator by the same factor.",
            ),
            "simplify_result": (
                "After adding numerators, simplify the result.",
                "Divide numerator and denominator by their greatest common factor.",
            ),
            "self_check": (
                "Estimate first to catch unreasonable answers.",
                "Check by converting to decimals to verify approximate size.",
            ),
        },
        style_snippets={
            "rule": (
                "Rule summary: common denominator -> rewrite -> add numerators -> simplify -> verify.",
            ),
            "worked_example": (
                "Example: 1/2 + 1/3 -> 3/6 + 2/6 = 5/6.",
                "Check: 0.5 + 0.333... ~= 0.833..., so 5/6 is reasonable.",
            ),
            "mistake_fix": (
                "Common mistake: adding denominators directly.",
                "Common mistake: forgetting to simplify at the end.",
            ),
        },
        quiz_snippets=(
            "Your turn: add 2/5 + 1/10 and simplify.",
            "Quick quiz: add 3/4 + 1/6 in simplest form.",
        ),
        feature_phrases={
            "common_denominator": ("common denominator", "lcd", "least common denominator"),
            "convert_numerators": (
                "equivalent fraction",
                "rewrite each fraction",
                "multiply numerator and denominator",
            ),
            "simplify_result": ("simplify", "reduce", "greatest common factor", "gcf"),
            "self_check": ("check your answer", "estimate", "verify", "common mistake"),
            "example": ("example", "->", "1/2 + 1/3"),
            "quiz": ("your turn", "quick quiz", "question:", "try this"),
        },
    ),
    "subject_verb_agreement": TopicSpec(
        topic_id="subject_verb_agreement",
        display_name="Subject-Verb Agreement",
        objective="Pick verbs that agree with the true grammatical subject.",
        skills=("singular_plural_subject", "identify_main_verb", "ignore_interrupting_phrase", "self_check"),
        skill_prompt_names={
            "singular_plural_subject": "detecting singular vs plural subject",
            "identify_main_verb": "matching the verb form to the subject",
            "ignore_interrupting_phrase": "ignoring interrupting phrases",
            "self_check": "checking sentence agreement after choosing",
        },
        skill_weights={
            "singular_plural_subject": 0.31,
            "identify_main_verb": 0.27,
            "ignore_interrupting_phrase": 0.27,
            "self_check": 0.15,
        },
        focus_snippets={
            "singular_plural_subject": (
                "Find the true subject first: singular subject uses singular verb.",
                "Plural subject uses plural verb.",
            ),
            "identify_main_verb": (
                "Match verb form directly to the subject number.",
                "Do not choose based on nearby nouns.",
            ),
            "ignore_interrupting_phrase": (
                "Ignore interrupting phrases between subject and verb.",
                "Agreement depends on the main subject, not inserted phrases.",
            ),
            "self_check": (
                "Reread sentence after filling the blank to check agreement.",
                "If it sounds awkward, re-check the subject number.",
            ),
        },
        style_snippets={
            "rule": (
                "Rule summary: identify subject -> ignore interrupters -> choose agreeing verb -> verify.",
            ),
            "worked_example": (
                "Example: 'The teacher, along with notes, is ready.'",
                "Main subject is 'teacher' (singular), so use 'is'.",
            ),
            "mistake_fix": (
                "Common mistake: matching verb to the nearest noun instead of subject.",
                "Common mistake: letting prepositional phrases change agreement.",
            ),
        },
        quiz_snippets=(
            "Your turn: 'The students with the tutor ___ prepared (is/are)'.",
            "Quick quiz: 'The notebook, as well as the pages, ___ missing (is/are)'.",
        ),
        feature_phrases={
            "singular_plural_subject": ("singular", "plural", "true subject"),
            "identify_main_verb": ("verb form", "match the verb", "subject number"),
            "ignore_interrupting_phrase": (
                "interrupting phrase",
                "ignore nearby nouns",
                "prepositional phrase",
            ),
            "self_check": ("check agreement", "reread", "common mistake", "verify"),
            "example": ("example", "teacher", "students"),
            "quiz": ("your turn", "quick quiz", "question:", "try this"),
        },
    ),
}

_TEST_MAKERS = {
    "decimal_to_binary": _make_binary_item,
    "fraction_addition": _make_fraction_item,
    "subject_verb_agreement": _make_sva_item,
}

DEFAULT_TOPIC_ID = "decimal_to_binary"


def get_topic(topic_id: str) -> TopicSpec:
    if topic_id not in TOPICS:
        valid = ", ".join(sorted(TOPICS))
        raise ValueError(f"Unknown topic_id '{topic_id}'. Valid topics: {valid}")
    return TOPICS[topic_id]


def list_topics() -> list[TopicSpec]:
    return [TOPICS[name] for name in sorted(TOPICS)]
