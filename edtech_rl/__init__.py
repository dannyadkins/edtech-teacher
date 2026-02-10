"""Teacher-student RL sandbox for token-efficient edtech experiments."""

from .domain import (
    LessonOutcome,
    StudentProfile,
    TeachingAction,
    action_space,
    approximate_token_count,
    build_teacher_prompt,
    evaluate_lesson,
    extract_lesson_features,
    parse_action_name,
    render_lesson,
    sample_weak_profile,
)
from .topics import DEFAULT_TOPIC_ID, TestItem, TopicSpec, get_topic, list_topics

__all__ = [
    "DEFAULT_TOPIC_ID",
    "LessonOutcome",
    "StudentProfile",
    "TeachingAction",
    "TestItem",
    "TopicSpec",
    "action_space",
    "approximate_token_count",
    "build_teacher_prompt",
    "evaluate_lesson",
    "extract_lesson_features",
    "get_topic",
    "list_topics",
    "parse_action_name",
    "render_lesson",
    "sample_weak_profile",
]
