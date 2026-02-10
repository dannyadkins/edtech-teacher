"""Teacher-student RL sandbox for token-efficient edtech experiments."""

from .domain import (
    SKILLS,
    LessonOutcome,
    StudentProfile,
    TeachingAction,
    action_space,
    approximate_token_count,
    build_teacher_prompt,
    evaluate_lesson,
    render_lesson,
    sample_weak_profile,
)

__all__ = [
    "SKILLS",
    "LessonOutcome",
    "StudentProfile",
    "TeachingAction",
    "action_space",
    "approximate_token_count",
    "build_teacher_prompt",
    "evaluate_lesson",
    "render_lesson",
    "sample_weak_profile",
]
