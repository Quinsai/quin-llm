"""Versioned prompts for teacher-generated SFT data.

Every task asks the teacher for one JSON object with an ``instruction`` and an
``output`` string.  Keeping the outer envelope machine-readable is what lets
the generator resume, filter, and audit records without re-parsing prose.

Bump ``PROMPT_VERSION`` whenever a template changes; request IDs include it, so
old cached responses are not silently reused with new prompts.
"""

from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "v1"


SYSTEM_PROMPT = """\
You generate supervised fine-tuning data for a small English language model.
Return exactly one JSON object with two string keys:
  "instruction": a self-contained request a user might type.
  "output": the ideal, helpful assistant answer.
Rules:
- Write natural English only. Keep both fields concise.
- The answer must directly satisfy the instruction; never mention these rules.
- Do not add greetings, apologies, or phrases like "Sure" or "Here is".
- Do not use markdown fences, headings, or bullet symbols unless explicitly asked.
- Do not reveal step-by-step reasoning; give the final answer only.
- Return valid JSON with escaped newlines; no text outside the JSON object."""


JSON_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "book",
        "fields": {
            "title": "string",
            "author": "string",
            "year": "integer",
            "available": "boolean",
        },
    },
    {
        "name": "weather",
        "fields": {
            "city": "string",
            "temperature_c": "number",
            "condition": "string",
            "humidity": "integer",
        },
    },
    {
        "name": "event",
        "fields": {
            "name": "string",
            "date": "string",
            "location": "string",
            "attendees": "integer",
        },
    },
    {
        "name": "product",
        "fields": {
            "name": "string",
            "price": "number",
            "in_stock": "boolean",
            "tags": "array of strings",
        },
    },
    {
        "name": "user",
        "fields": {
            "username": "string",
            "age": "integer",
            "active": "boolean",
            "roles": "array of strings",
        },
    },
    {
        "name": "recipe",
        "fields": {
            "name": "string",
            "servings": "integer",
            "ingredients": "array of strings",
            "steps": "array of strings",
        },
    },
    {
        "name": "flight",
        "fields": {
            "airline": "string",
            "flight_number": "string",
            "departure": "string",
            "arrival": "string",
            "price_usd": "number",
        },
    },
    {
        "name": "movie",
        "fields": {
            "title": "string",
            "director": "string",
            "release_year": "integer",
            "rating": "number",
            "genres": "array of strings",
        },
    },
]


EXTRACTION_FIELDS: list[dict[str, str]] = [
    {"name": "date", "description": "a calendar date such as 14 March 2021"},
    {"name": "person name", "description": "the full name of one person"},
    {"name": "email address", "description": "one email address"},
    {"name": "phone number", "description": "one phone number"},
    {"name": "company name", "description": "the name of one company"},
    {"name": "city", "description": "the name of one city"},
    {"name": "amount of money", "description": "one amount of money with a currency"},
    {"name": "product name", "description": "the name of one product"},
    {"name": "order number", "description": "one order number"},
    {"name": "meeting time", "description": "one clock time with a day"},
]


INSTRUCTION_CONSTRAINTS: list[dict[str, Any]] = [
    {
        "name": "max_words",
        "value": 20,
        "description": "The output must be at most 20 words long.",
    },
    {
        "name": "exact_sentences",
        "value": 1,
        "description": "The output must be exactly one sentence.",
    },
    {
        "name": "starts_with",
        "value": "First",
        "description": 'The output must start with the word "First".',
    },
    {
        "name": "bullet_count",
        "value": 3,
        "description": "The output must contain exactly three short bullet points.",
    },
    {
        "name": "includes_word",
        "value": "because",
        "description": 'The output must include the word "because".',
    },
    {
        "name": "lowercase",
        "value": True,
        "description": "The output must be entirely lowercase.",
    },
]


REASONING_TOPICS: list[str] = [
    "why leaves change color in autumn",
    "why the sky appears blue",
    "how a bicycle brake works",
    "why ice floats on water",
    "what causes ocean tides",
    "why we have seasons",
    "how plants use sunlight",
    "why metals feel cold to the touch",
    "how a simple thermometer works",
    "why bread dough rises",
    "how a lever makes lifting easier",
    "why sound travels slower than light",
]


def _schema_text(schema: dict[str, Any]) -> str:
    lines = [
        f'  "{field}": {kind}' for field, kind in schema["fields"].items()
    ]
    return "{\n" + ",\n".join(lines) + "\n}"


def _fact_prompt(seed: dict[str, Any], difficulty: str) -> str:
    entity = seed.get("entity", "").strip()
    relation = seed.get("relation", "").strip()
    expected = seed.get("expected") or []
    passage = seed.get("passage", "").strip()

    if passage:
        return (
            "Read this passage:\n"
            f'"{passage}"\n\n'
            "Write one short, natural question that is answered by the passage, "
            "and give the answer using only information from the passage. "
            "The question must be self-contained and must not mention the passage. "
            f"Difficulty: {difficulty}."
        )

    if expected:
        answers = " / ".join(str(x) for x in expected)
        relation_text = f"the {relation}" if relation else "this fact"
        return (
            f"Write one short, natural question about {entity} whose answer is "
            f"{relation_text}: {answers}. "
            "The question must be self-contained, must ask for exactly that fact, "
            "and must not hint at the answer. "
            "The answer must be one short sentence that states the fact directly. "
            f"Difficulty: {difficulty}."
        )

    return (
        f"Write one short, self-contained factual question about {entity}"
        + (f" related to {relation}." if relation else ".")
        + " The answer must be a single, verifiable fact in one short sentence. "
        f"Difficulty: {difficulty}."
    )


def _json_prompt(seed: dict[str, Any], difficulty: str) -> str:
    schema = seed.get("schema") or JSON_SCHEMAS[0]
    topic = seed.get("topic", "a general record")
    return (
        f"Create a realistic example about {topic}. "
        "Write an instruction asking to return the information as JSON, and "
        "provide the JSON answer that satisfies it. "
        f'The JSON object must use exactly this schema:\n{_schema_text(schema)}\n'
        'The "output" field of your response must be a string containing that '
        "JSON object, not a nested object. Do not use markdown fences. "
        f"Difficulty: {difficulty}."
    )


def _extraction_prompt(seed: dict[str, Any], difficulty: str) -> str:
    field_name = seed.get("field", "a date")
    description = seed.get("field_description", field_name)
    return (
        f"Write a short two- or three-sentence passage that contains exactly one "
        f"{description}. "
        f"Then write an instruction asking to extract the {field_name} from the "
        "passage. The instruction must include the passage. "
        f'The "output" field of your response must be a JSON array of strings '
        f"containing the extracted {field_name} values. "
        f"Difficulty: {difficulty}."
    )


def _instruction_prompt(seed: dict[str, Any], difficulty: str) -> str:
    topic = seed.get("topic", "a simple task")
    constraint = seed.get("constraint") or INSTRUCTION_CONSTRAINTS[0]
    return (
        f"Write a simple, useful instruction about {topic}. "
        f"Constraint: {constraint['description']} "
        "The instruction must state the constraint clearly enough that a reader "
        "can check it, and the answer must obey it exactly. "
        f"Difficulty: {difficulty}."
    )


def _reasoning_prompt(seed: dict[str, Any], difficulty: str) -> str:
    topic = seed.get("topic", "a simple everyday phenomenon")
    return (
        f"Write a question about {topic} that needs one short explanation. "
        "The answer must explain the main cause or mechanism in one to three "
        "sentences, without lists, equations, or step-by-step reasoning. "
        f"Difficulty: {difficulty}."
    )


_BUILDERS = {
    "fact_qa": _fact_prompt,
    "format_json": _json_prompt,
    "extraction": _extraction_prompt,
    "instruction": _instruction_prompt,
    "reasoning": _reasoning_prompt,
}


def build_user_prompt(task: dict[str, Any]) -> str:
    category = task.get("category", "")
    builder = _BUILDERS.get(category)
    if builder is None:
        raise ValueError(f"unknown task category: {category!r}")
    return builder(task.get("seed") or {}, task.get("difficulty", "easy"))


def build_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task)},
    ]


def schema_for_example(value: Any, schema: dict[str, Any]) -> list[str]:
    """Return human-readable schema violations for a parsed JSON example."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"expected an object, got {type(value).__name__}"]

    expected = schema.get("fields", {})
    missing = [key for key in expected if key not in value]
    extra = [key for key in value if key not in expected]
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"unexpected fields: {', '.join(extra)}")

    for key, kind in expected.items():
        if key not in value:
            continue
        item = value[key]
        if kind == "string" and not isinstance(item, str):
            errors.append(f"{key} must be a string")
        elif kind == "integer" and (
            not isinstance(item, int) or isinstance(item, bool)
        ):
            errors.append(f"{key} must be an integer")
        elif kind == "number" and (
            not isinstance(item, (int, float)) or isinstance(item, bool)
        ):
            errors.append(f"{key} must be a number")
        elif kind == "boolean" and not isinstance(item, bool):
            errors.append(f"{key} must be a boolean")
        elif kind == "array of strings" and (
            not isinstance(item, list)
            or not all(isinstance(x, str) for x in item)
        ):
            errors.append(f"{key} must be an array of strings")
    return errors


def example_for_schema(schema: dict[str, Any]) -> str:
    """A short example used in tests and dry-run output."""
    return json.dumps(
        {field: f"<{kind}>" for field, kind in schema["fields"].items()},
        ensure_ascii=False,
    )
