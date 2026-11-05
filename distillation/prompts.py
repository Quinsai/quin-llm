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

PROMPT_VERSION = "v2"


SYSTEM_PROMPT = """\
You generate supervised fine-tuning data for a small English language model.
Return exactly one JSON object with two string keys:
  "instruction": a self-contained request a real user might type to an assistant.
  "output": the ideal, helpful assistant answer.
Rules:
- The "instruction" must be the user's request, not a description of the data
  you were asked to write. Never copy or paraphrase the meta-instructions from
  the user message into "instruction".
- If the user message asks you to write a question, "instruction" is that
  question and "output" is its answer. If it asks for a passage plus an
  extraction request, "instruction" is the extraction request containing the
  passage, and "output" is the extracted value.
- For JSON tasks, "instruction" must ask for the data as JSON, and "output"
  must be the JSON text. For extraction tasks, "instruction" must include the
  passage.
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
    {
        "name": "order",
        "fields": {
            "vendor": "string",
            "items": "array of strings",
            "total": "number",
            "paid": "boolean",
        },
    },
    {
        "name": "shopping_list",
        "fields": {
            "store": "string",
            "items": "array of strings",
            "budget": "number",
            "complete": "boolean",
        },
    },
    {
        "name": "match",
        "fields": {
            "home_team": "string",
            "away_team": "string",
            "home_score": "integer",
            "away_score": "integer",
        },
    },
    {
        "name": "delivery",
        "fields": {
            "carrier": "string",
            "tracking_number": "string",
            "status": "string",
            "delivery_date": "string",
        },
    },
    {
        "name": "transaction",
        "fields": {
            "account": "string",
            "amount": "number",
            "currency": "string",
            "timestamp": "string",
        },
    },
    {
        "name": "rental",
        "fields": {
            "customer": "string",
            "car_model": "string",
            "pickup_date": "string",
            "return_date": "string",
            "daily_rate": "number",
        },
    },
    {
        "name": "pet",
        "fields": {
            "name": "string",
            "species": "string",
            "age": "integer",
            "adopted": "boolean",
        },
    },
    {
        "name": "playlist",
        "fields": {
            "name": "string",
            "creator": "string",
            "tracks": "array of strings",
            "public": "boolean",
        },
    },
    {
        "name": "ticket",
        "fields": {
            "ticket_id": "string",
            "subject": "string",
            "priority": "string",
            "status": "string",
        },
    },
    {
        "name": "album",
        "fields": {
            "title": "string",
            "owner": "string",
            "photo_count": "integer",
            "shared": "boolean",
        },
    },
    {
        "name": "bug_report",
        "fields": {
            "title": "string",
            "severity": "string",
            "status": "string",
            "reporter": "string",
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


# One distinct topic per reasoning task.  The generator prefers one task per
# topic before reusing a topic with another prompt style, so this list must be
# at least as long as the configured reasoning count.
REASONING_TOPICS: list[str] = [
    # Everyday physics and earth science (1-50)
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
    "why the sky turns red at sunset",
    "how soap removes grease",
    "why we get hiccups",
    "how a refrigerator keeps food cold",
    "why mountains form",
    "how earthquakes happen",
    "why volcanoes erupt",
    "how rainbows form",
    "why the ocean is salty",
    "how a compass works",
    "why we dream",
    "how muscles grow",
    "why hair turns gray",
    "how vaccines work",
    "why the moon has phases",
    "how a camera captures an image",
    "why some materials are magnetic",
    "how plants absorb water",
    "why deserts are dry",
    "how rivers shape valleys",
    "why thunder follows lightning",
    "how a microwave heats food",
    "why ice is slippery",
    "how birds migrate",
    "why we shiver when cold",
    "how a battery stores energy",
    "why the night sky is dark",
    "how a guitar produces sound",
    "why leaves fall in autumn",
    "how lava cools into rock",
    "why the sea has waves",
    "how a seed germinates",
    "why some animals hibernate",
    "how an airplane stays in the air",
    "why objects fall to the ground",
    "how a lock and key work",
    "why water boils at a lower temperature at high altitude",
    "how sunscreen protects skin",
    # Light, electricity, and sound (51-70)
    "why the moon looks bigger near the horizon",
    "why a mirror reverses left and right",
    "how polarized sunglasses reduce glare",
    "why a prism splits white light",
    "how a magnifying glass makes things look bigger",
    "why a straw looks bent in a glass of water",
    "how a periscope lets you see around corners",
    "how a solar panel turns light into electricity",
    "why a balloon sticks to a wall after rubbing",
    "how static electricity makes hair stand up",
    "why a lightning rod protects a building",
    "how a fuse protects an electrical circuit",
    "why a circuit breaker trips",
    "how a loudspeaker turns electricity into sound",
    "why a microphone picks up sound",
    "how noise-cancelling headphones work",
    "why echoes happen in empty rooms",
    "how sonar measures distance underwater",
    "how a submarine dives and rises",
    "why a boat floats even though it is heavy",
    # Machines and simple mechanics (71-90)
    "why a hot air balloon rises",
    "how a parachute slows a fall",
    "why a spinning top stays upright",
    "how a gyroscope helps navigation",
    "why a bicycle stays balanced when moving",
    "how gears change speed and force",
    "why a pulley reduces the effort needed to lift",
    "how a wheel and axle make work easier",
    "why a sharp knife cuts better than a dull one",
    "how a wedge splits wood",
    "why a screw holds things together",
    "how friction makes a match light",
    "why rubbing your hands warms them",
    "how a thermos keeps drinks hot",
    "why sweating cools your body",
    "how a fever helps fight infection",
    "why blood is red",
    "how the heart pumps blood",
    "why we need to breathe oxygen",
    "how the lungs exchange gases",
    # The human body (91-110)
    "why eating fiber helps digestion",
    "how the stomach breaks down food",
    "why we get thirsty",
    "how the kidneys filter blood",
    "why bones heal after a break",
    "how a scab forms over a cut",
    "why we get sunburned",
    "how a bruise forms",
    "why we sneeze",
    "why we yawn",
    "how our eyes adjust to darkness",
    "why some people need glasses",
    "how the ear detects sound",
    "why we lose hearing as we age",
    "how taste buds detect flavors",
    "why smell affects how food tastes",
    "how the skin senses touch",
    "why we get goosebumps",
    "how fingernails grow",
    "why baby teeth fall out and adult teeth grow in",
    # Living things (111-130)
    "how a caterpillar becomes a butterfly",
    "why birds can fly but humans cannot",
    "how fish breathe underwater",
    "why some animals change color",
    "how a spider spins a web",
    "why bees make honey",
    "how ants find their way",
    "why some plants eat insects",
    "how trees carry water to their leaves",
    "why cactus spines are not leaves",
    "how mushrooms grow without sunlight",
    "why mold grows on bread",
    "how yeast makes bread rise",
    "why milk spoils",
    "how salt preserves food",
    "why freezing food keeps it fresh",
    "why carbonated drinks fizz",
    "how popcorn pops",
    "why chocolate melts in your hand",
    "why honey never spoils",
    # Health and behavior (131-144)
    "how coffee wakes you up",
    "why tea contains caffeine",
    "how alcohol affects the brain",
    "why smoking harms the lungs",
    "how exercise strengthens the heart",
    "why sleep is important for memory",
    "how the brain forms habits",
    "why practice improves skill",
    "how children learn language",
    "why we forget things",
    "how optical illusions trick the brain",
    "how stress affects the body",
    "why laughter is contagious",
    "how music affects mood",
    # Everyday technology (145-179)
    "how a touchscreen detects your finger",
    "how a barcode scanner reads a code",
    "how GPS finds your location",
    "how Wi-Fi carries data through the air",
    "how a fiber optic cable transmits light",
    "how a computer stores data on a hard drive",
    "how a flash drive keeps data without power",
    "how a battery charger refills a battery",
    "how a wireless charger transfers power",
    "how an LED produces light",
    "how a fluorescent bulb produces light",
    "how an electric motor turns",
    "how a generator produces electricity",
    "how a transformer changes voltage",
    "how a power grid balances supply and demand",
    "how a smoke detector senses fire",
    "how a fire extinguisher puts out a flame",
    "how a seat belt protects passengers",
    "how an airbag deploys",
    "how anti-lock brakes prevent skidding",
    "how a car engine turns fuel into motion",
    "how a catalytic converter reduces pollution",
    "how an electric car recharges its battery",
    "how regenerative braking recovers energy",
    "how a jet engine produces thrust",
    "how a rocket lifts off the ground",
    "how a satellite stays in orbit",
    "how a telescope makes distant objects visible",
    "how a microscope magnifies tiny objects",
    "how an X-ray machine sees inside the body",
    "how an MRI scanner images soft tissue",
    "how ultrasound creates a picture of a baby",
    "how a laser cuts metal",
    "how a laser pointer works",
    "how a hologram appears three-dimensional",
    # Imaging, materials, and weather (180-200)
    "how a 3D printer builds an object layer by layer",
    "how a photocopier copies a page",
    "how an inkjet printer puts ink on paper",
    "how a digital camera sensor records light",
    "how a television screen shows moving pictures",
    "how a speaker produces bass and treble",
    "how sound travels through walls",
    "how a lens focuses light",
    "how snowflakes get their shape",
    "how frost forms on a cold window",
    "how dew forms on grass in the morning",
    "how fog forms near a lake",
    "how hail forms in a thunderstorm",
    "why the sky looks white on a cloudy day",
    "how a barometer predicts the weather",
    "how a depth gauge measures water pressure",
    "how an air conditioner cools a room",
    "how a heat pump warms a house",
    "how a water filter removes impurities",
    "how a desalination plant turns seawater into drinking water",
    "how a wind turbine generates electricity",
]


def _schema_text(schema: dict[str, Any]) -> str:
    lines = [
        f'  "{field}": {kind}' for field, kind in schema["fields"].items()
    ]
    return "{\n" + ",\n".join(lines) + "\n}"


# Each tuple entry is one phrasing variant.  The task builder combines these
# with the seed pool and difficulty to create unique task prompts.
FACT_PROMPT_STYLES: tuple[str, ...] = (
    "Write one short, natural question about {entity} whose answer is {answers}. "
    "The question should target {relation_text}.",
    "Ask a direct factual question about {entity} that has the answer {answers}. "
    "It should ask for {relation_text}.",
    "Create a quiz question about {entity} where the correct answer is {answers}; "
    "make it ask for {relation_text}.",
    "Write a simple question a curious student might ask about {entity} that is "
    "answered by {answers} and targets {relation_text}.",
    "Compose a self-contained question about {entity} whose answer is {answers}, "
    "focusing on {relation_text}.",
    "Write a brief, natural-language question about {entity} that asks for "
    "{relation_text}; the answer is {answers}.",
)

JSON_PROMPT_STYLES: tuple[str, ...] = (
    "Create a realistic example about {topic}.",
    "Write a short sample record about {topic}.",
    "Provide a plausible example for {topic}.",
    "Draft a small, realistic record for {topic}.",
    "Invent a brief example related to {topic}.",
    "Write a concise example entry for {topic}.",
)

EXTRACTION_PROMPT_STYLES: tuple[str, ...] = (
    "Write a short two- or three-sentence passage about {topic} that contains "
    "exactly one {description}.",
    "Write a brief two- or three-sentence note about {topic} that includes "
    "exactly one {description}.",
)

INSTRUCTION_PROMPT_STYLES: tuple[str, ...] = (
    "Write a simple, useful instruction about {topic}.",
    "Create a short user request about {topic}.",
)

REASONING_PROMPT_STYLES: tuple[str, ...] = (
    "Write a question about {topic} that needs one short explanation.",
    'Write a simple "why" or "how" question about {topic} that needs a brief explanation.',
    "Write a one-sentence question about {topic} whose answer explains the main "
    "cause or mechanism.",
    "Write a natural question a curious person might ask about {topic} that has "
    "a short explanatory answer.",
)


def _style_index(seed: dict[str, Any], styles: tuple[str, ...]) -> int:
    try:
        index = int(seed.get("style", 0))
    except (TypeError, ValueError):
        index = 0
    return index % len(styles)


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

    relation_text = f"the {relation}" if relation else "this fact"
    if expected:
        answers = " / ".join(str(x) for x in expected)
        lead = FACT_PROMPT_STYLES[_style_index(seed, FACT_PROMPT_STYLES)].format(
            entity=entity, relation_text=relation_text, answers=answers
        )
        return (
            f"{lead} "
            "The question must be self-contained, must ask for exactly that fact, "
            "and must not mention or hint at any acceptable answer. "
            "The answer must be one short sentence that states the fact directly. "
            f"Difficulty: {difficulty}."
        )

    if _style_index(seed, FACT_PROMPT_STYLES) % 2 == 0:
        lead = (
            f"Write one short, self-contained factual question about {entity}"
            + (f" related to {relation}." if relation else ".")
        )
    else:
        lead = (
            f"Write a brief factual question about {entity}"
            + (f" that asks about {relation}." if relation else ".")
        )
    return (
        f"{lead} The answer must be a single, verifiable fact in one short "
        f"sentence. Difficulty: {difficulty}."
    )


def _json_prompt(seed: dict[str, Any], difficulty: str) -> str:
    schema = seed.get("schema") or JSON_SCHEMAS[0]
    topic = seed.get("topic", "a general record")
    lead = JSON_PROMPT_STYLES[_style_index(seed, JSON_PROMPT_STYLES)].format(
        topic=topic
    )
    return (
        f"{lead} "
        "Write an instruction asking to return the information as JSON, and "
        "provide the JSON answer that satisfies it. "
        f'The JSON object must use exactly this schema:\n{_schema_text(schema)}\n'
        'The "output" field of your response must be a string containing that '
        "JSON object, not a nested object. Do not use markdown fences. "
        f"Difficulty: {difficulty}."
    )


def _extraction_prompt(seed: dict[str, Any], difficulty: str) -> str:
    topic = seed.get("topic", "a short scenario")
    field_name = seed.get("field", "a date")
    description = seed.get("field_description", field_name)
    lead = EXTRACTION_PROMPT_STYLES[
        _style_index(seed, EXTRACTION_PROMPT_STYLES)
    ].format(topic=topic, description=description)
    return (
        f"{lead} "
        f"Then write an instruction asking to extract the {field_name} from the "
        "passage. The instruction must include the passage. "
        f'The "output" field of your response must be a JSON array of strings '
        f"containing the extracted {field_name} values. "
        f"Difficulty: {difficulty}."
    )


def _instruction_prompt(seed: dict[str, Any], difficulty: str) -> str:
    topic = seed.get("topic", "a simple task")
    constraint = seed.get("constraint") or INSTRUCTION_CONSTRAINTS[0]
    description = constraint.get("description") or (
        f"The output must satisfy the {constraint.get('name', 'given')} constraint."
    )
    lead = INSTRUCTION_PROMPT_STYLES[
        _style_index(seed, INSTRUCTION_PROMPT_STYLES)
    ].format(topic=topic)
    return (
        f"{lead} "
        "Then give the assistant's answer to that request, not another "
        "instruction or a restatement of the request. "
        f"Constraint: {description} "
        "The instruction must state the constraint clearly enough that a reader "
        "can check it, and the answer must obey it exactly. "
        f"Difficulty: {difficulty}."
    )


def _reasoning_prompt(seed: dict[str, Any], difficulty: str) -> str:
    topic = seed.get("topic", "a simple everyday phenomenon")
    lead = REASONING_PROMPT_STYLES[
        _style_index(seed, REASONING_PROMPT_STYLES)
    ].format(topic=topic)
    return (
        f"{lead} "
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
