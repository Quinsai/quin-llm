# Distillation seed data

## `train_entities.jsonl`

246 entities, 1,132 `(entity, relation, expected)` facts.

- Country entries (186) were derived from the UN-member subset of
  [`mledoze/countries`](https://github.com/mledoze/countries)
  (`countries.json`, fetched 2026-11-05, SHA-256
  `0d92fb23b43ae080bd439f5dc5d61a8845395fdd0af1479b433c59fe079bd6fd`).
  Each country contributes capital, continent, currency, official language,
  and ISO 3166-1 alpha-2 country code, where available.
- Countries present in `eval_entities.jsonl` are deliberately excluded.
- The 60 non-country entries (people, landmarks, planets, rivers,
  organizations, and so on) are manually curated.
- Expected-answer lists include ASCII-folded aliases when the canonical value
  uses accents (for example `Brasília` / `Brasilia`).

Regenerating the file requires network access to the source above. Review any
new country facts before running generation: the upstream dataset is a
starting point, not a guarantee of political or linguistic accuracy.

## `eval_entities.jsonl`

Held-out entities used by the contamination check. `train_entities.jsonl` and
`eval_entities.jsonl` must remain disjoint; `task_builder.assert_disjoint_entities`
enforces this.

## `topics.jsonl`

30 everyday topics used by `format_json`, `extraction`, and `instruction`.
Each row has a `schemas` list naming the JSON schemas that make sense for that
topic. Keep this list aligned with `JSON_SCHEMAS` in `distillation/prompts.py`.
