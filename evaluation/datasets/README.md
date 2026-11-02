# Frozen evaluation sets

These JSONL files are frozen before teacher-data generation and must not be
edited while an experiment is running. `manifest.json` pins each file by
SHA256; verify it before training or evaluation:

```bash
conda run -n quin-llm python -m evaluation.freeze_datasets \
  --dir evaluation/datasets --check
```

To intentionally change a dataset, edit the JSONL, regenerate the manifest,
and re-run `evaluation.contamination_check` before training.

## Schema

```json
{
  "id": "fact_unseen_001",
  "category": "fact_qa",
  "split": "unseen_entity",
  "instruction": "What is the capital of Spain?",
  "reference": "Madrid",
  "aliases": ["Madrid"],
  "check": "alias_match",
  "metadata": {"entity": "Spain", "relation": "capital"}
}
```

Optional fields: `input`, `schema` (for `json_schema`), `constraint` (for
`constraint`), `pattern`/`flags` (for `regex`), `keywords` (for `keyword`),
`rubric` (for `llm_judge`/`manual`), `max_new_tokens`.

## Splits

| File | Split | Purpose |
|---|---|---|
| `fact_seen_entity.jsonl` | `seen_entity_seen_template` | Memorization of trained entities/templates |
| `fact_seen_unseen_template.jsonl` | `seen_entity_unseen_template` | Template generalization on trained entities |
| `fact_unseen_entity.jsonl` | `unseen_entity` | Entity generalization; entities disjoint from `distillation/seeds/train_entities.jsonl` |
| `fact_external_floor.jsonl` | `external_floor` | Common facts outside the seed pool; sanity floor |
| `behavior_json.jsonl` | `behavioral` | JSON validity + schema conformance |
| `behavior_extraction.jsonl` | `behavioral` | Span extraction set precision/recall/F1 |
| `behavior_instruction.jsonl` | `behavioral` | Rule-checkable instruction constraints |
| `retention.jsonl` | `retention` | Story/chat retention; LLM-judged |
| `retention_alpaca.jsonl` | `retention` | Alpaca-style held-out instruction retention |

`llm_judge` and `manual` records are generated but left unscored by the
evaluator; their `rubric` is for a blind judge or human pass.

## Contamination rules

- `fact_unseen_entity.jsonl` uses only entities from
  `distillation/seeds/eval_entities.jsonl`.
- `fact_seen_unseen_template.jsonl` changes both the question wording and, for
  several items, the relation (continent/currency/language) relative to the
  generation prompts.
- `evaluation/contamination_check.py` rejects training rows that match an eval
  instruction exactly, form a near-duplicate, or mention an entity from the
  `unseen_entity` split / `eval_entities.jsonl`. Seen-entity splits are not
  entity-checked (their entities are meant to be trained on); `external_floor`
  is protected by exact/near matching only, since some of its subjects
  ("gold", "1984") are generic words.
- `retention_alpaca.jsonl` is drawn from instruction styles not used by the
  teacher task builder.
