# Teacher 蒸馏 SFT 对比报告

生成时间：2026-11-05T13:40:56.984309+00:00

## 运行

- Baseline：`reports/eval_baseline_distill_v2.jsonl`（checkpoints/sft_baseline_distill_v2/best.pt）
- Mixed：`reports/eval_mixed_distill_v2.jsonl`（checkpoints/sft_distill_distill_v2/best.pt）
- 配对样本：78（Binary 64，Scored 64）
- Baseline-only ids：0；Mixed-only ids：0
- Bootstrap：1,000，seed=42

## 总体

- Baseline accuracy：0.031；Mixed accuracy：0.141；Δ = +0.109
- Δ accuracy 95% CI：[+0.016, +0.203]
- McNemar p：0.0654
- Baseline mean score：0.028；Mixed mean score：0.143；Δ = +0.115

### 分类别

| Group | n | Baseline acc | Mixed acc | Δ acc | 95% CI | McNemar p | Baseline score | Mixed score | Δ score |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| behavior_extraction | 6 | 0.000 | 0.167 | +0.167 | [+0.000, +0.500] | 1.0000 | 0.000 | 0.167 | +0.167 |
| behavior_instruction | 6 | 0.000 | 0.167 | +0.167 | [+0.000, +0.500] | 1.0000 | 0.000 | 0.167 | +0.167 |
| behavior_json | 6 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 1.0000 | 0.000 | 0.000 | +0.000 |
| fact_qa | 44 | 0.000 | 0.159 | +0.159 | [+0.068, +0.273] | 0.0156 | 0.000 | 0.159 | +0.159 |
| retention_alpaca | 2 | 1.000 | 0.000 | -1.000 | [-1.000, -1.000] | 0.5000 | 0.900 | 0.062 | -0.838 |
| retention_chat | 0 | — | — | — | — | — | — | — | — |
| retention_story | 0 | — | — | — | — | — | — | — | — |

### 分拆分

| Group | n | Baseline acc | Mixed acc | Δ acc | 95% CI | McNemar p | Baseline score | Mixed score | Δ score |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| behavioral | 18 | 0.000 | 0.111 | +0.111 | [+0.000, +0.278] | 0.5000 | 0.000 | 0.111 | +0.111 |
| external_floor | 8 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 1.0000 | 0.000 | 0.000 | +0.000 |
| retention | 2 | 1.000 | 0.000 | -1.000 | [-1.000, -1.000] | 0.5000 | 0.900 | 0.062 | -0.838 |
| seen_entity_seen_template | 12 | 0.000 | 0.250 | +0.250 | [+0.000, +0.500] | 0.2500 | 0.000 | 0.250 | +0.250 |
| seen_entity_unseen_template | 12 | 0.000 | 0.333 | +0.333 | [+0.083, +0.583] | 0.1250 | 0.000 | 0.333 | +0.333 |
| unseen_entity | 12 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 1.0000 | 0.000 | 0.000 | +0.000 |

### 类别 × 拆分

| Group | n | Baseline acc | Mixed acc | Δ acc | 95% CI | McNemar p | Baseline score | Mixed score | Δ score |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| behavior_extraction/behavioral | 6 | 0.000 | 0.167 | +0.167 | [+0.000, +0.500] | 1.0000 | 0.000 | 0.167 | +0.167 |
| behavior_instruction/behavioral | 6 | 0.000 | 0.167 | +0.167 | [+0.000, +0.500] | 1.0000 | 0.000 | 0.167 | +0.167 |
| behavior_json/behavioral | 6 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 1.0000 | 0.000 | 0.000 | +0.000 |
| fact_qa/external_floor | 8 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 1.0000 | 0.000 | 0.000 | +0.000 |
| fact_qa/seen_entity_seen_template | 12 | 0.000 | 0.250 | +0.250 | [+0.000, +0.500] | 0.2500 | 0.000 | 0.250 | +0.250 |
| fact_qa/seen_entity_unseen_template | 12 | 0.000 | 0.333 | +0.333 | [+0.083, +0.583] | 0.1250 | 0.000 | 0.333 | +0.333 |
| fact_qa/unseen_entity | 12 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 1.0000 | 0.000 | 0.000 | +0.000 |
| retention_alpaca/retention | 2 | 1.000 | 0.000 | -1.000 | [-1.000, -1.000] | 0.5000 | 0.900 | 0.062 | -0.838 |
| retention_chat/retention | 0 | — | — | — | — | — | — | — | — |
| retention_story/retention | 0 | — | — | — | — | — | — | — | — |

## 代表性变化

### Mixed 变好

- **behavior_extraction/behavioral · extract_006**
  - Q: Extract the amount of money from this passage as a JSON array of strings: The invoice totals $245.50, due within 30 days.
  - Ref: ['$245.50']
  - Baseline: The total number of strings is $2.50, which is $2.50.
  - Mixed: ["$245.50"]
- **fact_qa/seen_entity_seen_template · fact_seen_001**
  - Q: What is the capital of France?
  - Ref: Paris
  - Baseline: The capital of France is France.
  - Mixed: The capital of France is Paris.
- **fact_qa/seen_entity_seen_template · fact_seen_008**
  - Q: What is the capital of Germany?
  - Ref: Berlin
  - Baseline: The capital of Germany is Germany.
  - Mixed: The capital of Germany is Berlin.
- **fact_qa/seen_entity_seen_template · fact_seen_009**
  - Q: What is the capital of Italy?
  - Ref: Rome
  - Baseline: The capital of Italy is Italy.
  - Mixed: The capital of Italy is Rome.
- **fact_qa/seen_entity_unseen_template · fact_seen_tpl_002**
  - Q: On which continent is Japan located?
  - Ref: Asia
  - Baseline: The continent is Japan located in Japan.
  - Mixed: Nutsia is located in Asia.

### Mixed 变差（遗忘/回归）

- **retention_alpaca/retention · retain_alpaca_002**
  - Q: Classify the sentiment of this review as positive, negative, or neutral: The food was cold and the waiter was rude.
  - Ref: negative
  - Baseline: Negative
  - Mixed: The food was cold and the waiter was rude.
- **retention_alpaca/retention · retain_alpaca_004**
  - Q: Correct the grammar in this sentence: She go to the store yesterday and buyed some apples.
  - Ref: She went to the store yesterday and bought some apples.
  - Baseline: She goes to the store yesterday and buy some apples.
  - Mixed: Correct the grammar in the morning.

## 未自动判分的开放题（retention）

共 14 条。以下仅展示输出，quality 需人工或独立 judge 评分。

- **retain_alpaca_001** — Rewrite the following sentence to be more formal: The meeting was kinda boring and we didn't get much done.
  - Baseline: The meeting was a great way to get started.
  - Mixed: The meeting was a great way to get started.
- **retain_alpaca_003** — Summarize this paragraph in one sentence: Solar panels convert sunlight into electricity. They are increasingly used on rooftops because they reduce electricit…
  - Baseline: Solar panels convert sunlight into electricity.
  - Mixed: The solar panels convert sunlight into electricity.
- **retain_alpaca_005** — Give a short explanation of the difference between weather and climate.
  - Baseline: Solar and climate are both factors that can be seen in the past. climate is a major source of greenhouse gases, which can cause a variety of effects on the environment, including extreme weather events, extreme weather events, and extreme weather events.
  - Mixed: Spring is primarily cold and wet. The temperature of the sky is lower than the sun. The temperature of the sky is lower than the sun. The temperature of the sky is lower than the sun. The temperature of the sky is lower than the sun. The temperature of the sky is lower than the sun. The temperature…

## 说明与限制

- 规则判分优先；alias 匹配对短答案较稳，但对长句和否定句可能偏宽松。
- 开放题未自动评分，retention 结论需结合人工检查。
- 只有配对样本参与比较；缺失 id 不会补 0 分。
- 若两组训练步数、数据量或超参不同，本报告不能单独证明教师数据的作用。
