# quin-llm

一个从零手写的小型 decoder-only LLM：自己的 byte-level BPE、Transformer、训练循环和采样逻辑。目标是在单卡（V100 32GB）上训练出一个能生成通顺英文、并做简单指令跟随的小模型。

## 快速开始

### 1. 环境

项目所有 Python 依赖都在 `requirements.txt` 中。先创建环境，再用该文件安装：

```bash
conda create -n quin-llm python=3.11 -y
conda activate quin-llm
```

本地 CPU 安装：

```bash
pip install -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
```

V100 / CUDA 安装（PyPI 默认 torch wheel 带 CUDA）：

```bash
pip install -r requirements.txt
```

如果服务器需要显式指定 CUDA 12.4：

```bash
pip install -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
```

### 2. 本地 CPU 冒烟测试（mini）

`mini` 模式使用项目内自带的小语料，只用于验证代码能跑通，不会得到有意义的模型。

一键从 tokenize 跑到 sft：

```bash
./run_pipeline.sh --start tokenize --mode mini
```

或者分步执行：

```bash
python main.py tokenize --config configs/tokenizer_mini.yaml
python main.py prepare-pretrain --config configs/pretrain_mini.yaml
python main.py pretrain --config configs/pretrain_mini.yaml
python main.py prepare-sft --config configs/sft_mini.yaml --ckpt checkpoints/pretrain_mini/best.pt
python main.py sft --config configs/sft_mini.yaml --ckpt checkpoints/pretrain_mini/best.pt
python main.py generate --ckpt checkpoints/sft_mini/best.pt --prompt "What is a cat?"
```

进入 mini 模型的交互式对话：

```bash
./run_chat.sh --mode mini
```

单元测试：

```bash
python -m unittest discover -s tests -v
```

### 3. V100 上的 GPU 冒烟测试

正式训练前建议先跑一次，确认 CUDA、fp16 AMP、checkpoint 和生成都正常：

```bash
python scripts/gpu_smoke.py
```

## 入口与脚本

`main.py` 仍然保留全部子命令，两个 shell 脚本只是常用流程的封装。

### run_pipeline.sh

```bash
./run_pipeline.sh [--start STAGE] [--end STAGE] [--mode full|mini] [--save-log true|false]
```

- `start` 可选：`tokenize`、`prepare-pretrain`、`pretrain`、`prepare-sft`、`sft`
- `end` 可选：同上
- 执行闭区间 `[start, end]`，开始和结束阶段都包含
- `start` 默认是 `pretrain`
- `end` 默认是 `sft`
- `mode` 可选：`full`（默认，V100 完整配置）或 `mini`（纯 CPU 冒烟配置）
- `save-log` 可选：`true`（默认，除 tqdm 进度条外的全部输出写入 `log/pipeline_*.log`，进度条只显示在终端）或 `false`（不写日志文件）

示例：

```bash
./run_pipeline.sh                                          # full：pretrain -> sft
./run_pipeline.sh --start tokenize                        # full：tokenize -> sft
./run_pipeline.sh --start prepare-pretrain --end pretrain # full：只跑这两个阶段
./run_pipeline.sh --start tokenize --end pretrain --mode mini
./run_pipeline.sh --start prepare-sft --mode mini         # mini：prepare-sft -> sft
./run_pipeline.sh --save-log false                        # 不写日志文件
```

默认从 `pretrain` 开始时，前提是 tokenizer 和预训练数据已经准备好；如果需要从头生成，请显式传入 `--start tokenize`。

### run_chat.sh

```bash
./run_chat.sh                      # full 的 SFT checkpoint
./run_chat.sh --mode mini          # CPU 冒烟模型
./run_chat.sh --save-log false     # 不写日志文件
CKPT=checkpoints/sft_full/final.pt ./run_chat.sh
./run_chat.sh --temperature 0.7 --max-new-tokens 300
```

`run_pipeline.sh` 默认把除 tqdm 进度条外的全部输出同时写入终端和 `log/pipeline_*.log`；进度条直接写终端，不进入日志文件。没有控制终端（例如 `nohup`）时会自动关闭进度条。`run_chat.sh` 默认把终端输出同时写入 `log/chat_*.log`。两者传 `--save-log false` 都关闭日志文件。

### 配置文件的分工

- 根目录 `config.py`：按阶段定义配置类（`TokenizerConfig`、`PretrainConfig`、`SFTConfig` 等），负责把 YAML 的对应 section 解析成类型化对象。
- `configs/schema.py`：底层通用结构 `ModelConfig`、`TrainConfig`、`load_yaml`、`ensure_dir`。
- `configs/*.yaml`：各阶段实际使用的超参数和数据源。

## 完整训练流程

```bash
# 一键执行 full 模式：tokenize -> prepare-pretrain -> pretrain -> prepare-sft -> sft
./run_pipeline.sh --start tokenize

# 交互式生成
./run_chat.sh
```

也可以分步执行：

```bash
# 0. 下载语料到 data/raw（默认走 ModelScope，服务器可直接运行；可断点续传）
./scripts/download_data.sh

# 1. 训练 tokenizer（16384 BPE）
python main.py tokenize --config configs/tokenizer.yaml

# 2. 预处理预训练语料（TinyStories + wikitext-103）
python main.py prepare-pretrain --config configs/pretrain_full.yaml

# 3. 预训练（约 27M 参数）
python main.py pretrain --config configs/pretrain_full.yaml

# 4. 预处理 Alpaca 指令数据（token-level answer mask）
python main.py prepare-sft --config configs/sft_full.yaml --ckpt checkpoints/pretrain_full/best.pt

# 5. SFT
python main.py sft --config configs/sft_full.yaml --ckpt checkpoints/pretrain_full/best.pt

# 6. 交互式生成
python main.py chat --ckpt checkpoints/sft_full/best.pt
```

断点续训：

```bash
python main.py pretrain --config configs/pretrain_full.yaml \
    --resume checkpoints/pretrain_full/step_1000.pt
python main.py sft --config configs/sft_full.yaml \
    --ckpt checkpoints/pretrain_full/best.pt \
    --resume checkpoints/sft_full/step_500.pt
```

## 教师蒸馏实验（Baseline vs Mixed）

用 OpenAI-API-Compatible 教师（Qwen3.5）生成 SFT 数据，与 Alpaca 按 **answer
token 50:50** 混合，训练 Mixed 模型；同一起点、同一 `max_steps` 的 Alpaca-only
Baseline 作为对照。**不做 teacher-only SFT**。

### 1. 配置教师凭证

编辑 [configs/distill_teacher.yaml](configs/distill_teacher.yaml)，直接填：

```yaml
base_url: https://your-endpoint/v1
model: qwen3.5
api_key: "sk-..."
```

也可以把 key 放在单独文件（`api_key_file`）或环境变量（`api_key_env`）。优先级：
`api_key` > `api_key_file` > `api_key_env`。不想改受 git 管理的文件时：

```bash
cp configs/distill_teacher.yaml configs/distill_teacher.local.yaml
# 在 local 文件里填 api_key
export TEACHER_CONFIG=configs/distill_teacher.local.yaml
```

本地只测 API 不需要 GPU、checkpoint 或 Alpaca 数据：

```bash
# 只打印 prompt，不发请求
./scripts/distill/01_generate.sh --dry-run --limit 5

# 1 次 ping + 1 条真实生成，写到 data/distill/raw/api_smoke.jsonl
./scripts/distill/99_api_smoke.sh

# 多测几条 / 指定类别
./scripts/distill/99_api_smoke.sh --limit 5 --categories fact_qa,format_json

# 查看原始响应（每行一条 JSON）
head -1 data/distill/raw/api_smoke.jsonl
```

常见错误：401/403 检查 `api_key`；400 检查 `model` 名称和接口是否支持 `seed` /
`response_format`。如果响应里有很长的 `reasoning_content`（表现为 completion
token 远大于答案本身、请求很慢），说明 thinking 模式没关掉。多数 Qwen / vLLM
接口要用：

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: false
```

顶层 `enable_thinking: false` 可能被代理忽略。

### 2. 按编号执行

先进入 `quin-llm` 环境（或设置 `PYTHON` 指向该环境的解释器）：

```bash
conda activate quin-llm
```

所有步骤都在 [scripts/distill/](scripts/distill/)，默认 `RUN_ID=pilot_v1`：

| 脚本 | 作用 |
|---|---|
| `00_preflight.sh` | 检查依赖、数据、checkpoint、API；写 `baseline_budget.json` |
| `01_generate.sh` | 生成教师数据（默认 2,000 条 pilot） |
| `02_filter.sh` | 规则过滤、去重、污染检查，生成 `review.csv` |
| `03_apply_review.sh` | 把复核 CSV 的结论写回数据（不是再次审阅） |
| `04_build_dataset.sh` | 教师数据 + Alpaca 按 token 50:50 生成 parquet |
| `05_contamination_check.sh` | 最终训练集 vs 冻结评测集污染检查 |
| `06_make_configs.sh` | 生成受控 Baseline / Mixed 配置 |
| `07_train_baseline.sh` | `prepare-sft` + 训练 Baseline |
| `08_train_mixed.sh` | `prepare-sft` + 训练 Mixed |
| `09_eval_baseline.sh` | Baseline 评测 |
| `10_eval_mixed.sh` | Mixed 评测 |
| `11_compare.sh` | 生成对比报告 |
| `12_val_loss.sh` | 预训练验证集 loss（遗忘监控） |
| `99_api_smoke.sh` | 本地 API 连通性测试（ping + 1 条生成，不属于训练流程） |

执行顺序：

```bash
./scripts/distill/00_preflight.sh
./scripts/distill/01_generate.sh
./scripts/distill/02_filter.sh

# 人工复核：打开 data/distill/filtered/pilot_v1.review.csv
#   model_verdict / model_notes  第一遍（模型先筛）
#   human_verdict / human_notes  第二遍（人工确认；留空沿用上一遍）
./scripts/distill/03_apply_review.sh   # 只把 CSV 结论写回数据

./scripts/distill/04_build_dataset.sh
./scripts/distill/05_contamination_check.sh
./scripts/distill/06_make_configs.sh
./scripts/distill/07_train_baseline.sh
./scripts/distill/08_train_mixed.sh
./scripts/distill/09_eval_baseline.sh
./scripts/distill/10_eval_mixed.sh
./scripts/distill/11_compare.sh
./scripts/distill/12_val_loss.sh
```

脚本都支持追加参数，例如：

```bash
./scripts/distill/00_preflight.sh --ping
./scripts/distill/01_generate.sh --dry-run --limit 5
./scripts/distill/01_generate.sh --limit 5
```

> 生成只依赖 HTTP，本地和服务器都能跑。2,000 条 pilot 的 raw JSONL 约 6 MB、
> tasks 约 0.7 MB，过滤后约 1.6 MB。如果服务器不能访问教师 API，可以在本地跑
> `01_generate.sh`，把 `data/distill/raw/<run_id>.jsonl` 拷到服务器，再从
> `02_filter.sh` 继续（服务器上的 tokenizer 会让长度检查更准确）。

### 3. Pilot 门禁与 40k 主实验

Pilot 通过条件：事实类人工复核通过率 ≥ 90%、无污染、overlong/重复比例可接受、
Baseline/Mixed 步数一致且评测链路跑通、retention 无明显退化。

通过后先把 [configs/distill_generation.yaml](configs/distill_generation.yaml) 的
各类别 count 扩大到约 40,000，再用新的 run id 重跑同一套脚本：

```bash
export RUN_ID=main_v1
./scripts/distill/01_generate.sh
./scripts/distill/02_filter.sh
# 复核并 merge
./scripts/distill/03_apply_review.sh
./scripts/distill/04_build_dataset.sh
# 后续 05~12 相同
```

### 4. 可覆盖变量与关键产物

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RUN_ID` | `pilot_v1` | 数据 / checkpoint / 报告命名 |
| `PYTHON` | `python` | 未激活环境时指向 `quin-llm` 解释器 |
| `TEACHER_CONFIG` | `configs/distill_teacher.yaml` | 教师凭证配置 |
| `REPORT_DIR` | `reports` | 评测与报告输出目录 |
| `PRETRAIN_CKPT` | `checkpoints/pretrain_full/best.pt` | SFT 起点 |

```text
data/distill/raw/<run_id>.jsonl
data/distill/filtered/<run_id>.jsonl / .rejected.jsonl / .review.csv
data/distill/sft/<run_id>/train.parquet / manifest.json
configs/generated/sft_baseline_<run_id>.yaml / sft_distill_<run_id>.yaml
checkpoints/sft_baseline_<run_id>/ / checkpoints/sft_distill_<run_id>/
reports/eval_*_<run_id>.jsonl / distill_<run_id>.md
```

代码结构见 [distillation/](distillation/) 和 [evaluation/](evaluation/)；评测集已冻结，
可用 `python -m evaluation.freeze_datasets --dir evaluation/datasets --check` 校验。

## 模型与数据

默认 `configs/pretrain_full.yaml`：

| 项目 | 值 |
|---|---|
| 结构 | Pre-LN decoder-only Transformer |
| 层数 | 6 |
| hidden | 512 |
| heads | 8 |
| context | 512 |
| vocab | 16384（byte-level BPE） |
| 参数量 | 约 27.5M（非嵌入约 18.9M） |
| 预训练数据 | TinyStories 60% + wikitext-103 40% |
| SFT 数据 | Alpaca 52k |

训练要点：

- byte-level BPE，无 UNK；特殊 token：`<PAD>=256, <BOS>=257, <EOS>=258, <UNK>=259`
- 预训练使用随机窗口，文档之间插入 BOS/EOS
- SFT 先 tokenize 整段对话，再按 token 把 prompt 部分的 label 置为 `-100`
- V100 使用 fp16 + `GradScaler`（V100 不支持 bf16）
- AdamW + cosine schedule + warmup + grad clip；full 配置下预训练 `batch_size=128, lr=6e-3`，SFT `batch_size=32, lr=3e-4`
- 早停：`early_stop_patience` 为连续多少次评估 `val_loss` 无提升后停止，`null` 表示关闭；`min_delta` 是最小改善量
- checkpoint 保存完整 `model_config`、optimizer、step、tokenizer 路径

## 目录结构

```
config.py               各阶段配置类
configs/
  __init__.py           使 configs 成为 Python 包
  schema.py             底层 ModelConfig / TrainConfig / YAML 工具
  tokenizer.yaml        full tokenizer 配置
  pretrain_full.yaml   full 预训练配置
  sft_full.yaml        full SFT 配置
  tokenizer_mini.yaml   mini tokenizer 配置
  pretrain_mini.yaml    mini 预训练配置
  sft_mini.yaml         mini SFT 配置
  distill_teacher.yaml  教师 API 配置（key 可写在 YAML / 文件 / 环境变量）
  distill_generation.yaml  教师数据生成配比
  distill_filter.yaml   教师数据过滤阈值
  distill_dataset.yaml  Alpaca + teacher token 混合
  sft_baseline_controlled.yaml  受控 Baseline 模板
  sft_distill.yaml      Mixed 模板
  eval.yaml             冻结评测集与 greedy 解码
  generated/            preflight/make_configs 生成的受控配置（运行时）
distillation/           教师数据生成、过滤、复核、建集
  seeds/                训练 / held-out 实体和主题
  prompts.py            版本化 prompt
  teacher_client.py     OpenAI-compatible client
  task_builder.py       确定性任务列表
  data_generator.py     并发生成与断点续传
  data_filter.py        规则过滤、去重、污染检查、复核表
  dataset_builder.py    Alpaca + teacher 混合
  preflight.py          环境 / 数据 / 步数预检
  make_configs.py       生成受控训练配置
  run_*.py              各阶段 CLI
evaluation/             冻结评测集、指标、对比和污染检查
  datasets/             评测 JSONL + manifest + README
  metrics.py            规则指标与统计检验
  evaluator.py          checkpoint 推理与判分
  run_eval.py           评测 CLI
  compare_models.py     Baseline vs Mixed 报告
  contamination_check.py  污染检查
  freeze_datasets.py    评测集 hash 固化
  val_loss.py           预训练验证集 loss / perplexity
pipeline/               按阶段组织的流水线代码
  common.py             共享的 tokenizer / device / checkpoint 加载
  sources.py            本地文件 / HuggingFace 数据读取
  fixtures/
    tiny_corpus.txt     本地冒烟测试小语料
    tiny_sft_message.txt 本地冒烟测试 SFT 小语料
  tokenizer/
    bpe_tokenizer.py    byte-level BPE
    train.py            tokenizer 训练入口
  pretrain/
    prepare.py          预训练数据预处理
    dataset.py          memmap dataset
    train.py            预训练入口
  sft/
    prepare.py          SFT 数据预处理
    dataset.py          SFT / replay dataset
    train.py            prepare-sft 与 sft 入口
processed_data/         运行后生成的数据产物（可重新生成）
  tokenizer/            full 训练好的 tokenizer
  pretrain/             full train.bin / val.bin
  sft/                  full train.pt / val.pt
  tokenizer_mini/       mini tokenizer
  pretrain_mini/        mini train.bin / val.bin
  sft_mini/             mini train.pt / val.pt
data/distill/           教师任务、raw 响应、过滤结果、混合 parquet（运行时）
  tasks/               确定性任务列表
  raw/                 教师原始响应（可断点续传）
  filtered/            通过/拒绝数据、统计、复核 CSV
  sft/<run_id>/        train.parquet + manifest.json
checkpoints/            模型权重（受控实验使用独立目录）
reports/                评测输出与 Markdown 对比报告
log/                    运行日志（两个 shell 脚本默认写入）
legacy/                 旧版本文件（不参与训练）
model/
  model.py              TinyModel
  units/                attention / MLP / block / embeddings / head
trainer/                训练循环（AMP、eval、checkpoint、日志）
utils/                  日志、绘图、随机种子
scripts/gpu_smoke.py    V100 冒烟测试
scripts/distill/        教师蒸馏实验的编号步骤脚本（00~12）
tests/                  单元测试
main.py                 CLI 入口
run_pipeline.sh         按 start/end 闭区间执行阶段
run_chat.sh             加载 checkpoint 进入 chat
```

`pipeline/` 和 `processed_data/` 的分工：

- `pipeline/` 是代码，按阶段回答“数据怎么读、怎么 tokenize、怎么构造样本、怎么训练”。
- `processed_data/` 是产物，回答“tokenizer 和 token 张量在哪里”，里面的内容由 `tokenize` / `prepare-pretrain` / `prepare-sft` 生成，可以删除后重建。

## 与旧版本的区别

旧版本的两个核心问题已经移除：

1. **模型配置与 checkpoint 不一致**：现在 checkpoint 自带完整 `model_config`，推理时从 checkpoint 重建模型。
2. **SFT 的字符/token 混用**：旧实现按字符截断 128 字符、按字符索引 mask、按字符平移 x/y，并会被答案里的空行切碎。新实现全部按 token 处理，并按 `###Agent:` 的 token 位置 mask prompt。

旧 checkpoint、旧 tokenizer 不再兼容，也不需要兼容。

## 已知限制

- 约 27M 参数的模型容量有限，适合简单英文叙事和短指令，不适合复杂推理或知识问答。
- 默认输出英文；如果要中文，需要替换为中文预训练/SFT 语料并重新训练 tokenizer。
- `datasets` 首次运行会从 HuggingFace 下载数据，服务器需要能访问外网。
