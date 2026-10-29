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
./run_pipeline.sh tokenize mini
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
./run_pipeline.sh [phase] [mode]
```

- `phase` 可选：`tokenize`、`prepare-pretrain`、`pretrain`、`prepare-sft`、`sft`
- 从 `phase` 开始，按顺序执行到 `sft` 结束
- `phase` 默认是 `pretrain`
- `mode` 可选：`full`（默认，V100 完整配置）或 `mini`（纯 CPU 冒烟配置）
- 也可以用 `./run_pipeline.sh --mode mini tokenize` 这种写法

示例：

```bash
./run_pipeline.sh tokenize          # full：从 tokenize 跑到 sft
./run_pipeline.sh pretrain          # full：从 pretrain 跑到 sft
./run_pipeline.sh tokenize mini     # mini：从 tokenize 跑到 sft
./run_pipeline.sh prepare-sft mini  # mini：从 prepare-sft 跑到 sft
```

默认从 `pretrain` 开始时，前提是 tokenizer 和预训练数据已经准备好；如果需要从头生成，请显式传入 `tokenize`。

### run_chat.sh

```bash
./run_chat.sh                      # full 的 SFT checkpoint
./run_chat.sh --mode mini          # CPU 冒烟模型
CKPT=checkpoints/sft_full/final.pt ./run_chat.sh
./run_chat.sh --temperature 0.7 --max-new-tokens 300
```

### 配置文件的分工

- 根目录 `config.py`：按阶段定义配置类（`TokenizerConfig`、`PretrainConfig`、`SFTConfig` 等），负责把 YAML 的对应 section 解析成类型化对象。
- `configs/schema.py`：底层通用结构 `ModelConfig`、`TrainConfig`、`load_yaml`、`ensure_dir`。
- `configs/*.yaml`：各阶段实际使用的超参数和数据源。

## 完整训练流程

```bash
# 一键执行 full 模式：tokenize -> prepare-pretrain -> pretrain -> prepare-sft -> sft
./run_pipeline.sh tokenize

# 交互式生成
./run_chat.sh
```

也可以分步执行：

```bash
# 1. 训练 tokenizer（8192 BPE）
python main.py tokenize --config configs/tokenizer.yaml

# 2. 预处理预训练语料（TinyStories + wikitext-103）
python main.py prepare-pretrain --config configs/pretrain_full.yaml

# 3. 预训练（约 14M 参数）
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

## 模型与数据

默认 `configs/pretrain_full.yaml`：

| 项目 | 值 |
|---|---|
| 结构 | Pre-LN decoder-only Transformer |
| 层数 | 6 |
| hidden | 384 |
| heads | 6 |
| context | 512 |
| vocab | 8192（byte-level BPE） |
| 参数量 | 约 14M |
| 预训练数据 | TinyStories 60% + wikitext-103 40% |
| SFT 数据 | Alpaca 52k |

训练要点：

- byte-level BPE，无 UNK；特殊 token：`<PAD>=256, <BOS>=257, <EOS>=258, <UNK>=259`
- 预训练使用随机窗口，文档之间插入 BOS/EOS
- SFT 先 tokenize 整段对话，再按 token 把 prompt 部分的 label 置为 `-100`
- V100 使用 fp16 + `GradScaler`（V100 不支持 bf16）
- AdamW + cosine schedule + warmup + grad clip
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
data_pipeline/          数据处理的源代码（怎么造数据）
  sources.py            本地文件 / HuggingFace 数据读取
  tokenizer/            byte-level BPE
  pretrain/             预训练数据预处理与 memmap dataset
  sft/                  SFT 数据预处理与 dataset
  tiny_corpus.txt       本地冒烟测试小语料
  tiny_sft_message.txt  本地冒烟测试 SFT 小语料
processed_data/         运行后生成的数据产物（可重新生成）
  tokenizer/            full 训练好的 tokenizer
  pretrain/             full train.bin / val.bin
  sft/                  full train.pt / val.pt
  tokenizer_mini/       mini tokenizer
  pretrain_mini/        mini train.bin / val.bin
  sft_mini/             mini train.pt / val.pt
checkpoints/            模型权重
legacy/                 旧版本文件（不参与训练）
model/
  model.py              TinyModel
  units/                attention / MLP / block / embeddings / head
trainer/                训练循环（AMP、eval、checkpoint、日志）
utils/                  日志、绘图、随机种子
scripts/gpu_smoke.py    V100 冒烟测试
tests/                  单元测试
main.py                 CLI 入口
run_pipeline.sh         从指定阶段跑到 sft
run_chat.sh             加载 checkpoint 进入 chat
```

`data_pipeline/` 和 `processed_data/` 的分工：

- `data_pipeline/` 是代码，回答“数据怎么读、怎么 tokenize、怎么构造样本”。
- `processed_data/` 是产物，回答“tokenizer 和 token 张量在哪里”，里面的内容由 `tokenize` / `prepare-pretrain` / `prepare-sft` 生成，可以删除后重建。

## 与旧版本的区别

旧版本的两个核心问题已经移除：

1. **模型配置与 checkpoint 不一致**：现在 checkpoint 自带完整 `model_config`，推理时从 checkpoint 重建模型。
2. **SFT 的字符/token 混用**：旧实现按字符截断 128 字符、按字符索引 mask、按字符平移 x/y，并会被答案里的空行切碎。新实现全部按 token 处理，并按 `###Agent:` 的 token 位置 mask prompt。

旧 checkpoint、旧 tokenizer 不再兼容，也不需要兼容。

## 已知限制

- 14M 参数的模型容量有限，适合简单英文叙事和短指令，不适合复杂推理或知识问答。
- 默认输出英文；如果要中文，需要替换为中文预训练/SFT 语料并重新训练 tokenizer。
- `datasets` 首次运行会从 HuggingFace 下载数据，服务器需要能访问外网。
