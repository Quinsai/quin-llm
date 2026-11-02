"""Teacher-model data distillation for quin-llm.

Pipeline stages live in this package:

    task_builder -> data_generator -> data_filter -> dataset_builder

The package is intentionally independent of the training loop.  It writes
parquet/JSONL artifacts that the existing ``main.py prepare-sft`` and
``main.py sft`` commands consume unchanged.
"""
