"""Evaluation utilities for quin-llm checkpoints.

The evaluator reuses the training prompt template and greedy decoding, then
scores each frozen JSONL example with a rule-based check.  Open-ended retention
prompts are left unscored and saved for manual or independent LLM judging.
"""
