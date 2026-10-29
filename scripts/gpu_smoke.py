"""V100/CUDA smoke test: AMP forward/backward, checkpoint, generation.

Run from the repository root:
    python scripts/gpu_smoke.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ModelConfig  # noqa: E402
from model.model import TinyModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available on this machine")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}")
    print(f"memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    config = ModelConfig(
        vocab_size=8192,
        block_size=args.block_size,
        n_layer=6,
        n_embd=384,
        n_head=6,
        dropout=0.1,
        tie_weights=True,
        use_sdpa=True,
    )
    model = TinyModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    x = torch.randint(0, config.vocab_size, (args.batch_size, config.block_size), device=device)
    y = torch.randint(0, config.vocab_size, (args.batch_size, config.block_size), device=device)

    model.train()
    start = time.time()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 5 == 0:
            print(f"step {step}: loss={loss.item():.4f} grad_norm={grad_norm.item():.3f}")
        if not torch.isfinite(loss):
            raise SystemExit("loss is not finite; fp16 training is unstable")

    elapsed = time.time() - start
    tokens = args.steps * args.batch_size * config.block_size
    print(f"throughput: {tokens / elapsed:,.0f} tok/s")

    ckpt_path = ROOT / "checkpoints" / "gpu_smoke.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_config": config.to_dict()}, ckpt_path)
    reloaded = TinyModel(config)
    reloaded.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state"])
    reloaded.to(device).eval()
    print(f"checkpoint round-trip ok: {ckpt_path}")

    prompt = torch.randint(0, config.vocab_size, (1, 8), device=device)
    with torch.no_grad():
        out = reloaded.generate(prompt, max_new_tokens=8, temperature=0.8)
    print(f"generation ok: {tuple(out.shape)}")
    print("GPU smoke test passed")


if __name__ == "__main__":
    main()
