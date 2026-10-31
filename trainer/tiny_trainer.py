"""Training loop for pretraining and supervised fine-tuning."""

from __future__ import annotations

import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import TrainConfig
from utils.logger import CSVLogger
from utils.visualizer import draw_curves


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        cfg: TrainConfig,
        tokenizer_path: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.use_amp = bool(cfg.amp and self.device.type == "cuda")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError, RuntimeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.tokenizer_path = str(tokenizer_path or "")
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.train_history: list[dict] = []
        self.val_history: list[dict] = []

        print(
            f"trainer: device={self.device}, amp={self.use_amp}, "
            f"params={self.model.num_parameters():,} "
            f"({self.model.num_parameters(non_embedding=True):,} non-embedding)"
        )

    # ------------------------------------------------------------------
    # basics
    # ------------------------------------------------------------------
    def _autocast(self):
        if self.use_amp:
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @staticmethod
    def _worker_init(worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        # Python's fork copies the parent RNG state into every worker; reseed
        # so workers do not all sample the same pretraining windows.
        seed = info.seed % (2**31)
        random.seed(seed)
        dataset = info.dataset
        if hasattr(dataset, "seed_rng"):
            dataset.seed_rng(seed)

    def _make_loader(
        self,
        dataset: Dataset,
        shuffle: bool,
        drop_last: bool | None = None,
    ) -> DataLoader:
        if drop_last is None:
            drop_last = shuffle
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=self.cfg.num_workers > 0,
            worker_init_fn=self._worker_init if self.cfg.num_workers > 0 else None,
            generator=generator,
        )

    def _to_device(self, batch):
        x, y = batch
        return (
            x.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True),
        )

    def _lr_at(self, step: int, total_steps: int) -> float:
        if self.cfg.warmup_steps > 0 and step < self.cfg.warmup_steps:
            return self.cfg.lr * (step + 1) / self.cfg.warmup_steps
        progress = (step - self.cfg.warmup_steps) / max(1, total_steps - self.cfg.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return self.cfg.min_lr + 0.5 * (self.cfg.lr - self.cfg.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    # ------------------------------------------------------------------
    # evaluation / checkpoints
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for i, batch in enumerate(loader):
            if self.cfg.eval_batches and i >= self.cfg.eval_batches:
                break
            x, y = self._to_device(batch)
            with self._autocast():
                _, loss = self.model(x, y)
            total += loss.item()
            count += 1
        self.model.train()
        return total / max(1, count)

    def save_checkpoint(self, path: str | Path, step: int, val_loss: float | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_state": self.scaler.state_dict() if self.use_amp else None,
                "step": int(step),
                "val_loss": val_loss,
                "model_config": self.model.config.to_dict(),
                "tokenizer_path": self.tokenizer_path,
                "vocab_size": self.tokenizer.vocab_size,
                "special_ids": {k: int(v) for k, v in self.tokenizer.special_ids.items()},
            },
            path,
        )

    @staticmethod
    def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
        return torch.load(path, map_location=map_location, weights_only=False)

    def load_pretrained(self, path: str | Path) -> dict:
        checkpoint = self.load_checkpoint(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        return checkpoint

    def resume(self, path: str | Path) -> int:
        checkpoint = self.load_checkpoint(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.use_amp and checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        return int(checkpoint.get("step", 0))

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def _run(
        self,
        train_loader: Iterable,
        val_loader: DataLoader | None,
        max_steps: int,
        tag: str,
        start_step: int = 0,
    ) -> float:
        self.model.train()
        step = start_step
        accum_count = 0
        accum_loss = 0.0
        tokens_seen = 0
        best_val = float("inf")
        best_path = self.out_dir / "best.pt"
        if start_step > 0 and best_path.exists():
            previous = self.load_checkpoint(best_path)
            if previous.get("val_loss") is not None:
                best_val = float(previous["val_loss"])
        start_time = time.time()

        logger = CSVLogger(
            self.out_dir / f"{tag}_log.csv",
            ["step", "train_loss", "val_loss", "lr", "tokens_per_sec"],
        )
        pbar = tqdm(
            total=max_steps,
            initial=step,
            desc=tag,
            unit="step",
            dynamic_ncols=True,
        )
        try:
            while step < max_steps:
                for batch in train_loader:
                    x, y = self._to_device(batch)
                    with self._autocast():
                        _, loss = self.model(x, y)
                    self.scaler.scale(loss / self.cfg.grad_accum_steps).backward()
                    accum_loss += loss.item()
                    tokens_seen += x.numel()
                    accum_count += 1

                    if accum_count % self.cfg.grad_accum_steps != 0:
                        continue

                    lr = self._lr_at(step, max_steps)
                    for group in self.optimizer.param_groups:
                        group["lr"] = lr
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    step += 1
                    pbar.update(1)

                    train_loss = accum_loss / self.cfg.grad_accum_steps
                    accum_loss = 0.0
                    elapsed = max(1e-9, time.time() - start_time)
                    tokens_per_sec = tokens_seen / elapsed
                    pbar.set_postfix(
                        train_loss=f"{train_loss:.4f}",
                        lr=f"{lr:.1e}",
                        tok_s=f"{tokens_per_sec:,.0f}",
                    )

                    if step == 1 or step % self.cfg.log_interval == 0:
                        pbar.write(
                            f"[{tag}] step {step}/{max_steps} "
                            f"train_loss {train_loss:.4f} lr {lr:.2e} "
                            f"{tokens_per_sec:,.0f} tok/s"
                        )
                        logger.log(
                            step=step,
                            train_loss=train_loss,
                            val_loss="",
                            lr=lr,
                            tokens_per_sec=tokens_per_sec,
                        )
                        self.train_history.append(
                            {"step": step, "loss": train_loss}
                        )

                    if (
                        val_loader is not None
                        and self.cfg.eval_interval
                        and step % self.cfg.eval_interval == 0
                    ):
                        val_loss = self.evaluate(val_loader)
                        pbar.write(f"[{tag}] step {step} val_loss {val_loss:.4f}")
                        logger.log(
                            step=step,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            lr=lr,
                            tokens_per_sec=tokens_per_sec,
                        )
                        self.val_history.append(
                            {"step": step, "val_loss": val_loss}
                        )
                        if val_loss < best_val:
                            best_val = val_loss
                            self.save_checkpoint(self.out_dir / "best.pt", step, val_loss)
                        self.model.train()

                    if self.cfg.save_interval and step % self.cfg.save_interval == 0:
                        self.save_checkpoint(self.out_dir / f"step_{step}.pt", step)

                    if step >= max_steps:
                        break
        finally:
            pbar.close()
            logger.close()

        final_val = best_val if best_val < float("inf") else None
        self.save_checkpoint(self.out_dir / "final.pt", step, final_val)
        # Downstream commands default to best.pt; make sure it exists even when
        # evaluation was disabled or never reached.
        if not best_path.exists():
            self.save_checkpoint(best_path, step, final_val)

        if self.cfg.draw and (self.train_history or self.val_history):
            draw_curves(
                [h["loss"] for h in self.train_history],
                [h["val_loss"] for h in self.val_history],
                x=[h["step"] for h in self.train_history],
                val_x=[h["step"] for h in self.val_history],
                label_x="step",
                label_y="loss",
                title=f"{tag}_loss",
                img_path=self.out_dir,
            )
        print(f"[{tag}] done at step {step}; checkpoints in {self.out_dir}")
        return best_val

    # ------------------------------------------------------------------
    # public entry points
    # ------------------------------------------------------------------
    def train_pretrain(self, train_dataset: Dataset, val_dataset: Dataset | None = None,
                       resume_from: str | None = None) -> float:
        train_loader = self._make_loader(train_dataset, shuffle=True)
        if len(train_loader) == 0:
            raise ValueError(
                f"train dataset has {len(train_dataset)} samples, fewer than "
                f"batch_size={self.cfg.batch_size}"
            )
        # Shuffle validation windows so each evaluation samples the whole
        # val.bin instead of only its prefix (TinyStories before wikitext).
        val_loader = (
            self._make_loader(val_dataset, shuffle=True, drop_last=False)
            if val_dataset is not None and len(val_dataset) > 0
            else None
        )
        max_steps = self.cfg.max_steps or (
            len(train_loader) * (self.cfg.epochs or 1)
        )
        start_step = self.resume(resume_from) if resume_from else 0
        return self._run(train_loader, val_loader, max_steps, tag="pretrain", start_step=start_step)

    def train_sft(self, train_dataset: Dataset, val_dataset: Dataset | None = None,
                  epochs: int | None = None, pretrain_checkpoint: str | None = None,
                  resume_from: str | None = None) -> float:
        if pretrain_checkpoint and not resume_from:
            self.load_pretrained(pretrain_checkpoint)
            print(f"loaded pretrained weights from {pretrain_checkpoint}")
        train_loader = self._make_loader(train_dataset, shuffle=True)
        if len(train_loader) == 0:
            raise ValueError(
                f"train dataset has {len(train_dataset)} samples, fewer than "
                f"batch_size={self.cfg.batch_size}"
            )
        val_loader = (
            self._make_loader(val_dataset, shuffle=False)
            if val_dataset is not None and len(val_dataset) > 0
            else None
        )
        epochs = epochs or self.cfg.epochs or 1
        max_steps = self.cfg.max_steps or (len(train_loader) * epochs)
        start_step = self.resume(resume_from) if resume_from else 0
        return self._run(train_loader, val_loader, max_steps, tag="sft", start_step=start_step)
