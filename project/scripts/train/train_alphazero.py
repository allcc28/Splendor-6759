"""Train an AlphaZero-style Splendor agent with self-play + PUCT."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import yaml
from tqdm import tqdm

sys.path.insert(0, ".")
sys.path.insert(0, "modules")
sys.path.append("project/src")

from training.alphazero_trainer import AlphaZeroTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AlphaZero-style Splendor agent")
    parser.add_argument(
        "--config",
        type=str,
        default="project/configs/mcts/alphazero_v1_baseline.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint path to resume from",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Optional override for total training iterations",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.iterations is not None:
        config["training"]["iterations"] = int(args.iterations)

    run_name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("project", "logs", f"{run_name}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(log_dir, "config.yaml"))

    trainer = AlphaZeroTrainer(config)
    iterations = int(config["training"].get("iterations", 10))
    ckpt_freq = int(config["training"].get("checkpoint_every", 1))

    history = []
    start_iter = 1
    if args.resume.strip():
        resume_path = args.resume.strip()
        trainer.load_checkpoint(resume_path)
        print(f"Resumed from checkpoint: {resume_path}")

        # Try to infer iteration from filename suffix like alphazero_iter_0010.pt.
        stem = os.path.splitext(os.path.basename(resume_path))[0]
        if stem.startswith("alphazero_iter_"):
            maybe_iter = stem.replace("alphazero_iter_", "")
            if maybe_iter.isdigit():
                start_iter = int(maybe_iter) + 1

    pbar = tqdm(range(start_iter, iterations + 1), desc="AlphaZero Training", unit="iter")
    for it in pbar:
        iter_start = time.time()
        metrics = trainer.run_iteration()
        iter_elapsed = time.time() - iter_start
        metrics["iteration"] = it
        metrics["iter_time_sec"] = round(iter_elapsed, 1)
        history.append(metrics)

        buf_info = f" buf={metrics['buffer_size']}" if "buffer_size" in metrics else ""
        pbar.set_postfix_str(
            f"s={metrics['samples']}{buf_info} "
            f"ploss={metrics['policy_loss']:.3f} "
            f"vloss={metrics['value_loss']:.3f} "
            f"{iter_elapsed:.0f}s/iter"
        )

        if it % ckpt_freq == 0:
            checkpoint_path = os.path.join(log_dir, f"alphazero_iter_{it:04d}.pt")
            trainer.save_checkpoint(checkpoint_path)

    final_path = os.path.join(log_dir, "final_model.pt")
    trainer.save_checkpoint(final_path)

    with open(os.path.join(log_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history}, f, indent=2)

    print(f"Saved final checkpoint: {final_path}")
    print(f"Saved metrics: {os.path.join(log_dir, 'training_metrics.json')}")


if __name__ == "__main__":
    main()
