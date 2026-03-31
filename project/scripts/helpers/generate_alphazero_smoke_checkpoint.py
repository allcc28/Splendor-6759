"""Generate a minimal AlphaZero checkpoint for smoke tests.

This utility is useful when validating evaluator/tracking pipelines without
waiting for full self-play training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

import sys

sys.path.insert(0, ".")
sys.path.insert(0, "modules")
sys.path.append("project/src")

from nn.policy_value_net import SplendorPolicyValueNet
from nn.tensor_encoder import SplendorTensorEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate minimal AlphaZero smoke checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="project/configs/mcts/alphazero_v1_baseline.yaml",
        help="AlphaZero config used to initialize model architecture",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="project/logs/alphazero_smoke_init",
        help="Directory where config.yaml and final_model.pt are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    net_cfg = config.get("network", {})
    encoder = SplendorTensorEncoder()
    model = SplendorPolicyValueNet(
        input_channels=encoder.spec.channels,
        policy_size=int(net_cfg.get("policy_size", 256)),
        trunk_channels=int(net_cfg.get("trunk_channels", 128)),
        num_res_blocks=int(net_cfg.get("num_res_blocks", 3)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.yaml"
    checkpoint_path = output_dir / "final_model.pt"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    payload = {
        "model_state_dict": model.state_dict(),
        # This checkpoint is intended for smoke evaluation, not optimizer resume.
        "optimizer_state_dict": None,
        "action_indexer_state": None,
        "smoke_checkpoint": True,
        "config": config,
    }
    torch.save(payload, checkpoint_path)

    print(f"Generated smoke checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
