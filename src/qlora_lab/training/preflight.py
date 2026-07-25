from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from qlora_lab.training.config import TrainConfig


def estimated_minimum_vram_gb(model_id: str) -> float:
    normalized = model_id.upper()
    if "8B" in normalized:
        return 20.0
    if "4B" in normalized:
        return 12.0
    if any(size in normalized for size in ("0.6B", "1.7B", "2B")):
        return 7.0
    return 8.0


def collect_environment(config: TrainConfig | None = None) -> tuple[dict[str, object], bool]:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    ok = (3, 11) <= sys.version_info[:2] < (3, 13)

    try:
        import torch
    except ImportError:
        report["torch"] = "not installed"
        report["cuda_available"] = False
        return report, False

    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        vram_gb = round(properties.total_memory / 1024**3, 2)
        report.update(
            {
                "gpu": properties.name,
                "vram_gb": vram_gb,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
        if config is not None:
            minimum_vram = estimated_minimum_vram_gb(config.base_model_id)
            profile_fits = vram_gb >= minimum_vram
            report.update(
                {
                    "training_profile": config.base_model_id,
                    "estimated_minimum_vram_gb": minimum_vram,
                    "profile_fits_gpu": profile_fits,
                }
            )
            ok = ok and profile_fits
    else:
        ok = False

    try:
        import bitsandbytes

        report["bitsandbytes"] = bitsandbytes.__version__
    except Exception as exc:  # a failed CUDA DLL load is useful preflight information
        report["bitsandbytes"] = f"unavailable: {exc}"
        ok = False
    return report, ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the CUDA QLoRA environment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_8gb.yaml"),
    )
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    report, ok = collect_environment(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not ok:
        print(
            "\nPreflight failed. Use Python 3.11, a CUDA-enabled PyTorch build, "
            "and a working bitsandbytes installation.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
