from __future__ import annotations

import json
import platform
import sys


def collect_environment() -> tuple[dict[str, object], bool]:
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
        report.update(
            {
                "gpu": properties.name,
                "vram_gb": round(properties.total_memory / 1024**3, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
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
    report, ok = collect_environment()
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
