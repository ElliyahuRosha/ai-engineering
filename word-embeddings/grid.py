"""Run the configuration grid on a machine with time to spare, then one table.

Every run writes to its own `runs/<name>/` directory, so nothing overwrites the
model in `outputs/`. Re-running the script skips runs that already have metrics.
"""

import subprocess
import sys
from pathlib import Path

# Only one thing changes per step, except where a step is a deliberate bundle.
CONFIGS = [
    {"name": "base_e1", "dim": 100, "window": 5, "negatives": 5, "subsample": 0.0, "epochs": 1},
    {"name": "sub_e1", "dim": 100, "window": 5, "negatives": 5, "subsample": 1e-4, "epochs": 1},
    {"name": "sub_e3", "dim": 100, "window": 5, "negatives": 5, "subsample": 1e-4, "epochs": 3},
    {"name": "sub_e5", "dim": 100, "window": 5, "negatives": 5, "subsample": 1e-4, "epochs": 5},
    {"name": "sub_k10_e3", "dim": 100, "window": 5, "negatives": 10, "subsample": 1e-4, "epochs": 3},
    {"name": "sub_w2_e3", "dim": 100, "window": 2, "negatives": 5, "subsample": 1e-4, "epochs": 3},
    {"name": "d200_k10_e3", "dim": 200, "window": 5, "negatives": 10, "subsample": 1e-4, "epochs": 3},
    {"name": "d300_k15_e5", "dim": 300, "window": 5, "negatives": 15, "subsample": 1e-4, "epochs": 5},
]


def main() -> None:
    finished = []
    for config in CONFIGS:
        run = Path("runs") / config["name"]
        if (run / "metrics.json").exists():
            print(f"skip {config['name']}: already scored", flush=True)
            finished.append(str(run))
            continue

        print("=" * 72, flush=True)
        print(f"run {config['name']}: {config}", flush=True)
        command = [
            sys.executable, "train.py", "--fresh", "--out", str(run),
            "--dim", str(config["dim"]),
            "--window", str(config["window"]),
            "--negatives", str(config["negatives"]),
            "--subsample", str(config["subsample"]),
            "--epochs", str(config["epochs"]),
        ]
        if config.get("steps"):
            command += ["--steps", str(config["steps"])]
        subprocess.run(command, check=True)
        subprocess.run([sys.executable, "evaluate.py", str(run)], check=True)
        finished.append(str(run))

    print("=" * 72, flush=True)
    subprocess.run([sys.executable, "evaluate.py", *finished, "--compare"], check=True)


if __name__ == "__main__":
    main()
