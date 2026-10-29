"""Run all fixed offline scenarios and save computed outputs.

Run: python -m cra373.demo.demo_offline --output demo_results
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .scenarios import NAMES, load_scenario
from .pipeline import analyze

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("demo_results"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in NAMES:
        result = analyze(load_scenario(name))
        slug = name.lower().replace(" ", "_")
        (args.output / f"{slug}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        rows.append({"scenario": name, "recommendation": result["reasoning"]["selected_action"],
                     "valid_fraction": result["reasoning"]["features"]["ahead_valid_fraction"],
                     "traversable_fraction": result["reasoning"]["features"]["ahead_traversable_fraction"],
                     **{f"{k}_ms": v for k, v in result["latency_ms"].items()}})
        # Export a real visualization of computed geometry and spatial masks.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        height = np.array(result["height_m"], float)
        axes[0].imshow(height, origin="lower", cmap="terrain")
        axes[0].set_title("Height (m)")
        axes[1].imshow(result["maps"]["traversable"], origin="lower", vmin=0, vmax=1, cmap="Greens")
        axes[1].set_title("Traversable heuristic")
        overlay = np.zeros((*height.shape, 3))
        for key, color in [("foothold", (0.1, 0.7, 0.2)), ("obstacle", (1, 0.6, 0)),
                           ("gap", (0.7, 0, 0.7)), ("unknown", (0.4, 0.4, 0.4))]:
            overlay[np.array(result["maps"][key], bool)] = color
        axes[2].imshow(overlay, origin="lower")
        axes[2].set_title("Foothold / obstacle / gap")
        fig.suptitle(f"{name}: {result['reasoning']['selected_action']}")
        fig.tight_layout()
        fig.savefig(args.output / f"{slug}.png", dpi=130)
        plt.close(fig)
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} computed scenarios in {args.output.resolve()}")
    for row in rows:
        print(f"{row['scenario']}: {row['recommendation']}")

if __name__ == "__main__":
    main()
