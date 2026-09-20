"""
Build ecosystem_dashboard.html from a simulation run.

Usage:
    python build_dashboard.py --prefix run --out ecosystem_dashboard.html
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_population(path: Path) -> list[dict]:
    """Read population CSV into dictionaries."""

    if not path.exists():
        raise FileNotFoundError(
            f"Population file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        rows = []

        for row in reader:

            converted = {}

            for key, value in row.items():

                if value is None:
                    converted[key] = value
                    continue

                if key == "step":
                    converted[key] = int(float(value))

                elif key in {
                    "plants",
                    "grazers",
                    "predators"
                }:
                    converted[key] = int(float(value))

                else:
                    try:
                        converted[key] = float(value)
                    except ValueError:
                        converted[key] = value

            rows.append(converted)

        return rows


def read_traits(path: Path) -> list[dict]:
    """Read trait CSV into dictionaries."""

    if not path.exists():
        raise FileNotFoundError(
            f"Traits file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        rows = []

        for row in reader:

            converted = {}

            for key, value in row.items():

                if value is None:
                    converted[key] = value
                    continue

                if key == "step":
                    converted[key] = int(float(value))

                elif key == "mean":
                    converted[key] = float(value)

                else:
                    converted[key] = value

            rows.append(converted)

        return rows


def read_snapshots(path: Path) -> dict:
    """Read recorded simulation snapshots."""

    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def downsample_population(
    rows: list[dict],
    max_rows: int = 1500
) -> list[dict]:
    """Keep population data small enough for the browser."""

    if len(rows) <= max_rows:
        return rows

    stride = max(
        1,
        len(rows) // max_rows
    )

    sampled = rows[::stride]

    if rows[-1] not in sampled:
        sampled.append(rows[-1])

    return sampled


def downsample_traits(
    rows: list[dict],
    max_rows: int = 5000
) -> list[dict]:
    """Keep trait data small enough for the browser."""

    if len(rows) <= max_rows:
        return rows

    stride = max(
        1,
        len(rows) // max_rows
    )

    sampled = rows[::stride]

    if rows[-1] not in sampled:
        sampled.append(rows[-1])

    return sampled


def normalise_snapshot(snapshot: dict) -> dict:
    """
    Ensure every snapshot has the structure expected by the
    dashboard JavaScript.
    """

    return {
        "step": int(
            snapshot.get("step", 0)
        ),

        "plants": snapshot.get(
            "plants",
            []
        ),

        "grazers": snapshot.get(
            "grazers",
            []
        ),

        "predators": snapshot.get(
            "predators",
            []
        ),
    }


def build_data(
    prefix: Path
) -> dict:

    population_path = Path(
        f"{prefix}_population.csv"
    )

    traits_path = Path(
        f"{prefix}_traits.csv"
    )

    snapshots_path = Path(
        f"{prefix}_snapshots.json"
    )

    population = read_population(
        population_path
    )

    traits = read_traits(
        traits_path
    )

    raw_snapshots = read_snapshots(
        snapshots_path
    )

    # ---------------------------------------------------------
    # Snapshots
    # ---------------------------------------------------------

    raw_frames = raw_snapshots.get(
        "snapshots",
        []
    )

    # Some older simulation output may use "frames".
    # Support that too.
    if not raw_frames:

        raw_frames = raw_snapshots.get(
            "frames",
            []
        )

    snapshots = [
        normalise_snapshot(snapshot)
        for snapshot in raw_frames
    ]

    # ---------------------------------------------------------
    # Keep all recorded frames.
    #
    # The simulation already controls snapshot frequency.
    # We do NOT aggressively downsample these because the
    # replay needs the actual spatial states.
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Events
    # ---------------------------------------------------------

    events = raw_snapshots.get(
        "events",
        []
    )

    # ---------------------------------------------------------
    # World dimensions
    # ---------------------------------------------------------

    world_w = raw_snapshots.get(
        "world_w",
        1000
    )

    world_h = raw_snapshots.get(
        "world_h",
        600
    )

    # ---------------------------------------------------------
    # Final dashboard data
    # ---------------------------------------------------------

    return {
        "seed": raw_snapshots.get(
            "seed"
        ),

        "steps": raw_snapshots.get(
            "steps"
        ),

        "world_w": world_w,

        "world_h": world_h,

        "events": events,

        "population":
            downsample_population(
                population
            ),

        "traits":
            downsample_traits(
                traits
            ),

        "snapshots":
            snapshots,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prefix",
        default="run",
        help="Simulation output prefix"
    )

    parser.add_argument(
        "--out",
        default="ecosystem_dashboard.html",
        help="Output dashboard HTML"
    )

    args = parser.parse_args()

    project_root = Path(
        __file__
    ).resolve().parent

    template_path = (
        project_root /
        "dashboard_template.html"
    )

    output_path = (
        project_root /
        args.out
    )

    prefix = (
        project_root /
        args.prefix
    )

    if not template_path.exists():
        raise FileNotFoundError(
            f"Dashboard template not found: {template_path}"
        )

    data = build_data(
        prefix
    )

    template = template_path.read_text(
        encoding="utf-8"
    )

    data_json = json.dumps(
        data,
        separators=(",", ":"),
        ensure_ascii=False
    )

    if "__DATA_JSON__" not in template:
        raise RuntimeError(
            "dashboard_template.html does not contain "
            "__DATA_JSON__ placeholder."
        )

    html = template.replace(
        "__DATA_JSON__",
        data_json
    )

    output_path.write_text(
        html,
        encoding="utf-8"
    )

    print()
    print(
        f"Wrote {output_path}"
    )

    print(
        f"  Population rows : "
        f"{len(data['population']):,}"
    )

    print(
        f"  Trait rows      : "
        f"{len(data['traits']):,}"
    )

    print(
        f"  Playback frames : "
        f"{len(data['snapshots']):,}"
    )

    print(
        f"  World           : "
        f"{data['world_w']} × {data['world_h']}"
    )

    print(
        f"  Seed            : "
        f"{data['seed']}"
    )

    print()
    

if __name__ == "__main__":
    main()