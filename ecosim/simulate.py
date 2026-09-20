from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np

from .config import Config
from .world import World
from .genome import GRAZER_TRAITS, PREDATOR_TRAITS


def run(cfg: Config, out_prefix: str, verbose=True):
    world = World(cfg)

    pop_rows = []
    trait_rows = []
    snapshots = []

    event_markers = [
        {
            "start": s,
            "end": s + d,
            "label": "Drought",
        }
        for s, d, *_ in cfg.drought_events
    ]

    t0 = time.time()

    for step in range(cfg.n_steps):

        drought_on = world.step()

        # -----------------------------------------------------
        # Population and trait logging
        # -----------------------------------------------------

        if step % cfg.log_every == 0:

            pop_rows.append({
                "step": step,
                "plants": len(world.plants),
                "grazers": len(world.grazers),
                "predators": len(world.predators),
                "corpses": len(world.corpses),
                "drought": int(drought_on),
            })

            if world.grazers:

                for tname in GRAZER_TRAITS:

                    vals = [
                        g.traits[tname]
                        for g in world.grazers
                    ]

                    trait_rows.append({
                        "step": step,
                        "tier": "grazer",
                        "trait": tname,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    })

            if world.predators:

                for tname in PREDATOR_TRAITS:

                    vals = [
                        p.traits[tname]
                        for p in world.predators
                    ]

                    trait_rows.append({
                        "step": step,
                        "tier": "predator",
                        "trait": tname,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                    })

        # -----------------------------------------------------
        # Visual snapshot
        # -----------------------------------------------------

        if step % cfg.snapshot_every == 0:

            snap = {
                "step": step,

                "plants": [
                    [round(x, 1), round(y, 1)]
                    for x, y in world.plants[:400]
                ],

                "grazers": [
                    [
                        round(g.x, 1),
                        round(g.y, 1),
                        round(g.traits["camouflage"], 2),
                        round(g.energy / cfg.max_energy_grazer, 2),
                    ]
                    for g in world.grazers
                ],

                "predators": [
                    [
                        round(p.x, 1),
                        round(p.y, 1),
                        round(p.traits["ambush"], 2),
                        round(p.energy / cfg.max_energy_predator, 2),
                    ]
                    for p in world.predators
                ],
            }

            snapshots.append(snap)

        # -----------------------------------------------------
        # Console progress
        # -----------------------------------------------------

        if verbose and step % 200 == 0:

            elapsed = time.time() - t0

            print(
                f"step {step:5d} | "
                f"plants {len(world.plants):4d} | "
                f"grazers {len(world.grazers):4d} | "
                f"predators {len(world.predators):3d} | "
                f"drought={drought_on} | "
                f"{elapsed:.1f}s"
            )

        # -----------------------------------------------------
        # Extinction
        # -----------------------------------------------------

        if (
            len(world.grazers) == 0
            and len(world.predators) == 0
        ):
            print("Full extinction, stopping early.")
            break

    # ---------------------------------------------------------
    # Population CSV
    # ---------------------------------------------------------

    with open(
        f"{out_prefix}_population.csv",
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "plants",
                "grazers",
                "predators",
                "corpses",
                "drought",
            ],
        )

        writer.writeheader()
        writer.writerows(pop_rows)

    # ---------------------------------------------------------
    # Traits CSV
    # ---------------------------------------------------------

    with open(
        f"{out_prefix}_traits.csv",
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "tier",
                "trait",
                "mean",
                "std",
            ],
        )

        writer.writeheader()
        writer.writerows(trait_rows)

    # ---------------------------------------------------------
    # Snapshots JSON
    # ---------------------------------------------------------

    with open(
        f"{out_prefix}_snapshots.json",
        "w",
    ) as f:

        json.dump(
            {
                "seed": cfg.seed,
                "world_w": cfg.world_w,
                "world_h": cfg.world_h,
                "events": event_markers,
                "snapshots": snapshots,
            },
            f,
        )

    # ---------------------------------------------------------
    # Summary JSON
    # ---------------------------------------------------------

    with open(
        f"{out_prefix}_summary.json",
        "w",
    ) as f:

        json.dump(
            {
                "seed": cfg.seed,
                "population": pop_rows,
                "traits": trait_rows,
                "events": event_markers,
                "world_w": cfg.world_w,
                "world_h": cfg.world_h,
                "n_steps": step + 1,
            },
            f,
        )

    elapsed = time.time() - t0

    print(
        f"Done in {elapsed:.1f}s. "
        f"Final: grazers={len(world.grazers)} "
        f"predators={len(world.predators)} "
        f"plants={len(world.plants)} "
        f"seed={cfg.seed}"
    )

    return pop_rows, trait_rows, snapshots


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of simulation steps",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible simulation runs",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="run",
        help="Output file prefix",
    )

    args = parser.parse_args()

    cfg = Config()

    if args.steps is not None:
        cfg.n_steps = args.steps

    if args.seed is not None:
        cfg.seed = args.seed

    print(
        f"Starting simulation "
        f"(seed={cfg.seed}, steps={cfg.n_steps})"
    )

    run(
        cfg,
        args.out,
    )