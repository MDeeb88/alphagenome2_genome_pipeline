#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from run_species import final_outputs_complete


PROJECT_ROOT = Path(
    "/media/eternus1/nfs/projects/dpanchenko/"
    "alphagenome2_genome_pipeline"
)

SRC_DIR = PROJECT_ROOT / "src"

DEFAULT_INVENTORY = (
    PROJECT_ROOT
    / "inventory"
    / "CTCF_Figure7_genome_annotation_inventory.csv"
)

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "default.yaml"
)

DEFAULT_RESULTS = PROJECT_ROOT / "results"
DEFAULT_LOGS = PROJECT_ROOT / "logs"


def timestamp():
    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one_species(
    species_id,
    inventory,
    config,
    results_dir,
    run_log_dir,
    skip_tracks=False,
    no_cleanup=False,
    force=False,
):
    log_path = (
        run_log_dir
        / f"{species_id}.log"
    )

    cmd = [
        sys.executable,
        "-u",
        str(
            SRC_DIR
            / "run_species.py"
        ),
        "--species",
        species_id,
        "--inventory",
        str(inventory),
        "--config",
        str(config),
        "--results-dir",
        str(results_dir),
    ]

    if skip_tracks:
        cmd.append(
            "--skip-tracks"
        )

    if no_cleanup:
        cmd.append(
            "--no-cleanup"
        )

    if force:
        cmd.append(
            "--force"
        )

    started = time.perf_counter()

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log:

        log.write(
            "COMMAND:\n"
            + " ".join(cmd)
            + "\n\n"
        )

        log.flush()

        process = subprocess.run(
            cmd,
            cwd=str(
                PROJECT_ROOT
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    return {
        "species_id": species_id,
        "success": (
            process.returncode == 0
        ),
        "return_code": (
            process.returncode
        ),
        "elapsed_seconds": (
            elapsed
        ),
        "log": str(
            log_path
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple AlphaGenome 2 species "
            "with controlled parallelism."
        )
    )

    parser.add_argument(
        "--species",
        nargs="+",
        required=True,
        help=(
            "species_id values to process"
        ),
    )

    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--skip-tracks",
        action="store_true",
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show what would run without "
            "downloading or predicting anything."
        ),
    )

    args = parser.parse_args()

    # Resolve results directory from the selected assay config.
    # An explicit --results-dir overrides the YAML setting.
    if args.results_dir is None:
        path_config = load_config(args.config)
        args.results_dir = Path(
            path_config["paths"]["results_dir"]
        )

    cfg = load_config(
        args.config
    )

    max_parallel = (
        args.max_parallel
        if args.max_parallel is not None
        else int(
            cfg[
                "runtime"
            ][
                "max_parallel_species"
            ]
        )
    )

    if max_parallel < 1:
        raise ValueError(
            "max_parallel must be >= 1"
        )

    inventory = pd.read_csv(
        args.inventory
    )

    known_species = set(
        inventory[
            "species_id"
        ].astype(str)
    )

    requested = []

    for species_id in args.species:
        if species_id not in known_species:
            raise ValueError(
                f"Unknown species_id: "
                f"{species_id}"
            )

        if species_id not in requested:
            requested.append(
                species_id
            )

    build_tracks = (
        bool(
            cfg[
                "outputs"
            ].get(
                "build_bigwig",
                True,
            )
        )
        and not args.skip_tracks
    )

    runnable = []
    already_complete = []

    for species_id in requested:
        species_results = (
            args.results_dir
            / species_id
        )

        complete = (
            final_outputs_complete(
                species_results,
                require_tracks=(
                    build_tracks
                ),
            )
        )

        if (
            complete
            and not args.force
        ):
            already_complete.append(
                species_id
            )
        else:
            runnable.append(
                species_id
            )

    print("=" * 70)
    print("AlphaGenome 2 inventory scheduler")
    print("=" * 70)

    print(
        f"Requested species:  "
        f"{len(requested)}"
    )

    print(
        f"Already complete:   "
        f"{len(already_complete)}"
    )

    print(
        f"Would run:          "
        f"{len(runnable)}"
    )

    print(
        f"Max parallel:       "
        f"{max_parallel}"
    )

    print(
        f"Build BigWigs:      "
        f"{build_tracks}"
    )

    print(
        f"Cleanup:            "
        f"{not args.no_cleanup}"
    )

    if runnable:
        print(
            "\nSpecies to run:"
        )

        for species_id in runnable:
            row = inventory[
                inventory[
                    "species_id"
                ]
                == species_id
            ].iloc[0]

            print(
                f"  {species_id:35s} "
                f"{row['species_name']}"
            )

    if args.dry_run:
        print()
        print(
            "DRY RUN ONLY — "
            "nothing was executed."
        )
        return

    if not runnable:
        print()
        print(
            "Nothing to run."
        )
        return

    if not os.environ.get(
        "ALPHA_GENOME_API_KEY"
    ):
        raise RuntimeError(
            "ALPHA_GENOME_API_KEY is not set."
        )

    run_id = timestamp()

    run_log_dir = (
        DEFAULT_LOGS
        / "inventory_runs"
        / run_id
    )

    run_log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Scheduler logs:"
    )
    print(
        run_log_dir
    )
    print()

    started = (
        time.perf_counter()
    )

    results = []

    with ThreadPoolExecutor(
        max_workers=max_parallel
    ) as executor:

        futures = {}

        for species_id in runnable:
            print(
                f"[START] {species_id}"
            )

            future = executor.submit(
                run_one_species,
                species_id,
                args.inventory,
                args.config,
                args.results_dir,
                run_log_dir,
                args.skip_tracks,
                args.no_cleanup,
                args.force,
            )

            futures[
                future
            ] = species_id

        for future in as_completed(
            futures
        ):
            species_id = (
                futures[
                    future
                ]
            )

            try:
                result = (
                    future.result()
                )

            except Exception as exc:
                result = {
                    "species_id": (
                        species_id
                    ),
                    "success": False,
                    "return_code": None,
                    "elapsed_seconds": None,
                    "log": None,
                    "exception": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }

            results.append(
                result
            )

            if result[
                "success"
            ]:
                print(
                    f"[DONE]  "
                    f"{species_id}"
                )
            else:
                print(
                    f"[FAIL]  "
                    f"{species_id}"
                )

                if result.get(
                    "log"
                ):
                    print(
                        f"        log: "
                        f"{result['log']}"
                    )

    elapsed = (
        time.perf_counter()
        - started
    )

    results = sorted(
        results,
        key=lambda x: x[
            "species_id"
        ],
    )

    summary_df = pd.DataFrame(
        results
    )

    summary_csv = (
        run_log_dir
        / "run_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    summary_json = (
        run_log_dir
        / "run_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "run_id": run_id,
                "requested": requested,
                "already_complete": (
                    already_complete
                ),
                "attempted": runnable,
                "max_parallel": (
                    max_parallel
                ),
                "elapsed_seconds": (
                    elapsed
                ),
                "results": results,
            },
            f,
            indent=2,
        )

    success_count = sum(
        bool(x["success"])
        for x in results
    )

    failure_count = (
        len(results)
        - success_count
    )

    print()
    print("=" * 70)
    print("INVENTORY RUN COMPLETE")
    print("=" * 70)

    print(
        f"Successful:      "
        f"{success_count}"
    )

    print(
        f"Failed:          "
        f"{failure_count}"
    )

    print(
        f"Elapsed:         "
        f"{elapsed / 3600:.2f} h"
    )

    print(
        f"Summary:         "
        f"{summary_csv}"
    )

    if failure_count:
        sys.exit(1)


if __name__ == "__main__":
    main()