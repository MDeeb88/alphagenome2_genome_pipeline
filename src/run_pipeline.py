#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from api_key import (
    api_key_status,
    load_api_key,
    manage_api_key,
)
from run_species import (
    final_outputs_complete,
)


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

DEFAULT_RESULTS = (
    PROJECT_ROOT
    / "results"
)

DEFAULT_WORK = (
    PROJECT_ROOT
    / "work"
)


def load_config(path):
    with open(
        path,
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def read_json(path):
    if not path.exists():
        return None

    try:
        with open(
            path,
            encoding="utf-8",
        ) as f:
            return json.load(f)
    except Exception:
        return None


def status_for_species(
    species_id,
    results_dir,
    work_dir,
    require_tracks,
    config=None,
):
    result_dir = (
        results_dir
        / species_id
    )

    if final_outputs_complete(
        result_dir,
        require_tracks=require_tracks,
        expected_output_type=(
            str(config["alphagenome"].get(
                "output_type", "RNA_SEQ"
            )).upper()
            if config is not None else None
        ),
        expected_condition=(
            str(config["alphagenome"]["condition"])
            if config is not None else None
        ),
        allow_bigwig_only=(
            bool(config.get("cleanup", {}).get(
                "delete_consolidated_predictions", False
            ))
            if config is not None else False
        ),
    ):
        return "COMPLETE"

    status_path = (
        result_dir
        / "metadata"
        / "pipeline_status.json"
    )

    status = read_json(
        status_path
    )

    if status:
        if status.get(
            "error"
        ):
            return "FAILED"

        for stage in (
            status.get(
                "stages",
                {}
            ).values()
        ):
            if (
                stage.get(
                    "status"
                )
                == "failed"
            ):
                return "FAILED"

        if any(
            stage.get(
                "status"
            )
            == "running"
            for stage in (
                status.get(
                    "stages",
                    {}
                ).values()
            )
        ):
            return "INCOMPLETE"

    if (
        result_dir.exists()
        or (
            work_dir
            / species_id
        ).exists()
    ):
        return "INCOMPLETE"

    return "NOT_STARTED"


def build_status_table(
    inventory,
    results_dir,
    work_dir,
    require_tracks,
    config=None,
):
    rows = []

    for row in inventory.itertuples(
        index=False
    ):
        species_id = str(
            row.species_id
        )

        rows.append(
            {
                "species_id": (
                    species_id
                ),
                "species_name": str(
                    row.species_name
                ),
                "taxa": str(
                    row.taxa
                ),
                "status": (
                    status_for_species(
                        species_id,
                        results_dir,
                        work_dir,
                        require_tracks,
                        config=config,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def print_status_summary(
    status_df,
):
    print()
    print("=" * 70)
    print("PIPELINE STATUS")
    print("=" * 70)

    counts = (
        status_df[
            "status"
        ]
        .value_counts()
    )

    for status in [
        "COMPLETE",
        "FAILED",
        "INCOMPLETE",
        "NOT_STARTED",
    ]:
        print(
            f"{status:15s} "
            f"{int(counts.get(status, 0)):4d}"
        )

    print(
        f"{'TOTAL':15s} "
        f"{len(status_df):4d}"
    )


def find_species_matches(
    inventory,
    query,
):
    query = (
        query.strip().lower()
    )

    if not query:
        return inventory.iloc[
            0:0
        ]

    ids = (
        inventory[
            "species_id"
        ]
        .astype(str)
        .str.lower()
    )

    names = (
        inventory[
            "species_name"
        ]
        .astype(str)
        .str.lower()
    )

    exact = inventory[
        (ids == query)
        | (names == query)
    ]

    if len(exact):
        return exact

    return inventory[
        ids.str.contains(
            query,
            regex=False,
        )
        | names.str.contains(
            query,
            regex=False,
        )
    ]


def choose_one_species(
    inventory,
):
    while True:
        query = input(
            "\nType species name or search term: "
        ).strip()

        matches = find_species_matches(
            inventory,
            query,
        )

        if len(matches) == 0:
            print(
                "No matches."
            )
            continue

        if len(matches) == 1:
            row = matches.iloc[0]

            print()
            print(
                f"Selected: "
                f"{row['species_name']} "
                f"({row['species_id']})"
            )

            return str(
                row[
                    "species_id"
                ]
            )

        print(
            f"\n{len(matches)} matches:"
        )

        limited = matches.head(
            30
        ).reset_index(
            drop=True
        )

        for i, row in (
            limited.iterrows()
        ):
            print(
                f"{i + 1:2d}. "
                f"{row['species_name']} "
                f"[{row['species_id']}]"
            )

        if len(matches) > 30:
            print(
                "... refine your search "
                "to see more."
            )
            continue

        selection = input(
            "Choose number: "
        ).strip()

        try:
            n = int(
                selection
            )
        except ValueError:
            print(
                "Invalid selection."
            )
            continue

        if not (
            1
            <= n
            <= len(limited)
        ):
            print(
                "Invalid selection."
            )
            continue

        return str(
            limited.iloc[
                n - 1
            ][
                "species_id"
            ]
        )


def choose_multiple_species(
    inventory,
):
    selected = []

    print()
    print(
        "Select multiple species."
    )
    print(
        "Search by species name or species_id."
    )
    print(
        "From search results you can enter "
        "multiple numbers, e.g. 1,3,5."
    )
    print(
        "Type 'all' to add every displayed match."
    )
    print(
        "Press Enter on an empty search when finished."
    )

    while True:
        print()

        if selected:
            print(
                f"Currently selected: "
                f"{len(selected)} species"
            )

        query = input(
            "Search species: "
        ).strip()

        if query.lower() in {"b", "back"}:
            print("Returning to main menu.")
            return []

        if not query:
            break

        if query.lower() in {"list", "all"}:
            matches = inventory

        elif "," in query:
            for name in query.split(","):
                name = name.strip()
                if not name:
                    continue

                found = find_species_matches(inventory, name)

                if len(found) == 1:
                    species_id = str(found.iloc[0]["species_id"])
                    if species_id not in selected:
                        selected.append(species_id)
                        print(f"Added: {species_id}")
                elif len(found) == 0:
                    print(f"No matches: {name}")
                else:
                    print(
                        f"Ambiguous: {name}. "
                        "Search individually."
                    )
            continue

        else:
            matches = find_species_matches(inventory, query)

        if len(matches) == 0:
            print(
                "No matches."
            )
            continue

        limited = (
            matches.head(50)
            .reset_index(
                drop=True
            )
        )

        print(
            f"\n{len(matches)} matches:"
        )

        for i, row in (
            limited.iterrows()
        ):
            already = (
                " [SELECTED]"
                if str(
                    row["species_id"]
                ) in selected
                else ""
            )

            print(
                f"{i + 1:2d}. "
                f"{row['species_name']} "
                f"[{row['species_id']}]"
                f"{already}"
            )

        if len(matches) > 50:
            print(
                "\nShowing first 50 matches. "
                "Refine the search if needed."
            )

        choice = input(
            "\nChoose numbers "
            "(example: 1,3,5), "
            "'all', or Enter to cancel: "
        ).strip().lower()

        if choice in {"b", "back"}:
            print("Returning to main menu.")
            return []

        if not choice:
            continue

        if choice == "all":
            numbers = list(
                range(
                    1,
                    len(limited) + 1,
                )
            )

        else:
            try:
                numbers = [
                    int(x.strip())
                    for x in choice.split(",")
                    if x.strip()
                ]

            except ValueError:
                print(
                    "Invalid selection. "
                    "Use numbers separated by commas."
                )
                continue

        invalid = [
            n
            for n in numbers
            if not (
                1 <= n <= len(limited)
            )
        ]

        if invalid:
            print(
                "Invalid selection numbers:",
                invalid,
            )
            continue

        added = 0

        for n in numbers:
            species_id = str(
                limited.iloc[
                    n - 1
                ][
                    "species_id"
                ]
            )

            if species_id not in selected:
                selected.append(
                    species_id
                )
                added += 1

        print(
            f"Added {added} species."
        )

    if selected:
        print()
        print(
            f"Final selection: "
            f"{len(selected)} species"
        )

        selected_rows = inventory[
            inventory[
                "species_id"
            ].isin(
                selected
            )
        ]

        for species_id in selected:
            row = selected_rows[
                selected_rows[
                    "species_id"
                ]
                == species_id
            ].iloc[0]

            print(
                f"  - "
                f"{row['species_name']} "
                f"[{species_id}]"
            )

    return selected



# AG2_BACK_NAVIGATION_V1
class BackToMainMenu(Exception):
    """Cancel the current menu operation."""


def _menu_input(prompt):
    value = input(prompt)
    if value.strip().lower() in {"b", "back"}:
        raise BackToMainMenu()
    return value


def _menu_yes_no(prompt, default=False):
    suffix = " [Y/n]: " if default else " [y/N]: "
    value = _menu_input(prompt + suffix).strip().lower()

    if not value:
        return default

    return value in {"y", "yes"}


def _menu_call(function, fallback, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except BackToMainMenu:
        print("Returning to main menu.")
        return fallback


def choose_taxa(
    inventory,
):
    taxa = (
        inventory[
            "taxa"
        ]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    names = list(
        taxa.index
    )

    print()
    print("Taxonomic groups:")

    for i, name in enumerate(
        names,
        start=1,
    ):
        print(
            f"{i:2d}. "
            f"{name:30s} "
            f"{int(taxa[name])} species"
        )

    selection = _menu_input(
        "Choose number: "
    ).strip()

    try:
        n = int(
            selection
        )
    except ValueError:
        raise ValueError(
            "Invalid taxa selection."
        )

    if not (
        1
        <= n
        <= len(names)
    ):
        raise ValueError(
            "Invalid taxa selection."
        )

    return names[
        n - 1
    ]


def preview_selection(
    species_ids,
    inventory,
    status_df,
    max_parallel,
    build_tracks,
    cleanup,
):
    subset = inventory[
        inventory[
            "species_id"
        ].isin(
            species_ids
        )
    ].copy()

    status_map = dict(
        zip(
            status_df[
                "species_id"
            ],
            status_df[
                "status"
            ],
        )
    )

    print()
    print("=" * 70)
    print("RUN PREVIEW")
    print("=" * 70)

    print(
        f"Selected species:   "
        f"{len(species_ids)}"
    )

    print(
        f"Max parallel:       "
        f"{max_parallel}"
    )

    print(
        f"Generate BigWigs:   "
        f"{build_tracks}"
    )

    print(
        f"Cleanup on success: "
        f"{cleanup}"
    )

    print()

    for row in subset.itertuples(
        index=False
    ):
        status = status_map.get(
            str(row.species_id),
            "UNKNOWN",
        )

        print(
            f"{str(row.species_id):35s} "
            f"{status:12s} "
            f"{row.species_name}"
        )


def ask_yes_no(
    prompt,
    default=False,
):
    suffix = (
        " [Y/n]: "
        if default
        else " [y/N]: "
    )

    value = input(
        prompt + suffix
    ).strip().lower()

    if not value:
        return default

    return value in {
        "y",
        "yes",
    }


def get_runtime_options(
    default_parallel,
    multiple,
):
    if multiple:
        raw = _menu_input(
            f"Max parallel species "
            f"[{default_parallel}]: "
        ).strip()

        if raw:
            max_parallel = int(
                raw
            )
        else:
            max_parallel = (
                default_parallel
            )
    else:
        max_parallel = 1

    build_tracks = _menu_yes_no(
        "Generate BigWigs?",
        default=True,
    )

    cleanup = _menu_yes_no(
        "Delete temporary genomes/checkpoints "
        "after successful validation?",
        default=True,
    )

    return (
        max_parallel,
        build_tracks,
        cleanup,
    )


def execute_selection(
    species_ids,
    inventory_path,
    config_path,
    max_parallel,
    build_tracks,
    cleanup,
):
    if not species_ids:
        print(
            "No species selected."
        )
        return

    if not os.environ.get(
        "ALPHA_GENOME_API_KEY"
    ):
        raise RuntimeError(
            "ALPHA_GENOME_API_KEY is not set."
        )

    if len(species_ids) == 1:
        cmd = [
            sys.executable,
            str(
                SRC_DIR
                / "run_species.py"
            ),
            "--species",
            species_ids[0],
            "--inventory",
            str(
                inventory_path
            ),
            "--config",
            str(
                config_path
            ),
        ]

        if not build_tracks:
            cmd.append(
                "--skip-tracks"
            )

        if not cleanup:
            cmd.append(
                "--no-cleanup"
            )

    else:
        cmd = [
            sys.executable,
            str(
                SRC_DIR
                / "run_inventory.py"
            ),
            "--species",
            *species_ids,
            "--inventory",
            str(
                inventory_path
            ),
            "--config",
            str(
                config_path
            ),
            "--max-parallel",
            str(
                max_parallel
            ),
        ]

        if not build_tracks:
            cmd.append(
                "--skip-tracks"
            )

        if not cleanup:
            cmd.append(
                "--no-cleanup"
            )

    print()
    print(
        "Launching:"
    )

    print(
        " ".join(cmd)
    )

    subprocess.run(
        cmd,
        cwd=str(
            PROJECT_ROOT
        ),
        check=True,
        env=os.environ.copy(),
    )


def resolve_cli_species(
    inventory,
    values,
):
    selected = []

    for value in values:
        matches = find_species_matches(
            inventory,
            value,
        )

        if len(matches) != 1:
            raise ValueError(
                f"CLI species '{value}' must "
                f"resolve to exactly one species; "
                f"found {len(matches)}."
            )

        species_id = str(
            matches.iloc[0][
                "species_id"
            ]
        )

        if species_id not in selected:
            selected.append(
                species_id
            )

    return selected


def interactive_mode(
    inventory,
    inventory_path,
    config_path,
    config,
    status_df,
):
    # AG2_MENU_REFACTOR_V2
    config_path = Path(config_path)

    def refresh_status():
        return build_status_table(inventory, Path(config['paths']['results_dir']), Path(config['paths']['work_dir']), require_tracks=bool(config['outputs'].get('build_bigwig', True)), config=config)

    default_parallel = int(
        config[
            "runtime"
        ][
            "max_parallel_species"
        ]
    )

    while True:
        status_df = refresh_status()
        print()
        print("=" * 70)
        print("AlphaGenome 2 Genome Pipeline")
        print("=" * 70)

        print(
            f"Inventory: "
            f"{len(inventory)} species"
        )

        counts = (
            status_df[
                "status"
            ]
            .value_counts()
        )

        print(
            f"Complete: "
            f"{int(counts.get('COMPLETE', 0))}"
        )
        print(
            f"API key: "
            f"{api_key_status()}"
    )




        print(
            f"Active config: {config_path.name} | "
            f"{config['alphagenome'].get('output_type', 'RNA_SEQ')} | "
            f"{config['alphagenome'].get('condition', 'n/a')}"
        )
        print("1. Run selected species (one or multiple)")
        print("2. Run one taxonomic group")
        print("3. Run all unfinished species")
        print("4. Retry failed species")
        print("5. Download reference genome(s)")
        print("6. Configuration & pipeline status")
        print("7. Dry-run / preview")
        print("8. AlphaGenome API key")
        print("9. Exit")

        choice = input(
            "\nSelection: "
        ).strip()

        dry_run = False
        full_inventory = False

        if choice == "9":
            return
        if choice == "8":
           manage_api_key()
           continue


        if choice == "5":
            downloader = SRC_DIR / "download_reference_genomes.py"
            if not downloader.is_file():
                print(f"Downloader not found: {downloader}")
            else:
                try:
                    subprocess.run(
                        [sys.executable, str(downloader)],
                        cwd=str(PROJECT_ROOT),
                        check=True,
                    )
                except subprocess.CalledProcessError as exc:
                    print(f"Downloader failed: {exc}")
            continue

        if choice == "6":
            while True:
                configs = sorted(
                    (PROJECT_ROOT / "configs").glob("*.yaml")
                )
                print("\nConfiguration & pipeline status")
                print(f"Active: {config_path.name}")

                for i, file in enumerate(configs, 1):
                    marker = (
                        " [ACTIVE]"
                        if file.resolve() == config_path.resolve()
                        else ""
                    )
                    print(f"{i}. {file.name}{marker}")

                print("s. Show active pipeline status")
                print("b. Back to main menu (or Enter)")

                answer = input("Selection: ").strip().lower()

                if answer in {"", "b", "back"}:
                    break

                if answer == "s":
                    status_df = refresh_status()
                    print_status_summary(status_df)
                    if ask_yes_no(
                        "Show full species table?",
                        default=False,
                    ):
                        print(status_df.to_string(index=False))
                    continue

                if (
                    not answer.isdigit()
                    or not 1 <= int(answer) <= len(configs)
                ):
                    print("Invalid selection.")
                    continue

                target = configs[int(answer) - 1]

                try:
                    candidate = load_config(target)
                    candidate["alphagenome"]["output_type"]
                    candidate["paths"]["results_dir"]
                    candidate["paths"]["work_dir"]
                    parallel = int(
                        candidate["runtime"]["max_parallel_species"]
                    )
                except (
                    KeyError, TypeError, ValueError, OSError
                ) as exc:
                    print(f"Invalid configuration: {exc}")
                    continue

                config_path = target
                config = candidate
                default_parallel = parallel
                status_df = refresh_status()

                print(f"Switched to {config_path.name}")
                print_status_summary(status_df)
            continue


        if choice == "7":
            dry_run = True
            print("\nPreview scope:")
            print("1. Selected species")
            print("2. Taxonomic group")
            print("3. All unfinished species")
            print("4. Retry failed species")

            preview_choice = input(
                "Selection (b to go back): "
            ).strip().lower()

            if preview_choice in {"", "b", "back"}:
                continue

            choice = {
                "1": "1",
                "2": "2",
                "3": "3",
                "4": "4",
            }.get(preview_choice, "")

        if choice == "1":
            selected = choose_multiple_species(inventory)

        elif choice == "2":
            taxa = _menu_call(
                choose_taxa, None, inventory
            )
            if taxa is None:
                continue

            selected = (
                inventory[
                    inventory[
                        "taxa"
                    ].astype(str)
                    == taxa
                ][
                    "species_id"
                ]
                .astype(str)
                .tolist()
            )

            print(
                f"\nSelected taxon "
                f"{taxa}: "
                f"{len(selected)} species"
            )

        elif choice == "3":
            full_inventory = True

            unfinished = (
                status_df[
                    status_df[
                        "status"
                    ]
                    != "COMPLETE"
                ]
            )

            selected = (
                unfinished[
                    "species_id"
                ]
                .astype(str)
                .tolist()
            )

        elif choice == "4":
            failed = (
                status_df[
                    status_df[
                        "status"
                    ]
                    == "FAILED"
                ]
            )

            selected = (
                failed[
                    "species_id"
                ]
                .astype(str)
                .tolist()
            )

        else:
            print(
                "Invalid selection."
            )
            continue

        if not selected:
            print(
                "No species match this selection."
            )
            continue

        (
            max_parallel,
            build_tracks,
            cleanup,
        ) = _menu_call(
            get_runtime_options,
            (None, None, None),
            default_parallel,
            multiple=(
                len(selected) > 1
            ),
        )

        if max_parallel is None:
            continue

        preview_selection(
            selected,
            inventory,
            status_df,
            max_parallel,
            build_tracks,
            cleanup,
        )

        if dry_run:
            print()
            print(
                "DRY RUN ONLY — "
                "nothing was executed."
            )
            continue

        if full_inventory:
            if not ask_yes_no(
                "Continue with the full unfinished inventory?",
                default=False,
            ):
                print(
                    "Cancelled."
                )
                continue

            confirmation = input(
                f"\nType exactly:\n"
                f"RUN ALL {len(selected)}\n\n"
                f"Confirmation: "
            )

            if confirmation != (
                f"RUN ALL "
                f"{len(selected)}"
            ):
                print(
                    "Confirmation did not match. "
                    "Cancelled."
                )
                continue

        else:
            if not ask_yes_no(
                "Proceed with this run?",
                default=False,
            ):
                print(
                    "Cancelled."
                )
                continue

        execute_selection(
            selected,
            inventory_path,
            config_path,
            max_parallel,
            build_tracks,
            cleanup,
        )

        #
        # Refresh statuses after run.
        #
        status_df = (
            build_status_table(
                inventory,
                Path(config["paths"]["results_dir"]),
                Path(config["paths"]["work_dir"]),
                require_tracks=(
                    bool(
                        config[
                            "outputs"
                        ].get(
                            "build_bigwig",
                            True,
                        )
                    )
                ),
                config=config,
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Interactive and command-line controller "
            "for the AlphaGenome 2 genome pipeline."
        )
    )

    selection = (
        parser.add_mutually_exclusive_group()
    )

    selection.add_argument(
        "--species",
        nargs="+",
        help=(
            "Run one or more species."
        ),
    )

    selection.add_argument(
        "--taxa",
        help=(
            "Run all unfinished species "
            "within one taxa value."
        ),
    )

    selection.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run all unfinished species."
        ),
    )

    selection.add_argument(
        "--failed",
        action="store_true",
        help=(
            "Retry species currently marked FAILED."
        ),
    )

    selection.add_argument(
        "--status",
        action="store_true",
        help=(
            "Show pipeline status and exit."
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
        "--dry-run",
        action="store_true",
        help=(
            "Preview selection and settings "
            "without executing anything."
        ),
    )

    parser.add_argument(
        "--confirm-all",
        action="store_true",
        help=(
            "Required with --all for an unattended "
            "full-inventory run."
        ),
    )

    args = parser.parse_args()
    load_api_key()
    inventory = pd.read_csv(
        args.inventory
    )

    config = load_config(
        args.config
    )

    default_build_tracks = bool(
        config[
            "outputs"
        ].get(
            "build_bigwig",
            True,
        )
    )

    status_df = build_status_table(
        inventory,
        Path(config["paths"]["results_dir"]),
        Path(config["paths"]["work_dir"]),
        require_tracks=(
            default_build_tracks
        ),
        config=config,
    )

    #
    # No selection arguments means interactive mode.
    #
    if not any(
        [
            args.species,
            args.taxa,
            args.all,
            args.failed,
            args.status,
        ]
    ):
        interactive_mode(
            inventory,
            args.inventory,
            args.config,
            config,
            status_df,
        )

        return

    if args.status:
        print_status_summary(
            status_df
        )

        print()
        print(
            status_df.to_string(
                index=False
            )
        )

        return

    if args.species:
        selected = resolve_cli_species(
            inventory,
            args.species,
        )

    elif args.taxa:
        matches = inventory[
            inventory[
                "taxa"
            ].astype(str)
            == args.taxa
        ]

        if len(matches) == 0:
            raise ValueError(
                f"No taxa found: "
                f"{args.taxa}"
            )

        selected = (
            matches[
                "species_id"
            ]
            .astype(str)
            .tolist()
        )

    elif args.failed:
        selected = (
            status_df[
                status_df[
                    "status"
                ]
                == "FAILED"
            ][
                "species_id"
            ]
            .astype(str)
            .tolist()
        )

    elif args.all:
        selected = (
            status_df[
                status_df[
                    "status"
                ]
                != "COMPLETE"
            ][
                "species_id"
            ]
            .astype(str)
            .tolist()
        )

        if (
            not args.dry_run
            and not args.confirm_all
        ):
            raise RuntimeError(
                "--all requires --confirm-all "
                "unless --dry-run is used."
            )

    else:
        selected = []

    if not selected:
        print(
            "No species selected."
        )
        return

    max_parallel = (
        args.max_parallel
        if args.max_parallel is not None
        else int(
            config[
                "runtime"
            ][
                "max_parallel_species"
            ]
        )
    )

    build_tracks = (
        default_build_tracks
        and not args.skip_tracks
    )

    cleanup = (
        not args.no_cleanup
    )

    preview_selection(
        selected,
        inventory,
        status_df,
        max_parallel,
        build_tracks,
        cleanup,
    )

    if args.dry_run:
        print()
        print(
            "DRY RUN ONLY — "
            "nothing was executed."
        )
        return

    execute_selection(
        selected,
        args.inventory,
        args.config,
        max_parallel,
        build_tracks,
        cleanup,
    )


if __name__ == "__main__":
    main()