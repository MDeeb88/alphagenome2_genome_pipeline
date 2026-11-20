#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from api_key import load_api_key
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
import zarr


PROJECT_ROOT = Path(
    "/media/eternus1/nfs/projects/dpanchenko/"
    "alphagenome2_genome_pipeline"
)

SRC_DIR = PROJECT_ROOT / "src"

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

DEFAULT_INVENTORY = (
    PROJECT_ROOT
    / "inventory"
    / "CTCF_Figure7_genome_annotation_inventory.csv"
)

DEFAULT_WORK = PROJECT_ROOT / "work"
DEFAULT_RESULTS = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def load_yaml(path):
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


def atomic_write_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        tmp,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )

    os.replace(
        tmp,
        path,
    )


def clean_json_value(value):
    if pd.isna(value):
        return None

    if hasattr(
        value,
        "item",
    ):
        try:
            return value.item()
        except Exception:
            pass

    return value


# ---------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------

def load_species_row(
    inventory_path,
    species_id,
):
    df = pd.read_csv(
        inventory_path
    )

    match = df[
        df["species_id"]
        == species_id
    ]

    if len(match) == 0:
        raise ValueError(
            f"Species not found in inventory: "
            f"{species_id}"
        )

    if len(match) > 1:
        raise ValueError(
            f"Species occurs more than once "
            f"in inventory: {species_id}"
        )

    row = match.iloc[0]

    accession = str(
        row["ncbi_assembly_accession"]
    ).strip()

    if (
        not accession
        or accession.lower() == "nan"
    ):
        raise ValueError(
            f"No NCBI assembly accession "
            f"for {species_id}"
        )

    row_dict = {
        str(key): clean_json_value(
            value
        )
        for key, value
        in row.items()
    }

    return (
        row_dict,
        accession,
    )


# ---------------------------------------------------------------------
# Logging / command execution
# ---------------------------------------------------------------------

def run_logged(
    cmd,
    log_path,
    cwd=PROJECT_ROOT,
):
    """
    Run one pipeline command.

    Output is shown live in the terminal and simultaneously
    written to a species-specific log file.

    Output is forwarded as raw bytes so terminal control
    characters such as tqdm carriage returns are preserved.
    This keeps progress bars on one updating line.
    """

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("RUNNING")
    print("=" * 70)
    print(
        " ".join(
            str(x)
            for x in cmd
        )
    )
    print()

    header = (
        "\n"
        + "=" * 70
        + "\n"
        + f"STARTED UTC: {utc_now()}\n"
        + "COMMAND: "
        + " ".join(
            str(x)
            for x in cmd
        )
        + "\n"
        + "=" * 70
        + "\n"
    )

    with open(
        log_path,
        "ab",
        buffering=0,
    ) as log:

        log.write(
            header.encode(
                "utf-8"
            )
        )

        process = subprocess.Popen(
            [
                str(x)
                for x in cmd
            ],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )

        assert (
            process.stdout
            is not None
        )

        while True:
            chunk = os.read(
                process.stdout.fileno(),
                8192,
            )

            if not chunk:
                break

            #
            # Preserve tqdm's \r and ANSI terminal
            # control sequences exactly.
            #
            sys.stdout.buffer.write(
                chunk
            )

            sys.stdout.buffer.flush()

            log.write(
                chunk
            )

        return_code = (
            process.wait()
        )

        footer = (
            f"\nFINISHED UTC: "
            f"{utc_now()}\n"
            f"RETURN CODE: "
            f"{return_code}\n"
        )

        log.write(
            footer.encode(
                "utf-8"
            )
        )

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            cmd,
        )


def generate_sequence_report(
    accession,
    output_path,
    log_path,
):
    """
    Generate the NCBI per-sequence report needed for chromosome /
    scaffold filtering.

    The JSONL report itself goes to output_path.
    Diagnostic stderr goes into the pipeline log.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    if tmp_path.exists():
        tmp_path.unlink()

    cmd = [
        "datasets",
        "summary",
        "genome",
        "accession",
        accession,
        "--report",
        "sequence",
        "--as-json-lines",
    ]

    print()
    print("=" * 70)
    print("GENERATING NCBI SEQUENCE REPORT")
    print("=" * 70)
    print(
        " ".join(cmd)
    )

    with open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as output, open(
        log_path,
        "a",
        encoding="utf-8",
    ) as log:

        log.write(
            "\n"
            + "=" * 70
            + "\n"
        )

        log.write(
            f"STARTED UTC: {utc_now()}\n"
        )

        log.write(
            "COMMAND: "
            + " ".join(cmd)
            + "\n"
        )

        log.write(
            "=" * 70
            + "\n"
        )

        result = subprocess.run(
            cmd,
            cwd=str(
                PROJECT_ROOT
            ),
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.stderr:
            print(
                result.stderr,
                end="",
            )

            log.write(
                result.stderr
            )

        log.write(
            f"\nFINISHED UTC: "
            f"{utc_now()}\n"
        )

        log.write(
            f"RETURN CODE: "
            f"{result.returncode}\n"
        )

    if result.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink()

        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
        )

    if (
        not tmp_path.exists()
        or tmp_path.stat().st_size == 0
    ):
        raise RuntimeError(
            "NCBI sequence report "
            "was empty."
        )

    os.replace(
        tmp_path,
        output_path,
    )

    print(
        "Sequence report:",
        output_path,
    )


# ---------------------------------------------------------------------
# Permanent-output validation
# ---------------------------------------------------------------------

def consolidation_complete(
    species_results,
):
    store_path = (
        species_results
        / "predictions"
        / "predictions.zarr"
    )

    summary_path = (
        species_results
        / "metadata"
        / "consolidation_summary.json"
    )

    summary = read_json(
        summary_path
    )

    if not store_path.exists():
        return False

    if not summary:
        return False

    if not summary.get(
        "complete",
        False,
    ):
        return False

    try:
        root = zarr.open_group(
            str(store_path),
            mode="r",
        )

        if not root.attrs.get(
            "complete",
            False,
        ):
            return False

        if (
            int(
                root.attrs.get(
                    "selected_bases",
                    -1,
                )
            )
            != int(
                summary.get(
                    "selected_bases",
                    -2,
                )
            )
        ):
            return False

    except Exception:
        return False

    return True


def tracks_complete(
    species_results,
):
    tracks_dir = (
        species_results
        / "tracks"
    )

    summary_path = (
        tracks_dir
        / "track_build_summary.json"
    )

    summary = read_json(
        summary_path
    )

    if not summary:
        return False

    if not summary.get(
        "complete",
        False,
    ):
        return False

    if not summary.get(
        "validation_passed",
        False,
    ):
        return False

    required = [
        tracks_dir
        / "totalRNA_plus.bw",

        tracks_dir
        / "totalRNA_minus.bw",

        tracks_dir
        / "chrom.sizes",

        tracks_dir
        / "chrom_aliases.tsv",
    ]

    return all(
        path.exists()
        and path.stat().st_size > 0
        for path in required
    )


def final_outputs_complete(
    species_results,
    require_tracks,
):
    if not consolidation_complete(
        species_results
    ):
        return False

    if (
        require_tracks
        and not tracks_complete(
            species_results
        )
    ):
        return False

    return True


# ---------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------

def mark_stage(
    status,
    status_path,
    stage,
    state,
    message=None,
):
    status[
        "stages"
    ].setdefault(
        stage,
        {}
    )

    status[
        "stages"
    ][stage][
        "status"
    ] = state

    if state == "running":
        status[
            "stages"
        ][stage][
            "started_utc"
        ] = utc_now()

    if state in {
        "complete",
        "skipped",
        "failed",
    }:
        status[
            "stages"
        ][stage][
            "finished_utc"
        ] = utc_now()

    if message is not None:
        status[
            "stages"
        ][stage][
            "message"
        ] = str(
            message
        )

    atomic_write_json(
        status_path,
        status,
    )


# ---------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------

def cleanup_work(
    species_work,
    delete_genome,
    delete_tiles,
):
    if not species_work.exists():
        return

    if (
        delete_genome
        and delete_tiles
    ):
        #
        # All useful planning/assembly metadata has already been
        # copied into results/<species>/metadata by consolidation.
        #
        shutil.rmtree(
            species_work
        )

        return

    if delete_genome:
        for name in [
            "genome.fa",
            "genome.fa.fai",
        ]:
            path = (
                species_work
                / name
            )

            if path.exists():
                path.unlink()

    if delete_tiles:
        tile_dir = (
            species_work
            / "prediction_tiles"
        )

        if tile_dir.exists():
            shutil.rmtree(
                tile_dir
            )


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete AlphaGenome 2 pipeline "
            "for one species."
        )
    )

    parser.add_argument(
        "--species",
        required=True,
        help=(
            "species_id from "
            "CTCF_Figure7_genome_annotation_inventory.csv"
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
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
    )

    parser.add_argument(
        "--skip-tracks",
        action="store_true",
        help=(
            "Do not generate optional BigWig tracks."
        ),
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help=(
            "Keep downloaded genome and temporary "
            "prediction checkpoints after success."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run a species even if permanent "
            "outputs are already complete."
        ),
    )

    args = parser.parse_args()

    load_api_key()
    started_clock = (
        time.perf_counter()
    )

    cfg = load_yaml(
        args.config
    )

    species_row, accession = (
        load_species_row(
            args.inventory,
            args.species,
        )
    )

    config_build_tracks = bool(
        cfg.get(
            "outputs",
            {}
        ).get(
            "build_bigwig",
            True,
        )
    )

    build_tracks = (
        config_build_tracks
        and not args.skip_tracks
    )

    species_work = (
        args.work_dir
        / args.species
    )

    species_results = (
        args.results_dir
        / args.species
    )

    metadata_dir = (
        species_results
        / "metadata"
    )

    logs_dir = (
        species_results
        / "logs"
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        metadata_dir
        / "pipeline_status.json"
    )

    #
    # Preserve exactly which inventory record and configuration
    # produced this species result.
    #
    shutil.copy2(
        args.config,
        metadata_dir
        / "pipeline_config.yaml",
    )

    atomic_write_json(
        metadata_dir
        / "inventory_row.json",
        species_row,
    )

    status = {
        "species_id": args.species,
        "species_name": (
            species_row.get(
                "species_name"
            )
        ),
        "assembly_accession": (
            accession
        ),
        "pipeline_root": str(
            PROJECT_ROOT
        ),
        "inventory": str(
            args.inventory
        ),
        "config": str(
            args.config
        ),
        "build_tracks": bool(
            build_tracks
        ),
        "started_utc": utc_now(),
        "complete": False,
        "stages": {},
    }

    atomic_write_json(
        status_path,
        status,
    )

    print("=" * 70)
    print("AlphaGenome 2 species pipeline")
    print("=" * 70)

    print(
        "Species ID:       ",
        args.species,
    )

    print(
        "Species name:     ",
        species_row.get(
            "species_name"
        ),
    )

    print(
        "Taxa:             ",
        species_row.get(
            "taxa"
        ),
    )

    print(
        "Assembly:         ",
        accession,
    )

    print(
        "Generate tracks:  ",
        build_tracks,
    )

    print(
        "Cleanup work:     ",
        not args.no_cleanup,
    )

    print()

    #
    # If this species was already fully processed, do not download
    # the genome again.
    #
    if (
        not args.force
        and final_outputs_complete(
            species_results,
            require_tracks=build_tracks,
        )
    ):
        status[
            "complete"
        ] = True

        status[
            "finished_utc"
        ] = utc_now()

        status[
            "message"
        ] = (
            "Permanent outputs already complete. "
            "No work required."
        )

        atomic_write_json(
            status_path,
            status,
        )

        print(
            "Species already complete."
        )

        print(
            "Permanent result:",
            species_results,
        )

        return

    current_stage = None

    try:

        # ============================================================
        # 1. Download + index genome
        # ============================================================

        current_stage = "download_genome"

        genome_path = (
            species_work
            / "genome.fa"
        )

        genome_index = (
            species_work
            / "genome.fa.fai"
        )

        assembly_metadata = (
            species_work
            / "assembly_metadata.json"
        )

        if (
            genome_path.exists()
            and genome_index.exists()
            and assembly_metadata.exists()
            and not args.force
        ):
            mark_stage(
                status,
                status_path,
                current_stage,
                "skipped",
                (
                    "Prepared genome "
                    "already exists."
                ),
            )

            print(
                "Prepared genome already exists; "
                "download skipped."
            )

        else:
            mark_stage(
                status,
                status_path,
                current_stage,
                "running",
            )

            cmd = [
                sys.executable,
                SRC_DIR
                / "download_genome.py",
                "--species",
                args.species,
                "--inventory",
                args.inventory,
                "--work-dir",
                args.work_dir,
            ]

            if args.force:
                cmd.append(
                    "--force"
                )

            run_logged(
                cmd,
                logs_dir
                / "01_download.log",
            )

            mark_stage(
                status,
                status_path,
                current_stage,
                "complete",
            )

        # ============================================================
        # 2. NCBI sequence report
        # ============================================================

        current_stage = (
            "sequence_report"
        )

        sequence_report = (
            species_work
            / "sequence_report.jsonl"
        )

        if (
            sequence_report.exists()
            and sequence_report.stat().st_size > 0
            and not args.force
        ):
            mark_stage(
                status,
                status_path,
                current_stage,
                "skipped",
                (
                    "Sequence report "
                    "already exists."
                ),
            )

            print(
                "NCBI sequence report already exists; "
                "generation skipped."
            )

        else:
            mark_stage(
                status,
                status_path,
                current_stage,
                "running",
            )

            generate_sequence_report(
                accession=accession,
                output_path=(
                    sequence_report
                ),
                log_path=(
                    logs_dir
                    / "02_sequence_report.log"
                ),
            )

            mark_stage(
                status,
                status_path,
                current_stage,
                "complete",
            )

        # ============================================================
        # 3. Plan genome tiles
        # ============================================================

        current_stage = (
            "plan_tiles"
        )

        mark_stage(
            status,
            status_path,
            current_stage,
            "running",
        )

        run_logged(
            [
                sys.executable,
                SRC_DIR
                / "plan_genome_tiles.py",
                "--species",
                args.species,
                "--config",
                args.config,
                "--work-dir",
                args.work_dir,
            ],
            logs_dir
            / "03_plan_tiles.log",
        )

        mark_stage(
            status,
            status_path,
            current_stage,
            "complete",
        )

        # ============================================================
        # 4. AlphaGenome predictions
        # ============================================================

        current_stage = (
            "predict_genome"
        )

        if consolidation_complete(
            species_results
        ) and not args.force:
            #
            # Permanent predictions already exist, therefore no reason
            # to contact AlphaGenome again.
            #
            mark_stage(
                status,
                status_path,
                current_stage,
                "skipped",
                (
                    "Permanent consolidated predictions "
                    "already exist."
                ),
            )

            print(
                "Permanent predictions already exist; "
                "AlphaGenome prediction skipped."
            )

        else:
            if not os.environ.get(
                "ALPHA_GENOME_API_KEY"
            ):
                raise RuntimeError(
                    "ALPHA_GENOME_API_KEY is not set.\n"
                    "Set it before running a species that "
                    "requires new AlphaGenome predictions."
                )

            mark_stage(
                status,
                status_path,
                current_stage,
                "running",
            )

            run_logged(
                [
                    sys.executable,
                    SRC_DIR
                    / "predict_genome.py",
                    "--species",
                    args.species,
                    "--config",
                    args.config,
                    "--work-dir",
                    args.work_dir,
                    "--results-dir",
                    args.results_dir,
                ],
                logs_dir
                / "04_prediction.log",
            )

            mark_stage(
                status,
                status_path,
                current_stage,
                "complete",
            )

        # ============================================================
        # 5. Consolidate predictions
        # ============================================================

        current_stage = (
            "consolidate_predictions"
        )

        if (
            consolidation_complete(
                species_results
            )
            and not args.force
        ):
            mark_stage(
                status,
                status_path,
                current_stage,
                "skipped",
                (
                    "Validated permanent "
                    "prediction store already exists."
                ),
            )

            print(
                "Consolidated prediction store "
                "already valid; consolidation skipped."
            )

        else:
            mark_stage(
                status,
                status_path,
                current_stage,
                "running",
            )

            cmd = [
                sys.executable,
                SRC_DIR
                / "consolidate_predictions.py",
                "--species",
                args.species,
                "--work-dir",
                args.work_dir,
                "--results-dir",
                args.results_dir,
            ]

            if args.force:
                cmd.append(
                    "--force"
                )

            run_logged(
                cmd,
                logs_dir
                / "05_consolidation.log",
            )

            if not consolidation_complete(
                species_results
            ):
                raise RuntimeError(
                    "Consolidation command finished, "
                    "but permanent prediction validation "
                    "did not pass."
                )

            mark_stage(
                status,
                status_path,
                current_stage,
                "complete",
            )

        # ============================================================
        # 6. Build optional BigWig tracks
        # ============================================================

        current_stage = (
            "build_tracks"
        )

        if not build_tracks:
            mark_stage(
                status,
                status_path,
                current_stage,
                "skipped",
                (
                    "BigWig generation disabled."
                ),
            )

        elif (
            tracks_complete(
                species_results
            )
            and not args.force
        ):
            mark_stage(
                status,
                status_path,
                current_stage,
                "skipped",
                (
                    "Validated BigWig tracks "
                    "already exist."
                ),
            )

            print(
                "Validated BigWigs already exist; "
                "track generation skipped."
            )

        else:
            mark_stage(
                status,
                status_path,
                current_stage,
                "running",
            )

            cmd = [
                sys.executable,
                SRC_DIR
                / "build_tracks.py",
                "--species",
                args.species,
                "--results-dir",
                args.results_dir,
            ]

            if args.force:
                cmd.append(
                    "--force"
                )

            run_logged(
                cmd,
                logs_dir
                / "06_tracks.log",
            )

            if not tracks_complete(
                species_results
            ):
                raise RuntimeError(
                    "BigWig command finished, "
                    "but track validation did not pass."
                )

            mark_stage(
                status,
                status_path,
                current_stage,
                "complete",
            )

        # ============================================================
        # 7. Final permanent-output validation
        # ============================================================

        current_stage = (
            "final_validation"
        )

        mark_stage(
            status,
            status_path,
            current_stage,
            "running",
        )

        if not final_outputs_complete(
            species_results,
            require_tracks=build_tracks,
        ):
            raise RuntimeError(
                "Final permanent-output validation failed."
            )

        mark_stage(
            status,
            status_path,
            current_stage,
            "complete",
        )

        # ============================================================
        # 8. Cleanup temporary workspace
        # ============================================================

        current_stage = (
            "cleanup"
        )

        if args.no_cleanup:
            mark_stage(
                status,
                status_path,
                current_stage,
                "skipped",
                (
                    "--no-cleanup requested."
                ),
            )

        else:
            mark_stage(
                status,
                status_path,
                current_stage,
                "running",
            )

            cleanup_cfg = (
                cfg.get(
                    "cleanup",
                    {}
                )
            )

            delete_genome = bool(
                cleanup_cfg.get(
                    "delete_downloaded_genome",
                    True,
                )
            )

            delete_tiles = bool(
                cleanup_cfg.get(
                    "delete_temporary_predictions",
                    True,
                )
            )

            cleanup_work(
                species_work=species_work,
                delete_genome=delete_genome,
                delete_tiles=delete_tiles,
            )

            mark_stage(
                status,
                status_path,
                current_stage,
                "complete",
                (
                    f"delete_genome={delete_genome}; "
                    f"delete_tiles={delete_tiles}"
                ),
            )

        # ============================================================
        # Success
        # ============================================================

        elapsed = (
            time.perf_counter()
            - started_clock
        )

        status[
            "complete"
        ] = True

        status[
            "finished_utc"
        ] = utc_now()

        status[
            "elapsed_seconds"
        ] = float(
            elapsed
        )

        status[
            "result_directory"
        ] = str(
            species_results
        )

        atomic_write_json(
            status_path,
            status,
        )

        print()
        print("=" * 70)
        print("SPECIES PIPELINE COMPLETE")
        print("=" * 70)

        print(
            "Species:",
            args.species,
        )

        print(
            "Assembly:",
            accession,
        )

        print(
            "Result:",
            species_results,
        )

        print(
            "Elapsed:",
            f"{elapsed / 60:.1f} min",
        )

        print(
            "Cleanup:",
            (
                "completed"
                if not args.no_cleanup
                else "disabled"
            ),
        )

    except Exception as exc:

        if current_stage is not None:
            mark_stage(
                status,
                status_path,
                current_stage,
                "failed",
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        status[
            "complete"
        ] = False

        status[
            "failed_utc"
        ] = utc_now()

        status[
            "error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        atomic_write_json(
            status_path,
            status,
        )

        print()
        print("=" * 70)
        print("SPECIES PIPELINE FAILED")
        print("=" * 70)

        print(
            "Stage:",
            current_stage,
        )

        print(
            "Error:",
            exc,
        )

        print()
        print(
            "Temporary files were NOT automatically "
            "deleted."
        )

        print(
            "Fix the problem and run the same command "
            "again; checkpointed predictions can resume."
        )

        raise


if __name__ == "__main__":
    main()