#!/usr/bin/env python3

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import yaml
from tqdm import tqdm

from alphagenome.models.v2 import dna_client


PROJECT_ROOT = Path(
    "/media/eternus1/nfs/projects/dpanchenko/"
    "alphagenome2_genome_pipeline"
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_WORK = PROJECT_ROOT / "work"
DEFAULT_RESULTS = PROJECT_ROOT / "results"


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def atomic_save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            **arrays,
        )

    os.replace(
        tmp,
        path,
    )


def sanitize_sequence(sequence):
    sequence = sequence.upper()

    chars = np.frombuffer(
        sequence.encode("ascii"),
        dtype=np.uint8,
    )

    valid = (
        (chars == ord("A"))
        | (chars == ord("C"))
        | (chars == ord("G"))
        | (chars == ord("T"))
        | (chars == ord("N"))
    )

    replaced = int(
        (~valid).sum()
    )

    if replaced:
        chars = chars.copy()
        chars[~valid] = ord("N")
        sequence = chars.tobytes().decode("ascii")

    return sequence, replaced


def select_rna_tracks(track_data, cfg):
    metadata = track_data.metadata.copy()

    assay = cfg["alphagenome"]["assay_title"]
    imputation = cfg["alphagenome"]["imputation_mode"]
    endedness = cfg["alphagenome"]["endedness"]

    mask = (
        (metadata["Assay title"] == assay)
        & (metadata["imputation_mode"] == imputation)
        & (metadata["endedness"] == endedness)
        & (metadata["strand"].isin(["+", "-"]))
    )

    selected = metadata.loc[mask].copy()

    if len(selected) != 2:
        raise RuntimeError(
            "Expected exactly two RNA-seq tracks "
            f"(+ and -), found {len(selected)}.\n"
            f"{selected.to_string(index=False)}"
        )

    if set(selected["strand"]) != {"+", "-"}:
        raise RuntimeError(
            "Selected RNA tracks do not contain exactly "
            "one '+' and one '-' strand."
        )

    plus_idx = int(
        selected.index[
            selected["strand"] == "+"
        ][0]
    )

    minus_idx = int(
        selected.index[
            selected["strand"] == "-"
        ][0]
    )

    return (
        plus_idx,
        minus_idx,
        selected,
    )



def select_atac_tracks(track_data, cfg):
    """Select HEK293T ATAC tracks using output metadata."""
    metadata = track_data.metadata.copy()

    required = {
        "Assay title",
        "imputation_mode",
        "strand",
    }
    missing = required - set(metadata.columns)

    if missing:
        raise RuntimeError(
            f"ATAC metadata is missing columns: {sorted(missing)}"
        )

    assay = cfg["alphagenome"]["assay_title"]
    imputation = cfg["alphagenome"]["imputation_mode"]

    mask = (
        metadata["Assay title"].eq(assay)
        & metadata["imputation_mode"].eq(imputation)
        & metadata["strand"].eq(".")
    )

    indices = np.flatnonzero(mask.to_numpy()).tolist()
    selected = metadata.iloc[indices].copy()

    if not indices:
        raise RuntimeError(
            "No ATAC tracks match the configuration.\n"
            f"Requested assay: {assay}\n"
            f"Requested imputation: {imputation}\n"
            f"Available metadata:\n"
            f"{metadata.to_string(index=False)}"
        )

    if track_data.values.shape[1] != len(metadata):
        raise RuntimeError(
            "ATAC channel count does not match metadata."
        )

    return indices, selected


def validate_existing_tile(path, row, cfg):
    if not path.exists():
        return False

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as x:
            required = {
                "values",
                "track_strands",
                "fasta_name",
                "retained_start",
                "retained_end",
                "tile_index",
            }

            if not required.issubset(
                set(x.files)
            ):
                return False

            values = x["values"]

            expected_length = int(
                row["retained_length"]
            )

            assay_cfg = cfg["alphagenome"]
            output_type = assay_cfg.get(
                "output_type", "RNA_SEQ"
            ).upper()

            if (
                values.ndim != 2
                or values.shape[0] != expected_length
                or values.shape[1] < 1
            ):
                return False

            strands = x["track_strands"]
            if len(strands) != values.shape[1]:
                return False

            if output_type == "RNA_SEQ":
                if (
                    values.shape[1] != 2
                    or list(strands) != ["+", "-"]
                ):
                    return False
            elif output_type == "ATAC":
                if any(strand != "." for strand in strands):
                    return False
            else:
                return False

            if "output_type" in x.files:
                if str(x["output_type"].item()) != output_type:
                    return False
            elif output_type != "RNA_SEQ":
                return False

            for field, expected in (
                ("condition", assay_cfg["condition"]),
                ("assay_title", assay_cfg["assay_title"]),
                ("imputation_mode", assay_cfg["imputation_mode"]),
            ):
                if field not in x.files:
                    return False
                if str(x[field].item()) != str(expected):
                    return False

            if not np.all(
                np.isfinite(values)
            ):
                return False

            if int(
                x["tile_index"].item()
            ) != int(row["tile_index"]):
                return False

            if str(
                x["fasta_name"].item()
            ) != str(row["fasta_name"]):
                return False

            if int(
                x["retained_start"].item()
            ) != int(row["retained_start"]):
                return False

            if int(
                x["retained_end"].item()
            ) != int(row["retained_end"]):
                return False

        return True

    except Exception:
        return False


def build_input_sequence(
    fasta,
    row,
    input_length,
):
    fasta_name = str(
        row["fasta_name"]
    )

    genomic_start = int(
        row["genomic_input_start"]
    )

    genomic_end = int(
        row["genomic_input_end"]
    )

    pad_left = int(
        row["pad_left"]
    )

    pad_right = int(
        row["pad_right"]
    )

    sequence = fasta.fetch(
        fasta_name,
        genomic_start,
        genomic_end,
    )

    sequence, ambiguity_count = (
        sanitize_sequence(sequence)
    )

    sequence = (
        ("N" * pad_left)
        + sequence
        + ("N" * pad_right)
    )

    if len(sequence) != input_length:
        raise RuntimeError(
            f"Input length error for "
            f"{fasta_name}: "
            f"{len(sequence)} != "
            f"{input_length}"
        )

    return (
        sequence,
        ambiguity_count,
    )


def predict_with_retry(
    model,
    sequence,
    condition,
    output_type,
    max_retries,
    retry_seconds,
):
    requested_output = {
        "RNA_SEQ": dna_client.OutputType.RNA_SEQ,
        "ATAC": dna_client.OutputType.ATAC,
    }[output_type]

    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):
        start_time = time.perf_counter()

        try:
            output = model.predict_sequence(
                sequence=sequence,
                requested_outputs=[
                    requested_output
                ],
                condition=condition,
            )

            seconds = (
                time.perf_counter()
                - start_time
            )

            return output, seconds

        except Exception as exc:
            last_error = exc

            print()
            print(
                f"Prediction attempt "
                f"{attempt}/{max_retries} failed:"
            )
            print(
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < max_retries:
                sleep_time = (
                    retry_seconds
                    * attempt
                )

                print(
                    f"Retrying in "
                    f"{sleep_time} seconds..."
                )

                time.sleep(
                    sleep_time
                )

    raise RuntimeError(
        "AlphaGenome prediction failed after "
        f"{max_retries} attempts."
    ) from last_error


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable AlphaGenome 2 predictions "
            "for one planned genome."
        )
    )

    parser.add_argument(
        "--species",
        required=True,
        help="species_id from inventory",
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
        "--max-tiles",
        type=int,
        default=None,
        help=(
            "Optional limit for testing. "
            "Existing tiles count toward this limit."
        ),
    )

    args = parser.parse_args()

    cfg = load_config(
        args.config
    )

    condition = str(
        cfg["alphagenome"]["condition"]
    )

    output_type = str(
        cfg["alphagenome"].get(
            "output_type", "RNA_SEQ"
        )
    ).upper()

    if output_type not in {"RNA_SEQ", "ATAC"}:
        raise ValueError(
            f"Unsupported output type: {output_type}"
        )

    input_length = int(
        cfg["tiling"]["input_length"]
    )

    max_retries = int(
        cfg["prediction"]["max_retries"]
    )

    retry_seconds = int(
        cfg["prediction"]["retry_seconds"]
    )

    species_work = (
        args.work_dir
        / args.species
    )

    genome_path = (
        species_work
        / "genome.fa"
    )

    manifest_path = (
        species_work
        / "tile_manifest.csv"
    )

    if not genome_path.exists():
        raise FileNotFoundError(
            f"Missing genome: {genome_path}"
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing tile manifest: "
            f"{manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    if args.max_tiles is not None:
        manifest = manifest.iloc[
            : args.max_tiles
        ].copy()

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

    #
    # These are temporary checkpoint files.
    # They will later be consolidated into the
    # permanent prediction store.
    #
    tile_dir = (
        species_work
        / "prediction_tiles"
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tile_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    api_key = os.environ.get(
        "ALPHA_GENOME_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "ALPHA_GENOME_API_KEY is not set."
        )

    print("=" * 70)
    print("AlphaGenome 2 whole-genome predictor")
    print("=" * 70)
    print(
        "Species:       ",
        args.species,
    )
    print(
        "Genome:        ",
        genome_path,
    )
    print(
        "Tiles planned: ",
        len(manifest),
    )
    print(
        "Condition:     ",
        condition,
    )
    print(
        "Input length:  ",
        f"{input_length:,}",
    )
    print(
        "Checkpoint dir:",
        tile_dir,
    )
    print()

    model = dna_client.create(
        api_key
    )

    fasta = pysam.FastaFile(
        str(genome_path)
    )

    track_indices = None
    track_strands = None
    selected_metadata = None

    new_predictions = 0
    reused_predictions = 0
    ambiguity_total = 0

    prediction_records = []

    started = datetime.now(
        timezone.utc
    )

    try:
        progress = tqdm(
            manifest.itertuples(
                index=False
            ),
            total=len(manifest),
            desc=f"Predicting {args.species}",
            unit="tile",
        )

        for row_tuple in progress:
            row = row_tuple._asdict()

            tile_index = int(
                row["tile_index"]
            )

            fasta_name = str(
                row["fasta_name"]
            )

            retained_start = int(
                row["retained_start"]
            )

            retained_end = int(
                row["retained_end"]
            )

            filename = (
                f"tile_{tile_index:07d}__"
                f"{safe_name(fasta_name)}__"
                f"{retained_start}-{retained_end}.npz"
            )

            output_path = (
                tile_dir
                / filename
            )

            if validate_existing_tile(
                output_path,
                row,
                cfg,
            ):
                reused_predictions += 1

                progress.set_postfix(
                    new=new_predictions,
                    reused=reused_predictions,
                )

                continue

            sequence, ambiguity_count = (
                build_input_sequence(
                    fasta,
                    row,
                    input_length,
                )
            )

            ambiguity_total += (
                ambiguity_count
            )

            output, prediction_seconds = (
                predict_with_retry(
                    model=model,
                    sequence=sequence,
                    condition=condition,
                    output_type=output_type,
                    max_retries=max_retries,
                    retry_seconds=retry_seconds,
                )
            )

            track_data = (
                output.rna_seq
                if output_type == "RNA_SEQ"
                else output.atac
            )

            if track_data is None:
                raise RuntimeError(
                    f"No {output_type} output returned."
                )

            if track_indices is None:
                if output_type == "RNA_SEQ":
                    plus_idx, minus_idx, selected_metadata = (
                        select_rna_tracks(track_data, cfg)
                    )
                    track_indices = [plus_idx, minus_idx]
                    track_strands = ["+", "-"]
                    selected_metadata = (
                        track_data.metadata.iloc[track_indices].copy()
                    )
                else:
                    track_indices, selected_metadata = (
                        select_atac_tracks(track_data, cfg)
                    )
                    track_strands = (
                        selected_metadata["strand"]
                        .astype(str)
                        .tolist()
                    )

                selected_metadata.to_csv(
                    metadata_dir / "selected_track_metadata.csv",
                    index=False,
                )

            crop_start = int(row["crop_start"])
            crop_end = int(row["crop_end"])

            values = np.asarray(
                track_data.values[
                    crop_start:crop_end, :
                ][:, track_indices],
                dtype=np.float32,
            )

            expected_length = int(
                row["retained_length"]
            )

            if values.shape != (
                expected_length,
                len(track_indices),
            ):
                raise RuntimeError(
                    f"Unexpected cropped output "
                    f"shape for tile {tile_index}: "
                    f"{values.shape}"
                )

            if not np.all(
                np.isfinite(values)
            ):
                raise RuntimeError(
                    f"Non-finite prediction values "
                    f"in tile {tile_index}"
                )

            atomic_save_npz(
                output_path,
                values=values,
                track_strands=np.array(track_strands),
                track_indices=np.array(
                    track_indices, dtype=np.int32
                ),
                output_type=np.array(output_type),
                track_orientation=np.array(
                    "genomic_reference"
                ),
                species_id=np.array(
                    args.species
                ),
                tile_index=np.array(
                    tile_index
                ),
                fasta_name=np.array(
                    fasta_name
                ),
                chr_name=np.array(
                    str(row.get(
                        "chr_name",
                        "",
                    ))
                ),
                sequence_length=np.array(
                    int(
                        row[
                            "sequence_length"
                        ]
                    )
                ),
                retained_start=np.array(
                    retained_start
                ),
                retained_end=np.array(
                    retained_end
                ),
                retained_length=np.array(
                    expected_length
                ),
                genomic_input_start=np.array(
                    int(
                        row[
                            "genomic_input_start"
                        ]
                    )
                ),
                genomic_input_end=np.array(
                    int(
                        row[
                            "genomic_input_end"
                        ]
                    )
                ),
                pad_left=np.array(
                    int(row["pad_left"])
                ),
                pad_right=np.array(
                    int(row["pad_right"])
                ),
                crop_start=np.array(
                    crop_start
                ),
                crop_end=np.array(
                    crop_end
                ),
                condition=np.array(
                    condition
                ),
                assay_title=np.array(
                    cfg[
                        "alphagenome"
                    ][
                        "assay_title"
                    ]
                ),
                imputation_mode=np.array(
                    cfg[
                        "alphagenome"
                    ][
                        "imputation_mode"
                    ]
                ),
                endedness=np.array(cfg["alphagenome"].get("endedness", "")),
                input_length=np.array(
                    input_length
                ),
                ambiguous_bases_replaced=np.array(
                    ambiguity_count
                ),
                prediction_seconds=np.array(
                    prediction_seconds
                ),
            )

            new_predictions += 1

            prediction_records.append(
                {
                    "tile_index": tile_index,
                    "fasta_name": fasta_name,
                    "retained_start": retained_start,
                    "retained_end": retained_end,
                    "output_file": str(
                        output_path
                    ),
                    "prediction_seconds": (
                        prediction_seconds
                    ),
                    "ambiguous_bases_replaced": (
                        ambiguity_count
                    ),
                }
            )

            progress.set_postfix(
                new=new_predictions,
                reused=reused_predictions,
            )

    finally:
        fasta.close()

    finished = datetime.now(
        timezone.utc
    )

    #
    # Verify every planned tile now has a valid
    # checkpoint before declaring the run complete.
    #
    valid_tiles = 0

    for row_tuple in manifest.itertuples(
        index=False
    ):
        row = row_tuple._asdict()

        tile_index = int(
            row["tile_index"]
        )

        fasta_name = str(
            row["fasta_name"]
        )

        retained_start = int(
            row["retained_start"]
        )

        retained_end = int(
            row["retained_end"]
        )

        filename = (
            f"tile_{tile_index:07d}__"
            f"{safe_name(fasta_name)}__"
            f"{retained_start}-{retained_end}.npz"
        )

        if validate_existing_tile(
            tile_dir / filename,
            row,
            cfg,
        ):
            valid_tiles += 1

    complete = (
        valid_tiles
        == len(manifest)
    )

    run_info = {
        "species_id": args.species,
        "condition": condition,
        "output_type": output_type,
        "assay_title": cfg[
            "alphagenome"
        ]["assay_title"],
        "imputation_mode": cfg[
            "alphagenome"
        ]["imputation_mode"],
        "endedness": cfg["alphagenome"].get("endedness", ""),
        "input_length": input_length,
        "planned_tiles": int(
            len(manifest)
        ),
        "valid_tiles": int(
            valid_tiles
        ),
        "new_predictions": int(
            new_predictions
        ),
        "reused_predictions": int(
            reused_predictions
        ),
        "complete": bool(
            complete
        ),
        "ambiguous_bases_replaced_this_run": int(
            ambiguity_total
        ),
        "started_utc": (
            started.isoformat()
        ),
        "finished_utc": (
            finished.isoformat()
        ),
        "checkpoint_directory": str(
            tile_dir
        ),
    }

    with open(
        metadata_dir
        / "prediction_run.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_info,
            f,
            indent=2,
        )

    if prediction_records:
        pd.DataFrame(
            prediction_records
        ).to_csv(
            metadata_dir
            / "new_predictions_this_run.csv",
            index=False,
        )

    print()
    print("=" * 70)
    print("PREDICTION RUN COMPLETE")
    print("=" * 70)
    print(
        f"Planned tiles:       "
        f"{len(manifest):,}"
    )
    print(
        f"Valid tiles:         "
        f"{valid_tiles:,}"
    )
    print(
        f"New predictions:     "
        f"{new_predictions:,}"
    )
    print(
        f"Existing reused:     "
        f"{reused_predictions:,}"
    )
    print(
        f"Complete:            "
        f"{complete}"
    )
    print()

    if not complete:
        raise RuntimeError(
            "Prediction run ended without all "
            "planned tiles being valid."
        )


if __name__ == "__main__":
    main()