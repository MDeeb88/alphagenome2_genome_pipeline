#!/usr/bin/env python3

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm


PROJECT_ROOT = Path(
    "/media/eternus1/nfs/projects/dpanchenko/"
    "alphagenome2_genome_pipeline"
)

DEFAULT_WORK = PROJECT_ROOT / "work"
DEFAULT_RESULTS = PROJECT_ROOT / "results"

CHUNK_BASES = 524_288


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def tile_filename(row):
    tile_index = int(row["tile_index"])
    fasta_name = str(row["fasta_name"])
    start = int(row["retained_start"])
    end = int(row["retained_end"])

    return (
        f"tile_{tile_index:07d}__"
        f"{safe_name(fasta_name)}__"
        f"{start}-{end}.npz"
    )



def checkpoint_signature(x):
    """Identify the assay and selected channels of a checkpoint."""
    required = {
        "condition", "assay_title", "imputation_mode",
        "track_strands",
    }
    missing = required - set(x.files)
    if missing:
        raise RuntimeError(
            f"Checkpoint metadata missing: {sorted(missing)}"
        )

    def scalar(name, default=""):
        return str(x[name].item()) if name in x.files else default

    return {
        "output_type": scalar("output_type", "RNA_SEQ"),
        "condition": scalar("condition"),
        "assay_title": scalar("assay_title"),
        "imputation_mode": scalar("imputation_mode"),
        "endedness": scalar("endedness"),
        "track_strands": [str(v) for v in x["track_strands"]],
        "track_indices": (
            [int(v) for v in x["track_indices"]]
            if "track_indices" in x.files else []
        ),
    }


def load_and_validate_tile(path, row, expected_signature=None):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction tile: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as x:

        required = {
            "values",
            "track_strands",
            "track_orientation",
            "tile_index",
            "fasta_name",
            "retained_start",
            "retained_end",
            "retained_length",
        }

        missing = required - set(x.files)

        if missing:
            raise RuntimeError(
                f"{path.name}: missing fields {sorted(missing)}"
            )

        values = np.asarray(
            x["values"],
            dtype=np.float32,
        )

        expected_length = int(
            row["retained_length"]
        )

        if (
            values.ndim != 2
            or values.shape[0] != expected_length
            or values.shape[1] < 1
        ):
            raise RuntimeError(
                f"{path.name}: invalid values shape "
                f"{values.shape}; expected "
                f"({expected_length}, channels>=1)"
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise RuntimeError(
                f"{path.name}: contains NaN/Inf values"
            )

        strands = [
            str(v)
            for v in x["track_strands"]
        ]

        output_type = (
            str(x["output_type"].item())
            if "output_type" in x.files
            else "RNA_SEQ"
        )

        if len(strands) != values.shape[1]:
            raise RuntimeError(
                f"{path.name}: number of strand labels "
                "does not match the channel count."
            )

        if output_type == "RNA_SEQ":
            if strands != ["+", "-"]:
                raise RuntimeError(
                    f"{path.name}: invalid RNA strands "
                    f"{strands}"
                )

        elif output_type == "ATAC":
            if any(strand != "." for strand in strands):
                raise RuntimeError(
                    f"{path.name}: ATAC tracks must be "
                    f"unstranded; found {strands}"
                )

            if "track_indices" not in x.files:
                raise RuntimeError(
                    f"{path.name}: missing ATAC channel indices"
                )

            if len(x["track_indices"]) != values.shape[1]:
                raise RuntimeError(
                    f"{path.name}: ATAC channel index "
                    "count mismatch"
                )

        else:
            raise RuntimeError(
                f"{path.name}: unsupported output type "
                f"{output_type}"
            )

        orientation = str(
            x["track_orientation"].item()
        )

        if orientation != "genomic_reference":
            raise RuntimeError(
                f"{path.name}: unexpected orientation "
                f"{orientation}"
            )

        if expected_signature is not None:
            actual_signature = checkpoint_signature(x)
            if actual_signature != expected_signature:
                raise RuntimeError(
                    f"{path.name}: assay/channel metadata differs "
                    "from the first checkpoint."
                )

        checks = {
            "tile_index": (
                int(x["tile_index"].item()),
                int(row["tile_index"]),
            ),
            "fasta_name": (
                str(x["fasta_name"].item()),
                str(row["fasta_name"]),
            ),
            "retained_start": (
                int(x["retained_start"].item()),
                int(row["retained_start"]),
            ),
            "retained_end": (
                int(x["retained_end"].item()),
                int(row["retained_end"]),
            ),
        }

        for name, (actual, expected) in checks.items():
            if actual != expected:
                raise RuntimeError(
                    f"{path.name}: {name}={actual} "
                    f"but manifest says {expected}"
                )

    return values


def validate_manifest_coverage(manifest):
    """
    Verify that retained regions cover every selected sequence
    exactly once, continuously, from 0 to sequence_length.
    """

    summaries = []

    for fasta_name, group in manifest.groupby(
        "fasta_name",
        sort=False,
    ):
        group = group.sort_values(
            "retained_start"
        )

        lengths = group[
            "sequence_length"
        ].unique()

        if len(lengths) != 1:
            raise RuntimeError(
                f"{fasta_name}: inconsistent sequence lengths"
            )

        sequence_length = int(
            lengths[0]
        )

        expected_start = 0
        total_retained = 0

        for row in group.itertuples(
            index=False
        ):
            start = int(
                row.retained_start
            )

            end = int(
                row.retained_end
            )

            retained_length = int(
                row.retained_length
            )

            if start != expected_start:
                raise RuntimeError(
                    f"{fasta_name}: coverage gap/overlap. "
                    f"Expected start {expected_start}, "
                    f"found {start}"
                )

            if end <= start:
                raise RuntimeError(
                    f"{fasta_name}: invalid interval "
                    f"{start}-{end}"
                )

            if end - start != retained_length:
                raise RuntimeError(
                    f"{fasta_name}: retained length mismatch"
                )

            if end > sequence_length:
                raise RuntimeError(
                    f"{fasta_name}: tile exceeds "
                    f"sequence length"
                )

            total_retained += retained_length
            expected_start = end

        if expected_start != sequence_length:
            raise RuntimeError(
                f"{fasta_name}: final covered coordinate "
                f"{expected_start} != sequence length "
                f"{sequence_length}"
            )

        if total_retained != sequence_length:
            raise RuntimeError(
                f"{fasta_name}: total retained bases "
                f"{total_retained} != {sequence_length}"
            )

        summaries.append(
            {
                "fasta_name": fasta_name,
                "sequence_length": sequence_length,
                "tile_count": len(group),
            }
        )

    return summaries


def directory_size(path):
    return sum(
        p.stat().st_size
        for p in path.rglob("*")
        if p.is_file()
    )


def copy_metadata(
    species_work,
    metadata_dir,
):
    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    names = [
        "assembly_metadata.json",
        "sequence_report.jsonl",
        "sequence_selection.csv",
        "tile_manifest.csv",
        "tile_plan_summary.json",
    ]

    for name in names:
        source = species_work / name

        if source.exists():
            shutil.copy2(
                source,
                metadata_dir / name,
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate temporary AlphaGenome 2 tile "
            "predictions into a permanent Zarr store."
        )
    )

    parser.add_argument(
        "--species",
        required=True,
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
        "--force",
        action="store_true",
        help="Replace an existing consolidated store.",
    )

    args = parser.parse_args()

    species_work = (
        args.work_dir
        / args.species
    )

    tile_dir = (
        species_work
        / "prediction_tiles"
    )

    manifest_path = (
        species_work
        / "tile_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}"
        )

    if not tile_dir.exists():
        raise FileNotFoundError(
            f"Missing prediction tiles: {tile_dir}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    species_results = (
        args.results_dir
        / args.species
    )

    predictions_dir = (
        species_results
        / "predictions"
    )

    metadata_dir = (
        species_results
        / "metadata"
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_store = (
        predictions_dir
        / "predictions.zarr"
    )

    temp_store = (
        predictions_dir
        / "predictions.zarr.tmp"
    )

    if final_store.exists():
        if not args.force:
            raise RuntimeError(
                f"Final store already exists: "
                f"{final_store}\n"
                f"Use --force to rebuild it."
            )

        shutil.rmtree(
            final_store
        )

    if temp_store.exists():
        shutil.rmtree(
            temp_store
        )

    print("=" * 70)
    print("AlphaGenome 2 prediction consolidation")
    print("=" * 70)
    print("Species:      ", args.species)
    print("Manifest:     ", manifest_path)
    print("Tiles:        ", len(manifest))
    print("Tile dir:     ", tile_dir)
    print("Final store:  ", final_store)
    print()

    #
    # First validate genomic coverage before creating
    # any permanent data.
    #
    sequence_summaries = (
        validate_manifest_coverage(
            manifest
        )
    )

    selected_bases = sum(
        x["sequence_length"]
        for x in sequence_summaries
    )

    print(
        f"Sequences validated: "
        f"{len(sequence_summaries):,}"
    )

    print(
        f"Selected bases:       "
        f"{selected_bases:,}"
    )

    print()

    #
    # Zarr v3 store.
    #
    if manifest.empty:
        raise RuntimeError("Cannot consolidate an empty manifest.")

    first_row = manifest.iloc[0].to_dict()
    first_path = tile_dir / tile_filename(first_row)

    with np.load(first_path, allow_pickle=False) as x:
        signature = checkpoint_signature(x)

    output_type = signature["output_type"]
    track_strands = signature["track_strands"]
    n_channels = len(track_strands)

    if output_type not in {"RNA_SEQ", "ATAC"}:
        raise RuntimeError(f"Unsupported assay: {output_type}")

    if n_channels < 1:
        raise RuntimeError("No prediction channels found.")

    load_and_validate_tile(first_path, first_row, signature)

    print(f"Assay:         {output_type}")
    print(f"Channels:      {n_channels}")
    print(f"Condition:     {signature['condition']}")

    root = zarr.open_group(
        str(temp_store),
        mode="w",
        zarr_format=3,
    )

    root.attrs.update(
        {
            "species_id": args.species,
            "format": (
                "AlphaGenome2 whole-genome "
                "prediction store"
            ),
            "zarr_format": 3,
            "dtype": "float32",
            "track_strands": track_strands,
            "track_indices": signature["track_indices"],
            "output_type": output_type,
            "condition": signature["condition"],
            "assay_title": signature["assay_title"],
            "imputation_mode": signature["imputation_mode"],
            "track_orientation": (
                "genomic_reference"
            ),
            "selected_sequence_count": int(
                len(sequence_summaries)
            ),
            "selected_bases": int(
                selected_bases
            ),
            "planned_tiles": int(
                len(manifest)
            ),
            "complete": False,
        }
    )

    sequences_group = (
        root.create_group(
            "sequences"
        )
    )

    compressor = (
        zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=3,
            shuffle="bitshuffle",
        )
    )

    sequence_stats = []

    for sequence_summary in tqdm(
        sequence_summaries,
        desc="Consolidating sequences",
        unit="sequence",
    ):
        fasta_name = (
            sequence_summary[
                "fasta_name"
            ]
        )

        sequence_length = int(
            sequence_summary[
                "sequence_length"
            ]
        )

        group_rows = (
            manifest[
                manifest[
                    "fasta_name"
                ]
                == fasta_name
            ]
            .sort_values(
                "retained_start"
            )
        )

        first_row = (
            group_rows.iloc[0]
        )

        seq_group = (
            sequences_group.create_group(
                fasta_name
            )
        )

        seq_group.attrs.update(
            {
                "fasta_name": fasta_name,
                "chr_name": (
                    None
                    if pd.isna(
                        first_row[
                            "chr_name"
                        ]
                    )
                    else str(
                        first_row[
                            "chr_name"
                        ]
                    )
                ),
                "sequence_name": (
                    None
                    if pd.isna(
                        first_row[
                            "sequence_name"
                        ]
                    )
                    else str(
                        first_row[
                            "sequence_name"
                        ]
                    )
                ),
                "role": str(
                    first_row["role"]
                ),
                "sequence_length": (
                    sequence_length
                ),
                "track_strands": track_strands,
            }
        )

        chunk_bases = min(
            CHUNK_BASES,
            sequence_length,
        )

        values_array = (
            seq_group.create_array(
                "values",
                shape=(
                    sequence_length,
                    n_channels,
                ),
                dtype="float32",
                chunks=(
                    chunk_bases,
                    n_channels,
                ),
                compressors=[
                    compressor
                ],
                fill_value=np.nan,
            )
        )

        channel_min = np.full(n_channels, np.inf)
        channel_max = np.full(n_channels, -np.inf)
        channel_sum = np.zeros(n_channels, dtype=np.float64)

        bases_written = 0

        for row in group_rows.to_dict(
            orient="records"
        ):
            path = (
                tile_dir
                / tile_filename(row)
            )

            values = (
                load_and_validate_tile(
                    path,
                    row,
                    signature,
                )
            )

            start = int(
                row["retained_start"]
            )

            end = int(
                row["retained_end"]
            )

            #
            # Write into genomic coordinates.
            #
            values_array[
                start:end,
                :,
            ] = values

            #
            # Exact read-back verification.
            #
            written = np.asarray(
                values_array[
                    start:end,
                    :,
                ]
            )

            if not np.array_equal(
                values,
                written,
            ):
                raise RuntimeError(
                    f"Zarr read-back mismatch "
                    f"for {fasta_name}:"
                    f"{start}-{end}"
                )

            channel_min = np.minimum(
                channel_min, values.min(axis=0)
            )
            channel_max = np.maximum(
                channel_max, values.max(axis=0)
            )
            channel_sum += values.sum(
                axis=0, dtype=np.float64
            )

            bases_written += len(
                values
            )

        if bases_written != sequence_length:
            raise RuntimeError(
                f"{fasta_name}: wrote "
                f"{bases_written} bases, "
                f"expected {sequence_length}"
            )

        #
        # Check the final consolidated chromosome
        # for any unwritten NaN regions.
        #
        verification_step = (
            1_048_576
        )

        for start in range(
            0,
            sequence_length,
            verification_step,
        ):
            end = min(
                start
                + verification_step,
                sequence_length,
            )

            block = np.asarray(
                values_array[
                    start:end,
                    :,
                ]
            )

            if not np.all(
                np.isfinite(block)
            ):
                raise RuntimeError(
                    f"{fasta_name}: "
                    f"non-finite/unwritten data "
                    f"in {start}-{end}"
                )

        if output_type == "RNA_SEQ":
            statistics = {
                "plus_min": float(channel_min[0]),
                "plus_max": float(channel_max[0]),
                "plus_mean": float(channel_sum[0] / sequence_length),
                "minus_min": float(channel_min[1]),
                "minus_max": float(channel_max[1]),
                "minus_mean": float(channel_sum[1] / sequence_length),
            }
        else:
            statistics = {}
            for channel in range(n_channels):
                prefix = f"channel_{channel}"
                statistics[f"{prefix}_min"] = float(
                    channel_min[channel]
                )
                statistics[f"{prefix}_max"] = float(
                    channel_max[channel]
                )
                statistics[f"{prefix}_mean"] = float(
                    channel_sum[channel] / sequence_length
                )

        sequence_stats.append(
            {
                "fasta_name": fasta_name,
                "chr_name": (
                    None
                    if pd.isna(
                        first_row[
                            "chr_name"
                        ]
                    )
                    else str(
                        first_row[
                            "chr_name"
                        ]
                    )
                ),
                "sequence_length": (
                    sequence_length
                ),
                "tile_count": int(
                    len(group_rows)
                ),
                **statistics,
            }
        )

    root.attrs["complete"] = True

    #
    # Release references before moving the store.
    #
    del sequences_group
    del root

    #
    # Atomic finalization on the same filesystem.
    #
    temp_store.rename(
        final_store
    )

    #
    # Preserve all metadata required to reproduce
    # this prediction.
    #
    copy_metadata(
        species_work,
        metadata_dir,
    )

    stats_path = (
        metadata_dir
        / "sequence_prediction_stats.csv"
    )

    pd.DataFrame(
        sequence_stats
    ).to_csv(
        stats_path,
        index=False,
    )

    stored_bytes = (
        directory_size(
            final_store
        )
    )

    uncompressed_bytes = (
        selected_bases
        * n_channels
        * np.dtype(
            np.float32
        ).itemsize
    )

    summary = {
        "species_id": args.species,
        "complete": True,
        "zarr_format": 3,
        "track_strands": track_strands,
        "track_indices": signature["track_indices"],
        "channel_count": n_channels,
        "output_type": output_type,
        "condition": signature["condition"],
        "assay_title": signature["assay_title"],
        "imputation_mode": signature["imputation_mode"],
        "track_orientation": (
            "genomic_reference"
        ),
        "selected_sequence_count": int(
            len(sequence_summaries)
        ),
        "selected_bases": int(
            selected_bases
        ),
        "tile_count": int(
            len(manifest)
        ),
        "dtype": "float32",
        "uncompressed_prediction_bytes": int(
            uncompressed_bytes
        ),
        "stored_zarr_bytes": int(
            stored_bytes
        ),
        "compression_ratio": float(
            uncompressed_bytes
            / stored_bytes
            if stored_bytes
            else 0
        ),
        "prediction_store": str(
            final_store
        ),
    }

    summary_path = (
        metadata_dir
        / "consolidation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("CONSOLIDATION COMPLETE")
    print("=" * 70)

    print(
        f"Sequences:          "
        f"{len(sequence_summaries):,}"
    )

    print(
        f"Bases:              "
        f"{selected_bases:,}"
    )

    print(
        f"Tiles:              "
        f"{len(manifest):,}"
    )

    print(
        f"Uncompressed size:  "
        f"{uncompressed_bytes / 1024**3:.2f} GiB"
    )

    print(
        f"Stored Zarr size:   "
        f"{stored_bytes / 1024**3:.2f} GiB"
    )

    print(
        f"Compression ratio:  "
        f"{summary['compression_ratio']:.2f}x"
    )

    print()
    print(
        "Prediction store:",
        final_store,
    )

    print(
        "Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()