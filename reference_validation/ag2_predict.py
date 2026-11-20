#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import alphagenome
import numpy as np
import pandas as pd
import yaml
from Bio import SeqIO
from tqdm import tqdm

from alphagenome.models.v2 import dna_client


# ============================================================
# FASTA / identifier helpers
# ============================================================

HEADER_RE = re.compile(
    r".*::([^:]+):(\d+)-(\d+)(?:\(([+-])\))?$"
)


def parse_header(header):

    header = str(header).strip().lstrip(">")

    match = HEADER_RE.fullmatch(header)

    if match is None:
        raise ValueError(
            f"Cannot parse FASTA header: {header}"
        )

    chrom = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))
    strand = match.group(4) or ""

    if end <= start:
        raise ValueError(
            f"Invalid coordinates: {header}"
        )

    return chrom, start, end, strand


def safe_prediction_filename(sequence_id):

    safe = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(sequence_id),
    ).strip("_")

    digest = hashlib.sha1(
        str(sequence_id).encode("utf-8")
    ).hexdigest()[:10]

    return f"{safe}__{digest}.npz"


def count_fasta_records(path):

    count = 0

    with open(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1

    return count


# ============================================================
# Center padding
# ============================================================

def center_pad_sequence(
    sequence,
    input_length,
):

    sequence = str(sequence).upper()

    invalid = (
        set(sequence)
        - set("ACGTN")
    )

    if invalid:
        raise ValueError(
            "Sequence contains unsupported bases: "
            f"{sorted(invalid)}"
        )

    sequence_length = len(sequence)

    if sequence_length > input_length:
        raise ValueError(
            f"Sequence length {sequence_length:,} "
            f"exceeds model input "
            f"{input_length:,}"
        )

    total_padding = (
        input_length
        - sequence_length
    )

    padding_left = (
        total_padding // 2
    )

    padding_right = (
        total_padding
        - padding_left
    )

    padded = (
        ("N" * padding_left)
        + sequence
        + ("N" * padding_right)
    )

    if len(padded) != input_length:
        raise RuntimeError(
            "Internal padding error."
        )

    return (
        padded,
        padding_left,
        padding_right,
    )


# ============================================================
# AG2 target-track selection
# ============================================================

def select_target_rna_tracks(
    rna,
    condition,
    assay_title,
    imputation_mode,
    endedness,
):

    metadata = rna.metadata.copy()

    required_columns = [
        "strand",
        "biosample_curie",
        "Assay title",
        "imputation_mode",
    ]

    for column in required_columns:
        if column not in metadata.columns:
            raise RuntimeError(
                "AG2 RNA metadata missing "
                f"column: {column}"
            )

    mask = (
        metadata["biosample_curie"]
        .astype(str)
        .eq(str(condition))
        &
        metadata["Assay title"]
        .astype(str)
        .str.casefold()
        .eq(str(assay_title).casefold())
        &
        metadata["imputation_mode"]
        .astype(str)
        .str.casefold()
        .eq(str(imputation_mode).casefold())
    )

    if (
        endedness is not None
        and "endedness" in metadata.columns
    ):
        mask &= (
            metadata["endedness"]
            .astype(str)
            .str.casefold()
            .eq(str(endedness).casefold())
        )

    candidate_positions = np.flatnonzero(
        mask.to_numpy()
    )

    strand_to_position = {}

    for position in candidate_positions:

        strand = str(
            metadata.iloc[position]["strand"]
        )

        if strand not in {"+", "-"}:
            continue

        if strand in strand_to_position:
            raise RuntimeError(
                "More than one matching track "
                f"for strand {strand}"
            )

        strand_to_position[strand] = (
            int(position)
        )

    if set(strand_to_position) != {
        "+",
        "-",
    }:
        raise RuntimeError(
            "Expected exactly one '+' and "
            "one '-' observed total RNA-seq "
            "track. Found: "
            f"{sorted(strand_to_position)}"
        )

    positions = [
        strand_to_position["+"],
        strand_to_position["-"],
    ]

    selected_metadata = (
        metadata
        .iloc[positions]
        .copy()
        .reset_index(drop=True)
    )

    return (
        positions,
        selected_metadata,
    )


# ============================================================
# Raw prediction storage
# ============================================================

def save_prediction_atomic(
    output_path,
    values,
    job,
    condition,
    assay_title,
    input_length,
    elapsed,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = output_path.with_name(
        output_path.name + ".tmp.npz"
    )

    np.savez_compressed(
        temp_path,

        values=np.asarray(
            values,
            dtype=np.float32,
        ),

        track_strands=np.asarray(
            ["+", "-"]
        ),

        track_orientation=np.asarray(
            "input_sequence"
        ),

        sequence_id=np.asarray(
            job["sequence_id"]
        ),

        sequence_index=np.asarray(
            job["sequence_index"],
            dtype=np.int64,
        ),

        chrom=np.asarray(
            job["chrom"]
        ),

        start=np.asarray(
            job["start"],
            dtype=np.int64,
        ),

        end=np.asarray(
            job["end"],
            dtype=np.int64,
        ),

        fasta_strand=np.asarray(
            job["fasta_strand"]
        ),

        condition=np.asarray(
            condition
        ),

        assay_title=np.asarray(
            assay_title
        ),

        input_length=np.asarray(
            input_length,
            dtype=np.int64,
        ),

        padding_left=np.asarray(
            job["padding_left"],
            dtype=np.int64,
        ),

        padding_right=np.asarray(
            job["padding_right"],
            dtype=np.int64,
        ),

        prediction_seconds=np.asarray(
            elapsed,
            dtype=np.float64,
        ),
    )

    os.replace(
        temp_path,
        output_path,
    )


def raw_prediction_valid(path):

    try:

        with np.load(
            path,
            allow_pickle=False,
        ) as raw:

            required = {
                "values",
                "track_strands",
                "sequence_id",
                "chrom",
                "start",
                "end",
                "fasta_strand",
            }

            if not required.issubset(
                raw.files
            ):
                return False

            values = raw["values"]

            if (
                values.ndim != 2
                or values.shape[1] != 2
                or values.shape[0] == 0
            ):
                return False

            if not np.isfinite(
                values
            ).all():
                return False

        return True

    except Exception:
        return False


def manifest_row_from_raw(path):

    with np.load(
        path,
        allow_pickle=False,
    ) as raw:

        values = raw["values"]

        return {
            "sequence_index":
                int(raw["sequence_index"]),

            "sequence_id":
                str(raw["sequence_id"]),

            "chrom":
                str(raw["chrom"]),

            "start":
                int(raw["start"]),

            "end":
                int(raw["end"]),

            "fasta_strand":
                str(raw["fasta_strand"]),

            "sequence_length":
                int(values.shape[0]),

            "condition":
                str(raw["condition"]),

            "assay_title":
                str(raw["assay_title"]),

            "track_0":
                "+",

            "track_1":
                "-",

            "track_orientation":
                "input_sequence",

            "mean_plus":
                float(
                    np.mean(
                        values[:, 0]
                    )
                ),

            "mean_minus":
                float(
                    np.mean(
                        values[:, 1]
                    )
                ),

            "max_plus":
                float(
                    np.max(
                        values[:, 0]
                    )
                ),

            "max_minus":
                float(
                    np.max(
                        values[:, 1]
                    )
                ),

            "prediction_seconds":
                float(
                    raw[
                        "prediction_seconds"
                    ]
                ),

            "raw_prediction_file":
                str(path),

            "status":
                "complete",
        }


# ============================================================
# Manifest / state
# ============================================================

def load_manifest(path):

    if not path.exists():
        return {}

    df = pd.read_csv(path)

    if len(df) == 0:
        return {}

    return {
        str(row["sequence_id"]):
            row.to_dict()

        for _, row in df.iterrows()
    }


def write_manifest(
    path,
    manifest_rows,
):

    if not manifest_rows:
        return

    df = pd.DataFrame(
        manifest_rows.values()
    )

    if "sequence_index" in df.columns:
        df = df.sort_values(
            "sequence_index"
        )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        temp,
        index=False,
    )

    os.replace(
        temp,
        path,
    )


def write_failures(
    path,
    failures,
):

    df = pd.DataFrame(
        failures,
        columns=[
            "sequence_index",
            "sequence_id",
            "error",
        ],
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        temp,
        index=False,
    )

    os.replace(
        temp,
        path,
    )


def write_state(
    path,
    dataset_name,
    processed,
    skipped,
    failed,
    target_records,
    batch_size,
    max_workers,
):

    payload = {
        "dataset_name":
            dataset_name,

        "updated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "processed_this_run":
            processed,

        "skipped_existing":
            skipped,

        "failed_this_run":
            failed,

        "target_records":
            target_records,

        "batch_size":
            batch_size,

        "max_workers":
            max_workers,
    }

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temp,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
        )

    os.replace(
        temp,
        path,
    )


# ============================================================
# AG2 API calls
# ============================================================

def predict_batch_with_retries(
    model,
    sequences,
    condition,
    max_workers,
    max_retries,
    retry_seconds,
):

    conditions = [
        condition
        for _ in sequences
    ]

    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            return model.predict_sequences(
                sequences,

                conditions=conditions,

                requested_outputs=[
                    dna_client.OutputType.RNA_SEQ
                ],

                progress_bar=False,

                max_workers=max_workers,
            )

        except Exception as exc:

            last_error = exc

            if attempt >= max_retries:
                break

            delay = (
                retry_seconds
                * attempt
            )

            time.sleep(delay)

    raise RuntimeError(
        "Batch prediction failed after "
        f"{max_retries} attempts: "
        f"{last_error}"
    ) from last_error


def predict_one_with_retries(
    model,
    sequence,
    condition,
    max_retries,
    retry_seconds,
):

    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            return model.predict_sequence(
                sequence=sequence,

                requested_outputs=[
                    dna_client.OutputType.RNA_SEQ
                ],

                condition=condition,
            )

        except Exception as exc:

            last_error = exc

            if attempt >= max_retries:
                break

            delay = (
                retry_seconds
                * attempt
            )

            time.sleep(delay)

    raise RuntimeError(
        "Individual prediction failed after "
        f"{max_retries} attempts: "
        f"{last_error}"
    ) from last_error


# ============================================================
# Finalize one AG2 output
# ============================================================

def finalize_prediction(
    job,
    output,
    elapsed,
    raw_dir,
    metadata_path,
    manifest_rows,
    condition,
    assay_title,
    imputation_mode,
    endedness,
    input_length,
):

    rna = output.rna_seq

    if rna is None:
        raise RuntimeError(
            "AG2 returned no RNA_SEQ output."
        )

    (
        track_positions,
        selected_metadata,
    ) = select_target_rna_tracks(
        rna=rna,
        condition=condition,
        assay_title=assay_title,
        imputation_mode=imputation_mode,
        endedness=endedness,
    )

    crop_start = (
        job["padding_left"]
    )

    crop_end = (
        crop_start
        + job["sequence_length"]
    )

    values = np.asarray(
        rna.values[
            crop_start:
            crop_end,
            :
        ][
            :,
            track_positions
        ],
        dtype=np.float32,
    )

    expected_shape = (
        job["sequence_length"],
        2,
    )

    if values.shape != expected_shape:
        raise RuntimeError(
            "Unexpected prediction shape "
            f"{values.shape}; expected "
            f"{expected_shape}"
        )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "Prediction contains "
            "NaN or infinite values."
        )

    if not metadata_path.exists():

        selected_metadata.to_csv(
            metadata_path,
            index=False,
        )

    raw_path = (
        raw_dir
        / safe_prediction_filename(
            job["sequence_id"]
        )
    )

    save_prediction_atomic(
        output_path=raw_path,
        values=values,
        job=job,
        condition=condition,
        assay_title=assay_title,
        input_length=input_length,
        elapsed=elapsed,
    )

    manifest_rows[
        job["sequence_id"]
    ] = manifest_row_from_raw(
        raw_path
    )


# ============================================================
# Process one batch
# ============================================================

def process_batch(
    jobs,
    model,
    raw_dir,
    metadata_path,
    manifest_rows,
    failures,
    condition,
    assay_title,
    imputation_mode,
    endedness,
    input_length,
    max_workers,
    max_retries,
    retry_seconds,
):

    if not jobs:
        return 0

    sequences = [
        job["padded_sequence"]
        for job in jobs
    ]

    # --------------------------------------------------------
    # First try official parallel AG2 batch API
    # --------------------------------------------------------

    try:

        t0 = time.perf_counter()

        outputs = predict_batch_with_retries(
            model=model,
            sequences=sequences,
            condition=condition,
            max_workers=max_workers,
            max_retries=max_retries,
            retry_seconds=retry_seconds,
        )

        batch_elapsed = (
            time.perf_counter()
            - t0
        )

        if len(outputs) != len(jobs):
            raise RuntimeError(
                "AG2 batch returned "
                f"{len(outputs)} outputs for "
                f"{len(jobs)} inputs."
            )

        # Batch API does not expose exact runtime
        # for each sequence, so store the mean
        # wall-clock time per item.
        elapsed_per_item = (
            batch_elapsed
            / len(jobs)
        )

        completed = 0

        for job, output in zip(
            jobs,
            outputs,
        ):

            try:

                finalize_prediction(
                    job=job,
                    output=output,
                    elapsed=elapsed_per_item,
                    raw_dir=raw_dir,
                    metadata_path=metadata_path,
                    manifest_rows=manifest_rows,
                    condition=condition,
                    assay_title=assay_title,
                    imputation_mode=(
                        imputation_mode
                    ),
                    endedness=endedness,
                    input_length=input_length,
                )

                completed += 1

            except Exception as exc:

                failures.append(
                    (
                        job[
                            "sequence_index"
                        ],
                        job[
                            "sequence_id"
                        ],
                        str(exc),
                    )
                )

        return completed

    except Exception:

        # ----------------------------------------------------
        # If the whole batch fails, fall back to
        # individual requests so one problematic request
        # cannot lose the complete batch.
        # ----------------------------------------------------

        completed = 0

        for job in jobs:

            try:

                t0 = time.perf_counter()

                output = (
                    predict_one_with_retries(
                        model=model,
                        sequence=(
                            job[
                                "padded_sequence"
                            ]
                        ),
                        condition=condition,
                        max_retries=(
                            max_retries
                        ),
                        retry_seconds=(
                            retry_seconds
                        ),
                    )
                )

                elapsed = (
                    time.perf_counter()
                    - t0
                )

                finalize_prediction(
                    job=job,
                    output=output,
                    elapsed=elapsed,
                    raw_dir=raw_dir,
                    metadata_path=metadata_path,
                    manifest_rows=manifest_rows,
                    condition=condition,
                    assay_title=assay_title,
                    imputation_mode=(
                        imputation_mode
                    ),
                    endedness=endedness,
                    input_length=input_length,
                )

                completed += 1

            except Exception as exc:

                failures.append(
                    (
                        job[
                            "sequence_index"
                        ],
                        job[
                            "sequence_id"
                        ],
                        str(exc),
                    )
                )

        return completed


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Parallel AlphaGenome 2 "
            "FASTA prediction worker"
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help=(
            "Optional testing limit. "
            "Omit for full dataset."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute even when a valid "
            "NPZ already exists."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    config_path = Path(
        args.config
    ).resolve()

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as handle:

        cfg = yaml.safe_load(
            handle
        )

    dataset_cfg = cfg["dataset"]
    prediction_cfg = cfg["prediction"]
    output_cfg = cfg["output"]

    dataset_name = str(
        dataset_cfg["name"]
    )

    fasta_path = Path(
        dataset_cfg["fasta_path"]
    )

    expected_sequence_length = (
        dataset_cfg.get(
            "expected_sequence_length"
        )
    )

    if expected_sequence_length is not None:
        expected_sequence_length = int(
            expected_sequence_length
        )

    condition = str(
        prediction_cfg["condition"]
    )

    assay_title = str(
        prediction_cfg.get(
            "assay_title",
            "total RNA-seq",
        )
    )

    imputation_mode = str(
        prediction_cfg.get(
            "imputation_mode",
            "observed",
        )
    )

    endedness = (
        prediction_cfg.get(
            "endedness"
        )
    )

    input_length = int(
        prediction_cfg.get(
            "input_length",
            dna_client.SEQUENCE_LENGTH_1MB,
        )
    )

    padding_mode = str(
        prediction_cfg.get(
            "padding",
            "center",
        )
    )

    batch_size = int(
        prediction_cfg.get(
            "batch_size",
            5,
        )
    )

    max_workers = int(
        prediction_cfg.get(
            "max_workers",
            5,
        )
    )

    checkpoint_every = int(
        prediction_cfg.get(
            "checkpoint_every",
            25,
        )
    )

    max_retries = int(
        prediction_cfg.get(
            "max_retries",
            5,
        )
    )

    retry_seconds = float(
        prediction_cfg.get(
            "retry_seconds",
            5,
        )
    )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be >= 1"
        )

    if max_workers < 1:
        raise ValueError(
            "max_workers must be >= 1"
        )

    if padding_mode != "center":
        raise ValueError(
            "This pipeline requires "
            "padding: center"
        )

    dna_client.validate_sequence_length(
        input_length
    )

    if not fasta_path.exists():
        raise FileNotFoundError(
            fasta_path
        )

    # --------------------------------------------------------
    # API key
    # --------------------------------------------------------

    api_key = os.environ.get(
        "ALPHA_GENOME_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "ALPHA_GENOME_API_KEY "
            "is not set."
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    outdir = Path(
        output_cfg["dir"]
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_dir = (
        outdir
        / "raw_predictions"
        / "all"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        outdir
        / "manifest_all.csv"
    )

    failures_path = (
        outdir
        / "failures_all.csv"
    )

    state_path = (
        outdir
        / "state_all.json"
    )

    metadata_path = (
        outdir
        / "selected_track_metadata.csv"
    )

    run_info_path = (
        outdir
        / "run_info.json"
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    total_records = (
        count_fasta_records(
            fasta_path
        )
    )

    target_records = (
        total_records
        if args.max_sequences is None
        else min(
            total_records,
            args.max_sequences,
        )
    )

    # --------------------------------------------------------
    # Run info
    # --------------------------------------------------------

    run_info = {
        "dataset_name":
            dataset_name,

        "fasta_path":
            str(fasta_path),

        "total_fasta_records":
            total_records,

        "target_records":
            target_records,

        "condition":
            condition,

        "assay_title":
            assay_title,

        "imputation_mode":
            imputation_mode,

        "endedness":
            endedness,

        "input_length":
            input_length,

        "padding":
            padding_mode,

        "batch_size":
            batch_size,

        "max_workers":
            max_workers,

        "alphagenome_version":
            getattr(
                alphagenome,
                "__version__",
                "unknown",
            ),

        "track_orientation":
            "input_sequence",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    with open(
        run_info_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            run_info,
            handle,
            indent=2,
        )

    # --------------------------------------------------------
    # Resume state
    # --------------------------------------------------------

    manifest_rows = (
        load_manifest(
            manifest_path
        )
    )

    failures = []

    processed_this_run = 0
    skipped_existing = 0
    progress_since_checkpoint = 0

    # --------------------------------------------------------
    # Client
    # --------------------------------------------------------

    print("=" * 72)
    print("ALPHAGENOME 2 PARALLEL PREDICTION")
    print("=" * 72)

    print(
        "Dataset:",
        dataset_name,
    )

    print(
        "FASTA:",
        fasta_path,
    )

    print(
        "Total FASTA records:",
        total_records,
    )

    print(
        "Records targeted:",
        target_records,
    )

    print(
        "Condition:",
        condition,
    )

    print(
        "Assay:",
        assay_title,
    )

    print(
        "Input length:",
        f"{input_length:,}",
    )

    print(
        "Padding:",
        padding_mode,
    )

    print(
        "Batch size:",
        batch_size,
    )

    print(
        "Parallel workers:",
        max_workers,
    )

    print()
    print(
        "Creating AlphaGenome 2 client..."
    )

    model = dna_client.create(
        api_key
    )

    print("Client ready.")
    print()

    # --------------------------------------------------------
    # Checkpoint helper
    # --------------------------------------------------------

    def checkpoint():

        write_manifest(
            manifest_path,
            manifest_rows,
        )

        write_failures(
            failures_path,
            failures,
        )

        write_state(
            state_path,
            dataset_name,
            processed_this_run,
            skipped_existing,
            len(failures),
            target_records,
            batch_size,
            max_workers,
        )

    # --------------------------------------------------------
    # Batch helper
    # --------------------------------------------------------

    batch = []

    pbar = tqdm(
        total=target_records,
        desc=f"Predicting {dataset_name}",
        unit="seq",
        dynamic_ncols=True,
    )

    def flush_batch():

        nonlocal batch
        nonlocal processed_this_run
        nonlocal progress_since_checkpoint

        if not batch:
            return

        n_jobs = len(batch)

        completed = process_batch(
            jobs=batch,
            model=model,
            raw_dir=raw_dir,
            metadata_path=metadata_path,
            manifest_rows=manifest_rows,
            failures=failures,
            condition=condition,
            assay_title=assay_title,
            imputation_mode=(
                imputation_mode
            ),
            endedness=endedness,
            input_length=input_length,
            max_workers=max_workers,
            max_retries=max_retries,
            retry_seconds=retry_seconds,
        )

        processed_this_run += (
            completed
        )

        pbar.update(
            n_jobs
        )

        progress_since_checkpoint += (
            n_jobs
        )

        batch = []

    # --------------------------------------------------------
    # Stream FASTA
    # --------------------------------------------------------

    try:

        for sequence_index, record in enumerate(
            SeqIO.parse(
                str(fasta_path),
                "fasta",
            )
        ):

            if sequence_index >= target_records:
                break

            sequence_id = str(
                record.id
            )

            pbar.set_postfix_str(
                sequence_id[:45],
                refresh=False,
            )

            raw_path = (
                raw_dir
                / safe_prediction_filename(
                    sequence_id
                )
            )

            # -----------------------------------------------
            # Resume
            # -----------------------------------------------

            if (
                raw_path.exists()
                and not args.force
                and raw_prediction_valid(
                    raw_path
                )
            ):

                manifest_rows[
                    sequence_id
                ] = (
                    manifest_row_from_raw(
                        raw_path
                    )
                )

                skipped_existing += 1

                pbar.update(1)

                progress_since_checkpoint += 1

                if (
                    progress_since_checkpoint
                    >= checkpoint_every
                ):
                    checkpoint()
                    progress_since_checkpoint = 0

                continue

            # -----------------------------------------------
            # Validate / prepare record
            # -----------------------------------------------

            try:

                (
                    chrom,
                    start,
                    end,
                    fasta_strand,
                ) = parse_header(
                    record.id
                )

                sequence = str(
                    record.seq
                ).upper()

                sequence_length = len(
                    sequence
                )

                if (
                    sequence_length
                    != end - start
                ):
                    raise ValueError(
                        "Sequence length does not "
                        "match genomic interval."
                    )

                if (
                    expected_sequence_length
                    is not None
                    and sequence_length
                    != expected_sequence_length
                ):
                    raise ValueError(
                        f"Expected "
                        f"{expected_sequence_length} bp, "
                        f"found "
                        f"{sequence_length} bp."
                    )

                (
                    padded_sequence,
                    padding_left,
                    padding_right,
                ) = center_pad_sequence(
                    sequence,
                    input_length,
                )

                batch.append(
                    {
                        "sequence_index":
                            sequence_index,

                        "sequence_id":
                            sequence_id,

                        "chrom":
                            chrom,

                        "start":
                            start,

                        "end":
                            end,

                        "fasta_strand":
                            fasta_strand,

                        "sequence_length":
                            sequence_length,

                        "padding_left":
                            padding_left,

                        "padding_right":
                            padding_right,

                        "padded_sequence":
                            padded_sequence,
                    }
                )

            except Exception as exc:

                failures.append(
                    (
                        sequence_index,
                        sequence_id,
                        str(exc),
                    )
                )

                pbar.update(1)

                progress_since_checkpoint += 1

                continue

            # -----------------------------------------------
            # Execute complete batch
            # -----------------------------------------------

            if len(batch) >= batch_size:

                flush_batch()

                if (
                    progress_since_checkpoint
                    >= checkpoint_every
                ):
                    checkpoint()
                    progress_since_checkpoint = 0

        # ----------------------------------------------------
        # Final partial batch
        # ----------------------------------------------------

        flush_batch()

    finally:

        pbar.close()

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    checkpoint()

    print()
    print("=" * 72)
    print("PREDICTION COMPLETE")
    print("=" * 72)

    print(
        "New predictions:",
        processed_this_run,
    )

    print(
        "Existing predictions reused:",
        skipped_existing,
    )

    print(
        "Failures:",
        len(failures),
    )

    print()
    print(
        "Raw predictions:"
    )
    print(raw_dir)

    print()
    print("Manifest:")
    print(manifest_path)

    print()
    print("Failures:")
    print(failures_path)

    print()
    print("State:")
    print(state_path)


if __name__ == "__main__":
    main()
