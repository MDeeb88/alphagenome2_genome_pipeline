"""Build validated, unstranded BigWigs from ATAC predictions."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import zarr


WRITE_CHUNK = 250_000


def build_atac_tracks(store_path, manifest_path, tracks_dir, force=False):
    store = zarr.open_group(str(store_path), mode="r")

    if not store.attrs.get("complete", False):
        raise RuntimeError("Prediction store is incomplete.")

    if store.attrs.get("output_type") != "ATAC":
        raise RuntimeError("Prediction store is not ATAC-seq.")

    strands = list(store.attrs.get("track_strands", []))
    source_indices = list(store.attrs.get("track_indices", []))
    channel_count = len(strands)

    if not channel_count or strands != ["."] * channel_count:
        raise RuntimeError(f"Invalid ATAC strands: {strands}")

    if len(source_indices) != channel_count:
        raise RuntimeError("ATAC channel indices are inconsistent.")

    manifest = pd.read_csv(manifest_path)

    columns = [
        "fasta_name",
        "chr_name",
        "sequence_name",
        "sequence_length",
    ]

    sequences = manifest[columns].drop_duplicates("fasta_name")

    if sequences.empty:
        raise RuntimeError("Tile manifest contains no sequences.")

    zarr_sequences = store["sequences"]
    header = []

    for row in sequences.itertuples(index=False):
        name = str(row.fasta_name)
        length = int(row.sequence_length)

        if name not in zarr_sequences:
            raise RuntimeError(f"Missing Zarr sequence: {name}")

        shape = zarr_sequences[name]["values"].shape

        if shape != (length, channel_count):
            raise RuntimeError(
                f"{name}: invalid Zarr shape {shape}"
            )

        header.append((name, length))

    tracks_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        tracks_dir / (
            f"{tracks_dir.parent.name}_HEK293T_ATAC.bw"
            if channel_count == 1
            else f"{tracks_dir.parent.name}_HEK293T_ATAC_channel_{i + 1:02d}.bw"
        )
        for i in range(channel_count)
    ]

    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(
                f"Existing ATAC track: {path}"
            )

    temporary = [
        Path(str(path) + ".partial")
        for path in paths
    ]

    for path in temporary:
        path.unlink(missing_ok=True)

    minimum = np.full(channel_count, np.inf)
    maximum = np.full(channel_count, -np.inf)
    total = np.zeros(channel_count, dtype=np.float64)

    bases_written = 0
    validation_points = 0
    writers = []

    try:
        try:
            for path in temporary:
                writer = pyBigWig.open(str(path), "w")

                if writer is None:
                    raise RuntimeError(
                        f"Cannot open {path} for writing"
                    )

                writers.append(writer)
                writer.addHeader(header)

            for name, length in header:
                array = zarr_sequences[name]["values"]

                for start in range(0, length, WRITE_CHUNK):
                    end = min(start + WRITE_CHUNK, length)

                    values = np.asarray(
                        array[start:end, :],
                        dtype=np.float32,
                    )

                    if values.shape != (
                        end - start,
                        channel_count,
                    ):
                        raise RuntimeError(
                            f"{name}:{start}-{end} shape mismatch"
                        )

                    if not np.isfinite(values).all():
                        raise RuntimeError(
                            f"Non-finite values at {name}:{start}"
                        )

                    for i, writer in enumerate(writers):
                        writer.addEntries(
                            name,
                            start,
                            values=values[:, i].astype(float).tolist(),
                            span=1,
                            step=1,
                        )

                    minimum = np.minimum(
                        minimum, values.min(axis=0)
                    )
                    maximum = np.maximum(
                        maximum, values.max(axis=0)
                    )
                    total += values.sum(
                        axis=0, dtype=np.float64
                    )

                    bases_written += len(values)

        finally:
            for writer in writers:
                writer.close()

        expected_bases = sum(length for _, length in header)

        if bases_written != expected_bases:
            raise RuntimeError("ATAC BigWig base count mismatch.")

        # Reopen and validate each generated BigWig.
        for i, path in enumerate(temporary):
            reader = pyBigWig.open(str(path))

            if reader is None:
                raise RuntimeError(f"Cannot reopen {path}")

            try:
                if reader.chroms() != dict(header):
                    raise RuntimeError(
                        f"BigWig header mismatch: {path}"
                    )

                for name, length in header:
                    array = zarr_sequences[name]["values"]

                    positions = sorted({
                        0,
                        min(100, length - 1),
                        length // 2,
                        max(0, length - 101),
                        length - 1,
                    })

                    for pos in positions:
                        expected = float(
                            array[pos:pos + 1, i][0]
                        )

                        actual = reader.values(
                            name, pos, pos + 1
                        )[0]

                        if not np.isclose(
                            actual,
                            expected,
                            atol=1e-6,
                            rtol=0,
                        ):
                            raise RuntimeError(
                                f"Validation mismatch: "
                                f"{name}:{pos}, channel {i}"
                            )

                        validation_points += 1

            finally:
                reader.close()

        # Write chromosome metadata.
        with (tracks_dir / "chrom.sizes").open(
            "w", encoding="utf-8"
        ) as f:
            for name, length in header:
                f.write(f"{name}\t{length}\n")

        with (tracks_dir / "chrom_aliases.tsv").open(
            "w", encoding="utf-8"
        ) as f:
            f.write(
                "fasta_name\tchr_name\tsequence_name\n"
            )

            for row in sequences.itertuples(index=False):
                chr_name = (
                    ""
                    if pd.isna(row.chr_name)
                    else str(row.chr_name)
                )

                seq_name = (
                    ""
                    if pd.isna(row.sequence_name)
                    else str(row.sequence_name)
                )

                f.write(
                    f"{row.fasta_name}\t"
                    f"{chr_name}\t{seq_name}\n"
                )

        # Publish validated tracks.
        for source, destination in zip(temporary, paths):
            os.replace(source, destination)

        summary = {
            "complete": True,
            "validation_passed": True,
            "output_type": "ATAC",
            "condition": store.attrs.get("condition", ""),
            "track_strands": strands,
            "channel_count": channel_count,
            "selected_bases": bases_written,
            "validation_points": validation_points,
            "tracks": [
                {
                    "path": str(path),
                    "channel": i,
                    "source_track_index": int(source_indices[i]),
                    "min": float(minimum[i]),
                    "max": float(maximum[i]),
                    "mean": float(total[i] / bases_written),
                }
                for i, path in enumerate(paths)
            ],
        }

        summary_path = tracks_dir / "track_build_summary.json"
        temp_summary = Path(str(summary_path) + ".tmp")

        temp_summary.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        os.replace(temp_summary, summary_path)

        print(
            f"Validated {channel_count} ATAC BigWig track(s):"
        )

        for path in paths:
            print(f"  {path}")

    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
