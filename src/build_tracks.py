#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import zarr
from tqdm import tqdm


PROJECT_ROOT = Path(
    "/media/eternus1/nfs/projects/dpanchenko/"
    "alphagenome2_genome_pipeline"
)

DEFAULT_RESULTS = PROJECT_ROOT / "results"

WRITE_CHUNK = 250_000


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build strand-specific BigWigs from a consolidated "
            "AlphaGenome 2 Zarr prediction store."
        )
    )

    parser.add_argument(
        "--species",
        required=True,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    species_dir = (
        args.results_dir
        / args.species
    )

    store_path = (
        species_dir
        / "predictions"
        / "predictions.zarr"
    )

    metadata_dir = (
        species_dir
        / "metadata"
    )

    manifest_path = (
        metadata_dir
        / "tile_manifest.csv"
    )

    tracks_dir = (
        species_dir
        / "tracks"
    )

    tracks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plus_path = (
        tracks_dir
        / "totalRNA_plus.bw"
    )

    minus_path = (
        tracks_dir
        / "totalRNA_minus.bw"
    )

    chrom_sizes_path = (
        tracks_dir
        / "chrom.sizes"
    )

    aliases_path = (
        tracks_dir
        / "chrom_aliases.tsv"
    )

    summary_path = (
        tracks_dir
        / "track_build_summary.json"
    )

    if not store_path.exists():
        raise FileNotFoundError(
            f"Missing Zarr store: {store_path}"
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}"
        )

    assay_store = zarr.open_group(
        str(store_path),
        mode="r",
    )

    output_type = str(
        assay_store.attrs.get("output_type", "RNA_SEQ")
    ).upper()

    if output_type == "ATAC":
        from build_tracks_atac import build_atac_tracks

        build_atac_tracks(
            store_path=store_path,
            manifest_path=manifest_path,
            tracks_dir=tracks_dir,
            force=args.force,
        )
        return

    if output_type != "RNA_SEQ":
        raise ValueError(
            f"Unsupported output type: {output_type}"
        )

    for path in [
        plus_path,
        minus_path,
    ]:
        if path.exists():
            if args.force:
                path.unlink()
            else:
                raise RuntimeError(
                    f"Track already exists: {path}\n"
                    f"Use --force to rebuild."
                )

    root = zarr.open_group(
        str(store_path),
        mode="r",
    )

    if not root.attrs.get(
        "complete",
        False,
    ):
        raise RuntimeError(
            "Prediction store is not marked complete."
        )

    sequences = root["sequences"]

    manifest = pd.read_csv(
        manifest_path
    )

    #
    # Preserve genomic sequence order from the tile manifest.
    #
    sequence_order = (
        manifest[
            [
                "fasta_name",
                "chr_name",
                "sequence_name",
                "sequence_length",
            ]
        ]
        .drop_duplicates(
            subset=["fasta_name"]
        )
        .reset_index(drop=True)
    )

    header = []

    for row in sequence_order.itertuples(
        index=False
    ):
        fasta_name = str(
            row.fasta_name
        )

        sequence_length = int(
            row.sequence_length
        )

        if fasta_name not in sequences:
            raise RuntimeError(
                f"{fasta_name} missing from Zarr store"
            )

        arr = sequences[
            fasta_name
        ]["values"]

        if arr.shape != (
            sequence_length,
            2,
        ):
            raise RuntimeError(
                f"{fasta_name}: Zarr shape "
                f"{arr.shape} != "
                f"({sequence_length}, 2)"
            )

        header.append(
            (
                fasta_name,
                sequence_length,
            )
        )

    print("=" * 70)
    print("AlphaGenome 2 BigWig builder")
    print("=" * 70)
    print("Species:       ", args.species)
    print("Sequences:     ", len(header))
    print("Prediction DB: ", store_path)
    print("Plus track:    ", plus_path)
    print("Minus track:   ", minus_path)
    print()

    #
    # Write chromosome-size file.
    #
    with open(
        chrom_sizes_path,
        "w",
        encoding="utf-8",
    ) as f:
        for fasta_name, length in header:
            f.write(
                f"{fasta_name}\t{length}\n"
            )

    #
    # Write accession ↔ biological chromosome alias file.
    #
    with open(
        aliases_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "fasta_name\tchr_name\tsequence_name\n"
        )

        for row in sequence_order.itertuples(
            index=False
        ):
            chr_name = (
                ""
                if pd.isna(row.chr_name)
                else str(row.chr_name)
            )

            sequence_name = (
                ""
                if pd.isna(
                    row.sequence_name
                )
                else str(
                    row.sequence_name
                )
            )

            f.write(
                f"{row.fasta_name}\t"
                f"{chr_name}\t"
                f"{sequence_name}\n"
            )

    plus_bw = pyBigWig.open(
        str(plus_path),
        "w",
    )

    minus_bw = pyBigWig.open(
        str(minus_path),
        "w",
    )

    plus_bw.addHeader(
        header
    )

    minus_bw.addHeader(
        header
    )

    total_bases = 0

    plus_min = np.inf
    plus_max = -np.inf
    plus_sum = 0.0

    minus_min = np.inf
    minus_max = -np.inf
    minus_sum = 0.0

    try:
        for row in tqdm(
            sequence_order.itertuples(
                index=False
            ),
            total=len(sequence_order),
            desc="Writing BigWigs",
            unit="sequence",
        ):
            fasta_name = str(
                row.fasta_name
            )

            sequence_length = int(
                row.sequence_length
            )

            arr = sequences[
                fasta_name
            ]["values"]

            for start in range(
                0,
                sequence_length,
                WRITE_CHUNK,
            ):
                end = min(
                    start + WRITE_CHUNK,
                    sequence_length,
                )

                block = np.asarray(
                    arr[start:end, :],
                    dtype=np.float32,
                )

                if not np.all(
                    np.isfinite(block)
                ):
                    raise RuntimeError(
                        f"Non-finite values in "
                        f"{fasta_name}:{start}-{end}"
                    )

                plus = block[:, 0]
                minus = block[:, 1]

                #
                # Fixed-step BigWig entries:
                # one prediction value per genomic base.
                #
                plus_bw.addEntries(
                    fasta_name,
                    start,
                    values=plus.astype(
                        float
                    ).tolist(),
                    span=1,
                    step=1,
                )

                minus_bw.addEntries(
                    fasta_name,
                    start,
                    values=minus.astype(
                        float
                    ).tolist(),
                    span=1,
                    step=1,
                )

                total_bases += len(
                    block
                )

                plus_min = min(
                    plus_min,
                    float(plus.min()),
                )

                plus_max = max(
                    plus_max,
                    float(plus.max()),
                )

                plus_sum += float(
                    plus.sum(
                        dtype=np.float64
                    )
                )

                minus_min = min(
                    minus_min,
                    float(minus.min()),
                )

                minus_max = max(
                    minus_max,
                    float(minus.max()),
                )

                minus_sum += float(
                    minus.sum(
                        dtype=np.float64
                    )
                )

    finally:
        plus_bw.close()
        minus_bw.close()

    expected_bases = sum(
        length
        for _, length in header
    )

    if total_bases != expected_bases:
        raise RuntimeError(
            f"Written bases {total_bases} != "
            f"expected {expected_bases}"
        )

    #
    # Reopen BigWigs and validate against Zarr.
    #
    plus_bw = pyBigWig.open(
        str(plus_path)
    )

    minus_bw = pyBigWig.open(
        str(minus_path)
    )

    try:
        if plus_bw.chroms() != dict(
            header
        ):
            raise RuntimeError(
                "Plus BigWig chromosome header mismatch"
            )

        if minus_bw.chroms() != dict(
            header
        ):
            raise RuntimeError(
                "Minus BigWig chromosome header mismatch"
            )

        validation_records = []

        for row in sequence_order.itertuples(
            index=False
        ):
            fasta_name = str(
                row.fasta_name
            )

            sequence_length = int(
                row.sequence_length
            )

            arr = sequences[
                fasta_name
            ]["values"]

            positions = sorted(
                set(
                    [
                        0,
                        min(
                            100,
                            sequence_length - 1,
                        ),
                        sequence_length // 2,
                        max(
                            0,
                            sequence_length - 101,
                        ),
                        sequence_length - 1,
                    ]
                )
            )

            for pos in positions:
                zarr_values = np.asarray(
                    arr[
                        pos:pos + 1,
                        :,
                    ]
                )[0]

                bw_plus = plus_bw.values(
                    fasta_name,
                    pos,
                    pos + 1,
                )[0]

                bw_minus = minus_bw.values(
                    fasta_name,
                    pos,
                    pos + 1,
                )[0]

                plus_ok = np.isclose(
                    bw_plus,
                    zarr_values[0],
                    rtol=0,
                    atol=1e-7,
                )

                minus_ok = np.isclose(
                    bw_minus,
                    zarr_values[1],
                    rtol=0,
                    atol=1e-7,
                )

                if not (
                    plus_ok
                    and minus_ok
                ):
                    raise RuntimeError(
                        f"BigWig validation failed at "
                        f"{fasta_name}:{pos}"
                    )

                validation_records.append(
                    {
                        "fasta_name": fasta_name,
                        "position": pos,
                        "plus_ok": bool(
                            plus_ok
                        ),
                        "minus_ok": bool(
                            minus_ok
                        ),
                    }
                )

        plus_header = (
            plus_bw.header()
        )

        minus_header = (
            minus_bw.header()
        )

    finally:
        plus_bw.close()
        minus_bw.close()

    summary = {
        "species_id": args.species,
        "complete": True,
        "sequence_count": len(
            header
        ),
        "bases_written": int(
            total_bases
        ),
        "track_strands": [
            "+",
            "-",
        ],
        "chromosome_names": (
            "NCBI FASTA accessions"
        ),
        "plus": {
            "path": str(
                plus_path
            ),
            "min": float(
                plus_min
            ),
            "max": float(
                plus_max
            ),
            "mean": float(
                plus_sum
                / total_bases
            ),
            "header": plus_header,
        },
        "minus": {
            "path": str(
                minus_path
            ),
            "min": float(
                minus_min
            ),
            "max": float(
                minus_max
            ),
            "mean": float(
                minus_sum
                / total_bases
            ),
            "header": minus_header,
        },
        "validation_points": len(
            validation_records
        ),
        "validation_passed": True,
    }

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
    print("BIGWIG BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Sequences:       "
        f"{len(header)}"
    )

    print(
        f"Bases written:   "
        f"{total_bases:,}"
    )

    print(
        f"Validation pts:  "
        f"{len(validation_records)}"
    )

    print(
        "Validation:      PASSED"
    )

    print()
    print(
        "Plus BigWig: ",
        plus_path,
    )

    print(
        "Minus BigWig:",
        minus_path,
    )

    print(
        "Aliases:     ",
        aliases_path,
    )

    print(
        "Summary:     ",
        summary_path,
    )


if __name__ == "__main__":
    main()