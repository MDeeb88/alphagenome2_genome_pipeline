"""Synthetic integration test for the AlphaGenome 2 track pipeline."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_species import final_outputs_complete, remove_validated_atac_zarr


def run_command(*args):
    subprocess.run(
        [sys.executable, *map(str, args)],
        check=True,
    )


def run_case(temp_root, output_type, channel_count):
    case = temp_root / f"{output_type.lower()}_{channel_count}"
    work_root = case / "work"
    results_root = case / "results"
    species = "Synthetic_species"

    species_work = work_root / species
    tile_dir = species_work / "prediction_tiles"
    tile_dir.mkdir(parents=True)

    length = 500

    strands = (
        ["+", "-"]
        if output_type == "RNA_SEQ"
        else ["."] * channel_count
    )

    condition = (
        "EFO:0002824"
        if output_type == "RNA_SEQ"
        else "EFO:0001184"
    )

    assay_title = (
        "total RNA-seq"
        if output_type == "RNA_SEQ"
        else "ATAC-seq"
    )

    endedness = "single" if output_type == "RNA_SEQ" else ""

    expected = np.column_stack([
        np.linspace(
            0.0, 1.0, length, dtype=np.float32
        ) + i
        for i in range(channel_count)
    ])

    # Generate two adjacent prediction checkpoints.
    rows = []

    for tile_index, (start, end) in enumerate(
        ((0, 251), (251, 500))
    ):
        rows.append({
            "tile_index": tile_index,
            "fasta_name": "chrSynthetic",
            "chr_name": "chrSynthetic",
            "sequence_name": "synthetic",
            "role": "assembled-molecule",
            "assembly_unit": "Primary Assembly",
            "sequence_length": length,
            "retained_start": start,
            "retained_end": end,
            "retained_length": end - start,
        })

        filename = (
            f"tile_{tile_index:07d}__chrSynthetic__"
            f"{start}-{end}.npz"
        )

        np.savez_compressed(
            tile_dir / filename,
            values=expected[start:end, :],
            track_strands=np.array(strands),
            track_orientation=np.array("genomic_reference"),
            track_indices=np.arange(
                channel_count, dtype=np.int32
            ),
            output_type=np.array(output_type),
            condition=np.array(condition),
            assay_title=np.array(assay_title),
            imputation_mode=np.array("observed"),
            endedness=np.array(endedness),
            tile_index=np.array(tile_index),
            fasta_name=np.array("chrSynthetic"),
            retained_start=np.array(start),
            retained_end=np.array(end),
            retained_length=np.array(end - start),
        )

    pd.DataFrame(rows).to_csv(
        species_work / "tile_manifest.csv",
        index=False,
    )

    # Run the actual consolidation script.
    run_command(
        ROOT / "src" / "consolidate_predictions.py",
        "--species", species,
        "--work-dir", work_root,
        "--results-dir", results_root,
    )

    species_results = results_root / species
    store_path = (
        species_results / "predictions" / "predictions.zarr"
    )

    store = zarr.open_group(
        str(store_path), mode="r"
    )

    assert store.attrs["output_type"] == output_type

    actual_zarr = np.asarray(
        store["sequences"]["chrSynthetic"]["values"][:]
    )

    np.testing.assert_array_equal(
        actual_zarr, expected
    )

    del store

    # Run the actual BigWig dispatcher.
    run_command(
        ROOT / "src" / "build_tracks.py",
        "--species", species,
        "--results-dir", results_root,
    )

    track_dir = species_results / "tracks"

    if output_type == "RNA_SEQ":
        names = [
            "totalRNA_plus.bw",
            "totalRNA_minus.bw",
        ]
    elif channel_count == 1:
        names = ["Synthetic_species_HEK293T_ATAC.bw"]
    else:
        names = [
            f"Synthetic_species_HEK293T_ATAC_channel_{i + 1:02d}.bw"
            for i in range(channel_count)
        ]

    # Read every BigWig and compare all 500 values.
    for i, name in enumerate(names):
        bw = pyBigWig.open(
            str(track_dir / name)
        )

        assert bw is not None

        try:
            assert bw.chroms() == {
                "chrSynthetic": length
            }

            observed = np.asarray(
                bw.values("chrSynthetic", 0, length)
            )

            np.testing.assert_allclose(
                observed,
                expected[:, i],
                atol=1e-6,
                rtol=0,
            )
        finally:
            bw.close()

    summary = json.loads(
        (track_dir / "track_build_summary.json").read_text()
    )

    assert summary["complete"]
    assert summary["validation_passed"]

    # Test cleanup only for ATAC.
    if output_type == "ATAC":
        assert final_outputs_complete(
            species_results,
            require_tracks=True,
            expected_output_type="ATAC",
            expected_condition=condition,
            allow_bigwig_only=False,
        )

        remove_validated_atac_zarr(
            species_results, condition
        )

        assert not store_path.exists()

        assert final_outputs_complete(
            species_results,
            require_tracks=True,
            expected_output_type="ATAC",
            expected_condition=condition,
            allow_bigwig_only=True,
        )
    else:
        # RNA results must retain their Zarr store.
        assert store_path.exists()

        assert final_outputs_complete(
            species_results,
            require_tracks=True,
            expected_output_type="RNA_SEQ",
            expected_condition=condition,
            allow_bigwig_only=False,
        )

    print(
        f"PASS: {output_type}, "
        f"{channel_count} channel(s), 2 tiles"
    )


def main():
    with tempfile.TemporaryDirectory(
        prefix="ag2_integration_"
    ) as tmp:
        for output_type, channels in (
            ("RNA_SEQ", 2),
            ("ATAC", 1),
            ("ATAC", 2),
        ):
            run_case(
                Path(tmp),
                output_type,
                channels,
            )

    print("ALL INTEGRATION TESTS PASSED")


if __name__ == "__main__":
    main()
