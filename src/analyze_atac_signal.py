#!/usr/bin/env python3
"""Sample genomic regions and quantify AlphaGenome 2 ATAC signals."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "inventory/atac_hek293t_subset.csv"
DEFAULT_RESULTS = ROOT / "results/atac_hek293t"
DEFAULT_OUTPUT = ROOT / "analysis/atac_hek293t"


def analyze_species(species, results_dir, n_regions, width, rng):
    tracks_dir = results_dir / species / "tracks"
    bw_path = tracks_dir / f"{species}_HEK293T_ATAC.bw"
    summary_path = tracks_dir / "track_build_summary.json"

    summary = json.loads(summary_path.read_text())

    if not (
        summary.get("complete")
        and summary.get("validation_passed")
        and summary.get("output_type") == "ATAC"
        and summary.get("condition") == "EFO:0001184"
        and summary.get("channel_count") == 1
    ):
        raise RuntimeError(
            f"{species}: invalid ATAC track metadata"
        )

    records = []
    means = []

    bw = pyBigWig.open(str(bw_path))

    try:
        # Sampling probabilities are proportional to the
        # number of possible 250 bp window starting positions.
        chromosomes = [
            (chrom, length)
            for chrom, length in bw.chroms().items()
            if length >= max(width, 100_000)
        ]

        if not chromosomes:
            raise RuntimeError(
                f"{species}: no eligible chromosomes"
            )

        names = [name for name, _ in chromosomes]

        valid_starts = np.array(
            [length - width + 1 for _, length in chromosomes],
            dtype=np.float64,
        )

        probabilities = valid_starts / valid_starts.sum()

        chosen = rng.choice(
            len(chromosomes),
            size=n_regions,
            replace=True,
            p=probabilities,
        )

        for index in chosen:
            chrom = names[index]
            chrom_length = chromosomes[index][1]

            start = int(
                rng.integers(0, chrom_length - width + 1)
            )
            end = start + width

            # Exact per-base values, not BigWig zoom summaries.
            values = np.asarray(
                bw.values(chrom, start, end, numpy=True)
            )

            valid = np.isfinite(values)
            covered = int(valid.sum())

            mean = (
                float(values[valid].mean())
                if covered > 0 else np.nan
            )

            fully_covered = covered == width

            records.append({
                "species_id": species,
                "chromosome": chrom,
                "start": start,
                "end": end,
                "mean_signal": mean,
                "coverage_fraction": covered / width,
                "fully_covered": fully_covered,
                "human_control": species == "Homo_sapiens",
            })

            if fully_covered:
                means.append(mean)

    finally:
        bw.close()

    means = np.asarray(means, dtype=np.float64)

    species_summary = {
        "species_id": species,
        "sampled_regions": n_regions,
        "fully_covered_regions": len(means),
        "mean_signal": (
            float(means.mean()) if len(means) else np.nan
        ),
        "median_signal": (
            float(np.median(means)) if len(means) else np.nan
        ),
        "p95_signal": (
            float(np.percentile(means, 95))
            if len(means) else np.nan
        ),
        "zero_fraction": (
            float(np.mean(means == 0))
            if len(means) else np.nan
        ),
        "human_control": species == "Homo_sapiens",
    }

    return records, species_summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inventory", type=Path, default=DEFAULT_INVENTORY
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--samples", type=int, default=20_000
    )
    parser.add_argument(
        "--width", type=int, default=250
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()

    if args.samples < 1 or args.width < 1:
        parser.error("--samples and --width must be positive")

    inventory = pd.read_csv(args.inventory)
    species_ids = inventory["species_id"].tolist()

    if len(species_ids) != 14 or len(set(species_ids)) != 14:
        raise RuntimeError("Expected 14 distinct genomes")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    summaries = []

    for i, species in enumerate(species_ids):
        print(
            f"[{i + 1}/{len(species_ids)}] {species}",
            flush=True,
        )

        # Separate reproducible random stream per species.
        rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, i])
        )

        records, summary = analyze_species(
            species,
            args.results_dir,
            args.samples,
            args.width,
            rng,
        )

        all_records.extend(records)
        summaries.append(summary)

        print(
            f"  Sampled: {summary['sampled_regions']:,}; "
            f"fully covered: "
            f"{summary['fully_covered_regions']:,}",
            flush=True,
        )

    samples_df = pd.DataFrame(all_records)
    summary_df = pd.DataFrame(summaries)

    samples_df.to_csv(
        args.output_dir / "sampled_regions.csv",
        index=False,
    )

    summary_df.to_csv(
        args.output_dir / "signal_summary.csv",
        index=False,
    )

    settings = {
        "assay": "ATAC",
        "condition": "EFO:0001184",
        "human_control": "Homo_sapiens",
        "samples_per_genome": args.samples,
        "region_width_bp": args.width,
        "random_seed": args.seed,
        "sampling": "uniform over eligible genomic window starts",
        "missing_values": "excluded from mean; coverage recorded",
    }

    (args.output_dir / "sampling_settings.json").write_text(
        json.dumps(settings, indent=2)
    )

    print("\nANALYSIS COMPLETE")
    print("Total regions:", len(samples_df))
    print("Results:", args.output_dir)


if __name__ == "__main__":
    main()
