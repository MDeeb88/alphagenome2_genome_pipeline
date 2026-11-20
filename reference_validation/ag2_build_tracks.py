#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pyBigWig
from tqdm import tqdm


DEFAULT_RAW_DIR = Path(
    "/media/eternus1/nfs/projects/dpanchenko/"
    "alphagenome2_pipeline/results/astel2_2L_ag2/"
    "raw_predictions/all"
)

DEFAULT_OUT_DIR = Path(
    "/media/eternus1/nfs/projects/dpanchenko/"
    "alphagenome2_pipeline/results/astel2_2L_ag2/tracks"
)

DEFAULT_CHROM = "2L"
DEFAULT_CHROM_SIZE = 36_251_798


def scalar(npz, key):
    value = npz[key]
    if value.ndim == 0:
        return value.item()
    return value


def load_prediction(path, expected_chrom):
    with np.load(path, allow_pickle=False) as x:
        values = np.asarray(x["values"], dtype=np.float32)

        strands = [str(v) for v in x["track_strands"]]
        orientation = str(scalar(x, "track_orientation"))

        chrom = str(scalar(x, "chrom"))
        start = int(scalar(x, "start"))
        end = int(scalar(x, "end"))
        fasta_strand = str(scalar(x, "fasta_strand"))
        sequence_id = str(scalar(x, "sequence_id"))

    if chrom != expected_chrom:
        raise ValueError(
            f"{path.name}: expected chromosome {expected_chrom}, found {chrom}"
        )

    if orientation != "input_sequence":
        raise ValueError(
            f"{path.name}: unexpected track_orientation={orientation}"
        )

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(
            f"{path.name}: expected values shape (N, 2), found {values.shape}"
        )

    if end - start != values.shape[0]:
        raise ValueError(
            f"{path.name}: coordinate length {end-start} != "
            f"prediction length {values.shape[0]}"
        )

    if set(strands) != {"+", "-"}:
        raise ValueError(
            f"{path.name}: unexpected track_strands={strands}"
        )

    if fasta_strand not in {"+", "-"}:
        raise ValueError(
            f"{path.name}: unexpected fasta_strand={fasta_strand}"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{path.name}: prediction contains NaN or infinite values"
        )

    plus_idx = strands.index("+")
    minus_idx = strands.index("-")

    input_plus = values[:, plus_idx]
    input_minus = values[:, minus_idx]

    # AG2 tracks are relative to the sequence orientation supplied to AG2.
    #
    # '+' FASTA:
    #   input orientation == genomic orientation.
    #
    # '-' FASTA:
    #   sequence was already reverse-complemented in the FASTA.
    #   Therefore:
    #       input '+' -> genomic '-'
    #       input '-' -> genomic '+'
    #   and the arrays must be reversed back into increasing genomic
    #   coordinate order.
    if fasta_strand == "+":
        genomic_plus = input_plus
        genomic_minus = input_minus
    else:
        genomic_plus = input_minus[::-1]
        genomic_minus = input_plus[::-1]

    return {
        "sequence_id": sequence_id,
        "start": start,
        "end": end,
        "fasta_strand": fasta_strand,
        "plus": genomic_plus,
        "minus": genomic_minus,
    }


def write_tracks(
    chrom,
    chrom_size,
    sum_plus,
    sum_minus,
    counts,
    bedgraph_plus,
    bedgraph_minus,
    bigwig_plus,
    bigwig_minus,
    chunk_size=250_000,
):
    bw_plus = pyBigWig.open(str(bigwig_plus), "w")
    bw_minus = pyBigWig.open(str(bigwig_minus), "w")

    bw_plus.addHeader([(chrom, chrom_size)])
    bw_minus.addHeader([(chrom, chrom_size)])

    covered_total = 0

    plus_min = np.inf
    plus_max = -np.inf
    plus_sum = 0.0

    minus_min = np.inf
    minus_max = -np.inf
    minus_sum = 0.0

    try:
        with (
            open(bedgraph_plus, "w", buffering=1024 * 1024) as plus_bg,
            open(bedgraph_minus, "w", buffering=1024 * 1024) as minus_bg,
        ):
            for chunk_start in tqdm(
                range(0, chrom_size, chunk_size),
                desc="Writing tracks",
                unit="chunk",
            ):
                chunk_end = min(chunk_start + chunk_size, chrom_size)

                local_counts = counts[chunk_start:chunk_end]
                local_mask = local_counts > 0

                if not np.any(local_mask):
                    continue

                local_positions = np.flatnonzero(local_mask)
                positions = local_positions + chunk_start

                c = local_counts[local_mask].astype(np.float64)

                plus_values = (
                    sum_plus[chunk_start:chunk_end][local_mask] / c
                )
                minus_values = (
                    sum_minus[chunk_start:chunk_end][local_mask] / c
                )

                covered_total += len(positions)

                plus_min = min(plus_min, float(plus_values.min()))
                plus_max = max(plus_max, float(plus_values.max()))
                plus_sum += float(plus_values.sum())

                minus_min = min(minus_min, float(minus_values.min()))
                minus_max = max(minus_max, float(minus_values.max()))
                minus_sum += float(minus_values.sum())

                plus_bg.write(
                    "".join(
                        f"{chrom}\t{pos}\t{pos + 1}\t{value:.8g}\n"
                        for pos, value in zip(positions, plus_values)
                    )
                )

                minus_bg.write(
                    "".join(
                        f"{chrom}\t{pos}\t{pos + 1}\t{value:.8g}\n"
                        for pos, value in zip(positions, minus_values)
                    )
                )

                starts = positions.astype(np.int64).tolist()
                ends = (positions + 1).astype(np.int64).tolist()
                chroms = [chrom] * len(starts)

                bw_plus.addEntries(
                    chroms,
                    starts,
                    ends=ends,
                    values=plus_values.astype(float).tolist(),
                )

                bw_minus.addEntries(
                    chroms,
                    starts,
                    ends=ends,
                    values=minus_values.astype(float).tolist(),
                )

    finally:
        bw_plus.close()
        bw_minus.close()

    stats = {
        "covered_bases": covered_total,
        "coverage_fraction": (
            covered_total / chrom_size if chrom_size else 0.0
        ),
        "plus": {
            "min": float(plus_min) if covered_total else None,
            "max": float(plus_max) if covered_total else None,
            "mean": (
                float(plus_sum / covered_total)
                if covered_total
                else None
            ),
        },
        "minus": {
            "min": float(minus_min) if covered_total else None,
            "max": float(minus_max) if covered_total else None,
            "mean": (
                float(minus_sum / covered_total)
                if covered_total
                else None
            ),
        },
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build strand-specific genomic bedGraph and BigWig tracks "
            "from AlphaGenome 2 raw NPZ predictions."
        )
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    parser.add_argument(
        "--chrom",
        default=DEFAULT_CHROM,
    )

    parser.add_argument(
        "--chrom-size",
        type=int,
        default=DEFAULT_CHROM_SIZE,
    )

    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    chrom = args.chrom
    chrom_size = args.chrom_size

    files = sorted(raw_dir.glob("*.npz"))

    if not files:
        raise RuntimeError(
            f"No NPZ predictions found in {raw_dir}"
        )

    print("=" * 70)
    print("AlphaGenome 2 track builder")
    print("=" * 70)
    print(f"Raw directory: {raw_dir}")
    print(f"NPZ files:     {len(files)}")
    print(f"Chromosome:    {chrom}")
    print(f"Chrom size:    {chrom_size:,}")
    print()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Two signal sums plus one shared coverage-count array.
    #
    # float64 is used for accumulation to avoid unnecessary precision loss
    # when overlapping predictions are averaged.
    sum_plus = np.zeros(chrom_size, dtype=np.float64)
    sum_minus = np.zeros(chrom_size, dtype=np.float64)

    # uint16 is far beyond the expected overlap depth for these intervals.
    counts = np.zeros(chrom_size, dtype=np.uint16)

    plus_records = 0
    minus_records = 0

    for path in tqdm(
        files,
        desc="Accumulating predictions",
        unit="file",
    ):
        record = load_prediction(path, chrom)

        start = record["start"]
        end = record["end"]

        if start < 0 or end > chrom_size or start >= end:
            raise ValueError(
                f"{path.name}: invalid coordinates "
                f"{chrom}:{start}-{end}"
            )

        sum_plus[start:end] += record["plus"]
        sum_minus[start:end] += record["minus"]
        counts[start:end] += 1

        if record["fasta_strand"] == "+":
            plus_records += 1
        else:
            minus_records += 1

    covered = counts > 0

    covered_bases = int(covered.sum())
    max_overlap = int(counts.max())
    total_prediction_bases = int(counts.astype(np.uint64).sum())

    mean_depth_covered = (
        float(total_prediction_bases / covered_bases)
        if covered_bases
        else 0.0
    )

    print()
    print("Accumulation complete")
    print(f"Records on FASTA + strand: {plus_records}")
    print(f"Records on FASTA - strand: {minus_records}")
    print(f"Covered genomic bases:     {covered_bases:,}")
    print(
        f"Chromosome coverage:        "
        f"{100 * covered_bases / chrom_size:.2f}%"
    )
    print(f"Maximum overlap depth:      {max_overlap}")
    print(f"Mean depth where covered:   {mean_depth_covered:.3f}")
    print()

    bedgraph_plus = out_dir / "astel2_2L_totalRNA_plus.bedGraph"
    bedgraph_minus = out_dir / "astel2_2L_totalRNA_minus.bedGraph"

    bigwig_plus = out_dir / "astel2_2L_totalRNA_plus.bw"
    bigwig_minus = out_dir / "astel2_2L_totalRNA_minus.bw"

    chrom_sizes = out_dir / "2L.chrom.sizes"

    chrom_sizes.write_text(
        f"{chrom}\t{chrom_size}\n",
        encoding="utf-8",
    )

    stats = write_tracks(
        chrom=chrom,
        chrom_size=chrom_size,
        sum_plus=sum_plus,
        sum_minus=sum_minus,
        counts=counts,
        bedgraph_plus=bedgraph_plus,
        bedgraph_minus=bedgraph_minus,
        bigwig_plus=bigwig_plus,
        bigwig_minus=bigwig_minus,
    )

    summary = {
        "chromosome": chrom,
        "chromosome_size": chrom_size,
        "npz_files": len(files),
        "fasta_plus_records": plus_records,
        "fasta_minus_records": minus_records,
        "total_prediction_bases": total_prediction_bases,
        "covered_bases": covered_bases,
        "coverage_fraction": covered_bases / chrom_size,
        "maximum_overlap_depth": max_overlap,
        "mean_overlap_depth_on_covered_bases": mean_depth_covered,
        "overlap_resolution": "mean",
        "input_track_orientation": "input_sequence",
        "minus_strand_mapping": (
            "genomic_plus=input_minus[::-1]; "
            "genomic_minus=input_plus[::-1]"
        ),
        "tracks": stats,
        "outputs": {
            "plus_bedgraph": str(bedgraph_plus),
            "minus_bedgraph": str(bedgraph_minus),
            "plus_bigwig": str(bigwig_plus),
            "minus_bigwig": str(bigwig_minus),
            "chrom_sizes": str(chrom_sizes),
        },
    }

    summary_path = out_dir / "track_build_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print("TRACK BUILD COMPLETE")
    print("=" * 70)
    print(f"Plus bedGraph:  {bedgraph_plus}")
    print(f"Minus bedGraph: {bedgraph_minus}")
    print(f"Plus BigWig:    {bigwig_plus}")
    print(f"Minus BigWig:   {bigwig_minus}")
    print(f"Summary:        {summary_path}")


if __name__ == "__main__":
    main()