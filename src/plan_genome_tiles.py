#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(
    "/media/eternus1/nfs/projects/dpanchenko/"
    "alphagenome2_genome_pipeline"
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_WORK = PROJECT_ROOT / "work"


ORGANELLE_TERMS = {
    "mitochond",
    "chloroplast",
    "plastid",
    "apicoplast",
}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_fai(path):
    sequences = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip().split("\t")

            sequences.append(
                {
                    "fasta_name": fields[0],
                    "length": int(fields[1]),
                }
            )

    return sequences


def load_sequence_report(path):
    records = []
    aliases = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            records.append(x)

            for key in [
                "refseq_accession",
                "genbank_accession",
                "sequence_name",
                "ucsc_style_name",
            ]:
                value = x.get(key)

                if value:
                    aliases[str(value)] = x

    return records, aliases


def is_organelle(record):
    values = [
        record.get("assigned_molecule_location_type"),
        record.get("chr_name"),
        record.get("sequence_name"),
    ]

    text = " ".join(
        str(v).lower()
        for v in values
        if v is not None
    )

    return any(
        term in text
        for term in ORGANELLE_TERMS
    )


def decide_sequence(
    record,
    length,
    min_scaffold_length,
):
    role = str(
        record.get("role") or ""
    ).strip()

    assembly_unit = str(
        record.get("assembly_unit") or ""
    ).strip()

    if is_organelle(record):
        return False, "organelle"

    if role == "assembled-molecule":
        return True, "nuclear_assembled_molecule"

    if (
        role in {
            "unlocalized-scaffold",
            "unplaced-scaffold",
        }
        and assembly_unit == "Primary Assembly"
        and length >= min_scaffold_length
    ):
        return True, "long_primary_scaffold"

    if role in {
        "unlocalized-scaffold",
        "unplaced-scaffold",
    }:
        return False, "short_scaffold"

    return False, f"excluded_role:{role}"


def write_csv(path, rows, fieldnames):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plan AlphaGenome 2 whole-genome prediction tiles."
        )
    )

    parser.add_argument(
        "--species",
        required=True,
        help="species_id from the inventory",
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
        "--min-scaffold-length",
        type=int,
        default=None,
        help=(
            "Minimum length for unplaced/unlocalized scaffolds. "
            "Defaults to retained_length from config."
        ),
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    input_length = int(
        cfg["tiling"]["input_length"]
    )

    retained_length = int(
        cfg["tiling"]["retained_length"]
    )

    if input_length <= retained_length:
        raise ValueError(
            "input_length must be larger than retained_length"
        )

    if (input_length - retained_length) % 2 != 0:
        raise ValueError(
            "input_length - retained_length must be even"
        )

    flank = (
        input_length - retained_length
    ) // 2

    min_contig_length = int(
        cfg["tiling"].get("min_contig_length", 0)
    )

    if min_contig_length < 0:
        raise ValueError(
            "min_contig_length must be non-negative"
        )

    min_scaffold_length = (
        args.min_scaffold_length
        if args.min_scaffold_length is not None
        else cfg["tiling"].get(
            "min_contig_length",
            retained_length,
        )
    )

    species_dir = (
        args.work_dir
        / args.species
    )

    fai_path = (
        species_dir
        / "genome.fa.fai"
    )

    report_path = (
        species_dir
        / "sequence_report.jsonl"
    )

    if not fai_path.exists():
        raise FileNotFoundError(
            f"Missing FASTA index: {fai_path}"
        )

    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing sequence report: {report_path}"
        )

    fai_sequences = load_fai(
        fai_path
    )

    _, aliases = load_sequence_report(
        report_path
    )

    selection_rows = []
    selected_sequences = []
    unmatched = []

    total_bases = 0
    selected_bases = 0

    for seq in fai_sequences:
        fasta_name = seq["fasta_name"]
        length = seq["length"]

        total_bases += length

        record = aliases.get(
            fasta_name
        )

        if record is None:
            unmatched.append(
                fasta_name
            )
            continue

        if length < min_contig_length:
            keep, reason = False, "short_contig"
        else:
            keep, reason = decide_sequence(
                record,
                length,
                min_scaffold_length,
            )

        row = {
            "fasta_name": fasta_name,
            "length": length,
            "selected": keep,
            "reason": reason,
            "role": record.get("role"),
            "assembly_unit": record.get(
                "assembly_unit"
            ),
            "location_type": record.get(
                "assigned_molecule_location_type"
            ),
            "chr_name": record.get(
                "chr_name"
            ),
            "sequence_name": record.get(
                "sequence_name"
            ),
            "refseq_accession": record.get(
                "refseq_accession"
            ),
            "genbank_accession": record.get(
                "genbank_accession"
            ),
        }

        selection_rows.append(
            row
        )

        if keep:
            selected_sequences.append(
                {
                    "fasta_name": fasta_name,
                    "length": length,
                    "record": record,
                    "reason": reason,
                }
            )

            selected_bases += length

    if unmatched:
        print()
        print("ERROR: FASTA sequences missing from sequence report:")
        for name in unmatched[:20]:
            print(" ", name)

        if len(unmatched) > 20:
            print(
                f" ... and {len(unmatched) - 20} more"
            )

        raise RuntimeError(
            f"{len(unmatched)} FASTA sequences could not "
            "be mapped to the NCBI sequence report."
        )

    tile_rows = []

    tile_index = 0

    for seq in selected_sequences:
        fasta_name = seq["fasta_name"]
        sequence_length = seq["length"]
        record = seq["record"]

        sequence_tile_index = 0

        for retained_start in range(
            0,
            sequence_length,
            retained_length,
        ):
            retained_end = min(
                retained_start + retained_length,
                sequence_length,
            )

            valid_retained_length = (
                retained_end - retained_start
            )

            #
            # Desired AG2 input coordinates before clipping
            # to the chromosome/contig.
            #
            desired_input_start = (
                retained_start - flank
            )

            desired_input_end = (
                desired_input_start
                + input_length
            )

            #
            # Actual genome coordinates to extract.
            #
            genomic_input_start = max(
                0,
                desired_input_start,
            )

            genomic_input_end = min(
                sequence_length,
                desired_input_end,
            )

            pad_left = max(
                0,
                -desired_input_start,
            )

            pad_right = max(
                0,
                desired_input_end
                - sequence_length,
            )

            extracted_length = (
                genomic_input_end
                - genomic_input_start
            )

            final_input_length = (
                pad_left
                + extracted_length
                + pad_right
            )

            if final_input_length != input_length:
                raise RuntimeError(
                    f"Internal tiling error for "
                    f"{fasta_name}: "
                    f"{final_input_length} != "
                    f"{input_length}"
                )

            #
            # Coordinates inside the AG2 output array
            # that should be retained.
            #
            crop_start = flank

            crop_end = (
                crop_start
                + valid_retained_length
            )

            tile_rows.append(
                {
                    "tile_index": tile_index,
                    "sequence_tile_index": sequence_tile_index,
                    "species_id": args.species,
                    "fasta_name": fasta_name,
                    "chr_name": record.get(
                        "chr_name"
                    ),
                    "sequence_name": record.get(
                        "sequence_name"
                    ),
                    "role": record.get(
                        "role"
                    ),
                    "sequence_length": sequence_length,
                    "retained_start": retained_start,
                    "retained_end": retained_end,
                    "retained_length": valid_retained_length,
                    "desired_input_start": desired_input_start,
                    "desired_input_end": desired_input_end,
                    "genomic_input_start": genomic_input_start,
                    "genomic_input_end": genomic_input_end,
                    "pad_left": pad_left,
                    "pad_right": pad_right,
                    "crop_start": crop_start,
                    "crop_end": crop_end,
                }
            )

            tile_index += 1
            sequence_tile_index += 1

    selection_path = (
        species_dir
        / "sequence_selection.csv"
    )

    manifest_path = (
        species_dir
        / "tile_manifest.csv"
    )

    summary_path = (
        species_dir
        / "tile_plan_summary.json"
    )

    write_csv(
        selection_path,
        selection_rows,
        [
            "fasta_name",
            "length",
            "selected",
            "reason",
            "role",
            "assembly_unit",
            "location_type",
            "chr_name",
            "sequence_name",
            "refseq_accession",
            "genbank_accession",
        ],
    )

    write_csv(
        manifest_path,
        tile_rows,
        [
            "tile_index",
            "sequence_tile_index",
            "species_id",
            "fasta_name",
            "chr_name",
            "sequence_name",
            "role",
            "sequence_length",
            "retained_start",
            "retained_end",
            "retained_length",
            "desired_input_start",
            "desired_input_end",
            "genomic_input_start",
            "genomic_input_end",
            "pad_left",
            "pad_right",
            "crop_start",
            "crop_end",
        ],
    )

    summary = {
        "species_id": args.species,
        "input_length": input_length,
        "retained_length": retained_length,
        "flank": flank,
        "min_scaffold_length": min_scaffold_length,
        "assembly_sequence_count": len(
            fai_sequences
        ),
        "selected_sequence_count": len(
            selected_sequences
        ),
        "excluded_sequence_count": (
            len(fai_sequences)
            - len(selected_sequences)
        ),
        "assembly_total_bases": total_bases,
        "selected_bases": selected_bases,
        "excluded_bases": (
            total_bases - selected_bases
        ),
        "selected_fraction": (
            selected_bases / total_bases
            if total_bases
            else 0.0
        ),
        "ag2_prediction_calls": len(
            tile_rows
        ),
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
    print("GENOME TILE PLAN COMPLETE")
    print("=" * 70)

    print(
        f"Assembly sequences: "
        f"{len(fai_sequences):,}"
    )

    print(
        f"Selected sequences: "
        f"{len(selected_sequences):,}"
    )

    print(
        f"Excluded sequences: "
        f"{len(fai_sequences) - len(selected_sequences):,}"
    )

    print(
        f"Assembly bases:     "
        f"{total_bases:,}"
    )

    print(
        f"Selected bases:     "
        f"{selected_bases:,}"
    )

    print(
        f"Selected fraction:  "
        f"{100 * selected_bases / total_bases:.2f}%"
    )

    print(
        f"AG2 calls required: "
        f"{len(tile_rows):,}"
    )

    print()
    print("Outputs:")
    print(" ", selection_path)
    print(" ", manifest_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()