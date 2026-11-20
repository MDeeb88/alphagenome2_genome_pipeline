#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    "/media/eternus1/nfs/projects/dpanchenko/"
    "alphagenome2_genome_pipeline"
)

DEFAULT_INVENTORY = (
    PROJECT_ROOT
    / "inventory"
    / "CTCF_Figure7_genome_annotation_inventory.csv"
)

DEFAULT_WORK_DIR = PROJECT_ROOT / "work"


def run(cmd):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def load_species(inventory_path, species_id):
    df = pd.read_csv(inventory_path)

    match = df[df["species_id"] == species_id]

    if len(match) == 0:
        raise ValueError(
            f"Species not found in inventory: {species_id}"
        )

    if len(match) > 1:
        raise ValueError(
            f"Species occurs more than once in inventory: {species_id}"
        )

    row = match.iloc[0]

    accession = str(row["ncbi_assembly_accession"]).strip()

    if not accession or accession.lower() == "nan":
        raise ValueError(
            f"No NCBI assembly accession for {species_id}"
        )

    return row, accession


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare one genome from NCBI."
    )

    parser.add_argument(
        "--species",
        required=True,
        help="species_id from the inventory CSV",
    )

    parser.add_argument(
        "--inventory",
        type=Path,
        default=DEFAULT_INVENTORY,
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if genome.fa already exists",
    )

    args = parser.parse_args()

    row, accession = load_species(
        args.inventory,
        args.species,
    )

    species_dir = args.work_dir / args.species
    genome_path = species_dir / "genome.fa"
    genome_index = species_dir / "genome.fa.fai"

    zip_path = species_dir / "ncbi_dataset.zip"
    extract_dir = species_dir / "ncbi_dataset"

    metadata_path = species_dir / "assembly_metadata.json"

    species_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("Genome preparation")
    print("=" * 70)
    print("Species ID:   ", args.species)
    print("Species name: ", row["species_name"])
    print("Taxa:         ", row["taxa"])
    print("Assembly:     ", accession)
    print("Work dir:     ", species_dir)
    print()

    if (
        genome_path.exists()
        and genome_index.exists()
        and not args.force
    ):
        print("Prepared genome already exists.")
        print("Genome:", genome_path)
        print("Index: ", genome_index)
        return

    if args.force:
        if zip_path.exists():
            zip_path.unlink()

        if extract_dir.exists():
            shutil.rmtree(extract_dir)

        if genome_path.exists():
            genome_path.unlink()

        if genome_index.exists():
            genome_index.unlink()

    #
    # Download genome package.
    #
    run(
        [
            "datasets",
            "download",
            "genome",
            "accession",
            accession,
            "--include",
            "genome",
            "--filename",
            str(zip_path),
        ]
    )

    #
    # Extract package.
    #
    extract_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run(
        [
            "unzip",
            "-q",
            str(zip_path),
            "-d",
            str(extract_dir),
        ]
    )

    #
    # Locate the genomic FASTA.
    #
    fasta_files = list(
        extract_dir.rglob("*_genomic.fna")
    )

    if len(fasta_files) != 1:
        raise RuntimeError(
            f"Expected exactly one *_genomic.fna file, "
            f"found {len(fasta_files)}"
        )

    source_fasta = fasta_files[0]

    print()
    print("Downloaded FASTA:")
    print(source_fasta)

    #
    # Move into stable pipeline location.
    #
    shutil.copy2(
        source_fasta,
        genome_path,
    )

    #
    # Build FASTA index.
    #
    run(
        [
            "samtools",
            "faidx",
            str(genome_path),
        ]
    )

    #
    # Read chromosome/contig information.
    #
    sequences = []

    with open(genome_index) as f:
        for line in f:
            fields = line.rstrip().split("\t")

            sequences.append(
                {
                    "name": fields[0],
                    "length": int(fields[1]),
                }
            )

    total_bases = sum(
        x["length"]
        for x in sequences
    )

    #
    # Save reproducibility metadata.
    #
    metadata = {
        "species_id": args.species,
        "species_name": str(row["species_name"]),
        "taxa": str(row["taxa"]),
        "assembly_accession": accession,
        "assembly_accession_source": str(
            row["assembly_accession_source"]
        ),
        "original_inventory_genome_fasta": str(
            row["genome_fasta"]
        ),
        "genome_path": str(genome_path),
        "index_path": str(genome_index),
        "sequence_count": len(sequences),
        "total_bases": total_bases,
        "sequences": sequences,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    #
    # Remove download package/extraction.
    # Keep only genome.fa + genome.fa.fai + metadata.
    #
    zip_path.unlink()

    shutil.rmtree(
        extract_dir
    )

    print()
    print("=" * 70)
    print("GENOME PREPARATION COMPLETE")
    print("=" * 70)
    print("Genome:      ", genome_path)
    print("Index:       ", genome_index)
    print("Metadata:    ", metadata_path)
    print("Sequences:   ", len(sequences))
    print(f"Total bases: {total_bases:,}")


if __name__ == "__main__":
    main()