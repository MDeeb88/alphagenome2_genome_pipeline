
#!/usr/bin/env python3
"""Interactive reference genome downloader for AlphaGenome 2."""

import csv
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

INVENTORIES = [
    PROJECT / "inventory/CTCF_Figure7_genome_annotation_inventory.csv",
    PROJECT / "inventory/atac_hek293t_subset.csv",
]

REFERENCE_DIR = PROJECT / "data/igv_references"
ACCESSION_PATTERN = re.compile(r"GC[AF]_\d+\.\d+$")


def load_species():
    species = {}

    for inventory in INVENTORIES:
        if not inventory.is_file():
            continue

        with inventory.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("species_id") or "").strip()
                accession = (
                    row.get("ncbi_assembly_accession") or ""
                ).strip()

                if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                    continue
                if not ACCESSION_PATTERN.fullmatch(accession):
                    continue

                # Pilot inventory takes precedence for overlapping species.
                species[name] = {
                    "name": name,
                    "accession": accession,
                }

    if not species:
        raise RuntimeError("No assembly accessions found.")

    return sorted(species.values(), key=lambda x: x["name"])


def show_species(rows, indexes, limit=None):
    if limit is not None and len(rows) > limit:
        print(f"Showing {limit} of {len(rows)}. Type 'list' for all.")
        rows = rows[:limit]

    for row in rows:
        number = indexes[row["name"]]
        print(f"{number:3}. {row['name']:<32} {row['accession']}")


def select_species(rows):
    by_number = dict(enumerate(rows, 1))
    indexes = {row["name"]: i for i, row in by_number.items()}
    selected = set()

    print(f"\nAvailable reference genomes: {len(rows)}")
    show_species(rows, indexes, limit=20)

    print("\nSelection options:")
    print("  1, 2, 5          Select by number")
    print("  Apis_mellifera   Select by name")
    print("  ?drosophila      Search names")
    print("  list             Show all")
    print("  all              Select all")
    print("  Enter            Continue")

    while True:
        answer = input("\nSelect / add genomes: ").strip()

        if answer.lower() in {"b", "back"}:
            print("Returning to main menu.")
            return []

        if not answer:
            return [
                row for row in rows
                if row["name"] in selected
            ]

        if answer.lower() == "list":
            show_species(rows, indexes)
            continue

        if answer.startswith("?"):
            query = answer[1:].lower().replace("_", " ")
            matches = [
                row for row in rows
                if query in row["name"].lower().replace("_", " ")
            ]
            print(f"Found {len(matches)} match(es):")
            show_species(matches, indexes)
            continue

        if answer.lower() == "all":
            selected.update(row["name"] for row in rows)

        else:
            for item in answer.split(","):
                item = item.strip()
                if not item:
                    continue

                if item.isdigit():
                    row = by_number.get(int(item))
                    if row is None:
                        print(f"Invalid number: {item}")
                    else:
                        selected.add(row["name"])
                    continue

                query = item.lower().replace("_", " ")
                matches = [
                    row for row in rows
                    if query in row["name"].lower().replace("_", " ")
                ]

                exact = [
                    row for row in matches
                    if row["name"].lower().replace("_", " ") == query
                ]
                matches = exact or matches

                if len(matches) == 1:
                    selected.add(matches[0]["name"])
                elif len(matches) > 1:
                    print(f"Multiple matches for '{item}':")
                    show_species(matches, indexes)
                    print("Choose the corresponding numbers.")
                else:
                    print(f"No matching genome: {item}")

        print(f"Selected: {len(selected)} genome(s)")
        print("Press Enter when finished.")


def find_file(archive, accession, suffixes):
    prefix = f"ncbi_dataset/data/{accession}/"

    matches = [
        name for name in archive.namelist()
        if name.startswith(prefix)
        and name.lower().endswith(suffixes)
        and not name.endswith("/")
    ]

    return matches[0] if matches else None


def extract_file(archive, member, output):
    if member is None:
        return False

    temporary = output.with_name(output.name + ".partial")

    try:
        with archive.open(member) as source:
            with temporary.open("wb") as target:
                shutil.copyfileobj(source, target, 4 * 1024 * 1024)

        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    return True


def download_genome(row):
    species = row["name"]
    accession = row["accession"]

    folder = REFERENCE_DIR / species
    folder.mkdir(parents=True, exist_ok=True)

    fasta = folder / f"{species}.fna"
    annotation_gtf = folder / f"{species}.gtf"
    annotation_gff = folder / f"{species}.gff3"

    accession_file = folder / "assembly_accession.txt"
    zip_path = folder / "ncbi_dataset.zip"

    if accession_file.exists():
        existing_accession = accession_file.read_text().strip()
        if existing_accession != accession:
            raise RuntimeError(
                f"Assembly mismatch in {folder}: "
                f"{existing_accession} != {accession}"
            )

    if fasta.exists() and fasta.stat().st_size > 0:
        print(f"SKIPPED: {species} (reference already present)")
        return

    print(f"\nDownloading {species} ({accession})...")

    if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        zip_path.unlink(missing_ok=True)

        subprocess.run(
            [
                "datasets", "download", "genome",
                "accession", accession,
                "--include", "genome,gtf,gff3",
                "--filename", str(zip_path),
            ],
            check=True,
        )

    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(f"Invalid download archive: {species}")

    with zipfile.ZipFile(zip_path) as archive:
        genome_member = find_file(
            archive, accession, (".fna", ".fa", ".fasta")
        )

        if genome_member is None:
            raise RuntimeError(f"No genome FASTA: {species}")

        extract_file(archive, genome_member, fasta)

        annotation_types = [
            ((".gtf",), annotation_gtf, "GTF"),
            ((".gff3", ".gff"), annotation_gff, "GFF3"),
        ]

        for extensions, destination, label in annotation_types:
            member = find_file(archive, accession, extensions)

            if member:
                extract_file(archive, member, destination)
            else:
                print(f"  {label} not available from NCBI")

    accession_file.write_text(accession + "\n")
    zip_path.unlink()

    print(f"COMPLETED: {species}")
    print(f"Saved to: {folder}")


def main():
    if shutil.which("datasets") is None:
        raise RuntimeError("NCBI Datasets CLI not found in PATH.")

    rows = load_species()
    chosen = select_species(rows)

    if not chosen:
        print("No genomes selected. Returning.")
        return

    print(f"\nSelected genomes: {len(chosen)}")
    for row in chosen:
        print(f"  {row['name']} ({row['accession']})")

    print(f"\nDestination: {REFERENCE_DIR}")

    if input("\nStart downloads? [y/N]: ").strip().lower() != "y":
        print("Cancelled.")
        return

    failures = []

    for row in chosen:
        try:
            download_genome(row)
        except (
            OSError,
            RuntimeError,
            subprocess.CalledProcessError,
            zipfile.BadZipFile,
        ) as exc:
            print(f"FAILED: {row['name']} — {exc}", file=sys.stderr)
            failures.append(row["name"])

    print(f"\nFinished: {len(chosen) - len(failures)}/{len(chosen)}")
    if failures:
        print("Failed: " + ", ".join(failures))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f"Downloader error: {exc}") from exc
