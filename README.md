# AlphaGenome 2 Multi-Genome Prediction Pipeline

Reusable pipeline for running AlphaGenome 2 predictions across a comparative-genomics inventory.

The pipeline can download an NCBI genome assembly, select suitable nuclear sequences, plan AlphaGenome-compatible windows, run resumable predictions, consolidate predictions into Zarr, generate optional BigWig tracks, validate outputs, and remove temporary files only after successful validation.

## Project location on the validated server

```text
/media/eternus1/nfs/projects/dpanchenko/alphagenome2_genome_pipeline
```

Validated conda environment:

```text
alphagenome2
```

## Main entry point

From the project root:

```bash
cd /media/eternus1/nfs/projects/dpanchenko/alphagenome2_genome_pipeline
conda activate alphagenome2
python src/run_pipeline.py
```

The interactive controller provides:

```text
1. Run one species
2. Run selected species
3. Run one taxonomic group
4. Run all unfinished species
5. Retry failed species
6. Show pipeline status
7. Dry-run / preview
8. AlphaGenome API key
9. Exit
```

For large selections, use dry-run first.

## AlphaGenome API key

An AlphaGenome API key is required for new predictions.

Official AlphaGenome documentation:

https://www.alphagenomedocs.com/

### Recommended setup

Run:

```bash
python src/run_pipeline.py
```

Then choose:

```text
8. AlphaGenome API key
```

The menu allows you to:

1. Save or replace the API key.
2. Use a key for the current pipeline session only.
3. Remove a previously saved key.

A saved key is stored outside the project at:

```text
~/.config/alphagenome2/api_key
```

The file is created with Linux permission mode `600` and its directory with mode `700`. The file is plain text protected by the Linux user permissions; it is not encrypted.

The key is never printed by the pipeline and should never be committed to Git, placed in the inventory, or stored in project configuration files.

You can still configure a key manually for one shell session:

```bash
read -s ALPHA_GENOME_API_KEY
export ALPHA_GENOME_API_KEY
echo
```

## Installation

The recommended installation is the repository environment file:

```bash
conda env create -f environment.yml
conda activate alphagenome2
```

For an existing environment:

```bash
conda env update -n alphagenome2 -f environment.yml --prune
```

The environment includes the Python dependencies plus command-line tools used by the pipeline, including NCBI Datasets and samtools.

## Inventory

The main inventory is:

```text
inventory/CTCF_Figure7_genome_annotation_inventory.csv
```

The pipeline uses `species_id` as the internal species identifier and `ncbi_assembly_accession` to download the genome automatically.

### Adding a new species

Add one new row to:

```text
inventory/CTCF_Figure7_genome_annotation_inventory.csv
```

Keep all existing columns. The important fields for the prediction pipeline are:

```text
species_id
species_name
taxa
ncbi_assembly_accession
```

Example:

```text
species_id:              Example_species
species_name:            Example species
taxa:                    EXAMPLE_GROUP
ncbi_assembly_accession: GCF_XXXXXXXXX.X
```

After editing the inventory, verify the row without running anything:

```bash
python src/run_pipeline.py --species Example_species --dry-run
```

Do not manually download the genome. The pipeline downloads the assembly from NCBI using the accession in the inventory.

## Dry-run mode

Preview one species:

```bash
python src/run_pipeline.py \
  --species Drosophila_simulans \
  --dry-run
```

Preview all unfinished species:

```bash
python src/run_pipeline.py \
  --all \
  --dry-run
```

Dry-run performs no genome download and makes no AlphaGenome prediction calls.

## Command-line execution

Run one species:

```bash
python src/run_pipeline.py --species Drosophila_simulans
```

Run multiple species:

```bash
python src/run_pipeline.py \
  --species Drosophila_simulans Drosophila_yakuba \
  --max-parallel 2
```

Run a taxonomic group:

```bash
python src/run_pipeline.py \
  --taxa PROTOSTOMES \
  --max-parallel 5
```

Run all unfinished species only after previewing them:

```bash
python src/run_pipeline.py --all --dry-run
```

For an unattended full run, explicit confirmation is required:

```bash
python src/run_pipeline.py \
  --all \
  --confirm-all \
  --max-parallel 5
```

Completed species are detected and skipped automatically.

## Pipeline stages

For each species, the workflow is:

```text
inventory lookup
    ↓
NCBI genome download
    ↓
FASTA indexing
    ↓
NCBI sequence report
    ↓
sequence selection
    ↓
AG2 tile planning
    ↓
resumable AG2 predictions
    ↓
per-tile NPZ checkpoints
    ↓
Zarr consolidation
    ↓
permanent-output validation
    ↓
optional BigWig generation
    ↓
BigWig validation
    ↓
cleanup of temporary files
```

Temporary files are not removed after a failed run, allowing the same species to resume from valid checkpoints.

## Prediction configuration

The default configuration is:

```text
configs/default.yaml
```

Current validated target:

```text
AlphaGenome:       AlphaGenome 2
Condition:         EFO:0002824
Output:            RNA_SEQ
Assay:             total RNA-seq
Imputation mode:   observed
Endedness:         single
Tracks retained:   + and -
```

Current tiling design:

```text
AG2 input length:       1,048,576 bp
Retained center:          524,288 bp
Context on each side:     262,144 bp
```

## Results

Permanent species results are stored under:

```text
results/<species_id>/
```

Typical structure:

```text
results/<species_id>/
├── predictions/
│   └── predictions.zarr/
├── metadata/
├── tracks/
└── logs/
```

### Primary scientific output

The primary prediction output is:

```text
results/<species_id>/predictions/predictions.zarr
```

Each selected genomic sequence is stored under the Zarr `sequences` group with a two-column base-resolution array:

```text
column 0 = RNA +
column 1 = RNA -
```

The stored orientation is genomic reference orientation.

### BigWig tracks

When track generation is enabled:

```text
results/<species_id>/tracks/totalRNA_plus.bw
results/<species_id>/tracks/totalRNA_minus.bw
results/<species_id>/tracks/chrom.sizes
results/<species_id>/tracks/chrom_aliases.tsv
results/<species_id>/tracks/track_build_summary.json
```

BigWigs are derived visualization outputs; the Zarr prediction store remains the primary scientific result.

## Status

Interactive status:

```bash
python src/run_pipeline.py
```

Choose:

```text
6. Show pipeline status
```

Command-line status:

```bash
python src/run_pipeline.py --status
```

Statuses include:

```text
COMPLETE
FAILED
INCOMPLETE
NOT_STARTED
```

## Resume behavior

If a prediction run is interrupted, run the same species again:

```bash
python src/run_pipeline.py --species Drosophila_simulans
```

Valid existing tile checkpoints are reused.

## Parallel execution

The default species-level parallelism is configured in:

```text
configs/default.yaml
```

Current default:

```yaml
runtime:
  max_parallel_species: 5
```

Override it when needed:

```bash
python src/run_pipeline.py \
  --species Species_A Species_B Species_C \
  --max-parallel 2
```

## Temporary workspace

Temporary data are written under:

```text
work/<species_id>/
```

Cleanup occurs only after permanent outputs pass validation. Cleanup can be disabled with:

```bash
python src/run_pipeline.py \
  --species Drosophila_simulans \
  --no-cleanup
```

## Logs

Per-species stage logs:

```text
results/<species_id>/logs/
```

Parallel scheduler logs:

```text
logs/inventory_runs/<run_id>/
```

## Important interpretation note

AlphaGenome 2 is applied here across genomes beyond the model's original training species. These outputs are cross-species model predictions and should not be interpreted as direct experimental measurements.

The permanent predictions are intended for comparative downstream analyses, including the planned motif-combination analysis.
