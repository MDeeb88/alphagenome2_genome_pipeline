
# AlphaGenome 2 Multi-Genome Prediction Pipeline

Reusable pipeline for genome-wide AlphaGenome 2 predictions across a comparative-genomics inventory of 351 species.

The pipeline automates NCBI genome downloads, sequence filtering and tiling, resumable predictions, Zarr consolidation, BigWig generation, validation, and safe cleanup. It supports both RNA-seq and ATAC-seq predictions through separate configurations.

## Environment and execution

Validated server location:

```text
/media/eternus1/nfs/projects/dpanchenko/alphagenome2_genome_pipeline
```

Start the pipeline:

```bash
cd /media/eternus1/nfs/projects/dpanchenko/alphagenome2_genome_pipeline
conda activate alphagenome2
python src/run_pipeline.py
```

Create the environment if needed:

```bash
conda env create -f environment.yml
conda activate alphagenome2
```

## Interactive menu

```text
1. Run selected species (one or multiple)
2. Run one taxonomic group
3. Run all unfinished species
4. Retry failed species
5. Download reference genome(s)
6. Configuration & pipeline status
7. Dry-run / preview
8. AlphaGenome API key
9. Exit
```

Species can be selected by name, search term, or multiple numbers (e.g., `1,2,5`). Use `b` or `back` in supported selection prompts to return to the main menu.

Option 6 switches between prediction configurations and displays the corresponding pipeline status.

## Prediction configurations

Two configurations are available:

| Parameter | RNA-seq | ATAC-seq |
|---|---|---|
| Configuration | `configs/default.yaml` | `configs/atac_hek293t.yaml` |
| Output | `RNA_SEQ` | `ATAC` |
| Condition | `EFO:0002824` | `EFO:0001184` |
| Assay | Total RNA-seq | Chromatin accessibility |
| Cell type | Default configured condition | HEK293T |
| Mode | Observed | Observed |
| Tracks | Plus and minus strands | Unstranded |
| Storage | Zarr and optional BigWigs | Validated BigWig; intermediate Zarr removed |

Both workflows use:

```text
AG2 input length:      1,048,576 bp
Retained center:         524,288 bp
Context per side:        262,144 bp
Minimum contig length:   100,000 bp
```

Contigs shorter than 100 kb are excluded from tiling.

## Current progress

- **RNA-seq:** 12 genomes completed.
- **ATAC-seq:** 14 genomes completed, including *Anopheles gambiae* and the human hg38 control.
- All 14 ATAC BigWigs were validated successfully.
- RNA-seq results remain separate and preserved.
- The 14-genome ATAC run serves as a pilot before scaling to the full inventory.

## Inventory and reference genomes

Main inventory:

```text
inventory/CTCF_Figure7_genome_annotation_inventory.csv
```

ATAC pilot subset:

```text
inventory/atac_hek293t_subset.csv
```

The pipeline uses `species_id` and `ncbi_assembly_accession` to identify and retrieve the correct NCBI assemblies.

To add a species, add a row to the main inventory with the required identifiers and taxonomic group.

The interactive reference downloader (menu option 5) supports multiple species, name searches, and number-based selection. Downloaded FASTA files and available annotations are stored under:

```text
data/igv_references/<species_id>/
```

Reference genomes can also be downloaded independently using:

```bash
python src/download_reference_genomes.py
```

## AlphaGenome API key

An API key is required for new predictions.

Manage it through menu option 8 or configure it for the current shell:

```bash
read -s ALPHA_GENOME_API_KEY
export ALPHA_GENOME_API_KEY
echo
```

Saved keys are stored outside the repository:

```text
~/.config/alphagenome2/api_key
```

The key file uses permission mode `600` and must never be committed to Git.

Official documentation: https://www.alphagenomedocs.com/

## Command-line execution

Preview a species:

```bash
python src/run_pipeline.py --species Drosophila_simulans --dry-run
```

Run multiple species with RNA-seq:

```bash
python src/run_pipeline.py \
  --species Drosophila_simulans Drosophila_yakuba \
  --max-parallel 2
```

Run ATAC-seq:

```bash
python src/run_pipeline.py \
  --config configs/atac_hek293t.yaml \
  --species Drosophila_simulans \
  --max-parallel 2
```

Preview all unfinished species:

```bash
python src/run_pipeline.py --all --dry-run
```

Run all unfinished species with explicit confirmation:

```bash
python src/run_pipeline.py \
  --all --confirm-all --max-parallel 5
```

Use `--config configs/atac_hek293t.yaml` to select ATAC-seq in CLI runs.

## Pipeline workflow

```text
Inventory lookup
       ↓
NCBI genome download
       ↓
FASTA indexing and sequence filtering
       ↓
AlphaGenome-compatible tiling
       ↓
Resumable AlphaGenome 2 predictions
       ↓
Per-tile NPZ checkpoints
       ↓
Zarr consolidation
       ↓
BigWig generation and validation
       ↓
Cleanup after successful validation
```

Failed or interrupted runs retain valid checkpoints for resumption.

## Output structure

### RNA-seq

```text
results/<species_id>/
├── predictions/predictions.zarr/
├── metadata/
├── tracks/
│   ├── totalRNA_plus.bw
│   ├── totalRNA_minus.bw
│   ├── chrom.sizes
│   ├── chrom_aliases.tsv
│   └── track_build_summary.json
└── logs/
```

RNA-seq Zarr stores base-resolution predictions for the plus and minus genomic strands.

### ATAC-seq

```text
results/atac_hek293t/<species_id>/
├── metadata/
├── tracks/
│   ├── <species_id>_HEK293T_ATAC.bw
│   ├── chrom.sizes
│   ├── chrom_aliases.tsv
│   └── track_build_summary.json
└── logs/
```

ATAC-seq produces one unstranded BigWig for the selected observed HEK293T track. Filenames include the species and assay to simplify identification in IGV.

After validation, intermediate consolidated predictions can be deleted to reduce storage usage. Metadata and final BigWigs are retained.

Temporary work and logs are separated by configuration:

```text
work/<species_id>/
work/atac_hek293t/<species_id>/
logs/atac_hek293t/
```

## ATAC-seq signal analysis

The pilot analysis compares predicted ATAC-seq signals across 14 genomes using human hg38 as a control.

For each genome:

- Sample 20,000 random genomic windows of 250 bp.
- Calculate mean predicted ATAC signal per window.
- Apply `log1p` for visualization.
- Compare distributions using histograms, ECDFs, and boxplots.

Total sampled windows: **280,000**.

Analysis code:

```text
src/analyze_atac_signal.py
notebooks/atac_signal_distribution.ipynb
```

Outputs:

```text
analysis/atac_hek293t/
├── sampled_regions.csv
├── signal_summary.csv
├── sampling_settings.json
└── figures/
```

## Validation, status, and resumption

Available statuses:

```text
COMPLETE
FAILED
INCOMPLETE
NOT_STARTED
```

Check status through menu option 6 or:

```bash
python src/run_pipeline.py --status
```

To inspect ATAC status:

```bash
python src/run_pipeline.py \
  --config configs/atac_hek293t.yaml --status
```

Interrupted runs can be resumed by selecting the same species and configuration. Valid checkpoints are reused.

Integration tests:

```bash
python tests/test_integration_tracks.py
```

Validated cases include RNA-seq (2 channels), ATAC-seq (1 channel), and ATAC-seq (2 channels), including BigWig validation and intermediate-data cleanup.

## Scientific interpretation

AlphaGenome 2 predictions are applied to species beyond the model's primary training organisms. Cross-species differences in predicted signal are exploratory and should not be treated as direct experimental measurements.

Experimental validation is required before drawing biological conclusions about chromatin accessibility or regulatory activity.
