#!/usr/bin/env python3
"""Real 1 Mb hg38 HEK293T ATAC-seq pilot."""

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pyBigWig
import yaml

from alphagenome.models.v2 import dna_client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from api_key import load_api_key
from predict_genome import select_atac_tracks

CHROM = "NC_000001.11"
CHROM_LENGTH = 248_956_422

# Zero-based hg38 genomic coordinates.
START = 1_000_000
INPUT_LENGTH = 1_048_576
RETAINED_LENGTH = 524_288

FLANK = (INPUT_LENGTH - RETAINED_LENGTH) // 2

OUTDIR = ROOT / "work" / "atac_hek293t" / "_pilot_hg38_chr1"
OUTDIR.mkdir(parents=True, exist_ok=True)

BIGWIG_PATH = OUTDIR / "ATAC_pilot.bw"

if BIGWIG_PATH.exists():
    raise FileExistsError(
        f"Pilot already exists: {BIGWIG_PATH}"
    )

with (ROOT / "configs/atac_hek293t.yaml").open() as f:
    cfg = yaml.safe_load(f)

condition = cfg["alphagenome"]["condition"]

# Retrieve the exact human reference interval from NCBI.
params = urlencode({
    "db": "nuccore",
    "id": CHROM,
    "seq_start": START + 1,
    "seq_stop": START + INPUT_LENGTH,
    "rettype": "fasta",
    "retmode": "text",
})

url = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/efetch.fcgi?" + params
)

print("Downloading hg38 chromosome 1 pilot sequence...")

request = Request(
    url,
    headers={"User-Agent": "AlphaGenome2-research-pilot/1.0"},
)

with urlopen(request, timeout=120) as response:
    fasta_text = response.read().decode("utf-8")

lines = fasta_text.splitlines()

if not lines or not lines[0].startswith(">"):
    raise RuntimeError("NCBI did not return valid FASTA.")

sequence = "".join(lines[1:]).upper()

if len(sequence) != INPUT_LENGTH:
    raise RuntimeError(
        f"Incorrect sequence length: {len(sequence)}"
    )

if set(sequence) - set("ACGTN"):
    raise RuntimeError("Unexpected characters in FASTA.")

(OUTDIR / "pilot_sequence.fa").write_text(
    f">{CHROM}:{START + 1}-{START + INPUT_LENGTH}\n"
    + "\n".join(
        sequence[i:i + 80]
        for i in range(0, len(sequence), 80)
    )
    + "\n"
)

print("Sequence length:", len(sequence))
print("Condition:", condition)
print("Requesting one real ATAC prediction...")

load_api_key()

api_key = os.environ.get("ALPHA_GENOME_API_KEY")
if not api_key:
    raise RuntimeError("AlphaGenome API key unavailable.")

model = dna_client.create(api_key)

output = model.predict_sequence(
    sequence=sequence,
    requested_outputs=[dna_client.OutputType.ATAC],
    condition=condition,
)

atac = output.atac

if atac is None:
    raise RuntimeError("No ATAC output returned.")

indices, selected = select_atac_tracks(atac, cfg)

if len(indices) != 1:
    raise RuntimeError(
        f"Expected one observed ATAC channel, found {len(indices)}"
    )

crop_start = FLANK
crop_end = FLANK + RETAINED_LENGTH

values = np.asarray(
    atac.values[crop_start:crop_end, indices[0]],
    dtype=np.float32,
)

if len(values) != RETAINED_LENGTH:
    raise RuntimeError("Unexpected prediction length.")

if not np.isfinite(values).all():
    raise RuntimeError("Prediction contains NaN or Inf.")

genomic_start = START + FLANK
genomic_end = genomic_start + RETAINED_LENGTH

# Write genomic coordinates into a BigWig.
partial_path = Path(str(BIGWIG_PATH) + ".partial")

bw = pyBigWig.open(str(partial_path), "w")
try:
    bw.addHeader([(CHROM, CHROM_LENGTH)])
    bw.addEntries(
        CHROM,
        genomic_start,
        values=values.astype(float).tolist(),
        span=1,
        step=1,
    )
finally:
    bw.close()

# Verify all retained values, not just selected positions.
bw = pyBigWig.open(str(partial_path))
try:
    observed = np.asarray(
        bw.values(CHROM, genomic_start, genomic_end),
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        observed, values, rtol=0, atol=1e-6
    )
finally:
    bw.close()

os.replace(partial_path, BIGWIG_PATH)

summary = {
    "pilot": True,
    "assembly": "GRCh38.p14",
    "chromosome": CHROM,
    "input_start_0based": START,
    "input_length": INPUT_LENGTH,
    "retained_start_0based": genomic_start,
    "retained_end_0based": genomic_end,
    "condition": condition,
    "output_type": "ATAC",
    "selected_track": str(selected.iloc[0]["name"]),
    "bigwig": str(BIGWIG_PATH),
    "validation_passed": True,
}

(OUTDIR / "pilot_summary.json").write_text(
    json.dumps(summary, indent=2)
)

print()
print("REAL HG38 ATAC PILOT PASSED")
print("Selected track:", summary["selected_track"])
print("Retained region:", genomic_start, genomic_end)
print("Mean ATAC signal:", float(values.mean()))
print("BigWig:", BIGWIG_PATH)
