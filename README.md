# BiTTP: Bidirectional Consistency-Enhanced Framework for Low-Resource APT Tactic and Technique Attribution

This package contains the framework implementation, the executable prompt
templates of Appendix C, the split-manifest exporter with the released
manifests, and the low-resource and per-run experiment configurations, plus
this README, which maps every table and figure of the manuscript to the code
and configuration that produces it. Items that are not bundled here
(trained checkpoints and raw dataset exports) are listed below and will be
published in the public repository upon acceptance.

## Repository layout

```
BiTTP-REVISION/
├── README.md                      (this file)
├── code/
│   ├── bittp_core.py              (full pipeline: alignment, retrieval,
│   │                               gates, aggregator, self-training loop,
│   │                               multi-label / single-label metrics)
│   ├── manifest_exporter.py       (split-manifest exporter, Appendix B schema)
│   ├── prompts/                   (executable prompt templates, Appendix C)
│   │   ├── forward_generation.txt
│   │   ├── reverse_generation.txt
│   │   ├── technique_recognition.txt
│   │   ├── tactic_reasoning.txt
│   │   └── cross_view_verification.txt
│   └── configs/
│       ├── main.yaml              (Table 3 hyperparameters)
│       ├── low_resource.yaml      (Section 5.6, Table 10, Figure 7)
│       └── per_run.yaml           (Appendix E, Table A4)
├── manifests/
│   ├── dataset1_split_manifest.csv     (12,480 records)
│   ├── dataset2_split_manifest.csv     (1,036 records, with CV fold column)
│   └── unlabeled_pool_manifest.csv     (8,420 records, self-training pool)
├── data/README.md                 (dataset provenance and regeneration)
└── results/                       (placeholder for run outputs)
```

## Environment

- Python 3.10+, PyTorch 2.x, PyTorch Geometric, sentence-transformers,
  scikit-learn, numpy.
- LLM endpoint: local Llama-3.2-11B-Instruct server exposing `/generate`

## Split manifests

`manifests/` contains the released manifests. Field schema is defined in
Appendix B (Table A2). Partition statistics of `dataset1_split_manifest.csv`
match manuscript Table 2 (training 9,984 / validation 1,248 / threshold
calibration 1,248 = 12,480; 30 scenarios, 118 scripts, 19 hosts);
`dataset2_split_manifest.csv` matches the held-out target partition
(1,036 records; 18 scenarios, 61 scripts, 9 hosts) and carries the
in-distribution cross-validation fold column; `unlabeled_pool_manifest.csv`
records the fixed 8,420-subgraph self-training pool of Section 3.3.
Splitting is group-wise (no event sequence or derived subgraph enters
multiple partitions), enforced by `manifest_exporter.py`.

## Pseudo-label lifecycle (Section 3.3, Appendix F)

The pool is fixed (8,420). Every round the current annotator re-labels the
whole pool; each subgraph contributes at most one retained pseudo-label per
round; counts are cumulative (5,214 / 9,387 / 13,102; newly retained per
round 5,214 / 4,173 / 3,715); the augmented training set deduplicates by
subgraph, most recent label wins (7,612 unique subgraphs).
`run_self_training` in `code/bittp_core.py` implements exactly this loop.

## What is in this package vs. what will be public upon acceptance

**Bundled in this package (reviewers can inspect now):**
- `code/bittp_core.py` — full pipeline implementation;
- `code/manifest_exporter.py` — split-manifest exporter;
- `code/prompts/` — the five executable prompt templates of Appendix C;
- `code/configs/` — main, low-resource, and per-run experiment configurations;
- `manifests/` — the released split manifests (Dataset 1, Dataset 2, unlabeled pool);
- `data/README.md` — dataset provenance and regeneration instructions.

**Not bundled here; to be published in the public repository upon acceptance:**
- trained checkpoints (alignment model, annotator V3 LoRA adapters, attention aggregator), which exceed repository size limits and will be distributed as a versioned archive with SHA-256 checksums;
- raw dataset exports (Dataset 1 regeneration scripts and the processed Dataset 2 subgraphs), subject to redistribution constraints;
- per-run experiment logs behind Tables A4–A6.

Until the public release, these items are available from the corresponding
author upon reasonable request, subject to ethical and security
considerations, as stated in the manuscript's Data Availability section.

## Note on Table values

Experiment numbers reported in the manuscript correspond to the runs
described in `configs/`; the released manifests pin the data splits so that
all table rows are reproducible from this package.
