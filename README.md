# BiTTP: Bidirectional Consistency-Enhanced Framework for Low-Resource APT Tactic and Technique Attribution

This package contains the framework implementation, the executable prompt
templates of Appendix C, the split-manifest exporter with the released
manifests, and the low-resource and per-run experiment configurations. The full implementations are listed below and will be
published in the public repository upon acceptance.

## Repository layout

```
BiTTP-REVISION/
├── README.md                      (this file)
│── bittp_core.py              (full pipeline: alignment, retrieval,
│                               gates, aggregator, self-training loop,
│                               multi-label / single-label metrics)
│── manifest_exporter.py       (split-manifest exporter, Appendix B schema)
│── prompts/                   (executable prompt templates, Appendix C)
│   ├── forward_generation.txt
│   ├── reverse_generation.txt
│   ├── technique_recognition.txt
│   ├── tactic_reasoning.txt
│   └── cross_view_verification.txt
│── configs/
│       ├── main.yaml              (Table 3 hyperparameters)
│       ├── low_resource.yaml      (Section 5.6, Table 10, Figure 7)
│       └── per_run.yaml           (Appendix E, Table A4)
```

## Environment

- Python 3.10+, PyTorch 2.x, PyTorch Geometric, sentence-transformers,
  scikit-learn, numpy.
- LLM endpoint: local Llama-3.2-11B-Instruct server exposing `/generate`

**Data**
- **Dataset 1** is generated from the public Atomic Red Team repository
  (https://github.com/redcanaryco/atomic-red-team) and the DARPA Transparent
  Computing release, following the bidirectional generation procedure of
  Section 3.1. It contains 12,480 tactical subgraphs over 14 tactics and 187
  techniques. Regeneration instructions: run Stage 1 of `code/bittp_core.py`
  with `code/configs/main.yaml`.
- **Dataset 2** is derived from the public TREC benchmark audit traces
  following the TREC subgraph extraction procedure. It contains 1,036
  tactical subgraphs over 96 techniques.



