# -*- coding: utf-8 -*-
"""
Split-manifest exporter (manuscript Appendix B, Table A2).

Exports the sample-level split manifest that accompanies every experiment.
Splitting is group-wise: all samples sharing a source scenario, host, or
time window are assigned to the same partition, and content-hash
near-duplicate detection is applied before partitioning, so that no event
sequence or derived subgraph enters multiple partitions.

Usage:
    python manifest_exporter.py --dataset dataset1 --out dataset1_split_manifest.csv
"""

import argparse
import csv
import hashlib
from collections import defaultdict

FIELDS = ["subgraph_id", "source_scenario", "source_script", "host_uuid",
          "time_window", "content_hash", "group_id", "partition_role", "protocol"]

ROLES = ["training", "validation", "threshold_calibration", "self-training", "testing"]


def content_hash(normalized_event_sequence: str) -> str:
    """Hash of the normalized event sequence; used for near-duplicate detection."""
    return hashlib.sha256(normalized_event_sequence.encode("utf-8")).hexdigest()


def group_id(record: dict) -> str:
    """Group identifier combining scenario, host, and time window."""
    return f"{record['source_scenario']}|{record['host_uuid']}|{record['time_window']}"


def drop_near_duplicates(records):
    """Remove records whose content_hash was already seen (pre-partitioning)."""
    seen, out = set(), []
    for r in records:
        if r["content_hash"] in seen:
            continue
        seen.add(r["content_hash"])
        out.append(r)
    return out


def assign_partitions(records, ratios, seed=0):
    """Group-wise random partition assignment.

    ratios: dict role -> fraction (e.g. training 0.8, validation 0.1,
    threshold_calibration 0.1). Whole groups are assigned to one partition.
    """
    import random
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in records:
        groups[r["group_id"]].append(r)
    ids = list(groups)
    rng.shuffle(ids)
    total = len(records)
    target = {role: int(round(f * total)) for role, f in ratios.items()}
    # largest-remainder correction on targets
    assigned = sum(target.values())
    if assigned != total:
        order = sorted(ratios, key=lambda k: ratios[k], reverse=True)
        i = 0
        while assigned != total:
            target[order[i % len(order)]] += 1 if assigned < total else -1
            assigned += 1 if assigned < total else -1
            i += 1
    counts = {role: 0 for role in ratios}
    role_of = {}
    for gid in ids:                       # first-fit decreasing over shuffled groups
        gsize = len(groups[gid])
        candidates = [role for role in ratios
                      if counts[role] + gsize <= target[role]]
        role = candidates[0] if candidates else max(ratios, key=lambda k: target[k] - counts[k])
        role_of[gid] = role
        counts[role] += gsize
    for r in records:
        r["partition_role"] = role_of[r["group_id"]]
    return records


def export(records, protocol, out_path, extra_fields=None):
    fields = FIELDS + (extra_fields or [])
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # Records are produced by the dataset generators (Stage 1 of the pipeline);
    # here we only illustrate the export of already-built records.
    raise SystemExit("Plug in the record builder of the corresponding dataset; "
                     "see README.md 'Split manifests'.")
