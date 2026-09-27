# -*- coding: utf-8 -*-
"""
BiTTP core algorithm reference implementation.

This file implements the full BiTTP pipeline described in the manuscript:

  Stage 1  Bidirectional tactical subgraph construction      (data loader)
  Stage 2  Bidirectional dual-view alignment                 (InfoNCE)
  Stage 3  Retrieval-grounded two-stage reasoning            (LLM interface)
  Stage 4  Consistency-verified iterative self-training      (G1-G3 gates)
  Stage 5  Learned attention aggregator for tactic mapping

The LLM calls are isolated behind `GenerationAnnotator` so the code can be
wired to a local Llama-3.2-11B-Instruct server (see README.md).

Dependencies: torch, torch_geometric, numpy, scikit-learn, sentence_transformers
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------------------------------
# Hyperparameters (Table 3 of the manuscript)
# --------------------------------------------------------------------------
@dataclass
class BiTTPConfig:
    gat_layers: int = 2
    gat_heads: int = 4
    hidden_dim: int = 256
    embed_dim: int = 768
    info_ncetemperature: float = 0.07
    retrieval_k: int = 5
    tau_c: float = 0.90          # G2: confidence threshold
    tau_r: float = 0.82          # G3: cross-view consistency threshold
    temperatures: tuple = (0.4, 0.6, 0.8)   # G1: consensus temperatures
    self_training_rounds: int = 3
    aggregator_epochs: int = 10
    aggregator_lr: float = 1e-3
    device: str = "cuda"


# --------------------------------------------------------------------------
# Stage 2a: subgraph encoder (2-layer GAT over entity/relation embeddings)
# --------------------------------------------------------------------------
class SubgraphEncoder(nn.Module):
    def __init__(self, cfg: BiTTPConfig, num_relations: int):
        super().__init__()
        self.relation_emb = nn.Embedding(num_relations, cfg.hidden_dim)
        self.conv1 = GATConv(cfg.hidden_dim, cfg.hidden_dim // cfg.gat_heads,
                             heads=cfg.gat_heads)
        self.conv2 = GATConv(cfg.hidden_dim, cfg.hidden_dim // cfg.gat_heads,
                             heads=cfg.gat_heads)
        self.proj = nn.Linear(cfg.hidden_dim, cfg.embed_dim)

    def forward(self, x, edge_index, edge_type):
        # relation-conditioned message passing: add relation embedding to src
        x = x + self.relation_emb(edge_type.new_zeros(edge_type.size(0)))[0]
        h = F.elu(self.conv1(x, edge_index))
        h = F.elu(self.conv2(h, edge_index))
        z = self.proj(h.mean(dim=0))          # mean-pool nodes -> graph vector
        return F.normalize(z, dim=-1)


# --------------------------------------------------------------------------
# Stage 2b: definition encoder (frozen sentence encoder wrapper)
# --------------------------------------------------------------------------
class DefinitionEncoder:
    def __init__(self, cfg, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name).eval()
        self.cfg = cfg

    @torch.no_grad()
    def encode(self, definitions):
        """definitions: list[str] -> L2-normalized embedding tensor."""
        z = self.model.encode(definitions, convert_to_tensor=True)
        return F.normalize(z, dim=-1).to(self.cfg.device)


# --------------------------------------------------------------------------
# Stage 2c: bidirectional InfoNCE alignment
# --------------------------------------------------------------------------
def info_nce_loss(z_s, z_t, temperature):
    """z_s, z_t: (B, D) normalized. Symmetric (bidirectional) InfoNCE."""
    logits = z_s @ z_t.T / temperature
    labels = torch.arange(z_s.size(0), device=z_s.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.T, labels))


def train_alignment(subgraph_encoder, loader, definition_encoder, cfg, epochs=30):
    opt = torch.optim.AdamW(subgraph_encoder.parameters(), lr=3e-4)
    for _ in range(epochs):
        for batch, definitions in loader:
            z_s = subgraph_encoder(batch.x, batch.edge_index, batch.edge_type)
            z_t = definition_encoder.encode(definitions)
            loss = info_nce_loss(z_s, z_t, cfg.info_ncetemperature)
            opt.zero_grad(); loss.backward(); opt.step()


# --------------------------------------------------------------------------
# Stage 3: retrieval bank + grounded two-stage reasoning
# --------------------------------------------------------------------------
class DefinitionBank:
    def __init__(self, cfg, definitions: dict):
        """definitions: {technique_id: definition_text}"""
        self.cfg = cfg
        self.ids = list(definitions.keys())
        self.texts = [definitions[i] for i in self.ids]
        self.enc = DefinitionEncoder(cfg)
        self.emb = self.enc.encode(self.texts)
        self.nn = NearestNeighbors(n_neighbors=cfg.retrieval_k, metric="cosine")
        self.nn.fit(self.emb.cpu())

    def retrieve(self, z_s):
        """z_s: single normalized graph embedding -> top-k candidate ids."""
        idx = self.nn.kneighbors(z_s.detach().cpu().reshape(1, -1),
                                 return_distance=False)[0]
        return [self.ids[i] for i in idx]


class GenerationAnnotator:
    """LLM interface. Replace `generate` with a call to your local
    Llama-3.2-11B-Instruct endpoint (see README.md)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self, subgraph_text, candidate_defs, temperature):
        """Return (predicted_technique_id, confidence, reasoning_text)."""
        raise NotImplementedError("wire to local LLM endpoint")

    def predict_proba(self, subgraph_text, candidate_defs, temperature):
        """Softmax over candidate ids at one temperature -> (ids, probs)."""
        raise NotImplementedError("wire to local LLM endpoint")


def two_stage_reasoning(subgraph_text, bank, annotator, cfg):
    """Stage 3: retrieve candidates -> grounded generation + read-back."""
    z_s = bank.encode_subgraph(subgraph_text)            # frozen aligned space
    candidates = bank.retrieve(z_s)
    cand_defs = {c: bank.texts[bank.ids.index(c)] for c in candidates}
    y_hat, conf, rationale = annotator.generate(subgraph_text, cand_defs, 0.6)
    return y_hat, conf, rationale, candidates, z_s


# --------------------------------------------------------------------------
# Stage 4: consistency gates G1 (consensus) / G2 (confidence) / G3 (cross-view)
# --------------------------------------------------------------------------
def consensus_gate(annotator, subgraph_text, cand_defs, cfg):
    """G1: identical argmax label across all decoding temperatures."""
    preds = []
    for t in cfg.temperatures:
        ids, probs = annotator.predict_proba(subgraph_text, cand_defs, t)
        preds.append(ids[int(probs.argmax())])
    return preds if len(set(preds)) == 1 else None, preds


def cross_view_score(z_s, bank, y_hat):
    """G3: cosine between graph embedding and definition embedding."""
    j = bank.ids.index(y_hat)
    return float(z_s @ bank.emb[j].to(z_s.device))


def accept_pseudo_label(subgraph_text, bank, annotator, cfg):
    """Apply G1-G3 conjunctively. Return (x, y, confidence) or None."""
    z_s = bank.encode_subgraph(subgraph_text)
    candidates = bank.retrieve(z_s)
    cand_defs = {c: bank.texts[bank.ids.index(c)] for c in candidates}
    consensus, all_preds = consensus_gate(annotator, subgraph_text, cand_defs, cfg)
    if consensus is None:
        return None
    y_hat = consensus[0]
    ids, probs = annotator.predict_proba(subgraph_text, cand_defs, 0.6)
    conf = float(probs[ids.index(y_hat)]) if y_hat in ids else 0.0
    if conf < cfg.tau_c:                              # G2
        return None
    if cross_view_score(z_s, bank, y_hat) < cfg.tau_r:  # G3
        return None
    return (subgraph_text, y_hat, conf)


# --------------------------------------------------------------------------
# Stage 5: learned attention aggregator (tactic mapping over candidates)
# --------------------------------------------------------------------------
class AttentionAggregator(nn.Module):
    """0.59M params: scores top-k technique candidates, aggregates their
    tactic posteriors with attention over retrieval ranks."""

    def __init__(self, cfg, num_tactics, num_techniques):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(cfg.embed_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.tactic_head = nn.Linear(cfg.embed_dim, num_tactics)
        self.cfg = cfg

    def forward(self, cand_embs, tactic_of):
        """cand_embs: (k, D) retrieved definition embeddings;
        tactic_of: (k,) tactic index of each candidate technique."""
        attn = torch.softmax(self.score(cand_embs), dim=0)       # (k,1)
        tactic_post = torch.softmax(self.tactic_head(cand_embs), dim=-1)
        return (attn * tactic_post).sum(dim=0), attn             # (T,), (k,1)


# --------------------------------------------------------------------------
# Stage 4 (outer loop): consistency-verified iterative self-training
# --------------------------------------------------------------------------
def self_training_round(annotator, bank, pool, cfg, round_idx):
    """One round: relabel the WHOLE fixed pool; each subgraph contributes at
    most one retained pseudo-label. Returns list of (x, y, conf)."""
    retained = []
    for x in pool:                                    # 8,420 subgraphs
        rec = accept_pseudo_label(x, bank, annotator, cfg)
        if rec is not None:
            retained.append(rec)
    return retained


def run_self_training(annotator, bank, pool, seed_train_set, cfg):
    """Lifecycle (Section 3.3 / Appendix F):
      pool is FIXED (8,420); every round relabels the whole pool;
      per-round newly retained labels DECREASE (5,214 / 4,173 / 3,715);
      CUMULATIVE retained labels GROW (5,214 / 9,387 / 13,102);
      the augmented training set deduplicates by subgraph (most recent
      label wins), so cumulative labels can exceed the pool size.
    """
    train_set = list(seed_train_set)
    cumulative = []
    for k in range(cfg.self_training_rounds):
        newly = self_training_round(annotator, bank, pool, cfg, k)
        cumulative.extend(newly)
        by_x = {x: (x, y, c) for (x, y, c) in cumulative}   # dedup, latest wins
        train_set = list(seed_train_set) + list(by_x.values())
        annotator = retrain_annotator(train_set)            # V_{k+1}
    return annotator, train_set


def retrain_annotator(labeled_set):
    """LoRA fine-tune Llama-3.2-11B-Instruct on the augmented set."""
    raise NotImplementedError("wire to local training pipeline")


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def multilabel_tactic_metrics(gold_sets, pred_sets):
    """Tactic task: micro-F1 over class-instance pairs + subset accuracy."""
    tp = fp = fn = 0
    subset_correct = 0
    for g, p in zip(gold_sets, pred_sets):
        g, p = set(g), set(p)
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        subset_correct += int(g == p)
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return micro_f1, subset_correct / len(gold_sets)


def singlelabel_metrics(gold, pred):
    """Technique task: single-label, so micro-F1 == accuracy (verified)."""
    gold = np.asarray(gold); pred = np.asarray(pred)
    acc = float((gold == pred).mean())
    # micro-F1 over classes for single-label reduces to accuracy
    tp = fp = fn = 0
    for c in np.unique(gold):
        tp += int(((pred == c) & (gold == c)).sum())
        fp += int(((pred == c) & (gold != c)).sum())
        fn += int(((pred != c) & (gold == c)).sum())
    micro_f1 = 2 * tp / (2 * tp + fp + fn)
    assert abs(micro_f1 - acc) < 1e-9, "micro-F1 must equal accuracy here"
    return micro_f1
