"""The reranker score_paths scores every path with.

A transformer over a path as a sequence,

    [CLS]  e0  r1  e1  r2  e2 ... rn  en

each entity token a projection of its node's name and category embeddings
side by side, each relation token a projection of its hop's phrase, with
learned position and type embeddings. A head reads [CLS]'s output beside
the path's scalars (`SCALAR_NAMES`) and gives one score. Every path length
shares the model; each length's scores are read on one scale, its mean and
spread over the paths the model trained on (`length_scale`).
"""

from __future__ import annotations

import io
from pathlib import Path

import torch
from torch import Tensor, nn

# Scale for the hop count, so it sits O(1) next to the other scalars.
N_HOPS_SCALE = 8.0

# The per-path scalars the head reads, in order. The retrieval cost and the
# off-backbone count are never given a deployed model, and stay zero.
SCALAR_NAMES = [
    "n_hops_scaled",
    "g_cost",
    "g_present",
    "off_backbone_hops",
    # from the node context table
    "deg_out_max", "deg_out_mean",
    "deg_in_max", "deg_in_mean",
    "ic_mean", "ic_present_frac",
    # from the edge context table; mean = support along the path, min = the
    # weakest hop
    "edge_count_mean", "edge_count_min",
    "pubs_mean", "pubs_min",
    "primary_sources_mean", "primary_sources_min",
    "distinct_sources_mean", "distinct_sources_min",
    "attributes_seen_mean", "attributes_seen_min",
]

#  What a training run saves its model as, under its output directory
CHECKPOINT = "best.pt"


class PathSequenceScorer(nn.Module):
    """The transformer over `[CLS] e0 r1 e1 ... rn en` and its head.

    An n-hop path fills positions 0..2n+1 and the padding after it is never
    read, so a path's score is independent of its batch.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        head_hidden: int = 256,
        n_scalars: int = len(SCALAR_NAMES),
        metapath_vocab_size: int = 0,
        metapath_dim: int = 32,
        max_positions: int = 64,
    ):
        super().__init__()
        if metapath_vocab_size:
            raise ValueError("a metapath embedding cannot be read from a "
                             "message, so this reranker has none")
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.n_scalars = n_scalars
        self.max_positions = max_positions

        self.node_proj = nn.Linear(2 * embed_dim, d_model)
        self.edge_proj = nn.Linear(embed_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(max_positions, d_model) * 0.02)
        # 0 = ENTITY, 1 = RELATION
        self.type_embed = nn.Parameter(torch.randn(2, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.Linear(d_model + n_scalars, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        node_x: Tensor,        # (B, max_hops+1, 2*embed_dim)
        edge_x: Tensor,        # (B, max_hops, embed_dim)
        n_hops: Tensor,        # (B,) int
        scalars: Tensor,       # (B, n_scalars)
        metapath_ids: Tensor,  # (B,) int, unread
    ) -> Tensor:
        return self.score_tokens(self.node_proj(node_x), self.edge_proj(edge_x),
                                 n_hops, scalars, metapath_ids)

    @torch.jit.export
    def score_tokens(
        self,
        node_t: Tensor,        # (B, max_hops+1, d_model)
        edge_t: Tensor,        # (B, max_hops, d_model)
        n_hops: Tensor,        # (B,) int
        scalars: Tensor,       # (B, n_scalars)
        metapath_ids: Tensor,  # (B,) int, unread
    ) -> Tensor:
        """The scores of paths whose nodes and hops are already projected."""
        b, n_nodes, _ = node_t.shape
        max_hops = edge_t.shape[1]
        seq_len = 1 + n_nodes + max_hops  # == 2*max_hops + 2
        if seq_len > self.max_positions:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_positions="
                f"{self.max_positions}; raise max_positions for paths this long")

        h = node_t.new_zeros((b, seq_len, self.d_model))
        h[:, 0] = self.cls_token
        h[:, 1::2] = node_t + self.type_embed[0]
        h[:, 2::2] = edge_t + self.type_embed[1]
        h = h + self.pos_embed[:seq_len]

        # An n-hop path fills positions 0..2n+1, so padding is the suffix.
        positions = torch.arange(seq_len, device=h.device)
        pad_mask = positions.unsqueeze(0) >= (2 * n_hops + 2).unsqueeze(1)
        if self.training:
            h = self.encoder(h, src_key_padding_mask=pad_mask)
        else:
            # Only [CLS] is read, so the last layer gives [CLS] alone: the
            # same pre-norm GELU layer, attending to every position, without
            # its dropout.
            last = len(self.encoder.layers) - 1
            for i, layer in enumerate(self.encoder.layers):
                if i < last:
                    h = layer(h, src_key_padding_mask=pad_mask)
                else:
                    x = layer.norm1(h)
                    h = h[:, :1] + layer.self_attn(
                        x[:, :1], x, x, key_padding_mask=pad_mask,
                        need_weights=False)[0]
                    h = h + layer.linear2(nn.functional.gelu(
                        layer.linear1(layer.norm2(h))))
        return self.head(torch.cat([h[:, 0], scalars], dim=1)).squeeze(-1)


def save_scorer(path: Path, model: PathSequenceScorer, model_kwargs: dict,
                extra: dict) -> None:
    """Write the weights and everything needed to rebuild the scorer."""
    torch.save({"model": model.state_dict(), "model_kwargs": model_kwargs,
                "scalar_names": SCALAR_NAMES, "metapath_vocab": {}, **extra},
               path)


def load_scorer(path: Path, device: str = "cpu") -> tuple[PathSequenceScorer, dict]:
    """`(model, checkpoint)` from a saved scorer."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    kwargs = ckpt["model_kwargs"]
    if kwargs.get("encoder", "transformer") != "transformer":
        raise SystemExit(f"{path} reads its paths with a {kwargs['encoder']}; "
                         "only the transformer is served")
    model = PathSequenceScorer(**{k: v for k, v in kwargs.items()
                                  if k != "encoder"}).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def deployment_problems(ckpt: dict) -> list[str]:
    """Why a checkpoint cannot score messages, or nothing: the worker reads
    these scalars and both context tables, and no metapath embedding or
    retrieval cost, and needs a length scale."""
    trained = ckpt.get("args") or {}
    problems = []
    if list(ckpt.get("scalar_names") or []) != list(SCALAR_NAMES):
        problems.append(f"it reads the scalars {ckpt.get('scalar_names')}")
    if ckpt["model_kwargs"]["metapath_vocab_size"]:
        problems.append("it reads a metapath embedding")
    if trained.get("backbone") or trained.get("weight_cache"):
        problems.append("it reads the retrieval cost g, which a message "
                        "does not carry")
    if "length_scale" not in ckpt:
        problems.append("it has no length scale to rank lengths together")
    for name in ("node_context", "edge_context"):
        if not trained.get(name):
            problems.append(f"it was trained without the {name.replace('_', ' ')}")
    return problems


def scripted(model: PathSequenceScorer) -> bytes:
    """The model as TorchScript, as the worker loads it with torch.jit.load."""
    buffer = io.BytesIO()
    torch.jit.save(torch.jit.script(model.cpu().eval()), buffer)
    return buffer.getvalue()
