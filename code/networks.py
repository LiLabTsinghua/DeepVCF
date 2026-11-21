"""DeepVCF neural network modules aligned with the paper terminology."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import normalize


# =============================================================================
# Deep Representation Network (DRN)
# =============================================================================

class DeepRepresentationNetwork(nn.Module):
    """Deep Representation Network (DRN) with DistMult scoring.

    Stores one FoldUnit's node embeddings and scores triples via DistMult.
    """

    def __init__(
            self,
            num_nodes: int,
            num_relations: int,
            hidden_dim: int):
        """Initialize the Deep Representation Network.

        Args:
            num_nodes (int): Number of nodes in the graph.
            num_relations (int): Number of relations in the graph.
            hidden_dim (int): Dimensionality of the node embeddings.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.emb_dim = hidden_dim

        self.node_emb = Parameter(torch.empty(num_nodes, self.emb_dim))
        self.decoder = DistMult(num_relations, self.emb_dim)
        # DistMult initializes itself during construction; initialize the node
        # embeddings separately to preserve the original RNG draw order.
        self._reset_node_parameters()

    def _reset_node_parameters(self):
        """Reset only the node embeddings."""
        torch.nn.init.xavier_uniform_(self.node_emb)

    def reset_parameters(self):
        """Reset all learnable parameters of the module."""
        self._reset_node_parameters()
        self.decoder.reset_parameters()

    def kge_forward(self, edge_index: Tensor, edge_type: Tensor):
        """Compute triple scores using the DistMult decoder.

        Args:
            edge_index (Tensor): Shape (2, E), edge indices.
            edge_type (Tensor): Shape (E,), relation types.

        Returns:
            Tensor: Scores for each edge.
        """
        x = self.node_emb
        score = self.decoder(x, edge_index, edge_type)
        return score


# =============================================================================
# DistMult Decoder
# =============================================================================

class DistMult(torch.nn.Module):
    """DistMult: score(h,r,t) = <h, r, t> (element-wise product summed)."""
    def __init__(self, num_relations: int, hidden_channels: int):
        """Initialize a DistMult decoder for knowledge graph embeddings.

        Args:
            num_relations (int): Number of relations in the graph.
            hidden_channels (int): Dimensionality of the relation embeddings.
        """
        super().__init__()
        self.rel_emb = Parameter(torch.empty(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z: Tensor, edge_index: Tensor, edge_type: Tensor):
        """Compute scores using the DistMult scoring function: score(h,r,t) = <h, r, t>.

        Args:
            z (Tensor): Node embeddings.
            edge_index (Tensor): Shape (2, E), edge indices.
            edge_type (Tensor): Shape (E,), relation types.

        Returns:
            Tensor: Scores for each edge.
        """
        z = normalize(z, p=2, dim=1)
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)


# =============================================================================
# Target Prediction Network
# =============================================================================

class TargetPredictionNetwork(nn.Module):
    """Target Prediction Network (TPN).

    Three stages map a pair of DRN embeddings to a perturbation-outcome
    probability:

    1. Augmentation layer (``augment_*``): projects the source and destination
       embeddings and three interaction features (difference, squared
       difference, element-wise product), then concatenates them.
    2. Fusion layer (``fusion_layer``): an MLP that fuses the concatenated
       augmentation features into a single hidden representation.
    3. Classification layer (``cls_layer``): separate heads read the same
       fused representation and produce one logit per task relation and one
       for the background class.
    """

    def __init__(self, hidden_dim: int, hidden_channels: int, task_rel: Tensor,
                 dropout_prob: float = 0.1, ln: bool = True):
        """Initialize the Target Prediction Network.

        Args:
            hidden_dim (int): Dimensionality of the input DRN embeddings.
            hidden_channels (int): Dimensionality of the internal hidden layers.
            task_rel (Tensor): Exactly two distinct task-relation indices. The
                third output class represents background/negative examples.
            dropout_prob (float): Dropout probability applied in projection layers.
            ln (bool): Whether to use LayerNorm in projection layers.

        Raises:
            ValueError: If ``hidden_channels`` is smaller than 3, or if
                ``task_rel`` does not contain exactly two distinct relation
                indices.
        """
        super().__init__()

        if hidden_channels < 3:
            raise ValueError(
                'TargetPredictionNetwork requires hidden_channels >= 3.'
            )
        if (task_rel is None or task_rel.ndim != 1 or task_rel.numel() != 2
                or torch.unique(task_rel).numel() != 2):
            raise ValueError(
                'TargetPredictionNetwork requires exactly two distinct '
                'task-relation indices.'
            )

        inter_dim = hidden_channels // 3

        def _proj_block(out_dim):
            return nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.LayerNorm(out_dim) if ln else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob),
            )

        # Augmentation layer
        # Source/destination projections plus three interaction features.
        self.augment_src    = _proj_block(hidden_channels)
        self.augment_dst    = _proj_block(hidden_channels)
        self.augment_diff   = _proj_block(inter_dim)   # kge_src - kge_dst
        self.augment_sqdiff = _proj_block(inter_dim)   # (kge_src - kge_dst) ** 2
        self.augment_prod   = _proj_block(inter_dim)   # kge_src * kge_dst

        # Fusion layer
        # Concat layout: src + dst + 3 × inter → 2*hidden_channels + 3*inter_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_channels * 2 + inter_dim * 3, hidden_channels),
            nn.LayerNorm(hidden_channels) if ln else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels) if ln else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )

        # Classification layer
        # Two task-relation classes plus one background/negative class.
        self.num_classes = 3

        def _make_cls_layer():
            return nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 3),
                    nn.LayerNorm(hidden_channels // 3) if ln else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_prob),
                    nn.Linear(hidden_channels // 3, 1),
                )
                for _ in range(self.num_classes)
            ])

        self.cls_layer = _make_cls_layer()

        # Runtime cache derived from DRN parameters. It follows device moves
        # but is excluded from state_dict to avoid duplicating DRN embeddings.
        self.register_buffer('kge_emb', torch.empty(0), persistent=False)

        # Edge type → output-class index. Kept as a dict for readability and
        # mirrored in an integer LUT buffer for fast batched lookup.
        self.edge_type_to_output_index = self.get_output_index(task_rel)
        lut_size = int(task_rel.max().item()) + 1
        lut = torch.full((lut_size,), -1, dtype=torch.long)
        for et, out_idx in self.edge_type_to_output_index.items():
            lut[et] = out_idx
        self.register_buffer('rel_to_output_idx', lut)

    def reset_cls_layer(self):
        """Reset only the classification layer parameters.

        The augmentation and fusion layers retain their learned parameters.
        """
        for module in self.cls_layer.modules():
            reset_parameters = getattr(module, 'reset_parameters', None)
            if reset_parameters is not None:
                reset_parameters()

    def get_output_index(self, rel_task: Tensor):
        """Build a mapping from edge type integer to output logit index.

        Args:
            rel_task (Tensor): Two task-relation indices.

        Returns:
            dict: Mapping from edge type int to output column index.
        """
        unique_edge_types = torch.unique(rel_task, sorted=True)
        return {et.item(): idx for idx, et in enumerate(unique_edge_types)}

    def _augment(self, edge_index: Tensor) -> list:
        """Augmentation layer: project src/dst embeddings and interaction features.

        Args:
            edge_index (Tensor): Shape [2, E], head and tail node indices.

        Returns:
            list[Tensor]: Five augmentation features (src, dst, diff, squared
            diff, product) to be concatenated by the caller.
        """
        kge_emb = self.kge_emb
        kge_src = kge_emb[edge_index[0]]
        kge_dst = kge_emb[edge_index[1]]
        diff = kge_src - kge_dst

        return [
            self.augment_src(kge_src),
            self.augment_dst(kge_dst),
            self.augment_diff(diff),
            self.augment_sqdiff(diff ** 2),
            self.augment_prod(kge_src * kge_dst),
        ]

    def _fuse(self, concat: Tensor) -> Tensor:
        """Fusion layer: fuse the concatenated augmentation features.

        Args:
            concat (Tensor): Concatenated augmentation features, shape
                [E, 2*hidden_channels + 3*inter_dim].

        Returns:
            Tensor: Fused hidden representation, shape [E, hidden_channels].
        """
        return self.fusion_layer(concat)

    def _classify(self, fused: Tensor) -> Tensor:
        """Apply each classification head to the fused representation.

        Args:
            fused (Tensor): Fused hidden representation, shape [E, hidden_channels].

        Returns:
            Tensor: Raw logits, shape [E, num_classes].
        """
        return torch.cat([head(fused) for head in self.cls_layer], dim=1)

    def forward_logits(self, edge_index: Tensor) -> Tensor:
        """Return raw class logits (before softmax) for cross-entropy training.

        Args:
            edge_index (Tensor): Shape [2, E], head and tail node indices.

        Returns:
            Tensor: Raw logits, shape [E, num_classes].
        """
        concat = torch.cat(self._augment(edge_index), dim=1)
        fused = self._fuse(concat)
        return self._classify(fused)

    def forward(self, edge_index: Tensor) -> Tensor:
        """Return the complete three-class probability distribution.

        Computes per-class logits via the augment → concat → fuse → classify
        pipeline and applies softmax. Columns 0 and 1 correspond to the two
        task relations according to ``edge_type_to_output_index``; column 2 is
        the background/negative class.

        Args:
            edge_index (Tensor): Shape [2, E], head and tail node indices.

        Returns:
            Tensor: Three-class probabilities with shape [E, 3].
        """
        logits = self.forward_logits(edge_index)
        return F.softmax(logits, dim=1)

    def predict_edge_probability(
            self,
            edge_index: Tensor,
            edge_type: Tensor) -> Tensor:
        """Return the probability corresponding to each edge's relation.

        Args:
            edge_index (Tensor): Shape [2, E], head and tail node indices.
            edge_type (Tensor): Shape [E], queried relation type per edge.

        Returns:
            Tensor: Queried-relation probabilities with shape [E, 1].

        Raises:
            ValueError: If edge shapes are inconsistent or ``edge_type``
                contains a relation that is not one of the TPN task relations.
        """
        probabilities = self.forward(edge_index)
        edge_type = edge_type.to(probabilities.device)
        if edge_type.ndim != 1 or edge_type.numel() != edge_index.size(1):
            raise ValueError('edge_type must have one entry per edge.')
        if edge_type.numel() and (
                int(edge_type.min()) < 0
                or int(edge_type.max()) >= self.rel_to_output_idx.numel()):
            raise ValueError('edge_type contains an index outside the TPN lookup.')

        class_idx = self.rel_to_output_idx[edge_type]
        if class_idx.numel() and torch.any(class_idx < 0):
            raise ValueError(
                'edge_type contains a relation that is not a TPN task relation.'
            )

        # For each edge, pick the probability of its queried task relation.
        rows = torch.arange(probabilities.size(0), device=probabilities.device)
        queried_probability = probabilities[rows, class_idx]
        return queried_probability.unsqueeze(1)
