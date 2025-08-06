from typing import Optional, Tuple
import random
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter
import torch.nn as nn
from torch_geometric.nn import RGCNConv, RGATConv
from torch.nn.functional import normalize

class KGE_Model(torch.nn.Module):
    def __init__(self, 
                 num_nodes: int,
                 num_relations: int,
                 hidden_dim: int, # for node emb
                 decoder_name: str = 'DistMult'):
        """
        Initializes a Knowledge Graph Embedding (KGE) model.

        Args:
            num_nodes (int): Number of nodes in the graph.
            num_relations (int): Number of relations in the graph.
            hidden_dim (int): Dimensionality of the node embeddings.
            decoder_name (str, optional): Name of the decoder to use ('DistMult', 'TransE', etc.). Defaults to 'DistMult'.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.decoder_name = decoder_name
        
        # Initialize node embeddings
        self.node_emb = Parameter(torch.empty(num_nodes, hidden_dim))
        
        # Select and initialize decoder
        self.decoder = self.select_decoder(decoder_name, num_relations, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.node_emb)

    def select_decoder(self, decoder_name: str, num_relations: int, hidden_channel: int):
        """
        Selects and initializes the decoder based on the given name.

        Args:
            decoder_name (str): The name of the decoder to use.
            num_relations (int): Number of relations in the graph.
            hidden_channel (int): Dimensionality of the hidden layer.

        Returns:
            Module: Initialized decoder.
        """
        if decoder_name == 'DistMult':
            return DistMultDecoder(num_relations, hidden_channel)
        elif decoder_name == 'TransE':
            return TransEDecoder(num_relations, hidden_channel)
        elif decoder_name == 'RotatE':
            return RotatEDecoder(num_relations, hidden_channel)
        elif decoder_name == 'ComplEx':
            return ComplExDecoder(num_relations, hidden_channel)
        else:
            raise ValueError(f"Unknown decoder: {decoder_name}")

    def kge_forward(self, edge_index: Tensor, edge_type: Tensor):
        """
        Forward pass for computing scores using the selected decoder.

        Args:
            edge_index (Tensor): Indices of the edges.
            edge_type (Tensor): Types of the edges.

        Returns:
            Tensor: Scores for each edge.
        """
        x = self.node_emb
        score = self.decoder(x, edge_index, edge_type)
        return score


class DeepME_Model(torch.nn.Module):
    def __init__(self, 
                 num_nodes: int,
                 hidden_dim: int, # for node emb
                 hidden_channels: int,
                 num_relations: int,
                 num_rgcn_layers: int = 2,
                 decoder_name: str = 'TransE',
                 dropout: float = 0.1,
                 task_rel: Optional[Tensor] = None,
                 kge_emb: Optional[Tensor] = None,
                 rand_emb: Optional[Tensor] = None,
                 pcm_emb: Optional[Tensor] = None):
        """
        Initializes a Deep Multi-Embedding (DeepME) model.

        Args:
            num_nodes (int): Number of nodes in the graph.
            hidden_dim (int): Dimensionality of the node embeddings.
            hidden_channels (int): Number of hidden channels.
            num_relations (int): Number of relations in the graph.
            num_rgcn_layers (int, optional): Number of RGCN layers. Defaults to 2.
            decoder_name (str, optional): Name of the decoder to use. Defaults to 'TransE'.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            task_rel (Optional[Tensor], optional): Task-specific relations. Defaults to None.
            kge_emb (Optional[Tensor], optional): Pretrained KGE embeddings. Defaults to None.
            rand_emb (Optional[Tensor], optional): Random embeddings. Defaults to None.
            pcm_emb (Optional[Tensor], optional): PCM embeddings. Defaults to None.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.task_rel = task_rel
        self.decoder_name = decoder_name
        
        # Select and initialize decoder
        self.decoder = self.select_decoder(decoder_name, num_relations, hidden_channels, kge_emb=kge_emb, rand_emb=rand_emb, pcm_emb=pcm_emb)

    def select_decoder(self, decoder_name: str, num_relations: int, hidden_channel: int, **kwargs):
        """
        Selects and initializes the decoder based on the given name and additional arguments.

        Args:
            decoder_name (str): The name of the decoder to use.
            num_relations (int): Number of relations in the graph.
            hidden_channel (int): Dimensionality of the hidden layer.
            **kwargs: Additional keyword arguments specific to each decoder.

        Returns:
            Module: Initialized decoder.
        """
        if decoder_name == 'kge_emb':
            return Decoder0(hidden_channel, dropout_prob=self.dropout, task_rel=self.task_rel, kge_emb=kwargs['kge_emb'])
        elif decoder_name == 'rand_emb':
            return Decoder1(hidden_channel, dropout_prob=self.dropout, task_rel=self.task_rel, rand_emb=kwargs['rand_emb'])
        elif decoder_name == 'pcm_emb':
            return Decoder2(hidden_channel, dropout_prob=self.dropout, task_rel=self.task_rel, pcm_emb=kwargs['pcm_emb'])
        elif decoder_name == 'wo_distance':
            return Decoder3(hidden_channel, dropout_prob=self.dropout, task_rel=self.task_rel, kge_emb=kwargs['kge_emb'])
        else:
            raise ValueError(f"Unknown decoder: {decoder_name}")

    def deepme_forward(self, edge_index: Tensor, edge_type: Tensor, ms_edge_index: Tensor, ms_edge_type: Tensor, return_embedding: bool = False):
        """
        Forward pass for computing scores using the selected decoder.

        Args:
            edge_index (Tensor): Indices of the edges.
            edge_type (Tensor): Types of the edges.
            ms_edge_index (Tensor): Additional edge indices.
            ms_edge_type (Tensor): Additional edge types.
            return_embedding (bool, optional): Whether to return embeddings. Defaults to False.

        Returns:
            Tensor: Scores for each edge.
        """
        score = self.decoder(edge_index, edge_type)
        return score

# KGE Decoders implementation
class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations: int, hidden_channels: int):
        """
        Initializes a DistMult decoder for knowledge graph embeddings.

        Args:
            num_relations (int): Number of relations in the graph.
            hidden_channels (int): Dimensionality of the hidden layer.
        """
        super().__init__()
        self.rel_emb = Parameter(torch.empty(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z: Tensor, edge_index: Tensor, edge_type: Tensor):
        """
        Forward pass to compute scores using the DistMult scoring function.

        Args:
            z (Tensor): Node embeddings.
            edge_index (Tensor): Indices of the edges.
            edge_type (Tensor): Types of the edges.

        Returns:
            Tensor: Scores for each edge.
        """
        z = normalize(z, p=2, dim=1)
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)


class TransEDecoder(torch.nn.Module):
    def __init__(self, num_relations: int, hidden_channels: int):
        """
        Initializes a TransE decoder for knowledge graph embeddings.

        Args:
            num_relations (int): Number of relations in the graph.
            hidden_channels (int): Dimensionality of the hidden layer.
        """
        super().__init__()
        self.rel_emb = Parameter(torch.empty(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def l2_dissimilarity(self, a: Tensor, b: Tensor):
        """Compute dissimilarity between rows of `a` and `b` as ||a-b||_2^2."""
        assert len(a.shape) == len(b.shape)
        return (a - b).norm(p=2, dim=-1) ** 2

    def forward(self, z: Tensor, edge_index: Tensor, edge_type: Tensor):
        """
        Forward pass to compute scores using the TransE scoring function.

        Args:
            z (Tensor): Node embeddings.
            edge_index (Tensor): Indices of the edges.
            edge_type (Tensor): Types of the edges.

        Returns:
            Tensor: Scores for each edge.
        """
        z = normalize(z, p=2, dim=1)
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return -self.l2_dissimilarity(z_src + z_dst, rel)


class RotatEDecoder(torch.nn.Module):
    def __init__(self, num_relations: int, hidden_channels: int):
        """
        Initializes a RotatE decoder for knowledge graph embeddings.

        Args:
            num_relations (int): Number of relations in the graph.
            hidden_channels (int): Dimensionality of the hidden layer.
        """
        super().__init__()
        self.phase_rel = torch.nn.Parameter(torch.empty(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        torch.nn.init.uniform_(self.phase_rel, -3.141592653589793, 3.141592653589793)

    def forward(self, z: Tensor, edge_index: Tensor, edge_type: Tensor):
        """
        Forward pass to compute scores using the RotatE scoring function.

        Args:
            z (Tensor): Node embeddings.
            edge_index (Tensor): Indices of the edges.
            edge_type (Tensor): Types of the edges.

        Returns:
            Tensor: Scores for each edge.
        """
        z = normalize(z, p=2, dim=1)
        z = torch.view_as_complex(torch.stack([z, torch.zeros_like(z)], dim=-1))
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        phase_rel = torch.view_as_complex(torch.stack([torch.cos(self.phase_rel), torch.sin(self.phase_rel)], dim=-1))
        rel = phase_rel[edge_type]
        return torch.real(torch.sum(z_src * rel * torch.conj(z_dst), dim=1))


class ComplExDecoder(torch.nn.Module):
    def __init__(self, num_relations: int, hidden_channels: int):
        """
        Initializes a ComplEx decoder for knowledge graph embeddings.

        Args:
            num_relations (int): Number of relations in the graph.
            hidden_channels (int): Dimensionality of the hidden layer.
        """
        super().__init__()
        self.rel_re = torch.nn.Parameter(torch.empty(num_relations, hidden_channels // 2))
        self.rel_im = torch.nn.Parameter(torch.empty(num_relations, hidden_channels // 2))
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        torch.nn.init.xavier_uniform_(self.rel_re)
        torch.nn.init.xavier_uniform_(self.rel_im)

    def forward(self, z: Tensor, edge_index: Tensor, edge_type: Tensor):
        """
        Forward pass to compute scores using the ComplEx scoring function.

        Args:
            z (Tensor): Node embeddings.
            edge_index (Tensor): Indices of the edges.
            edge_type (Tensor): Types of the edges.

        Returns:
            Tensor: Scores for each edge.
        """
        z = normalize(z, p=2, dim=1)
        z_re, z_im = torch.chunk(z, 2, dim=-1)
        z_src_re, z_src_im = z_re[edge_index[0]], z_im[edge_index[0]]
        z_dst_re, z_dst_im = z_re[edge_index[1]], z_im[edge_index[1]]
        rel_re, rel_im = self.rel_re[edge_type], self.rel_im[edge_type]
        score = (z_src_re * rel_re - z_src_im * rel_im) * z_dst_re + \
                (z_src_re * rel_im + z_src_im * rel_re) * z_dst_im
        return torch.sum(score, dim=1)


# TASK NEURAL NETWORK Decoders implementation
class Decoder0(nn.Module):
    def __init__(self, hidden_channels: int, dropout_prob: float = 0.1, ln: bool = True,
                 task_rel: Optional[Tensor] = None, kge_emb: Optional[Tensor] = None):
        """
        Base decoder that uses KGE (Knowledge Graph Embedding) for edge prediction.

        Args:
            hidden_channels (int): Dimension of the hidden layer.
            dropout_prob (float): Dropout probability.
            ln (bool): Whether to use LayerNorm.
            task_rel (Optional[Tensor]): Task-specific relation types.
            kge_emb (Optional[Tensor]): Pretrained KGE embeddings.
        """
        super().__init__()
        self.device = kge_emb.device if kge_emb is not None else 'cpu'

        # Feature interaction projections
        self.inter_projection_1 = nn.Sequential(
            nn.Linear(hidden_channels, int(hidden_channels / 3)),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(hidden_channels / 3)) if ln else nn.Identity()
        )
        self.inter_projection_2 = nn.Sequential(
            nn.Linear(hidden_channels, int(hidden_channels / 3)),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(hidden_channels / 3)) if ln else nn.Identity()
        )
        self.inter_projection_3 = nn.Sequential(
            nn.Linear(hidden_channels, int(hidden_channels / 3)),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(hidden_channels / 3)) if ln else nn.Identity()
        )

        # Source and destination node feature projectors
        self.src_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_channels) if ln else nn.Identity()
        )
        self.dst_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_channels) if ln else nn.Identity()
        )
        self.rxn_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_channels) if ln else nn.Identity()
        )
        # Final MLP to combine all features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_channels) if ln else nn.Identity(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 3)
        )

        self.kge_emb = kge_emb
        self.edge_type_to_output_index = self.get_output_index(task_rel)

    def get_output_index(self, rel_task: Tensor):
        """
        Maps each unique edge type to an index in output logits.
        """
        if rel_task is None:
            return {}
        unique_edge_types = torch.unique(rel_task, sorted=True)
        return {et.item(): idx for idx, et in enumerate(unique_edge_types)}

    def forward(self, edge_index: Tensor, edge_type: Tensor):
        """
        Forward pass using KGE embeddings.

        Args:
            edge_index (Tensor): Shape [2, E], source and destination indices.
            edge_type (Tensor): Shape [E], edge types.

        Returns:
            Tensor: Predicted probabilities for selected edge types.
        """
        kge_emb = self.kge_emb
        src_idx, dst_idx = edge_index[0], edge_index[1]
        kge_src = kge_emb[src_idx]
        kge_dst = kge_emb[dst_idx]

        inter1 = self.inter_projection_1(kge_src - kge_dst)
        inter2 = self.inter_projection_2((kge_src - kge_dst) ** 2)
        inter3 = self.inter_projection_3(kge_src * kge_dst)

        src_x = self.src_projection(kge_src)
        dst_x = self.dst_projection(kge_dst)

        concat_x = torch.cat([src_x, dst_x, inter1, inter2, inter3], dim=1)
        output = self.fusion_mlp(concat_x)
        probs = F.softmax(output, dim=1)

        mapped_indices = torch.tensor([self.edge_type_to_output_index[e.item()] for e in edge_type],
                                      device=kge_emb.device)
        p = probs[torch.arange(probs.size(0)), mapped_indices]
        return p.unsqueeze(1)


class Decoder1(Decoder0):
    def __init__(self, hidden_channels: int, dropout_prob: float = 0.1, ln: bool = True,
                 task_rel: Optional[Tensor] = None, rand_emb: Optional[Tensor] = None):
        """
        Decoder using randomly initialized embeddings instead of KGE.

        Args:
            hidden_channels (int): Dimension of the hidden layer.
            dropout_prob (float): Dropout probability.
            ln (bool): Whether to use LayerNorm.
            task_rel (Optional[Tensor]): Task-specific relation types.
            rand_emb (Optional[Tensor]): Randomly initialized embeddings.
        """
        super().__init__(hidden_channels, dropout_prob, ln, task_rel, kge_emb=rand_emb)
        self.rand_emb = rand_emb
        self.device = rand_emb.device if rand_emb is not None else 'cpu'

    def forward(self, edge_index: Tensor, edge_type: Tensor):
        """
        Forward pass using random embeddings.

        Args:
            edge_index (Tensor): Shape [2, E], source and destination indices.
            edge_type (Tensor): Shape [E], edge types.

        Returns:
            Tensor: Predicted probabilities for selected edge types.
        """
        rand_emb = self.rand_emb
        src_idx, dst_idx = edge_index[0], edge_index[1]
        rand_src = rand_emb[src_idx]
        rand_dst = rand_emb[dst_idx]

        inter1 = self.inter_projection_1(rand_src - rand_dst)
        inter2 = self.inter_projection_2((rand_src - rand_dst) ** 2)
        inter3 = self.inter_projection_3(rand_src * rand_dst)

        src_x = self.src_projection(rand_src)
        dst_x = self.dst_projection(rand_dst)

        concat_x = torch.cat([src_x, dst_x, inter1, inter2, inter3], dim=1)
        output = self.fusion_mlp(concat_x)
        probs = F.softmax(output, dim=1)

        mapped_indices = torch.tensor([self.edge_type_to_output_index[e.item()] for e in edge_type],
                                      device=rand_emb.device)
        p = probs[torch.arange(probs.size(0)), mapped_indices]
        return p.unsqueeze(1)


class Decoder2(Decoder0):
    def __init__(self, hidden_channels: int, dropout_prob: float = 0.1, ln: bool = True,
                 task_rel: Optional[Tensor] = None, pcm_emb: Optional[Tensor] = None):
        """
        Decoder using PCM (Physicochemical or molecular) embeddings.

        Args:
            hidden_channels (int): Dimension of the hidden layer.
            dropout_prob (float): Dropout probability.
            ln (bool): Whether to use LayerNorm.
            task_rel (Optional[Tensor]): Task-specific relation types.
            pcm_emb (Optional[Tensor]): Physicochemical/molecular embeddings.
        """
        super().__init__(hidden_channels, dropout_prob, ln, task_rel, kge_emb=pcm_emb)
        self.pcm_emb = pcm_emb
        self.device = pcm_emb.device if pcm_emb is not None else 'cpu'

    def forward(self, edge_index: Tensor, edge_type: Tensor):
        """
        Forward pass using PCM embeddings.

        Args:
            edge_index (Tensor): Shape [2, E], source and destination indices.
            edge_type (Tensor): Shape [E], edge types.

        Returns:
            Tensor: Predicted probabilities for selected edge types.
        """
        pcm_emb = self.pcm_emb
        src_idx, dst_idx = edge_index[0], edge_index[1]
        pcm_src = pcm_emb[src_idx]
        pcm_dst = pcm_emb[dst_idx]

        inter1 = self.inter_projection_1(pcm_src - pcm_dst)
        inter2 = self.inter_projection_2((pcm_src - pcm_dst) ** 2)
        inter3 = self.inter_projection_3(pcm_src * pcm_dst)

        src_x = self.src_projection(pcm_src)
        dst_x = self.dst_projection(pcm_dst)

        concat_x = torch.cat([src_x, dst_x, inter1, inter2, inter3], dim=1)
        output = self.fusion_mlp(concat_x)
        probs = F.softmax(output, dim=1)

        mapped_indices = torch.tensor([self.edge_type_to_output_index[e.item()] for e in edge_type],
                                      device=pcm_emb.device)
        p = probs[torch.arange(probs.size(0)), mapped_indices]
        return p.unsqueeze(1)


class Decoder3(Decoder0):
    def __init__(self, hidden_channels: int, dropout_prob: float = 0.1, ln: bool = True,
                 task_rel: Optional[Tensor] = None, kge_emb: Optional[Tensor] = None):
        """
        Decoder without distance-based enhancement, focusing on node embeddings directly.

        Args:
            hidden_channels (int): Dimension of the hidden layer.
            dropout_prob (float): Dropout probability.
            ln (bool): Whether to use LayerNorm.
            task_rel (Optional[Tensor]): Task-specific relation types.
            kge_emb (Optional[Tensor]): Pretrained KGE embeddings.
        """
        super().__init__(hidden_channels, dropout_prob, ln, task_rel, kge_emb=kge_emb)
        
        # Remove unnecessary interaction projections
        del self.inter_projection_1
        del self.inter_projection_2
        del self.inter_projection_3

        # Adjust fusion MLP to handle fewer inputs
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # Adjust input size accordingly
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_channels) if ln else nn.Identity(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 3)
        )

        self.device = kge_emb.device if kge_emb is not None else 'cpu'

    def forward(self, edge_index: Tensor, edge_type: Tensor):
        """
        Forward pass without distance-based enhancement.

        Args:
            edge_index (Tensor): Shape [2, E], source and destination indices.
            edge_type (Tensor): Shape [E], edge types.

        Returns:
            Tensor: Predicted probabilities for selected edge types.
        """
        kge_emb = self.kge_emb
        src_idx, dst_idx = edge_index[0], edge_index[1]
        kge_src = kge_emb[src_idx]
        kge_dst = kge_emb[dst_idx]

        # Directly project source and destination embeddings
        src_x = self.src_projection(kge_src)
        dst_x = self.dst_projection(kge_dst)

        # Concatenate and fuse them
        concat_x = torch.cat([src_x, dst_x], dim=1)
        output = self.fusion_mlp(concat_x)
        probs = F.softmax(output, dim=1)

        mapped_indices = torch.tensor([self.edge_type_to_output_index[e.item()] for e in edge_type],
                                      device=self.device)
        p = probs[torch.arange(probs.size(0)), mapped_indices]
        return p.unsqueeze(1)