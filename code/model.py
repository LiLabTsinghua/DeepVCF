from typing import Optional, Tuple
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn import Parameter
import torch.nn as nn
from torch.nn.functional import normalize
from data_utils import EdgeDataLoader, EdgeLDataLoader
from metric import compute_binary_metrics, compute_ranking_metrics, compute_rank
from tqdm import tqdm
import os

class DeepVCF_Model(torch.nn.Module):
    def __init__(self,
                config, 
                num_nodes: int,
                num_relations: int,
                task_rel: Tensor,
                coverage: dict,
                ):
        """
        Initializes the DeepVCF model components.

        :param config: Dictionary or object containing model configuration 
                       parameters (e.g., 'hidden_dim', 'dropout', 'drn_method', 'ensemble', 'k').
        :type config: dict or object
        :param num_nodes: The total number of entities/nodes in the knowledge graph.
        :type num_nodes: int
        :param num_relations: The total number of relation types in the knowledge graph.
        :type num_relations: int
        :param task_rel: Tensor representing the relation types specific to the prediction task.
        :type task_rel: torch.Tensor
        :param coverage: Dictionary detailing the coverage information for the knowledge graph.
        :type coverage: dict
        """

        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.task_rel = task_rel
        self.coverage = coverage
        
        self.drn = KGE_Model(self.num_nodes, self.num_relations, self.config['hidden_dim'], self.config['drn_method'])

        if config['ensemble']:
            self.tpns = ModuleList([
                Decoder(self.config['hidden_dim'], self.config['hidden_channels'],
                        dropout_prob=self.config['dropout'],
                        task_rel=self.task_rel)
                for _ in range(self.config['k'])
            ])
        else:
            self.tpns = ModuleList([
                Decoder(self.config['hidden_dim'], self.config['hidden_channels'],
                        dropout_prob=self.config['dropout'],
                        task_rel=self.task_rel)
            ])

    def general_negative_samples(self, num_neg_samples, edge_index, edge_type, num_nodes):
        """
        Generates general negative samples for Knowledge Graph Embedding (KGE) training and evaluation.

        Negative sampling corrupts either the head (source) or the tail (target) entity of a positive 
        triple (h, r, t) with a randomly selected entity from the graph, creating a corrupted triple 
        that is assumed to be false. 

        :param num_neg_samples: The number of negative samples to generate for *each* positive edge.
        :type num_neg_samples: int
        :param edge_index: A torch.Tensor of shape (2, num_edges) representing the (head, tail) 
                        indices of the positive triples.
        :type edge_index: torch.Tensor
        :param edge_type: A torch.Tensor of shape (num_edges,) representing the relation type 'r' 
                        for each positive triple.
        :type edge_type: torch.Tensor
        :param num_nodes: The total number of nodes (entities) in the knowledge graph.
        :type num_nodes: int
        :returns: A tuple containing:
                - neg_edge_index (torch.Tensor): The negative edge index of shape (2, num_edges * num_neg_samples).
                - edge_type (torch.Tensor): The corresponding relation types, repeated for the negative samples.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """

        num_edges = edge_index.size(1)

        # Expand edge_index and edge_type for multiple negative samples per edge
        edge_index = edge_index.repeat_interleave(num_neg_samples, dim=1)
        edge_type = edge_type.repeat_interleave(num_neg_samples)

        # Create negative edge index by corrupting either the source or the target node
        neg_edge_index = edge_index.clone()

        # Sample mask to decide whether to corrupt source or target
        mask = torch.rand(neg_edge_index.size(1)) < 0.5

        # Define the sampling range for negative nodes
        valid_nodes = torch.arange(num_nodes, device=edge_index.device)

        # Create negative edge by corrupting source or target nodes
        neg_edge_index[0, mask] = valid_nodes[torch.randint(len(valid_nodes), (mask.sum(),), device=edge_index.device)]
        neg_edge_index[1, ~mask] = valid_nodes[torch.randint(len(valid_nodes), ((~mask).sum(),), device=edge_index.device)]

        # Assert that the indices are within valid range
        assert (neg_edge_index[0] < num_nodes).all(), "Negative edge index out of bounds!"
        assert (neg_edge_index[1] < num_nodes).all(), "Negative edge index out of bounds!"

        return neg_edge_index, edge_type

    def genome_negative_samples(self, dst_index, task_rel, eval_range):
        """
        Generates a set of 'negative' samples for genome-scale prediction, primarily 
        used for comprehensive evaluation where all possible head entities are checked 
        against specific tail entities and relations.
        
        This function implements a specific corruption strategy: it holds the tail (dst)
        and relation (task_rel) constant while iterating over all head entities 
        (eval_range), creating all possible (head, relation, tail) triples for evaluation.
        
        :param dst_index: A 1D tensor of tail node indices (e.g., metabolites) to be evaluated.
        :type dst_index: torch.Tensor
        :param task_rel: A 1D tensor of relation types specific to the prediction task.
        :type task_rel: torch.Tensor
        :param eval_range: A 1D tensor of all head node indices (e.g., proteins/genes) 
                        that should be checked against the tail nodes (dst_index).
        :type eval_range: torch.Tensor
        :returns: A tuple containing:
                - all_edge_index (torch.Tensor): The negative/evaluation edge index of 
                    shape (2, num_dst * num_valid_nodes * num_task_rel). It represents 
                    all generated (source, destination) pairs.
                - edge_type (torch.Tensor): The corresponding relation types, expanded 
                    to match the number of generated edges.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        # dst is metaboliet
        num_dst = len(dst_index)
        num_valid_nodes = len(eval_range)
        num_task_rel = len(task_rel)

        # Create negative edges by replacing the source node (head corruption)
        all_edge_src_index = eval_range.repeat(num_dst * num_task_rel).to(dst_index.device)
        all_edge_dst_index = dst_index.repeat_interleave(num_valid_nodes * num_task_rel)
        all_edge_index = [all_edge_src_index,all_edge_dst_index]
        all_edge_index = torch.stack(all_edge_index, dim=0)
        # print(eval_range)
        # print(all_edge_index.shape)

        # Expand edge_type to match the expanded edge_index
        edge_type = task_rel.repeat_interleave(num_valid_nodes).repeat(num_dst).to(dst_index.device)
        # print(edge_type)
        return all_edge_index, edge_type

    def train_drn(self, knowledge):
        """
        Trains the Deep Representation Network (DRN), which is the Knowledge Graph Embedding (KGE) 
        component, using positive and negative samples derived from the knowledge graph.
        
        The training uses early stopping based on the Mean Reciprocal Rank (MRR) metric 
        on the validation set.

        :param knowledge: Data object containing the knowledge graph triples (train, validation).
        :type knowledge: object
        :param add_train: Flag to potentially include additional training data (currently unused in implementation).
        :type add_train: bool
        :returns: None
        """
        print('-' * 200)
        print('Deep representation network training started!')
        print('-' * 200)

        # Initialize the AdamW optimizer for the DRN parameters.
        optimizer = torch.optim.AdamW(self.drn.parameters(), lr=self.config['drn_lr'], weight_decay=self.config['drn_wd'])

        knowledge = knowledge.to(self.device)

        kg_train_edge_index = knowledge.kg_train_edge_index
        kg_train_edge_type = knowledge.kg_train_edge_type
        kg_val_edge_index = knowledge.kg_val_edge_index
        kg_val_edge_type = knowledge.kg_val_edge_type

        # Create data loaders for batching edges during training and validation.
        train_edge_loader = EdgeDataLoader(kg_train_edge_index, kg_train_edge_type, self.config['drn_batch_size']) 
        val_edge_loader = EdgeDataLoader(kg_val_edge_index, kg_val_edge_type, 128) 

        best_mrr = 0
        patience_counter = 0
        best_state_dict = None  # Dictionary to save the best model parameters

        for epoch in range(1, self.config['drn_num_epochs'] + 1):
            batch_loss = []
            for batch_edges, batch_edge_types in tqdm(train_edge_loader):
                # Perform a single training step
                loss = self.train_drn_step(optimizer, batch_edges, batch_edge_types, num_nodes=self.num_nodes)
                batch_loss.append(loss)

            batch_loss = np.mean(batch_loss)
            print(f'Epoch: {epoch:05d}, Loss: {batch_loss:.4f}')

            # Evaluate on validation set after a warm-up period and at intervals.
            if (epoch > 10) and (epoch % self.config['drn_eval_interval'] == 0):
                mrr = self.test_drn(val_edge_loader)
                if mrr > best_mrr:
                    best_mrr = mrr
                    patience_counter = 0

                    # Save the best model parameters (deep copy state dict)
                    best_state_dict = {k: v.clone() for k, v in self.drn.state_dict().items()}
                    print(f"Improved MRR: {mrr:.4f}. Best model updated.")
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience counter: {patience_counter}/{self.config['drn_patience']}")
                    if patience_counter >= self.config['drn_patience']:
                        print("Early stopping triggered.")
                        break

        print('-' * 200)
        print("Training finished. Restoring best model parameters...")

        # Load the best saved parameters for the DRN.
        if best_state_dict is not None:
            self.drn.load_state_dict(best_state_dict)
            print("Best model restored.")
        else:
            print("Warning: best_state_dict is None, no update occurred!")

        print('-' * 200)
    
    def train_drn_step(self, optimizer, batch_edges, batch_edge_types, num_nodes=None):
        """
        Performs a single training step for the DRN using mini-batch positive edges.

        It generates negative samples, calculates the ranking loss (Margin Ranking Loss), 
        performs backpropagation, and updates the weights.

        :param optimizer: The optimizer used for weight updates.
        :type optimizer: torch.optim.Optimizer
        :param batch_edges: A tensor (2, batch_size) of positive edge indices (head, tail).
        :type batch_edges: torch.Tensor
        :param batch_edge_types: A tensor (batch_size,) of relation types for the edges.
        :type batch_edge_types: torch.Tensor
        :param num_nodes: Total number of nodes, used for negative sampling.
        :type num_nodes: int
        :returns: The loss value for the current batch.
        :rtype: float
        """
        self.drn.train()
        optimizer.zero_grad()

        # Forward pass for positive samples
        pos_out = self.drn.kge_forward(batch_edges, batch_edge_types)

        # Generate negative samples by corrupting head or tail
        neg_edge_index, neg_edge_types = self.general_negative_samples(self.config['drn_num_neg'], batch_edges, batch_edge_types, num_nodes)
        
        # Forward pass for negative samples
        neg_out = self.drn.kge_forward(neg_edge_index, neg_edge_types)

        # Ranking loss (e.g., Margin Ranking Loss) to maximize the score difference 
        # between positive and negative triples.
        pos_out_repeated = pos_out.repeat_interleave(self.config['drn_num_neg'])
        ranking_loss = F.margin_ranking_loss(pos_out_repeated, neg_out, torch.ones_like(pos_out_repeated), margin=1)

        # The total loss is currently just the ranking loss.
        loss = ranking_loss

        # Backpropagation and gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.drn.parameters(), 1.0)
        optimizer.step()

        return float(loss)

    @torch.no_grad()
    def test_drn(self, val_edge_loader):
        """
        Evaluates the DRN model's performance (link prediction) on the validation set.
        
        It calculates the Mean Reciprocal Rank (MRR) based on scoring positive triples 
        against a large set of sampled negative triples.

        :param val_edge_loader: DataLoader for the validation edge indices and types.
        :type val_edge_loader: EdgeDataLoader
        :returns: The Mean Reciprocal Rank (MRR) metric.
        :rtype: float
        """
        self.drn.eval()

        mrr_list = []
        for batch_edges, batch_edge_types in tqdm(val_edge_loader):
            batch_size = batch_edges.size(1)

            # Positive sample scores
            pos_scores = self.drn.kge_forward(batch_edges, batch_edge_types)

            # Generate negative samples for robust evaluation (10000 per positive edge)
            neg_edge_index, neg_edge_types = self.general_negative_samples(
                10000, # Number of negative samples per positive edge
                batch_edges,
                batch_edge_types,
                num_nodes=self.num_nodes
            )

            # Negative sample scores
            neg_scores = self.drn.kge_forward(neg_edge_index, neg_edge_types)
            # Reshape scores to [batch_size, num_neg_samples]
            neg_scores = neg_scores.view(batch_size, 10000)

            # Calculate the rank of the positive sample: count how many negative scores are greater than or equal to the positive score, then add 1.
            ranks = (neg_scores >= pos_scores.view(-1, 1)).sum(dim=1) + 1

            # Convert to float for calculation
            ranks = ranks.float()
            # Compute Reciprocal Rank (1/rank) for each positive sample
            mrr_list.extend((1.0 / ranks).tolist())

        # Final Metrics: Mean of all reciprocal ranks
        mrr = np.mean(mrr_list)

        print(f"MRR: {mrr:.4f}")

        return mrr

    def get_embedding(self):
        """
        Retrieves the learned node embeddings from the DRN.

        The embeddings are detached from the computation graph and L2-normalized.

        :returns: The L2-normalized KGE node embeddings.
        :rtype: torch.Tensor
        """
        # Detach to prevent gradient flow from TPN to DRN during TPN training
        kge_emb = self.drn.node_emb.detach()
        # L2 normalization of the embeddings
        kge_emb = F.normalize(kge_emb, p=2, dim=1)
        return kge_emb

    def train_tpn(self, data_list):
        """
        Trains the Target Prediction Network(s) (TPN) using the learned KGE embeddings.

        If ensemble mode is enabled, it trains multiple TPNs sequentially. Training uses 
        Binary Cross-Entropy (BCE) loss and employs early stopping based on Area Under 
        the Precision-Recall Curve (AUPR) on the validation set.

        :param data_list: A list of data objects, potentially one for each TPN in the ensemble.
        :type data_list: list[object]
        :returns: None
        """
        print('-' * 200)
        print('Target prediction network training started!')
        print('-' * 200)

        # Get or initialize the node embeddings.
        if self.config['use_drn']:
            kge_emb = self.get_embedding()
        else:
            # Fallback to random initialization if DRN embeddings are not used.
            kge_emb = torch.randn(self.num_nodes, self.config['hidden_dim'])

        # Set the embedding tensor for all TPN modules.
        for tpn in self.tpns:
            tpn.kge_emb = kge_emb.to(self.device)

        # Train each TPN model sequentially.
        for model_id, tpn in enumerate(self.tpns):
            # Select the corresponding data object (assumes data_list is aligned or repeated).
            data = data_list[model_id].to(self.device)

            print(f"\n======= Training TPN Model #{model_id+1} =======")
            # Initialize optimizer for the current TPN.
            optimizer = torch.optim.AdamW(
                tpn.parameters(),
                lr=self.config['tpn_lr'],
                weight_decay=self.config['tpn_wd']
            )

            best_aupr = 0
            patience_counter = 0
            best_state_dict = None  # Dictionary to save the best model parameters

            # Create data loader for TPN training edges.
            edge_loader = EdgeLDataLoader(
                data.train_edge_index, data.train_edge_type, data.train_label,
                self.config['tpn_batch_size']
            )

            for epoch in range(1, self.config['tpn_num_epochs'] + 1):
                batch_loss = []
                for batch_edges, batch_edge_types, batch_label in tqdm(edge_loader):
                    # Perform a single training step for TPN.
                    loss = self.train_tpn_step(tpn, optimizer, batch_edges, batch_edge_types, batch_label)
                    batch_loss.append(loss)

                batch_loss = np.mean(batch_loss)
                print(f'Model {model_id+1} | Epoch: {epoch:05d}, Loss: {batch_loss:.4f}')

                # Validation check
                if epoch % self.config['tpn_eval_interval'] == 0:
                    # Evaluate the single TPN on the validation set.
                    _, _, metrics = self.test_tpn_single(tpn, data)
                    aupr = metrics["auprc"]

                    # Early stopping logic based on AUPR
                    if aupr > best_aupr:
                        best_aupr = aupr
                        patience_counter = 0

                        # Save the best model state
                        best_state_dict = {k: v.clone() for k, v in tpn.state_dict().items()}
                        print(f"Improved AUPR for model {model_id+1}: {aupr:.4f}. Best model updated.")

                    else:
                        patience_counter += 1
                        print(f"No improvement. Patience {patience_counter}/{self.config['tpn_patience']}")

                        if patience_counter >= self.config['tpn_patience']:
                            print("Early stopping triggered.")
                            break

            # Restore the best model parameters for the current TPN after training/early stopping.
            if best_state_dict is not None:
                tpn.load_state_dict(best_state_dict)
                print(f"Model {model_id+1} restored to best AUPR = {best_aupr:.4f}")
            else:
                print(f"Model {model_id+1} had no improvement, no best state saved.")

        print('-' * 200)
    
    def train_tpn_step(self, tpn, optimizer, batch_edges, batch_edge_types, batch_label):
        """
        Performs a single training step for a TPN module.

        It calculates the Binary Cross-Entropy (BCE) loss on the target prediction task, 
        performs backpropagation, and updates the weights.

        :param tpn: The specific TPN module being trained.
        :type tpn: torch.nn.Module
        :param optimizer: The optimizer used for weight updates.
        :type optimizer: torch.optim.Optimizer
        :param batch_edges: A tensor (2, batch_size) of edge indices.
        :type batch_edges: torch.Tensor
        :param batch_edge_types: A tensor (batch_size,) of relation types.
        :type batch_edge_types: torch.Tensor
        :param batch_label: A tensor (batch_size,) of binary labels (0 or 1).
        :type batch_label: torch.Tensor
        :returns: The loss value for the current batch.
        :rtype: float
        """
        tpn.train()
        optimizer.zero_grad()

        # Forward pass through the TPN
        out = tpn(batch_edges, batch_edge_types)
        
        # Calculate Binary Cross-Entropy (BCE) Loss. The label tensor is unsqueezed 
        # to match the output shape.
        loss = F.binary_cross_entropy(out, batch_label.to(out.device).unsqueeze(dim=1))

        # Backpropagation and gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tpn.parameters(), 1.0)
        optimizer.step()

        return float(loss)

    @torch.no_grad()
    def test_tpn(self, data_list, test=False):
        """
        Evaluates the Target Prediction Network(s) on the validation or test set.

        It computes the predictions and metrics for both the ensemble mean and 
        each individual TPN model.

        :param data_list: A list containing the data objects (valid/test edge indices and labels).
        :type data_list: list[object]
        :param test: If True, uses the test set; otherwise, uses the validation set.
        :type test: bool
        :returns: A tuple containing:
                  - all_out (torch.Tensor): The ensemble mean prediction scores.
                  - all_label (torch.Tensor): The true labels.
                  - ensemble_metrics (dict): Metrics (e.g., AUPR, AUROC) for the ensemble.
                  - indiv_outputs (list[torch.Tensor]): List of predictions from each TPN.
                  - indiv_metrics (list[dict]): List of metrics for each individual TPN.
        :rtype: tuple[torch.Tensor, torch.Tensor, dict, list, list]
        """
        total_pred = None

        # Select the appropriate dataset and create DataLoader
        data = data_list[0].to(self.device)
        if test:
            all_label = data.test_label
            loader = EdgeDataLoader(data.test_edge_index, data.test_edge_type, 128, shuffle=False)
        else:
            all_label = data.valid_label
            loader = EdgeDataLoader(data.valid_edge_index, data.valid_edge_type, 128, shuffle=False)

        indiv_outputs = []
        indiv_metrics = []

        # Loop over ensemble models (self.tpns) to get predictions.
        for tpn in self.tpns: 
            tpn.eval()
            model_pred = []
            for edges, edge_types in loader:
                model_pred.append(tpn(edges, edge_types))

            model_pred = torch.cat(model_pred, dim=0)
            indiv_outputs.append(model_pred)

            # Accumulate predictions for the ensemble mean.
            if total_pred is None:
                total_pred = model_pred.clone()
            else:
                total_pred += model_pred

        # === Ensemble output ===
        all_out = total_pred / len(self.tpns)
        all_label = all_label.to(all_out.device)

        # Calculate ensemble metrics
        ensemble_metrics = compute_binary_metrics(all_out, all_label)

        # === Individual model metrics ===
        for pred in indiv_outputs:
            indiv_metrics.append(compute_binary_metrics(pred, all_label))

        return all_out, all_label, ensemble_metrics, indiv_outputs, indiv_metrics

    @torch.no_grad()
    def test_tpn_single(self, tpn, data, test=False):
        """
        Evaluates a single TPN module on the validation or test set.

        :param tpn: The specific TPN module to evaluate.
        :type tpn: torch.nn.Module
        :param data: The data object containing the valid/test edge indices and labels.
        :type data: object
        :param test: If True, uses the test set; otherwise, uses the validation set.
        :type test: bool
        :returns: A tuple containing:
                  - preds (torch.Tensor): The prediction scores.
                  - all_label (torch.Tensor): The true labels.
                  - metrics (dict): Metrics (e.g., AUPR, AUROC) for the single model.
        :rtype: tuple[torch.Tensor, torch.Tensor, dict]
        """
        tpn.eval()
        
        # Select the appropriate dataset and create DataLoader
        if test:
            all_label = data.test_label
            loader = EdgeDataLoader(data.test_edge_index, data.test_edge_type, 128, shuffle=False)
        else:
            all_label = data.valid_label
            loader = EdgeDataLoader(data.valid_edge_index, data.valid_edge_type, 128, shuffle=False)

        preds = []
        for edges, edge_types in loader:
            preds.append(tpn(edges, edge_types))

        preds = torch.cat(preds, dim=0)

        # Compute metrics based on predictions and true labels.
        metrics = compute_binary_metrics(preds, all_label.to(preds.device))
        return preds, all_label, metrics

    def save_model(self):
        """
        Saves the entire DeepVCF model state to a file.

        The saved dictionary includes model configuration, structural parameters, 
        and the state dictionaries and KGE embeddings for the DRN and all TPNs.

        :returns: None
        """
        save_dir = self.config.get("model_saved_dir", "./")
        model_name = self.config.get("model_name", "DeepVCF")

        # Ensure the save directory exists.
        if not os.path.exists(save_dir):
            print(f"[Info] Save path {save_dir} does not exist. Creating directory...")
            os.makedirs(save_dir, exist_ok=True)

        # Compose the full save path.
        save_path = os.path.join(save_dir, model_name + '.pt')

        # Create a dictionary containing all necessary model components for saving.
        save_dict = {
            "config": self.config,
            "num_nodes": self.num_nodes,
            "num_relations": self.num_relations,
            "task_rel": self.task_rel.cpu(),
            "coverage": self.coverage,

            "drn_state_dict": self.drn.state_dict(),
            "tpn_state_dicts": [tpn.state_dict() for tpn in self.tpns],

            # Save the KGE embedding tensor associated with each TPN.
            "tpn_kge_embs": [tpn.kge_emb.cpu() for tpn in self.tpns],
        }

        torch.save(save_dict, save_path)
        print(f"[OK] Model saved → {save_path}")
    
    @staticmethod
    def load_model(path: str, device="cpu"):
        """
        Loads the DeepVCF model state from a checkpoint file.

        This is a static method that rebuilds the model structure from the saved 
        configuration and loads the learned weights and embeddings onto the specified device.

        :param path: The file path to the saved model checkpoint (.pt file).
        :type path: str
        :param device: The device (e.g., "cpu" or "cuda:0") to load the model onto.
        :type device: str
        :returns: The fully reconstructed and loaded DeepVCF_Model instance.
        :rtype: DeepVCF_Model
        """
        checkpoint = torch.load(path, map_location=device)

        # 1. Reconstruct the model structure using saved initialization arguments.
        model = DeepVCF_Model(
            config=checkpoint["config"],
            num_nodes=checkpoint["num_nodes"],
            num_relations=checkpoint["num_relations"],
            task_rel=checkpoint["task_rel"],
            coverage=checkpoint["coverage"],
        )
        model.to(device)

        # 2. Load the state dictionary for the Deep Representation Network (DRN).
        model.drn.load_state_dict(checkpoint["drn_state_dict"])

        # 3. Load the state dictionaries for all Target Prediction Networks (TPN).
        for tpn, state_dict in zip(model.tpns, checkpoint["tpn_state_dicts"]):
            tpn.load_state_dict(state_dict)

        # 4. Restore the KGE embeddings associated with each TPN.
        saved_embs = checkpoint["tpn_kge_embs"]
        assert len(saved_embs) == len(model.tpns), \
            "Mismatch: number of TPNs in checkpoint and model does not match."

        for tpn, emb in zip(model.tpns, saved_embs):
            tpn.kge_emb = emb.to(device)

        print(f"[OK] Model loaded ← {path}")
        return model
    
    def finetune_tpn(self, data_list, epochs: int = 20, lr: float = 1e-5, wd: float = 1e-5, batch_size: int = 8,):
        """
        Performs fine-tuning on the TPN module(s) using new or limited training data.

        This method is typically used after loading a pre-trained model to adapt it 
        to a new, smaller dataset. It trains without early stopping validation checks.

        :param data_list: A list containing the data object for fine-tuning.
        :type data_list: list[object]
        :param epochs: The number of fine-tuning epochs.
        :type epochs: int
        :param lr: Learning rate for the fine-tuning optimizer.
        :type lr: float
        :param wd: Weight decay for the fine-tuning optimizer.
        :type wd: float
        :param batch_size: Batch size for the fine-tuning data loader.
        :type batch_size: int
        :returns: None
        """
        # NOTE: This fine-tuning implementation uses only the first data object in data_list.
        print('-' * 200)
        print('Target prediction network finetuning started!')
        print('-' * 200)

        # Train each TPN model sequentially.
        for model_id, tpn in enumerate(self.tpns):
            data = data_list[0].to(self.device)  # Use the first data object
            
            print(f"\n======= Training TPN Model #{model_id+1} =======")
            # Initialize optimizer with fine-tuning hyperparameters.
            optimizer = torch.optim.AdamW(tpn.parameters(), lr=lr, weight_decay=wd)

            # Note: Best AUPR and patience are not tracked in this simplified fine-tuning.

            edge_loader = EdgeLDataLoader(
                data.train_edge_index, data.train_edge_type, data.train_label,
                batch_size
            )

            for epoch in range(1, epochs + 1):
                batch_loss = []
                for batch_edges, batch_edge_types, batch_label in tqdm(edge_loader):
                    loss = self.train_tpn_step(tpn, optimizer, batch_edges, batch_edge_types, batch_label)
                    batch_loss.append(loss)

                batch_loss = np.mean(batch_loss)
                print(f'Model {model_id+1} | Epoch: {epoch:05d}, Loss: {batch_loss:.4f}')

        print('-' * 200)

    @torch.no_grad()
    def genome_scale_prediction(self, query_met, query_gene_prefix, gene_type, 
                                entity2idx, relation2idx,
                                output_dir):
        """
        Performs large-scale link prediction for target identification in a genome context.

        It generates all possible triples involving specified metabolites and candidate genes, 
        performs ensemble prediction using TPNs, and calculates both the mean score and 
        prediction uncertainty (variance) for each triple, saving the results to CSV files.

        :param query_met: List of metabolite names to query (tail entities).
        :type query_met: list[str]
        :param query_gene_prefix: Prefix used to filter genes/proteins (head entities).
        :type query_gene_prefix: str
        :param gene_type: Type of genes to consider ('whole genome', 'metabolic gene', 'non-metabolic gene').
        :type gene_type: str
        :param entity2idx: Dictionary mapping entity names to their ID indices.
        :type entity2idx: dict
        :param relation2idx: Dictionary mapping relation names to their ID indices.
        :type relation2idx: dict
        :param output_dir: Directory path to save the prediction results.
        :type output_dir: str
        :returns: None
        :raises ValueError: If a query metabolite is not found or an unsupported gene type is provided.
        """
        # --- Input validation and index mapping ---
        if any(x not in entity2idx.keys() for x in query_met):
            raise ValueError(f"query_met {query_met} not found in entity2idx")
        # Recommendation/Check for expected metabolite naming convention
        if any(not x.endswith('_c') for x in query_met):
            raise ValueError(f"we recommend use query_met {query_met} with '_c'")

        query_met_tensor = torch.LongTensor([entity2idx[x] for x in query_met]).to(self.device)

        # Select candidate genes based on the specified type and prefix.
        if gene_type == 'whole genome':
            gene = [x for x in self.coverage['all_gene'] if x.startswith(query_gene_prefix)]
        elif gene_type == 'metabolic gene':
            gene = [x for x in self.coverage['metabolic_gene'] if x.startswith(query_gene_prefix)]
        elif gene_type == 'non-metabolic gene':
            gene = [x for x in self.coverage['non_metabolic_gene'] if x.startswith(query_gene_prefix)]
        else:
            raise ValueError(f"gene_type {gene_type} not supported")

        gene_tensor = torch.LongTensor([entity2idx[g] for g in gene]).to(self.device)

        # Generate exhaustive triples (all gene candidates x all query metabolites x all task relations).
        all_edge_index, all_edge_type = self.genome_negative_samples(
            dst_index=query_met_tensor,
            task_rel=self.task_rel,
            eval_range=gene_tensor
        )

        # Use a large batch size for efficient inference.
        edge_loader = EdgeDataLoader(all_edge_index, all_edge_type, 10000, shuffle=False)

        # --- Ensemble prediction and Uncertainty Estimation ---
        indiv_outputs = []  # List of prediction scores from each TPN
        total_pred = None

        for tpn in self.tpns:
            tpn.eval()
            model_pred = []
            for edges, edge_types in edge_loader:
                out = tpn(edges, edge_types)
                model_pred.append(out.squeeze(1))

            model_pred = torch.cat(model_pred, dim=0)
            indiv_outputs.append(model_pred)

            # Accumulate predictions for the ensemble mean.
            if total_pred is None:
                total_pred = model_pred.clone()
            else:
                total_pred += model_pred

        # Calculate Ensemble Mean Score (Prediction)
        ensemble_mean = total_pred / len(self.tpns)
        ensemble_mean = ensemble_mean.cpu().numpy()
        
        # Calculate Ensemble Variance (Uncertainty)
        indiv_outputs_tensor = torch.stack(indiv_outputs, dim=0)  # [num_models, num_edges]
        ensemble_var = indiv_outputs_tensor.var(dim=0, unbiased=False)
        ensemble_var = ensemble_var.cpu().numpy()

        # print(ensemble_mean.shape)
        # print(ensemble_var.shape)

        # --- Save results ---
        os.makedirs(output_dir, exist_ok=True)
        # Assuming self.coverage['modification'] contains the relation names corresponding to self.task_rel
        num_candidate = len(gene) * len(self.coverage['modification']) 
        
        # Save results for each query metabolite separately.
        for idx, x in enumerate(query_met):
            df = pd.DataFrame({
                'qeury metabbolite': x,
                # Gene names and modification names are repeated to match the total number of candidates
                'gene': [x.split(':')[1] for x in gene] * len(self.coverage['modification']),
                'gene_type':['metabolic' if x in self.coverage['metabolic_gene'] else 'non-metabolic' for x in gene] * 2,
                'modification': [m for m in self.coverage['modification'] for _ in range(len(gene))],
                'DeepVCF score': ensemble_mean[idx * num_candidate : (idx + 1) * num_candidate],
                'var': ensemble_var[idx * num_candidate : (idx + 1) * num_candidate],
            })
            
            # Sort by prediction score (DeepVCF score) in descending order.
            df = df.sort_values(by='DeepVCF score', ascending=False)
            df.to_csv(os.path.join(output_dir, f'{x}.csv'), sep='\t', index=False)

    @property
    def device(self):
        """
        A property to get the device (CPU/GPU) of the model's parameters.

        :returns: The torch device of the model.
        :rtype: torch.device
        """
        return next(self.parameters()).device
    

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
        return -self.l2_dissimilarity(z_src + rel, z_dst)


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

class Decoder(nn.Module):
    def __init__(self, hidden_dim:int, 
                hidden_channels: int, dropout_prob: float = 0.1, ln: bool = True,
                task_rel: Tensor = None, ):

        super().__init__()

        # Feature interaction projections
        self.inter_projection_1 = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_channels / 3)),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(hidden_channels / 3)) if ln else nn.Identity()
        )
        self.inter_projection_2 = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_channels / 3)),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(hidden_channels / 3)) if ln else nn.Identity()
        )
        self.inter_projection_3 = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_channels / 3)),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(hidden_channels / 3)) if ln else nn.Identity()
        )

        # Source and destination node feature projectors
        self.src_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_channels),
            nn.Dropout(dropout_prob, inplace=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_channels) if ln else nn.Identity()
        )
        self.dst_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_channels),
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

        self.kge_emb = None

        # output_index
        self.edge_type_to_output_index = self.get_output_index(task_rel)

    def get_output_index(self, rel_task: Tensor):
        """
        Maps each unique edge type to an index in output logits.
        """
        if rel_task is None:
            return {}
        unique_edge_types = torch.unique(rel_task, sorted=True)
        return {et.item(): idx for idx, et in enumerate(unique_edge_types)} # 10→1，11→2

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
        # print("kge_src device:", kge_src.device)
        # print("kge_dst device:", kge_dst.device)

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