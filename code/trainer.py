import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import copy
from sampling import NegativeSampling
from metric import compute_binary_metrics, compute_ranking_metrics, compute_rank
from data_utils import EdgeDataLoader, EdgeLDataLoader, EdgeEDataLoader
import numpy as np


class Trainer:
    def __init__(self, model, config, negative_sampling=None):
        """
        Trainer class for Graph Autoencoder models with early stopping.

        Args:
            model: The KGE or DeepME model.
            config (dict): Configuration dictionary containing hyperparameters for training.
            negative_sampling: The negative sampling strategy used for training.
        """
        # Model setup
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])
        self.negative_sampling = negative_sampling
        self.config = config

        # Hyperparameters
        self.pretrain_batch_size = config.get('pretrain_batch_size', 4096)
        self.finetune_batch_size = config.get('finetune_batch_size', 1024)
        self.num_epochs = config.get('num_epochs', 1000)
        self.eval_interval = config.get('eval_interval', 1)
        self.num_neg_samples = config.get('num_neg_samples', 5)
        self.patience = config.get('patience', 20)
        self.val_metric = config.get('val_metric', 'auprc')

        # Early stopping variables
        self.best_val_metric = -float('inf')  # Higher MRR is better
        self.patience_counter = 0
        self.best_model_state_dict = None

    def kge_train_step(self, batch_edges, batch_edge_types, num_nodes=None):
        """
        Perform a single training step for Knowledge Graph Embedding (KGE) model.

        Args:
            batch_edges: A batch of edges in the graph.
            batch_edge_types: Corresponding edge types for the batch of edges.
            num_nodes: Number of nodes in the graph.

        Returns:
            float: The computed loss for the current batch.
        """
        self.model.train()
        self.optimizer.zero_grad()
        pos_out = self.model.kge_forward(batch_edges, batch_edge_types)

        # Generate negative samples
        neg_edge_index, neg_edge_types = self.negative_sampling.general_negative_samples(batch_edges, batch_edge_types,
                                                                                         num_nodes)

        # Forward pass for negative samples
        neg_out = self.model.kge_forward(neg_edge_index, neg_edge_types)

        # Ranking loss to ensure positive samples are ranked higher than negative samples
        pos_out_repeated = pos_out.repeat_interleave(self.num_neg_samples)
        ranking_loss = F.margin_ranking_loss(pos_out_repeated, neg_out, torch.ones_like(pos_out_repeated), margin=1)

        # Total loss (weighted combination of cross-entropy and ranking loss)
        loss = ranking_loss

        # Backpropagation and gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss)

    def deepme_train_step(self, batch_edges, batch_edge_types, batch_label, ms_edge_index, ms_edge_type):
        """
        Perform a single training step for R-GCN model.

        Args:
            batch_edges: A batch of edges for training.
            batch_edge_types: Corresponding edge types for the batch of edges.
            batch_label: Labels for the batch of edges.
            ms_edge_index: Edge index for multi-source graph.
            ms_edge_type: Edge types for the multi-source graph.

        Returns:
            float: The computed loss for the current batch.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass for positive samples
        out = self.model.deepme_forward(batch_edges, batch_edge_types, ms_edge_index, ms_edge_type)

        # Binary cross-entropy loss
        cross_entropy_loss = F.binary_cross_entropy(out, batch_label.to(out.device).unsqueeze(dim=1))

        # Total loss
        loss = cross_entropy_loss

        # Backpropagation and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return float(loss)

    def train(self, data):
        """
        Train the model based on the specified training option ('kge' or 'deepme').

        Args:
            data: Input data containing edge indices, edge types, and splits.
        """
        if self.config['training'] == 'kge':
            self.kge_training(data)
        elif self.config['training'] == 'deepme':
            self.deepme_training(data)
        else:
            raise ValueError("Unknown training option.")

    def kge_training(self, data, monitor='metric', add_train=False):
        """
        Train the KGE model with early stopping based on MRR.

        Args:
            data: Input data containing edge indices, edge types, and splits.
            monitor (str): Metric to monitor for early stopping ('metric' or 'loss').
            add_train (bool): Whether to combine edge indices for pretraining.
        """
        print('-' * 50)
        print('KGE Training started!')
        print('-' * 50)
        times = []

        if add_train:
            combined_edge_index = torch.cat([data.edge_index, data.train_edge_index], dim=1)
            combined_edge_type = torch.cat([data.edge_type, data.train_edge_type], dim=0)
        else:
            combined_edge_index = data.edge_index
            combined_edge_type = data.edge_type

        edge_loader = EdgeDataLoader(combined_edge_index, combined_edge_type, self.pretrain_batch_size)

        for epoch in range(1, self.num_epochs + 1):
            start = time.time()
            batch_loss = []
            for batch_edges, batch_edge_types in tqdm(edge_loader):
                loss = self.kge_train_step(batch_edges, batch_edge_types, num_nodes=data.num_nodes)
                batch_loss.append(loss)

            batch_loss = np.mean(batch_loss)
            print(f'Epoch: {epoch:05d}, Loss: {batch_loss:.4f}')
            times.append(time.time() - start)

            if epoch % self.eval_interval == 0:
                if monitor == 'loss':
                    auroc, auprc = 0, 0
                    flag = self.early_stopping(auroc, auprc, -batch_loss, 'neg_loss')
                else:
                    raise ValueError("Unknown monitor option.")

            if flag:
                break

        print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
        self.reset_early_stopping_vars()

    def deepme_training(self, data):
        """
        Train the R-GCN model with early stopping based on MRR.

        Args:
            data: Input data containing edge indices, edge types, and splits.
        """
        print('-' * 50)
        print('DeepME Training started!')
        print('-' * 50)
        times = []

        edge_loader = EdgeLDataLoader(data.train_edge_index, data.train_edge_type, data.train_label, self.finetune_batch_size)

        flag = False
        for epoch in range(1, self.num_epochs + 1):
            start = time.time()
            batch_loss = []
            for batch_edges, batch_edge_types, batch_label in tqdm(edge_loader):
                loss = self.deepme_train_step(batch_edges, batch_edge_types, batch_label, data.edge_index, data.edge_type)
                batch_loss.append(loss)

            batch_loss = np.mean(batch_loss)
            print(f'Epoch: {epoch:05d}, Loss: {batch_loss:.4f}')
            times.append(time.time() - start)

            if self.eval_interval > 0 and epoch % self.eval_interval == 0:
                _, _, metrics = self.bc_test(data)
                auroc = metrics.get("auroc", 0.0)
                auprc = metrics.get("auprc", 0.0)
                flag = self.early_stopping(auroc, auprc, -batch_loss, self.val_metric)

                _, _, metrics = self.bc_test(data, test=True)
                print(
                    f"Testing|AUROC: {metrics['auroc']:.4f} |AUPRC: {metrics['auprc']:.4f} |Accuracy: {metrics['accuracy']:.4f} |"
                    f"Precision: {metrics['precision']:.4f} |Recall: {metrics['recall']:.4f} |F1-Score: {metrics['f1_score']:.4f} |"
                    f"MCC: {metrics['mcc']:.4f}"
                )

            if flag:
                break

        print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
        self.reset_early_stopping_vars()

    def reset_early_stopping_vars(self):
        """
        Reset early stopping variables for further training.
        """
        self.best_val_metric = -float('inf')
        self.patience_counter = 0

    def early_stopping(self, auroc, auprc, neg_loss, monitor):
        """
        Implement early stopping based on validation metric (MRR, Hit@k, or loss).

        Args:
            auroc (float): Area under ROC curve.
            auprc (float): Area under precision-recall curve.
            neg_loss (float): The negative loss value used for early stopping.
            monitor (str): The validation metric to monitor for early stopping.

        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        flag = False

        # Select the validation metric
        if monitor == 'auroc':
            valid_metric = auroc
        elif monitor == 'auprc':
            valid_metric = auprc
        elif monitor == 'neg_loss':
            valid_metric = neg_loss
        else:
            raise ValueError(f"Unknown validation metric: {monitor}")

        if monitor == 'neg_loss':
            if valid_metric > self.best_val_metric + 0.01:
                self.best_val_metric = valid_metric
                self.patience_counter = 0
                self.best_model_state_dict = self.model.state_dict()
                print(f"Improved {monitor}: {valid_metric:.4f}. Saving model.")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience counter: {self.patience_counter}/{self.patience}")
        else:
            if valid_metric > self.best_val_metric:
                self.best_val_metric = valid_metric
                self.patience_counter = 0
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                print(f"Improved {monitor}: {valid_metric:.4f}. Saving model.")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience counter: {self.patience_counter}/{self.patience}")

        if self.patience_counter >= self.patience:
            print("Early stopping triggered. Restoring best model.")
            self.model.load_state_dict(self.best_model_state_dict)
            flag = True

        return flag

    @torch.no_grad()
    def bc_test(self, data, test=False):
        """
        Evaluate the model using binary classification metrics.

        Args:
            data: Input data containing edge indices, edge types, and labels.
            test (bool): Whether to evaluate on the test set or validation set.

        Returns:
            tuple: All predictions, all labels, and evaluation metrics.
        """
        if test:
            all_label = data.test_label
            valid_edge_loader = EdgeDataLoader(data.test_edge_index, data.test_edge_type, 128, shuffle=False)
        else:
            all_label = data.valid_label
            valid_edge_loader = EdgeDataLoader(data.valid_edge_index, data.valid_edge_type, 128, shuffle=False)

        all_out = []
        for batch_edges, batch_edge_types in tqdm(valid_edge_loader):
            out = self.bc_test_step(batch_edges, batch_edge_types, data.edge_index, data.edge_type)
            all_out.append(out)

        all_out = torch.cat(all_out, dim=0)
        all_label = all_label.to(all_out.device)
        metrics = compute_binary_metrics(all_out, all_label)

        return all_out, all_label, metrics

    @torch.no_grad()
    def bc_test_step(self, batch_edges, batch_edge_types, ms_edge_index, ms_edge_type):
        """
        Perform a single evaluation step for the validation or test set.

        Args:
            batch_edges: A batch of edges to evaluate.
            batch_edge_types: Corresponding edge types for the batch.
            ms_edge_index: Multi-source graph edges.
            ms_edge_type: Multi-source graph edge types.

        Returns:
            Tensor: Predicted output.
        """
        self.model.eval()
        if self.config['training'] in ['kge']:
            out = self.model.kge_forward(batch_edges, batch_edge_types)
        elif self.config['training'] in ['deepme']:
            out = self.model.deepme_forward(batch_edges, batch_edge_types, ms_edge_index, ms_edge_type)
        else:
            raise ValueError("Unknown test mode.")

        return out

    @torch.no_grad()
    def get_embedding(self, data, test=False):
        """
        Get embeddings for validation or test set.

        Args:
            data: Input data containing edge indices and types.
            test (bool): Whether to evaluate on the test set or validation set.

        Returns:
            Tensor: Concatenated embeddings.
        """
        self.model.eval()
        if test:
            valid_edge_loader = EdgeDataLoader(data.test_edge_index, data.test_edge_type, 128, shuffle=False)
        else:
            valid_edge_loader = EdgeDataLoader(data.valid_edge_index, data.valid_edge_type, 128, shuffle=False)

        all_embedding = []
        for batch_edges, batch_edge_types in tqdm(valid_edge_loader):
            embedding = self.model.decoder.get_embedding(batch_edges)
            all_embedding.append(embedding)

        all_embedding = torch.cat(all_embedding, dim=0)
        return all_embedding


class Ensemble_Tester:
    def __init__(self, ensemble_model: list, negative_sampling):
        """
        Ensemble Tester class for evaluating multiple models with early stopping based on MRR.

        Args:
            ensemble_model (list): List of models to be ensembled.
            negative_sampling: The negative sampling strategy used for evaluation.
        """
        self.ensemble_model = ensemble_model
        self.negative_sampling = negative_sampling

    @torch.no_grad()
    def bc_test(self, data):
        """
        Evaluate the ensemble of models using binary classification metrics on the test set.

        Args:
            data: Input data containing edge indices, edge types, and labels.

        Returns:
            tuple: Ensemble predictions, all labels, and evaluation metrics.
        """
        all_label = data.test_label
        valid_edge_loader = EdgeDataLoader(data.test_edge_index, data.test_edge_type, 128, shuffle=False)
        
        ensemble_out = []
        for model in self.ensemble_model:
            all_out = []
            for batch_edges, batch_edge_types in tqdm(valid_edge_loader):
                out = self.bc_test_step(model, batch_edges, batch_edge_types, data.edge_index, data.edge_type)
                all_out.append(out)
            all_out = torch.cat(all_out, dim=0)
            m = compute_binary_metrics(all_out, all_label)
            print(m)
            ensemble_out.append(all_out)

        ensemble_out = torch.stack(ensemble_out, dim=0)
        ensemble_out = torch.mean(ensemble_out, dim=0)
        all_label = all_label.to(ensemble_out.device)
        metrics = compute_binary_metrics(ensemble_out, all_label)
        print('after ensemble')
        print(metrics)
        return ensemble_out, all_label, metrics

    @torch.no_grad()
    def bc_test_step(self, model, batch_edges, batch_edge_types, ms_edge_index, ms_edge_type):
        """
        Perform a single evaluation step for the validation or test set.

        Args:
            model: The model to evaluate.
            batch_edges: A batch of edges to evaluate.
            batch_edge_types: Corresponding edge types for the batch.
            ms_edge_index: Multi-source graph edges.
            ms_edge_type: Multi-source graph edge types.

        Returns:
            Tensor: Predicted output from the model.
        """
        model.eval()
        out = model.deepme_forward(batch_edges, batch_edge_types, ms_edge_index, ms_edge_type)

        return out

    @torch.no_grad()
    def test_step(self, model, batch_edges, ms_edge_index, ms_edge_type, eval_range, task_rel):
        """
        Perform a single evaluation step for the validation or test set with specific range and task relation.

        Args:
            model: The model to evaluate.
            batch_edges: A batch of edges to evaluate.
            ms_edge_index: Multi-source graph edges.
            ms_edge_type: Multi-source graph edge types.
            eval_range: Evaluation range.
            task_rel: Task-specific relations.

        Returns:
            Tensor: Predicted output from the model.
        """
        model.eval()
        all_edge_index, all_edge_types = self.negative_sampling.evaluation_with_range_2(batch_edges, task_rel, eval_range=eval_range)
        out = model.deepme_forward(all_edge_index, all_edge_types, ms_edge_index, ms_edge_type)
        return out

    @torch.no_grad()
    def genome_scale_predict(self, data, met_tensor):
        """
        Evaluate the ensemble of models on genome-scale.

        Args:
            data: Input data containing edge indices, edge types, and other necessary tensors.
            met_tensor: Tensor containing metabolite queries.

        Returns:
            tuple: All gene names, mean predicted scores, and variance scores.
        """
        num = len(met_tensor)
        valid_edge_loader = EdgeEDataLoader(met_tensor, 1, shuffle=False)
        ensemble_out = []

        for model in self.ensemble_model:
            all_out = []
            for batch_edges in tqdm(valid_edge_loader):
                out = self.test_step(model, batch_edges, data.edge_index, data.edge_type, data.all_gene_tensor, data.task_rel)
                all_out.append(out)
            all_out = torch.cat(all_out, dim=0)
            ensemble_out.append(all_out)

        ensemble_out = torch.stack(ensemble_out, dim=0)  # [n_models, n_samples, n_genes]
        mean_out = torch.mean(ensemble_out, dim=0)       # [n_samples, n_genes]
        var_out = torch.var(ensemble_out, dim=0)         # [n_samples, n_genes]

        return data.all_gene, mean_out, var_out

    @torch.no_grad()
    def predict_all(self, data):
        """
        Evaluate the ensemble of models on all data.

        Args:
            data: Input data containing edge indices, edge types, and other necessary tensors.

        Returns:
            Tensor: Ensemble predictions.
        """
        num = len(data.all_met_tensor)
        valid_edge_loader = EdgeEDataLoader(data.all_met_tensor, 16, shuffle=False)
        ensemble_out = []

        for model in self.ensemble_model:
            all_out = []
            for batch_edges in tqdm(valid_edge_loader):
                out = self.test_step(model, batch_edges, data.edge_index, data.edge_type, data.all_gene_tensor, data.task_rel)
                all_out.append(out)
            all_out = torch.cat(all_out, dim=0)
            ensemble_out.append(all_out)

        ensemble_out = torch.stack(ensemble_out, dim=0)
        ensemble_out = torch.mean(ensemble_out, dim=0)

        return ensemble_out