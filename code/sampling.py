import torch

class NegativeSampling:
    def __init__(self, entity2idx, num_neg_samples=None):
        """
        Initialize the NegativeSampling class.

        Parameters:
        - entity2idx: A dictionary mapping entities to their indices (e.g., {'entity_name': index}).
        - num_neg_samples: Number of negative samples to generate per positive edge. If None, generate one negative sample per positive edge.
        """
        self.entity2idx = entity2idx
        self.num_neg_samples = num_neg_samples or 1

    def general_negative_samples(self, edge_index, edge_type, num_nodes):
        """
        Perform negative sampling by corrupting either the source or the target of each edge.

        Parameters:
        - edge_index: Tensor of shape (2, num_edges), containing the edge indices.
        - edge_type: Tensor of shape (num_edges,), containing the edge types.
        - num_nodes: Number of nodes in the graph.

        Returns:
        - neg_edge_index: Tensor of shape (2, num_edges * num_neg_samples), containing the negative edge indices.
        - neg_edge_type: Tensor of shape (num_edges * num_neg_samples,), containing the negative edge types.

        NOTE: this function is used for kge or rgcn pretrain
        """
        num_edges = edge_index.size(1)

        # Expand edge_index and edge_type for multiple negative samples per edge
        edge_index = edge_index.repeat_interleave(self.num_neg_samples, dim=1)
        edge_type = edge_type.repeat_interleave(self.num_neg_samples)

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

    def task_negative_samples_with_prefix(self, edge_index, edge_type, cand_prefix, type):
        """
        Perform negative sampling by corrupting either the source or the target of each edge.
        
        Parameters:
        - edge_index: Tensor of shape (2, num_edges), containing the edge indices.
        - edge_type: Tensor of shape (num_edges,), containing the edge types.
        - num_nodes: Number of nodes in the graph.
        - cand_prefix: Prefix to filter candidate entities.
        - type: 'head' or 'tail', indicating whether to corrupt the head or tail node.

        Returns:
        - neg_edge_index: Tensor of shape (2, num_edges * num_neg_samples), containing the negative edge indices.
        - neg_edge_type: Tensor of shape (num_edges * num_neg_samples,), containing the negative edge types.
        """
        num_edges = edge_index.size(1)

        # Expand edge_index and edge_type for multiple negative samples per edge
        edge_index = edge_index.repeat_interleave(self.num_neg_samples, dim=1)
        edge_type = edge_type.repeat_interleave(self.num_neg_samples)

        # Create negative edge index by corrupting either the source or the target node
        neg_edge_index = edge_index.clone()

        # Define the sampling range for negative nodes
        sample_range = self.get_candidate(cand_prefix)
        valid_nodes = torch.tensor(sample_range, device=edge_index.device)

        # Ensure valid nodes are not empty
        if len(valid_nodes) == 0:
            raise ValueError(f"No valid nodes found for the prefix {cand_prefix}.")

        # Create negative edge by corrupting source or target nodes
        if type == 'head':
            neg_edge_index[0] = valid_nodes[torch.randint(len(valid_nodes), (neg_edge_index.shape[1],), device=edge_index.device)]
        elif type == 'tail':
            neg_edge_index[1] = valid_nodes[torch.randint(len(valid_nodes), (neg_edge_index.shape[1],), device=edge_index.device)]
        else:
            raise ValueError("Invalid type specified! Use 'head' or 'tail'.")

        return neg_edge_index, edge_type

    def task_negative_samples_with_range(self, edge_index, edge_type, task_rel, train_range, type='head'):
        """
        Perform negative sampling by corrupting either the source or the target of each edge.
        
        Parameters:
        - edge_index: Tensor of shape (2, num_edges), containing the edge indices.
        - edge_type: Tensor of shape (num_edges,), containing the edge types.
        - num_nodes: Number of nodes in the graph.
        - cand_prefix: Prefix to filter candidate entities.
        - type: 'head' or 'tail', indicating whether to corrupt the head or tail node.

        Returns:
        - neg_edge_index: Tensor of shape (2, num_edges * num_neg_samples), containing the negative edge indices.
        - neg_edge_type: Tensor of shape (num_edges * num_neg_samples,), containing the negative edge types.
        """
        num_edges = edge_index.size(1)

        # Expand edge_index and edge_type for multiple negative samples per edge
        edge_index = edge_index.repeat_interleave(self.num_neg_samples, dim=1)
        edge_type = edge_type.repeat_interleave(self.num_neg_samples)

        # Create negative edge index by corrupting either the source or the target node
        neg_edge_index = edge_index.clone()
        # Define the sampling range for negative nodes
        valid_src_nodes = train_range[0]
        valid_dst_nodes = train_range[1]

        # Create negative edge by corrupting source or target nodes
        if type == 'head':
            neg_edge_index[0] = valid_src_nodes[torch.randint(len(valid_src_nodes), (neg_edge_index.shape[1],), device=edge_index.device)]
        elif type == 'tail':
            neg_edge_index[1] = valid_dst_nodes[torch.randint(len(valid_dst_nodes), (neg_edge_index.shape[1]), device=edge_index.device)]

        return neg_edge_index, edge_type

    def evaluation_with_range(self, edge_index, edge_type, task_rel, eval_range):
        """
        Generate negative samples for evaluation within a specified range.

        Parameters:
        - edge_index: Tensor of shape (2, num_edges), containing the edge indices.
        - edge_type: Tensor of shape (num_edges,), containing the edge types.
        - task_rel: List or Tensor of task-specific relations to consider.
        - eval_range: List or Tensor of valid nodes for sampling.

        Returns:
        - all_edge_index: Tensor of shape (2, num_edges * len(task_rel) * len(eval_range)),
                        containing the edge indices for both positive and negative samples.
        - edge_type: Tensor of shape (num_edges * len(task_rel) * len(eval_range)),
                    containing the edge types for the expanded edges.
        """
        num_edges = edge_index.size(1)
        num_valid_nodes = len(eval_range)
        num_task_rel = len(task_rel)

        # Expand edge_index for all combinations of valid nodes and task relations
        edge_index = edge_index.repeat_interleave(num_valid_nodes * num_task_rel, dim=1)

        # Create negative edges by replacing the source node (head corruption)
        all_edge_index = edge_index.clone()
        all_edge_index[0] = eval_range.repeat(num_edges * num_task_rel)

        # Expand edge_type to match the expanded edge_index
        edge_type = task_rel.repeat_interleave(num_valid_nodes).repeat(num_edges)

        return all_edge_index, edge_type
    
    def evaluation_with_range_2(self, dst_index, task_rel, eval_range):
        """
        Generate negative samples for evaluation within a specified range.

        Parameters:
        - edge_index: Tensor of shape (2, num_edges), containing the edge indices.
        - edge_type: Tensor of shape (num_edges,), containing the edge types.
        - task_rel: List or Tensor of task-specific relations to consider.
        - eval_range: List or Tensor of valid nodes for sampling.

        Returns:
        - all_edge_index: Tensor of shape (2, num_edges * len(task_rel) * len(eval_range)),
                        containing the edge indices for both positive and negative samples.
        - edge_type: Tensor of shape (num_edges * len(task_rel) * len(eval_range)),
                    containing the edge types for the expanded edges.
        """
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
    
    def task_hard_negative_samples_from_train(self, edge_index, edge_type, type, node_ids, weights):
        """
        Perform negative sampling by corrupting either the source or the target of each edge.
        
        Parameters:
        - edge_index: Tensor of shape (2, num_edges), containing the edge indices.
        - edge_type: Tensor of shape (num_edges,), containing the edge types.
        - num_nodes: Number of nodes in the graph.
        - cand_prefix: Prefix to filter candidate entities.
        - type: 'head' or 'tail', indicating whether to corrupt the head or tail node.

        Returns:
        - neg_edge_index: Tensor of shape (2, num_edges * num_neg_samples), containing the negative edge indices.
        - neg_edge_type: Tensor of shape (num_edges * num_neg_samples,), containing the negative edge types.

        NOTE:This function is discard now
        """
        num_edges = edge_index.size(1)

        # Expand edge_index and edge_type for multiple negative samples per edge
        edge_index = edge_index.repeat_interleave(self.num_neg_samples, dim=1)
        edge_type = edge_type.repeat_interleave(self.num_neg_samples)

        # Create negative edge index by corrupting either the source or the target node
        neg_edge_index = edge_index.clone()

        # Create negative edge by corrupting source or target nodes
        if type == 'head_hard':
            neg_edge_index[0] = self.weighted_negative_sampling(node_ids, weights, neg_edge_index.shape[1], device=edge_index.device)
        else:
            raise ValueError("Invalid type specified! Use 'head' or 'tail'.")

        return neg_edge_index, edge_type
    
    def get_candidate(self, cand_prefix):
        """
        Get the list of candidate entities based on a given prefix.
        
        Parameters:
        - cand_prefix: The prefix used to filter entities.

        Returns:
        - A list of indices of entities that match the given prefix.
        """
        return [v for k, v in self.entity2idx.items() if k.startswith(cand_prefix)]

    def weighted_negative_sampling(self, node_ids, weights, num_neg_samples, device):
        """
        Perform negative sampling based on node frequencies. The negative samples
        are selected with a probability proportional to the frequency of each node
        in the provided source node IDs.
        
        Args:
            src_node_ids (torch.Tensor): A tensor containing the source node IDs 
                                          from the training edge index.
            num_neg_samples (int): The number of negative samples to generate.
            device (str): The device on which the tensor should be allocated ('cpu' or 'cuda').
        
        Returns:
            torch.Tensor: A tensor containing the indices of the negative samples 
                          (same shape as `num_neg_samples`).
        """
        # Step 5: Use torch.multinomial to sample negative nodes based on weights
        neg_src_index = torch.multinomial(weights, num_neg_samples, replacement=True)

        # Step 6: Map sampled indices back to node IDs
        neg_src_index = torch.tensor([node_ids[i] for i in neg_src_index], device=device)

        return neg_src_index