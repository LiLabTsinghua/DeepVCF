import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class DeepVCF_Knowledge:
    """
    Knowledge Graph (KG) processing class.
    Handles global entity/relation indexing and static background knowledge construction.
    """
    def __init__(self, kg_path, train_path, dict_saved_path, no_inverse_relations=None, seed=42):
        """
        Args:
            kg_path (str): Path to the KG file (tab-separated h, r, t).
            train_path (str): Path to training data.
            dict_saved_path (str): Directory to save/load index dictionaries (.npy).
            no_inverse_relations (set): Relations that should NOT have an inverse edge.
            seed (int): Random seed for reproducibility.
        """
        self.kg_path = kg_path
        self.train_path = train_path
        self.dict_saved_path = dict_saved_path
        self.entity2idx = {}
        self.relation2idx = {}
        self.no_inverse_relations = no_inverse_relations if no_inverse_relations else set()
        self.seed = seed

    def read_file(self):
        """
        Reads KG file and categorizes entities into Metabolites and Proteins.
        
        Returns:
            tuple: (kg_df, all_met, all_gene, metabolic_gene, non_metabolic_gene)
        """
        kg = pd.read_csv(self.kg_path, sep='\t', names=['h', 'r', 't']).drop_duplicates()
        
        # Categorize entities based on naming conventions
        all_met = {x for x in set(kg['h']).union(set(kg['t'])) if 'Metabolite' in x}
        all_gene = {x for x in set(kg['h']).union(set(kg['t'])) if 'Protein' in x}
        
        # Identify metabolic genes (those that 'Catalyzes' something)
        metabolic_gene = set(kg[kg['r'] == 'Catalyzes']['h'])
        non_metabolic_gene = all_gene - metabolic_gene
        
        print(f'Count - Total Met: {len(all_met)}, Total Gene: {len(all_gene)}')
        print(f'Count - Metabolic Gene: {len(metabolic_gene)}, Non-metabolic: {len(non_metabolic_gene)}')

        return kg, all_met, all_gene, metabolic_gene, non_metabolic_gene
    
    def get_index_dict(self, kg):
        """
        Loads or creates mapping dictionaries for entities and relations.
        Includes original relations, inverse relations, and specific task relations.
        """
        index_dict_dir = os.path.join(self.dict_saved_path, 'index_dict')
        ent_dict_path = os.path.join(index_dict_dir, 'entity2idx.npy')
        rel_dict_path = os.path.join(index_dict_dir, 'relation2idx.npy')
        task_rel = ['knock out', 'overexpress'] # Core perturbation relations

        if os.path.exists(ent_dict_path) and os.path.exists(rel_dict_path):
            print("Loading existing index dictionaries...")
            self.entity2idx = np.load(ent_dict_path, allow_pickle=True).item()
            self.relation2idx = np.load(rel_dict_path, allow_pickle=True).item()
        else:
            print("Creating new index dictionaries...")
            os.makedirs(index_dict_dir, exist_ok=True)
            ent = set(kg['h']).union(set(kg['t']))
            rel = set(kg['r'])
            
            # Generate inverse relation labels (e.g., 'Catalyzes' -> 'Catalyzes_inverse')
            rel_add_inverse = [x + '_inverse' for x in rel if x not in self.no_inverse_relations]
            all_rel = list(rel) + rel_add_inverse + list(task_rel)
            
            self.entity2idx = {x: idx for idx, x in enumerate(ent)}
            self.relation2idx = {x: idx for idx, x in enumerate(all_rel)}
            
            np.save(ent_dict_path, self.entity2idx)
            np.save(rel_dict_path, self.relation2idx)

        task_rel_tensor = torch.LongTensor([self.relation2idx[x] for x in task_rel])
        return task_rel, task_rel_tensor
    
    def load_and_index_kg(self, df):
        """
        Maps raw triples to indices and adds inverse edges for message passing.
        
        Returns:
            tuple: (edge_index tensor, edge_type tensor)
        """
        df = df.copy()
        
        # Map forward edges
        idx_h = df['h'].map(self.entity2idx).values
        idx_t = df['t'].map(self.entity2idx).values
        idx_r = df['r'].map(self.relation2idx).values
        
        edge_index = torch.from_numpy(np.vstack([idx_h, idx_t])).long()
        edge_type = torch.tensor(idx_r, dtype=torch.long)

        # Map inverse edges
        df_inv = df[~df['r'].isin(self.no_inverse_relations)].copy()
        df_inv['r_inv'] = (df_inv['r'] + '_inverse').map(self.relation2idx)
        
        inv_idx_h = df_inv['h'].map(self.entity2idx).values
        inv_idx_t = df_inv['t'].map(self.entity2idx).values
        
        inv_edge_index = torch.from_numpy(np.vstack([inv_idx_t, inv_idx_h])).long()
        inv_edge_type = torch.tensor(df_inv['r_inv'].values, dtype=torch.long)

        # Concatenate forward and backward edges
        full_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)
        full_edge_type = torch.cat([edge_type, inv_edge_type])

        return full_edge_index, full_edge_type

    def train_val_split(self, kg):
        """Splits KG triples for internal validation (link prediction style)."""
        # FIXME:can also use bagging ensemble strategy
        return train_test_split(kg, test_size=0.02, random_state=self.seed)

    def process(self):
        """Main pipeline for background knowledge processing."""
        kg, all_met, all_gene, metabolic_gene, non_metabolic_gene = self.read_file()
        task_rel, task_rel_tensor = self.get_index_dict(kg)
        
        # Meta information for model coverage analysis
        coverage = {
            'all_met': all_met, 
            'all_gene': all_gene, 
            'metabolic_gene': metabolic_gene, 
            'non_metabolic_gene': non_metabolic_gene, 
            'modification': task_rel
        }

        kg_train, kg_val = self.train_val_split(kg)
        train_idx, train_type = self.load_and_index_kg(kg_train)
        val_idx, val_type = self.load_and_index_kg(kg_val)

        print('Number of triples:{}'.format(len(train_type)))
        print('Number of val triples:{}'.format(len(val_type)))

        # Package into PyG Data object
        knowledge_data = Data(
            kg_train_edge_index=train_idx,
            kg_train_edge_type=train_type,
            kg_val_edge_index=val_idx,
            kg_val_edge_type=val_type,
            num_nodes=len(self.entity2idx),
            num_edge_type=len(self.relation2idx),
            task_rel=task_rel_tensor,
        )
        return knowledge_data, coverage

class DeepVCF_Data:
    """
    Task Data processing class.
    Handles experimental data (perturbation labels) and prepares it for training/testing.
    Supports K-Fold Cross Validation.
    """
    def __init__(self, train_path, test_path, dict_saved_path, ensemble=True, k=5, use_valid=True, seed=42):
        self.train_path = train_path
        self.test_path = test_path
        self.dict_saved_path = dict_saved_path
        self.ensemble = ensemble
        self.k = k
        self.use_valid = use_valid
        self.seed = seed
    
    def read_file(self):
        """Loads training and testing CSVs with labels."""
        train = pd.read_csv(self.train_path, sep='\t', names=['h', 'r', 't', 'label'])
        test = pd.read_csv(self.test_path, sep='\t', names=['h', 'r', 't', 'label'])
        return train, test
    
    def get_index_dict(self):
        """Loads existing index dictionaries from the knowledge process."""
        index_dict_dir = os.path.join(self.dict_saved_path, 'index_dict')
        ent_path = os.path.join(index_dict_dir, 'entity2idx.npy')
        rel_path = os.path.join(index_dict_dir, 'relation2idx.npy')

        if not os.path.exists(ent_path) or not os.path.exists(rel_path):
            raise FileNotFoundError("[ERROR] Mappings not found. Run DeepVCF_Knowledge first.")

        print("Loading index dictionaries for task data...")
        self.entity2idx = np.load(ent_path, allow_pickle=True).item()
        self.relation2idx = np.load(rel_path, allow_pickle=True).item()
    
    def load_and_index_data(self, df):
        """Maps experimental data triples to indices and validates against dictionary."""
        df = df.copy()

        # Check for OOD (Out of Dictionary) entities/relations
        missing_h = set(df['h']) - set(self.entity2idx.keys())
        missing_t = set(df['t']) - set(self.entity2idx.keys())
        missing_r = set(df['r']) - set(self.relation2idx.keys())

        if missing_h or missing_t or missing_r:
            raise KeyError(f"[ERROR] Found entities/relations not in KG dictionary. H:{len(missing_h)}, T:{len(missing_t)}, R:{len(missing_r)}")

        edge_index = torch.from_numpy(
            np.vstack([df['h'].map(self.entity2idx).values, 
                       df['t'].map(self.entity2idx).values])
        ).long()
        edge_type = torch.tensor(df['r'].map(self.relation2idx).values, dtype=torch.long)
        
        return edge_index, edge_type
    
    def train_val_split(self, train):
        """Generates cross-validation folds or a single split."""
        if self.ensemble:
            kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
            return [(train.iloc[t_idx], train.iloc[v_idx]) for t_idx, v_idx in kf.split(train)]
        else:
            return train_test_split(train, test_size=0.2, random_state=self.seed)
    
    def process(self):
        """Main pipeline for experimental data processing."""
        train, test = self.read_file()
        self.get_index_dict()
        data_list = []

        if self.use_valid:
            splits = self.train_val_split(train)
            # Handle list of folds or single tuple
            if not self.ensemble: splits = [splits]

            for train_fold, valid_fold in splits:
                tr_idx, tr_tp = self.load_and_index_data(train_fold)
                va_idx, va_tp = self.load_and_index_data(valid_fold)
                ts_idx, ts_tp = self.load_and_index_data(test)
                print('Train:{}; Valid:{}; Test:{}'.format(
                        len(tr_tp), len(va_tp), len(ts_tp)))
                data_list.append(Data(
                    train_edge_index=tr_idx, train_edge_type=tr_tp,
                    train_label=torch.FloatTensor(train_fold['label'].values),
                    valid_edge_index=va_idx, valid_edge_type=va_tp,
                    valid_label=torch.FloatTensor(valid_fold['label'].values),
                    test_edge_index=ts_idx, test_edge_type=ts_tp,
                    test_label=torch.FloatTensor(test['label'].values)
                ))
        else:
            # Simple Train/Test split
            tr_idx, tr_tp = self.load_and_index_data(train)
            ts_idx, ts_tp = self.load_and_index_data(test)
            print('Train:{}; Test:{}'.format(len(tr_tp), len(ts_tp)))
            data_list.append(Data(
                train_edge_index=tr_idx, train_edge_type=tr_tp,
                train_label=torch.FloatTensor(train['label'].values),
                test_edge_index=ts_idx, test_edge_type=ts_tp,
                test_label=torch.FloatTensor(test['label'].values)
            ))
        return data_list


# --- DataLoader Section ---

class EdgeDataset(Dataset):
    """
    Dataset class for Knowledge Graph triples without labels.
    Used for unsupervised tasks or link prediction background.
    """
    def __init__(self, edge_index, edge_type):
        """
        Args:
            edge_index (LongTensor): [2, E] tensor where row 0 is head and row 1 is tail.
            edge_type (LongTensor): [E] tensor representing relation IDs.
        """
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_edges = edge_index.size(1)

    def __len__(self):
        """Returns the total number of edges in the dataset."""
        return self.num_edges

    def __getitem__(self, idx):
        """
        Retrieves a single edge triple by index.
        Returns: (head_tail_indices, relation_id)
        """
        return self.edge_index[:, idx], self.edge_type[idx]

class EdgeDataLoader(DataLoader):
    """
    Custom DataLoader for unlabeled edges.
    Simplifies the batching process by automatically applying a collation function.
    """
    def __init__(self, edge_index, edge_type, batch_size, shuffle=True):
        dataset = EdgeDataset(edge_index, edge_type)
        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=self.collate_fn
        )

    @staticmethod
    def collate_fn(batch):
        """
        Custom collation to stack multiple samples into a single batch tensor.
        
        Args:
            batch (list): List of tuples from EdgeDataset.__getitem__
        Returns:
            batch_edges (LongTensor): [2, BatchSize]
            batch_edge_types (LongTensor): [BatchSize]
        """
        # Dim=1 used for edge_index to keep [2, BatchSize] structure
        batch_edges = torch.stack([item[0] for item in batch], dim=1)
        batch_edge_types = torch.tensor([item[1] for item in batch])
        return batch_edges, batch_edge_types

# --- Labeled Data Section (for Downstream Tasks) ---

class EdgeLDataset(Dataset):
    """
    Dataset class for labeled edges (Experimental triples).
    Used for supervised learning where each perturbation has a target value (label).
    """
    def __init__(self, edge_index, edge_type, label):
        """
        Args:
            edge_index (LongTensor): [2, E] tensor of head and tail IDs.
            edge_type (LongTensor): [E] tensor of relation IDs.
            label (FloatTensor): [E] tensor of experimental results/values.
        """
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.label = label
        self.num_edges = edge_index.size(1)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, idx):
        """
        Returns: (head_tail_pair, relation, label)
        """
        return self.edge_index[:, idx], self.edge_type[idx], self.label[idx]

class EdgeLDataLoader(DataLoader):
    """
    Custom DataLoader for labeled experimental data.
    Ensures that labels are correctly batched alongside graph indices.
    """
    def __init__(self, edge_index, edge_type, label, batch_size, shuffle=True):
        dataset = EdgeLDataset(edge_index, edge_type, label)
        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=self.collate_fn
        )

    @staticmethod
    def collate_fn(batch):
        """
        Stacks samples into batch tensors including labels.
        
        Returns:
            batch_edges (LongTensor): [2, BatchSize]
            batch_edge_types (LongTensor): [BatchSize]
            batch_label (FloatTensor): [BatchSize]
        """
        batch_edges = torch.stack([item[0] for item in batch], dim=1)
        batch_edge_types = torch.tensor([item[1] for item in batch])
        batch_label = torch.tensor([item[2] for item in batch])
        return batch_edges, batch_edge_types, batch_label