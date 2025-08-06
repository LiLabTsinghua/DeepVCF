import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

class DeepMEData:
    def __init__(self, kg_path, train_path, test_path, dict_saved_path, no_inverse_relations=None, ensemble=True, use_valid=True):
        """
        Initialize the DeepMEData object with paths to knowledge graph and training/testing data.
        
        Args:
            kg_path (str): Path to the knowledge graph file containing edges (h, r, t).
            train_path (str): Path to the training data file containing edges (h, r, t, label).
            test_path (str): Path to the testing data file containing edges (h, r, t, label).
            dict_saved_path (str): Directory path where entity2idx and relation2idx dictionaries will be saved.
            no_inverse_relations (set, optional): Set of relations that should not have inverse edges. Defaults to None.
            ensemble (bool, optional): Whether to perform ensemble learning through cross-validation. Defaults to True.
        """
        self.kg_path = kg_path
        self.train_path = train_path
        self.test_path = test_path
        self.dict_saved_path = dict_saved_path
        self.entity2idx = {}
        self.relation2idx = {}
        self.no_inverse_relations = no_inverse_relations if no_inverse_relations else set()
        self.ensemble = ensemble
        self.use_valid = use_valid
    
    def read_file(self):
        """
        Load data from files and return the knowledge graph and edge data (train, valid, test).

        Returns:
            tuple: Contains DataFrame objects for kg, train, and test datasets.
        """
        kg = pd.read_csv(self.kg_path, sep='\t', names=['h', 'r', 't']).drop_duplicates()
        train = pd.read_csv(self.train_path, sep='\t', names=['h', 'r', 't','label'])
        test = pd.read_csv(self.test_path, sep='\t', names=['h', 'r', 't','label'])
        return kg, train, test
    
    def get_index_dict(self, kg, train, test):
        """
        Constructs or loads mappings from entities and relations to their indices.

        Args:
            kg (pd.DataFrame): DataFrame containing the knowledge graph triples.
            train (pd.DataFrame): DataFrame containing the training triples.
            test (pd.DataFrame): DataFrame containing the testing triples.

        Returns:
            tuple: A tuple containing task-related relations and their corresponding tensor indices.
        """
        index_dict_dir = os.path.join(self.dict_saved_path, 'index_dict')
        ent_dict_path = os.path.join(index_dict_dir, 'entity2idx.npy')
        rel_dict_path = os.path.join(index_dict_dir, 'relation2idx.npy')
        task_rel = ['knock out', 'overexpress']

        if os.path.exists(ent_dict_path) and os.path.exists(rel_dict_path):
            print("Loading existing index dictionaries...")
            self.entity2idx = np.load(ent_dict_path, allow_pickle=True).item()
            self.relation2idx = np.load(rel_dict_path, allow_pickle=True).item()
        else:
            print("Creating new index dictionaries...")
            os.makedirs(index_dict_dir, exist_ok=True)
            ent = set(kg['h']).union(set(kg['t'])).union(set(train['h'])).union(set(train['t'])).union(set(test['h'])).union(set(test['t']))
            rel = set(kg['r'])
            rel_add_inverse = [x + '_inverse' for x in rel if x not in self.no_inverse_relations]
            all_rel = list(rel) + rel_add_inverse + list(task_rel)
            self.entity2idx = {x: idx for idx, x in enumerate(sorted(ent))}
            self.relation2idx = {x: idx for idx, x in enumerate(sorted(all_rel))}
            np.save(ent_dict_path, self.entity2idx)
            np.save(rel_dict_path, self.relation2idx)
            print(f"Index dictionaries saved to: {index_dict_dir}")

        task_rel_tensor = torch.LongTensor([self.relation2idx[x] for x in task_rel])
        return task_rel, task_rel_tensor
    
    def load_and_index_kg(self, df):
        """
        Load data from a DataFrame, map entities and relations to indices, and convert to tensors.

        Args:
            df (pd.DataFrame): The DataFrame containing knowledge graph triples.

        Returns:
            tuple: Tensors containing edge indices and types.
        """
        df = df.copy()
        index_df = pd.DataFrame()
        index_df['h'] = df['h'].map(self.entity2idx)
        index_df['t'] = df['t'].map(self.entity2idx)
        index_df['r'] = df['r'].map(self.relation2idx)
        edge_index = torch.tensor([index_df['h'].values, index_df['t'].values], dtype=torch.long)
        edge_type = torch.tensor(index_df['r'].values, dtype=torch.long)

        # Add inverse edges for non-excluded relations
        df_inv = df[~df['r'].isin(self.no_inverse_relations)]
        df_inv['r'] = [x + '_inverse' for x in list(df_inv['r'])]
        index_df_inv = pd.DataFrame()
        index_df_inv['h'] = df_inv['h'].map(self.entity2idx)
        index_df_inv['t'] = df_inv['t'].map(self.entity2idx)
        index_df_inv['r'] = df_inv['r'].map(self.relation2idx)
        inv_edge_index = torch.tensor([index_df_inv['t'].values, index_df_inv['h'].values], dtype=torch.long)
        inv_edge_type = torch.tensor(index_df_inv['r'].values, dtype=torch.long)

        # Concatenate original edges with inverse edges
        edge_index = torch.cat([edge_index, inv_edge_index], dim=1)
        edge_type = torch.cat([edge_type, inv_edge_type])

        return edge_index, edge_type
    
    def load_and_index_data(self, df, filter_label=False):
        """
        Loads data from a DataFrame, maps entities and relations to indices, and converts to tensors.

        Args:
            df (pd.DataFrame): The DataFrame containing data.
            filter_label (bool, optional): Whether to filter data by label. Defaults to False.

        Returns:
            tuple: Tensors containing edge indices and types.
        """
        df = df.copy()
        if filter_label:
            df = df[df['label'] == 1]
        index_df = pd.DataFrame()
        index_df['h'] = df['h'].map(self.entity2idx)
        index_df['t'] = df['t'].map(self.entity2idx)
        index_df['r'] = df['r'].map(self.relation2idx)
        edge_index = torch.tensor([index_df['h'].values, index_df['t'].values], dtype=torch.long)
        edge_type = torch.tensor(index_df['r'].values, dtype=torch.long)
        return edge_index, edge_type
    
    def train_val_split(self, train):
        """
        Splits the training data into training and validation sets.

        Args:
            train (pd.DataFrame): Training dataset.

        Returns:
            tuple or list: If ensemble is True, returns a list of tuples (train_split, val_split). Otherwise, returns a single tuple.
        """
        if self.ensemble:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = []
            for train_idx, val_idx in kf.split(train):
                train_split = train.iloc[train_idx]
                val_split = train.iloc[val_idx]
                folds.append((train_split, val_split))
            return folds
        else:
            train_split, val_split = train_test_split(train, test_size=0.2, random_state=42)
            return train_split, val_split
    
    def process(self,):
        """
        Process the data files and create PyG Data objects containing the knowledge graph and edge data.

        Returns:
            list or Data: List of Data objects if ensemble is True, otherwise a single Data object.
        """
        kg, train, test = self.read_file()
        _, task_rel_tensor = self.get_index_dict(kg, train, test)
        kg_edge_index, kg_edge_type = self.load_and_index_kg(kg)
        num_nodes = len(self.entity2idx)
        num_edge_type = len(self.relation2idx)

        if self.use_valid:
            if self.ensemble:
                folds = self.train_val_split(train)
                data_list = []
                for i in range(5):
                    train_fold, valid_fold = folds[i]
                    train_edge_index, train_edge_type = self.load_and_index_data(train_fold)
                    valid_edge_index, valid_edge_type = self.load_and_index_data(valid_fold)
                    test_edge_index, test_edge_type = self.load_and_index_data(test)
                    train_label = torch.FloatTensor(train_fold['label'].values)
                    valid_label = torch.FloatTensor(valid_fold['label'].values)
                    test_label = torch.FloatTensor(test['label'].values)
                    data_list.append(Data(
                        edge_index=kg_edge_index,
                        num_nodes=num_nodes,
                        edge_type=kg_edge_type,
                        num_edge_type=num_edge_type,
                        train_edge_index=train_edge_index,
                        train_edge_type=train_edge_type,
                        train_label=train_label,
                        valid_edge_index=valid_edge_index,
                        valid_edge_type=valid_edge_type,
                        valid_label=valid_label,
                        test_edge_index=test_edge_index,
                        test_edge_type=test_edge_type,
                        test_label=test_label,
                        task_rel=task_rel_tensor,
                    ))
                return data_list
            else:
                train_split, valid_split = self.train_val_split(train)
                train_edge_index, train_edge_type = self.load_and_index_data(train_split)
                valid_edge_index, valid_edge_type = self.load_and_index_data(valid_split)
                test_edge_index, test_edge_type = self.load_and_index_data(test)
                train_label = torch.FloatTensor(train_split['label'].values)
                valid_label = torch.FloatTensor(valid_split['label'].values)
                test_label = torch.FloatTensor(test['label'].values)
                print('Train:{}; Valid:{}; Test:{}'.format(len(train_edge_type), len(valid_edge_type), len(test_edge_type)))
                data = [Data(
                    edge_index=kg_edge_index,
                    num_nodes=num_nodes,
                    edge_type=kg_edge_type,
                    num_edge_type=num_edge_type,
                    train_edge_index=train_edge_index,
                    train_edge_type=train_edge_type,
                    train_label=train_label,
                    valid_edge_index=valid_edge_index,
                    valid_edge_type=valid_edge_type,
                    valid_label=valid_label,
                    test_edge_index=test_edge_index,
                    test_edge_type=test_edge_type,
                    test_label=test_label,
                    task_rel=task_rel_tensor,
                )]
                return data
        
        else:
            train_edge_index, train_edge_type = self.load_and_index_data(train)
            test_edge_index, test_edge_type = self.load_and_index_data(test)
            train_label = torch.FloatTensor(train['label'].values)
            test_label = torch.FloatTensor(test['label'].values)
            print('Train:{}; Test:{}'.format(len(train_edge_type), len(test_edge_type)))
            data = [Data(
                edge_index=kg_edge_index,
                num_nodes=num_nodes,
                edge_type=kg_edge_type,
                num_edge_type=num_edge_type,
                train_edge_index=train_edge_index,
                train_edge_type=train_edge_type,
                train_label=train_label,
                test_edge_index=test_edge_index,
                test_edge_type=test_edge_type,
                test_label=test_label,
                task_rel=task_rel_tensor,
            )]
            return data

    def fn_process(self,):
        """
        Process the data files and create a PyG Data object containing the knowledge graph and edge data.

        Returns:
        - data (PyG Data object): The processed graph data.
        FIXME:
        """

        kg, train, test,  = self.read_file()
        all_data = pd.concat([train,test])

        task_rel, task_rel_tensor = self.get_index_dict(kg,train,test)
        # 1.get kg edge
        kg_edge_index, kg_edge_type = self.load_and_index_kg(kg,)

        # Infer the number of nodes and edge types from the mappings
        num_nodes = len(self.entity2idx)
        num_edge_type = len(self.relation2idx)# neg range used in training


        # Process the training set (no inverse edges)
        train_edge_index, train_edge_type = self.load_and_index_data(train,)
        test_edge_index, test_edge_type = self.load_and_index_data(test, )
        
        # LABEL
        train_label = torch.FloatTensor([x for x in list(train['label'])])
        test_label = torch.FloatTensor([x for x in list(test['label'])])

        print('train:{};test:{}'.format(len(train_edge_type),len(test_edge_type)))


        data = Data(
            edge_index=kg_edge_index,
            num_nodes=num_nodes,
            edge_type=kg_edge_type,
            num_edge_type=num_edge_type,
            train_edge_index=train_edge_index,
            train_edge_type=train_edge_type,
            train_label=train_label,
            test_edge_index=test_edge_index,
            test_edge_type=test_edge_type,
            test_label=test_label,
            task_rel = task_rel_tensor,
        )

        return data


class DeepMEData_test:
    def __init__(self, kg_path, train_path,
                dict_saved_path,
                no_inverse_relations=None):
        """
        Initialize the KGDataProcessor with file paths and optional configurations.

        Parameters:
        - kg_file (str): Path to the file containing the full knowledge graph (h, r, t).
        - train_file (str): Path to the file containing the training edges (h, r, t).
        - valid_file (str): Path to the file containing the validation edges (h, r, t).
        - test_file (str, optional): Path to the file containing the test edges (h, r, t). Default is None.
        - no_inverse_relations (list, optional): List of relations that should not have inverse edges (default: None).
        - use_dup (str, optional): Whether to duplicate heads, tails, or none ('head', 'tail', 'none'). Default is 'head'.
        """
        self.kg_path = kg_path
        self.train_path = train_path

        # save path
        self.dict_saved_path = dict_saved_path
        
        self.no_inverse_relations = set(no_inverse_relations) if no_inverse_relations else set()

        # Initialize mappings for entities and relations
        self.entity2idx = {}  # Mapping from entities to indices
        self.relation2idx = {}  # Mapping from relations to indices

    def read_file(self):
        """
        Load data from files and return the knowledge graph and edge data (train, valid, test).

        Returns:
        - kg (DataFrame): The knowledge graph data.
        - train (DataFrame): The training data.
        - valid (DataFrame): The validation data.
        - test (DataFrame, optional): The test data, if available.
        """
        kg = pd.read_csv(self.kg_path, sep='\t', names=['h', 'r', 't'])
        train = pd.read_csv(self.train_path, sep='\t', names=['h', 'r', 't','label'])

        return kg, train

    def get_index_dict(self, train):
        # 定义路径
        index_dict_dir = os.path.join(self.dict_saved_path, 'index_dict')
        ent_dict_path = os.path.join(index_dict_dir, 'entity2idx.npy')
        rel_dict_path = os.path.join(index_dict_dir, 'relation2idx.npy')

        task_rel = ['knock out', 'overexpress']
        # 如果文件存在，加载已有字典

        print("Loading existing index dict...")
        self.entity2idx = np.load(ent_dict_path, allow_pickle=True).item()
        self.relation2idx = np.load(rel_dict_path, allow_pickle=True).item()

        task_rel_tensor = torch.LongTensor([self.relation2idx[x] for x in task_rel])
        
        print(self.relation2idx)
        print(task_rel)

        return task_rel, task_rel_tensor
    
    def load_and_index_kg(self, df,):
        """
        Load data from a file, map entities and relations to indices, and convert to tensors.

        Parameters:
        - df (DataFrame): The data to be processed (h, r, t).
        - add_inverse_edge (bool, optional): Whether to add inverse edges. Default is False.
        - dup_on (bool, optional): Whether to duplicate entities in the data. Default is False.

        Returns:
        - edge_index (Tensor): Tensor containing head and tail indices.
        - edge_type (Tensor): Tensor containing relation types.
        """
        df = df.copy()

        # Map entities and relations to their respective indices
        index_df = pd.DataFrame()
        index_df['h'] = df['h'].map(self.entity2idx)
        index_df['t'] = df['t'].map(self.entity2idx)
        index_df['r'] = df['r'].map(self.relation2idx)

        # Create edge_index and edge_type tensors
        edge_index = torch.tensor([index_df['h'].values, index_df['t'].values], dtype=torch.long)
        edge_type = torch.tensor(index_df['r'].values, dtype=torch.long)
        print('vinilla kg have {} triples'.format(edge_type.shape[0]))

        # Filter out relations that should not have inverse edges
        df = df[~df['r'].isin(self.no_inverse_relations)]
        df['r'] = [x + '_inverse' for x in list(df['r'])]

        # Re-map entities and relations for inverse edges
        index_df = pd.DataFrame()
        index_df['h'] = df['h'].map(self.entity2idx)
        index_df['t'] = df['t'].map(self.entity2idx)
        index_df['r'] = df['r'].map(self.relation2idx)

        # Create inverse edge_index and edge_type tensors
        inv_edge_index = torch.tensor([index_df['t'].values, index_df['h'].values], dtype=torch.long)  # Inverted direction
        inv_edge_type = torch.tensor(index_df['r'].values, dtype=torch.long)

        # Concatenate original edges with inverse edges
        edge_index = torch.cat([edge_index, inv_edge_index], dim=1)
        edge_type = torch.cat([edge_type, inv_edge_type])
        print('kg with inversed edges have {} triples'.format(edge_type.shape[0]))

        return edge_index, edge_type

    def process(self, h_prefix):
        """
        Process the data files and create a PyG Data object containing the knowledge graph and edge data.

        Returns:
        - data (PyG Data object): The processed graph data.
        """

        kg, train,   = self.read_file()

        task_rel, task_rel_tensor = self.get_index_dict(train)
        # 1.get kg edge
        kg_edge_index, kg_edge_type = self.load_and_index_kg(kg,)
        # print(kg_edge_type)
        # Infer the number of nodes and edge types from the mappings
        num_nodes = len(self.entity2idx)
        num_edge_type = len(self.relation2idx)# neg range used in training

        # neg range used in evaluation
        # DEFINE DATA_RANGE AND NEG_INDEX
        all_gene = [k for k,v in self.entity2idx.items() if k.startswith(h_prefix)]
        all_gene_tensor = torch.LongTensor([self.entity2idx[entity] for entity in all_gene]) 

        all_met = [k for k,v in self.entity2idx.items() if (k.startswith('Metabolite:')) & (k.endswith('_c'))]
        all_met_tensor = torch.LongTensor([self.entity2idx[entity] for entity in all_met]) 

        data = Data(
            edge_index=kg_edge_index,
            num_nodes=num_nodes,
            edge_type=kg_edge_type,
            num_edge_type=num_edge_type,
            task_rel = task_rel_tensor,
            all_gene = all_gene,
            all_gene_tensor = all_gene_tensor,
            all_met = all_met,
            all_met_tensor = all_met_tensor,
        )

        return data

#dataloder
from torch.utils.data import DataLoader, Dataset
class EdgeDataset(Dataset):
    def __init__(self, edge_index, edge_type):
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.num_edges = edge_index.size(1)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, idx):
        return self.edge_index[:, idx], self.edge_type[idx]

class EdgeDataLoader(DataLoader):
    def __init__(self, edge_index, edge_type, batch_size, shuffle=True):
        dataset = EdgeDataset(edge_index, edge_type)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        batch_edges = torch.stack([item[0] for item in batch], dim=1)
        batch_edge_types = torch.tensor([item[1] for item in batch])
        return batch_edges, batch_edge_types
    
# with label
class EdgeLDataset(Dataset):
    def __init__(self, edge_index, edge_type, label):
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.label = label
        self.num_edges = edge_index.size(1)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, idx):
        return self.edge_index[:, idx], self.edge_type[idx], self.label[idx]

class EdgeLDataLoader(DataLoader):
    def __init__(self, edge_index, edge_type, label, batch_size, shuffle=True):
        dataset = EdgeLDataset(edge_index, edge_type, label)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        batch_edges = torch.stack([item[0] for item in batch], dim=1)
        batch_edge_types = torch.tensor([item[1] for item in batch])
        batch_label = torch.tensor([item[2] for item in batch])
        return batch_edges, batch_edge_types, batch_label

# use for evalutaion
class EdgeEDataset(Dataset):
    def __init__(self, edge_index, ):
        self.edge_index = edge_index
        self.num_edges = len(edge_index)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, idx):
        return self.edge_index[idx]

class EdgeEDataLoader(DataLoader):
    def __init__(self, edge_index, batch_size, shuffle=True):
        dataset = EdgeEDataset(edge_index, )
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        batch_edges = torch.tensor([item for item in batch])
        return batch_edges