import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

class DeepVCF_Knowledge:
    def __init__(self, kg_path, train_path, dict_saved_path, no_inverse_relations=None, seed=42):
        self.kg_path = kg_path
        self.train_path = train_path
        self.dict_saved_path = dict_saved_path
        self.entity2idx = {}
        self.relation2idx = {}
        self.no_inverse_relations = no_inverse_relations if no_inverse_relations else set()
        self.seed = seed

    def read_file(self):
        kg = pd.read_csv(self.kg_path, sep='\t', names=['h', 'r', 't']).drop_duplicates()
        all_met = {x for x in set(kg['h']).union(set(kg['t'])) if 'Metabolite' in x}
        all_gene = {x for x in set(kg['h']).union(set(kg['t'])) if 'Protein' in x}
        metabolic_gene = set(kg[kg['r'] == 'Catalyzes']['h'])
        non_metabolic_gene = all_gene - metabolic_gene
        print(f'num of all_met: {len(all_met)}')
        print(f'num of all_gene: {len(all_gene)}')
        print(f'num of metabolic_gene: {len(metabolic_gene)}')
        print(f'num of non_metabolic_gene: {len(non_metabolic_gene)}')

        # cf data is another knowledge
        # train = pd.read_csv(self.train_path, sep='\t', names=['h', 'r', 't', 'label'])
        # train = train_df['h','r','t']
        # label = train_df['label']

        return kg, all_met, all_gene, metabolic_gene, non_metabolic_gene
    
    def get_index_dict(self, kg, ):
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
            ent = set(kg['h']).union(set(kg['t']))
            rel = set(kg['r'])
            rel_add_inverse = [x + '_inverse' for x in rel if x not in self.no_inverse_relations]
            all_rel = list(rel) + rel_add_inverse + list(task_rel)
            self.entity2idx = {x: idx for idx, x in enumerate(ent)}
            self.relation2idx = {x: idx for idx, x in enumerate(all_rel)}
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
        edge_index = torch.from_numpy(
            np.vstack([index_df['h'].values, index_df['t'].values])
        ).long()
        edge_type = torch.tensor(index_df['r'].values, dtype=torch.long)

        # Add inverse edges for non-excluded relations
        df_inv = df[~df['r'].isin(self.no_inverse_relations)].copy()
        df_inv['r'] = [x + '_inverse' for x in list(df_inv['r'])]
        index_df_inv = pd.DataFrame()
        index_df_inv['h'] = df_inv['h'].map(self.entity2idx)
        index_df_inv['t'] = df_inv['t'].map(self.entity2idx)
        index_df_inv['r'] = df_inv['r'].map(self.relation2idx)
        inv_edge_index = torch.from_numpy(
            np.vstack([index_df_inv['t'].values, index_df_inv['h'].values])
        ).long()
        inv_edge_type = torch.tensor(index_df_inv['r'].values, dtype=torch.long)

        # Concatenate original edges with inverse edges
        edge_index = torch.cat([edge_index, inv_edge_index], dim=1)
        edge_type = torch.cat([edge_type, inv_edge_type])

        return edge_index, edge_type

    def train_val_split(self, kg):
        train_split, val_split = train_test_split(kg, test_size=0.02, random_state=self.seed)
        return train_split, val_split

    def process(self,):
        kg, all_met, all_gene, metabolic_gene, non_metabolic_gene = self.read_file()
        task_rel, task_rel_tensor = self.get_index_dict(kg, )
        coverage = {'all_met':all_met, 'all_gene':all_gene, 'metabolic_gene':metabolic_gene, 'non_metabolic_gene':non_metabolic_gene, 'modification':task_rel}

        kg_train, kg_val = self.train_val_split(kg)
        kg_train_edge_index, kg_train_edge_type = self.load_and_index_kg(kg_train)
        kg_val_edge_index, kg_val_edge_type = self.load_and_index_kg(kg_val)

        num_nodes = len(self.entity2idx)
        num_edge_type = len(self.relation2idx)
        print('Number of triples:{}'.format(len(kg_train_edge_type)))
        print('Number of val triples:{}'.format(len(kg_val_edge_type)))
        
        knowledge = Data(
            kg_train_edge_index=kg_train_edge_index,
            kg_train_edge_type=kg_train_edge_type,
            kg_val_edge_index=kg_val_edge_index,
            kg_val_edge_type=kg_val_edge_type,
            num_nodes=num_nodes,
            num_edge_type=num_edge_type,
            task_rel=task_rel_tensor,
        )
        return knowledge, coverage

class DeepVCF_Data:
    def __init__(self, train_path, test_path, dict_saved_path, ensemble=True, k=5, use_valid=True, seed=42):
        self.train_path = train_path
        self.test_path = test_path
        self.dict_saved_path = dict_saved_path
        self.entity2idx = {}
        self.relation2idx = {}
        self.ensemble = ensemble
        self.k = k
        self.use_valid = use_valid
        self.seed = seed
    
    def read_file(self):
        train = pd.read_csv(self.train_path, sep='\t', names=['h', 'r', 't','label'])
        test = pd.read_csv(self.test_path, sep='\t', names=['h', 'r', 't','label'])
        return train, test
    
    def get_index_dict(self):
        index_dict_dir = os.path.join(self.dict_saved_path, 'index_dict')
        ent_dict_path = os.path.join(index_dict_dir, 'entity2idx.npy')
        rel_dict_path = os.path.join(index_dict_dir, 'relation2idx.npy')

        # ----------- 新增：检查字典是否存在 -----------
        if not os.path.exists(ent_dict_path) or not os.path.exists(rel_dict_path):
            raise FileNotFoundError(
                f"[ERROR] 映射字典文件不存在！请先运行 DeepVCF_knowledge 生成映射字典。\n"
                f"缺失路径：\nentity2idx: {ent_dict_path}\nrelation2idx: {rel_dict_path}"
            )

        print("Loading existing index dictionaries...")
        self.entity2idx = np.load(ent_dict_path, allow_pickle=True).item()
        self.relation2idx = np.load(rel_dict_path, allow_pickle=True).item()
    
    def load_and_index_data(self, df):
        """
        Loads data from a DataFrame, maps entities and relations to indices, and converts to tensors.
        """
        df = df.copy()

        # ----------- 新增：检查所有实体和关系是否在字典中 -----------
        missing_entities_h = set(df['h']) - set(self.entity2idx.keys())
        missing_entities_t = set(df['t']) - set(self.entity2idx.keys())
        missing_relations = set(df['r']) - set(self.relation2idx.keys())

        error_message = ""
        if missing_entities_h:
            error_message += f"Head实体不存在于entity2idx：{missing_entities_h}\n"
        if missing_entities_t:
            error_message += f"Tail实体不存在于entity2idx：{missing_entities_t}\n"
        if missing_relations:
            error_message += f"关系不存在于relation2idx：{missing_relations}\n"

        if error_message:
            raise KeyError(
                "[ERROR] 映射失败！存在不在字典中的实体或关系。\n"
                + error_message +
                "请检查数据或重新生成字典"
            )

        index_df = pd.DataFrame()
        index_df['h'] = df['h'].map(self.entity2idx)
        index_df['t'] = df['t'].map(self.entity2idx)
        index_df['r'] = df['r'].map(self.relation2idx)

        edge_index = torch.from_numpy(
            np.vstack([index_df['h'].values, index_df['t'].values])
        ).long()
        edge_type = torch.tensor(index_df['r'].values, dtype=torch.long)
        return edge_index, edge_type
    
    def train_val_split(self, train):
        if self.ensemble:
            kf = KFold(n_splits=self.k, shuffle=True, random_state=self.seed)
            folds = []
            for train_idx, val_idx in kf.split(train):
                train_split = train.iloc[train_idx]
                val_split = train.iloc[val_idx]
                folds.append((train_split, val_split))
            return folds
        else:
            train_split, val_split = train_test_split(train, test_size=0.2, random_state=self.seed)
            return train_split, val_split
    
    def process(self):
        train, test = self.read_file()
        self.get_index_dict()

        if self.use_valid:
            if self.ensemble:
                folds = self.train_val_split(train)
                data_list = []
                for i in range(self.k):
                    train_fold, valid_fold = folds[i]
                    train_edge_index, train_edge_type = self.load_and_index_data(train_fold)
                    valid_edge_index, valid_edge_type = self.load_and_index_data(valid_fold)
                    test_edge_index, test_edge_type = self.load_and_index_data(test)
                    train_label = torch.FloatTensor(train_fold['label'].values)
                    valid_label = torch.FloatTensor(valid_fold['label'].values)
                    test_label = torch.FloatTensor(test['label'].values)
                    print('Train:{}; Valid:{}; Test:{}'.format(
                        len(train_edge_type), len(valid_edge_type), len(test_edge_type)))
                    data_list.append(Data(
                        train_edge_index=train_edge_index,
                        train_edge_type=train_edge_type,
                        train_label=train_label,
                        valid_edge_index=valid_edge_index,
                        valid_edge_type=valid_edge_type,
                        valid_label=valid_label,
                        test_edge_index=test_edge_index,
                        test_edge_type=test_edge_type,
                        test_label=test_label,
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
                print('Train:{}; Valid:{}; Test:{}'.format(
                    len(train_edge_type), len(valid_edge_type), len(test_edge_type)))
                data_list = [Data(
                    train_edge_index=train_edge_index,
                    train_edge_type=train_edge_type,
                    train_label=train_label,
                    valid_edge_index=valid_edge_index,
                    valid_edge_type=valid_edge_type,
                    valid_label=valid_label,
                    test_edge_index=test_edge_index,
                    test_edge_type=test_edge_type,
                    test_label=test_label,
                )]
                return data_list
        
        else:
            train_edge_index, train_edge_type = self.load_and_index_data(train)
            test_edge_index, test_edge_type = self.load_and_index_data(test)
            train_label = torch.FloatTensor(train['label'].values)
            test_label = torch.FloatTensor(test['label'].values)
            print('Train:{}; Test:{}'.format(len(train_edge_type), len(test_edge_type)))
            data_list = [Data(
                train_edge_index=train_edge_index,
                train_edge_type=train_edge_type,
                train_label=train_label,
                test_edge_index=test_edge_index,
                test_edge_type=test_edge_type,
                test_label=test_label,
            )]
            return data_list


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