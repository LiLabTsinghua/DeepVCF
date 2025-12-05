import os.path as osp
import time
import pickle
import sys
import os
code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../code'))
sys.path.append(code_path)
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import tqdm

from model import DeepVCF_Model
from data_utils import DeepVCF_Knowledge, DeepVCF_Data
from utils import set_seeds

# change config
#-----------------------------------------------------
config = {
    'device': 'cuda:4',
    'seed':53,

    'kg_path': '/home/shengkun/lab_github/DeepVCF/data/KG/ECO/kg_enhanced.txt',
    'mechanistic_pretrain_path':'/home/shengkun/lab_github/DeepVCF/data/me_data/train_data/eco_fseof_1_6.txt',
    'train_path':'/home/shengkun/lab_github/DeepVCF/data/me_data/train_data/train.txt',
    'test_path':'/home/shengkun/lab_github/DeepVCF/data/me_data/metabolic_gene/combined_test.txt',

    'use_drn':True,
    'hidden_dim':300,
    'drn_method':'DistMult',
    'drn_lr':1e-3,
    'drn_wd':1e-5,
    'drn_batch_size':1024,
    'drn_num_epochs':200,
    'drn_eval_interval':1,
    'drn_patience':20,
    'drn_num_neg':100,

    'ensemble':True,
    'k':5,
    'hidden_channels':300,
    'dropout':0.1,
    'tpn_lr':5e-5,
    'tpn_wd':1e-5,
    'tpn_batch_size':128,
    'tpn_num_epochs':500,
    'tpn_eval_interval':1,
    'tpn_patience':30,

    'model_saved_dir':'../trained_model/ECO/',
    'model_name':'DeepVCF_k5_53',
}

print('-'*100)
print('print hyperparameter!!!')
for key, value in config.items():
    print(f'{key}:{value}')
print('-'*100)

# set seed
#----------------------------------------------------
set_seeds(config['seed'])

# load knoweldge
k_processor = DeepVCF_Knowledge(config['kg_path'], config['train_path'], config['model_saved_dir'],no_inverse_relations=['PPI'],seed=config['seed'])
knowledge,coverage = k_processor.process()
num_nodes, num_edge_type, task_rel = knowledge.num_nodes, knowledge.num_edge_type, knowledge.task_rel.to(config['device'])

# load data
d1_processor = DeepVCF_Data(config['mechanistic_pretrain_path'], config['test_path'],
                        config['model_saved_dir'],
                        ensemble=True,
                        k=config['k'],
                        seed=config['seed'],
                        )
pre_data_list = d1_processor.process()

d_processor = DeepVCF_Data(config['train_path'], config['test_path'],
                        config['model_saved_dir'],
                        ensemble=config['ensemble'],
                        k=config['k'],
                        seed=config['seed'],
                        )
true_data_list = d_processor.process()

# initial model
model = DeepVCF_Model(config, num_nodes, num_edge_type, task_rel,
                    coverage).to(config['device'])

# drn training
model.train_drn(knowledge)

# pre-train tpn
# model.train_tpn(pre_data_list)

# tpn training
model.train_tpn(true_data_list)

# tpn testing
all_out, all_label, ensemble_metrics, indiv_outputs, indiv_metrics = model.test_tpn(true_data_list,test=True)   
print(ensemble_metrics)
print(indiv_metrics)

# save
model.save_model()
#-----------------------------------------