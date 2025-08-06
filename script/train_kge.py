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

from model import KGE_Model
from sampling import NegativeSampling
from trainer import Trainer
from data_utils import DeepMEData
from utils import load_file_path, set_seeds
# change config
#-----------------------------------------------------
config = {
    'kg_name': 'kg_enhanced',
    'device': 'cuda:3',
    
    'num_epochs': 200,
    'eval_interval': 1,
    'patience': 20,    
    'pretrain_batch_size': 1024,
    'finetune_batch_size': 128,
    'val_metric':'auprc',

    'gnn_layer': 3,   # discard
    'kge':'DistMult',
    'decoder':'kge_emb',
    'hid_dim':300,
    'lr':1e-3,
    'wd': 1e-5,
    'num_neg_samples': 100,

    'split':'application1_eco2023',
    'training':'kge',
    'run_name':'kg_enhanced'
}

print('-'*100)
print('print hyperparameter!!!')
for key, value in config.items():
    print(f'{key}:{value}')
print('-'*100)
#----------------------------------------------------


# load path
#-----------------------------------------------------
kg_path, train_path, test_path, model_saved_path = load_file_path(config)
#-----------------------------------------------------

# train kge
#-----------------------------------------------------
set_seeds(46)

test_results = []
processor = DeepMEData(kg_path, train_path, test_path,
                        model_saved_path,
                        no_inverse_relations=['PPI'],
                        ensemble=True)
data_list = processor.process()


data = data_list[0].to(config['device'])
model = KGE_Model(data.num_nodes,  config['hid_dim'], # for node emb
    config['hid_dim'], # for node emb
    decoder_name=config['kge'],).to(config['device'])
negative_sampler = NegativeSampling(processor.entity2idx, config['num_neg_samples'])

trainer = Trainer(
    model, config, negative_sampling=negative_sampler
)
# train and test
trainer.kge_training(data, monitor='loss',add_train=False)
kge_emb = trainer.model.node_emb.data
kge_emb = F.normalize(kge_emb, p=2, dim=1)

torch.save(kge_emb, model_saved_path + '{}_emb.pkl'.format(config['kge']))
#-----------------------------------------