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

from model import KGE_Model, DeepME_Model
from sampling import NegativeSampling
from trainer import Trainer
from data_utils import DeepMEData
from utils import load_file_path, set_seeds
from metric import compute_binary_metrics

# change config
#-----------------------------------------------------
config = {
    'kg_name': 'kg_enhanced',
    'device': 'cuda:4',
    
    'num_epochs': 100,
    'eval_interval': 200,
    'patience': 30,    
    'pretrain_batch_size': 1024,
    'finetune_batch_size': 2048,
    'val_metric':'auprc',

    'gnn_layer': 3,   # discard
    'kge':'DistMult',
    'decoder':'kge_emb',
    'hid_dim':300,
    'lr':1e-5,
    'wd': 1e-5,
    'num_neg_samples': 100,

    'split':'application1_eco2023',
    'training':'deepme',
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

# train model
#-----------------------------------------------------
set_seeds(10)

test_results = []
processor = DeepMEData(kg_path, train_path, test_path,
                        model_saved_path,
                        no_inverse_relations=['PPI'],
                        ensemble=False,
                        use_valid=False,)
data_list = processor.process()

metrics = []
all_out = []

kge_emb = torch.load(model_saved_path + '{}_emb.pkl'.format(config['kge'])).to(config['device'])
# rand_emb = torch.load(model_saved_path + 'rand_emb.pkl').to(config['device'])
for i in range(1):
    data = data_list[i].to(config['device'])
    
    # stage 2
    model = DeepME_Model(data.num_nodes,  config['hid_dim'], # for node emb
        config['hid_dim'], data.num_edge_type, num_rgcn_layers=config['gnn_layer'],
        decoder_name=config['decoder'],
        dropout=0.1,
        task_rel=data.task_rel,
        kge_emb=kge_emb,
        pcm_emb=None,
        rand_emb=None).to(config['device'])

    trainer = Trainer(
        model, config
    )
    trainer.train(data,)
    out, all_label, metric = trainer.bc_test(data, test=True)
    metrics.append(metric)
    all_out.append(out)
    print('_'*100)

    torch.save(trainer.model.state_dict(), model_saved_path + 'DeepME_m_{}.pkl'.format(i))
#-----------------------------------------


#print result
#-----------------------------------------
print(test_path)
print('before ensemble')
print(metrics)
print('ensemble')
all_out = torch.stack(all_out,dim=0)
all_out = torch.mean(all_out,dim=0)
metrics = compute_binary_metrics(all_out, all_label)
print(metrics)
#-----------------------------------------