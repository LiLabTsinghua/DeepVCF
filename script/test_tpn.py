import os.path as osp
import time
import pickle
import sys
import pandas as pd
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

# test
#-----------------------------------------------------
set_seeds(42)

test_results = []
processor = DeepMEData(kg_path, train_path, test_path,
                        model_saved_path,
                        no_inverse_relations=['PPI'],
                        ensemble=True)
data_list = processor.process()

metrics = []
all_out = []

kge_emb = torch.load(model_saved_path + '{}_emb.pkl'.format(config['kge'])).to(config['device'])

for i in range(5):
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

    # load trained model
    model.load_state_dict(torch.load(model_saved_path + 'DeepME_{}.pkl'.format(i),weights_only=True))

    trainer = Trainer(
        model, config
    )
    out, all_label, metric = trainer.bc_test(data, test=True)
    
    #save single run
    # df = pd.DataFrame({'score':out.squeeze(dim=1).cpu().numpy(),
    #                     'label':all_label.cpu().numpy()})
    # df.to_csv(model_saved_path + 'results/DeepME_{}_result_app1_test3.txt'.format(i),sep='\t',index=False)

    metrics.append(metric)
    all_out.append(out)
    print('_'*100)
#-----------------------------------------


#save_results
#-----------------------------------------
print(test_path)
print('before ensemble')
print(metrics)
print('ensemble')
all_out = torch.stack(all_out,dim=0)
all_out = torch.mean(all_out,dim=0)
df = pd.DataFrame({'score':all_out.squeeze(dim=1).cpu().numpy(),
                    'label':all_label.cpu().numpy()})
# df.to_csv(model_saved_path + 'results/DeepME_result_app1_unseen_met_head_new.txt',sep='\t',index=False)
metrics = compute_binary_metrics(all_out, all_label)
print(metrics)
#-----------------------------------------