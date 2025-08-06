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

from torch_geometric.nn import GAE, VGAE

from model import KGE_Model, DeepME_Model
from sampling import NegativeSampling
from trainer import Trainer
from data_utils import DeepMEData
from utils import load_file_path, set_seeds
from metric import compute_binary_metric

config = {
    'kg_name': 'kg_enhanced',
    'device': 'cuda:3',
    
    'num_epochs': 20,
    'eval_interval': 1,
    'patience': 10,    
    'pretrain_batch_size': 1024,
    'finetune_batch_size': 2,
    'val_metric':'auprc',

    'gnn_layer': 3,   # discard
    'kge':'DistMult',
    'decoder':'kge_emb',
    'hid_dim':300,
    'lr':1e-5,
    'wd': 1e-5,
    'num_neg_samples': 100,

    'split':'finetune_fang',
    'training':'deepme',

    'run_name':'kg_enhanced'
}

# load path
#-----------------------------------------------------
kg_path, train_path, test_path, model_saved_path = load_file_path(config)
#-----------------------------------------------------

# train model
#-----------------------------------------------------
set_seeds(42)

test_results = []
processor = DeepMEData(kg_path, train_path, test_path,
                        model_saved_path,
                        no_inverse_relations=['PPI'],
                        ensemble=True)
data = processor.fn_process().to(config['device'])

metrics = []
all_out = []

kge_emb = torch.load(model_saved_path + '{}_emb.pkl'.format(config['kge'])).to(config['device'])
for i in range(5):

    # stage 2
    model = DeepME_Model(data.num_nodes,  config['hid_dim'], # for node emb
        config['hid_dim'], data.num_edge_type, num_rgcn_layers=config['gnn_layer'],
        decoder_name=config['decoder'],
        dropout=0.1,
        task_rel=data.task_rel,
        kge_emb=kge_emb,
        pcm_emb=None,).to(config['device'])
    model.load_state_dict(torch.load(model_saved_path + 'DeepME_{}.pkl'.format(i),weights_only=True))

    trainer = Trainer(
        model, config
    )
    fn_record = trainer.deepme_finetune(data,)
    torch.save(trainer.model.state_dict(), model_saved_path + 'DeepME_fang_{}.pkl'.format(i))
#-----------------------------------------