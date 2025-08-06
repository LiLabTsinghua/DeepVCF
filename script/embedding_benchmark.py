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

# change config
#-----------------------------------------------------
config = {
    'kg_name': 'kg_gpr',
    'device': 'cuda:2',
    
    'num_epochs': 500,
    'eval_interval': 1,
    'patience': 30,    
    'pretrain_batch_size': 1024,
    'finetune_batch_size': 32,
    'val_metric':'auprc',

    'kge':'DistMult',
    'decoder':'kge_emb',
    'hid_dim':300,
    'gnn_layer': 3,
    'lr':1e-5,
    'wd': 1e-5,
    'num_neg_samples': 100,

    'training':'deepme',
    'run_name':'kg_gpr'
}

print('-'*100)
print('print hyperparameter!!!')
for key, value in config.items():
    print(f'{key}:{value}')
print('-'*100)
#----------------------------------------------------
all_results = {}
for split in ['random','random_rev','metabolite_hold_out','amino_acid_hold_out',
              'carbohydrate_hold_out','cofactors_and_vitamins_hold_out','lipid_hold_out','nucleotide_hold_out',
              'secondary_metabolites_hold_out','gene_hold_out_1']:
# for split in ['nucleotide_hold_out','gene_hold_out_2']:
    config['split'] = split
    # load path
    #-----------------------------------------------------
    kg_path, train_path, test_path, model_saved_path = load_file_path(config)
    #-----------------------------------------------------

    # train model
    #-----------------------------------------------------
    set_seeds(41)

    test_results = []
    processor = DeepMEData(kg_path, train_path, test_path,
                            model_saved_path,
                            no_inverse_relations=['PPI'],
                            ensemble=True)
    data_list = processor.process()

    metrics = []
    all_out = []

    kge_emb = torch.load(model_saved_path + '{}_emb.pkl'.format(config['kge'])).to(config['device'])
    # pcm_emb = torch.load(model_saved_path + 'pcm_emb.pkl').to(config['device'])
    # rand_emb = torch.load(model_saved_path + 'rand_emb.pkl').to(config['device'])
    for i in range(5):
        data = data_list[i].to(config['device'])
        
        # stage 2
        model = DeepME_Model(data.num_nodes,  config['hid_dim'], # for node emb
            config['hid_dim'], data.num_edge_type, num_rgcn_layers=config['gnn_layer'],
            decoder_name=config['decoder'],
            dropout=0.1,
            task_rel=data.task_rel,
            kge_emb=kge_emb,
            rand_emb=None,
            pcm_emb=None,).to(config['device'])

        trainer = Trainer(
            model, config
        )
        trainer.train(data,)
        out, all_label, metric = trainer.bc_test(data, test=True)
        metrics.append(metric)
        all_out.append(out)
        print('_'*100)

    #print result
    #-----------------------------------------
    print(test_path)
    print('before ensemble')
    print(metrics)
    all_results[split] = metrics
    #-----------------------------------------

print(all_results)