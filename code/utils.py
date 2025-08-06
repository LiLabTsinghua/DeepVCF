import os
from time import time
import pandas as pd
import numpy as np
import tqdm
import random
from collections import defaultdict
from scipy.stats import rankdata
import requests

import torch

from model import DeepME_Model
from data_utils import DeepMEData_test
from sampling import NegativeSampling
from trainer import Ensemble_Tester

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def load_file_path(config):
    if config['split'] == 'random':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/embedding_benchmark/random/train.txt'
        test_path = '../data/me_data/train_data/embedding_benchmark/random/test.txt'
        model_saved_path = '../trained_model/embedding_benchmark/{}/'.format(config['run_name'])
    
    elif config['split'] == 'random_rev':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/embedding_benchmark/random_rev/train.txt'
        test_path = '../data/me_data/train_data/embedding_benchmark/random_rev/test.txt'
        model_saved_path = '../trained_model/embedding_benchmark/{}/'.format(config['run_name'])

    elif config['split'] == 'metabolite_hold_out':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/embedding_benchmark/metabolite_hold_out/train.txt'
        test_path = '../data/me_data/train_data/embedding_benchmark/metabolite_hold_out/test.txt'
        model_saved_path = '../trained_model/embedding_benchmark/{}/'.format(config['run_name'])

    elif config['split'] == 'amino_acid_hold_out':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/embedding_benchmark/amino_acid_hold_out/train.txt'
        test_path = '../data/me_data/train_data/embedding_benchmark/amino_acid_hold_out/test.txt'
        model_saved_path = '../trained_model/embedding_benchmark/{}/'.format(config['run_name'])
    
    elif config['split'] == 'carbohydrate_hold_out':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/embedding_benchmark/carbohydrate_hold_out/train.txt'
        test_path = '../data/me_data/train_data/embedding_benchmark/carbohydrate_hold_out/test.txt'
        model_saved_path = '../trained_model/embedding_benchmark/{}/'.format(config['run_name'])
    
    elif config['split'] == 'cofactors_and_vitamins_hold_out':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/embedding_benchmark/cofactors_and_vitamins_hold_out/train.txt'
        test_path = '../data/me_data/train_data/embedding_benchmark/cofactors_and_vitamins_hold_out/test.txt'
        model_saved_path = '../trained_model/embedding_benchmark/{}/'.format(config['run_name'])
    
    elif config['split'] == 'lipid_hold_out':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/embedding_benchmark/lipid_hold_out/train.txt'
        test_path = '../data/me_data/train_data/embedding_benchmark/lipid_hold_out/test.txt'
        model_saved_path = '../trained_model/embedding_benchmark/{}/'.format(config['run_name'])
    
    elif config['split'] == 'nucleotide_hold_out':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/embedding_benchmark/nucleotide_hold_out/train.txt'
        test_path = '../data/me_data/train_data/embedding_benchmark/nucleotide_hold_out/test.txt'
        model_saved_path = '../trained_model/embedding_benchmark/{}/'.format(config['run_name'])
    
    elif config['split'] == 'secondary_metabolites_hold_out':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/embedding_benchmark/secondary_metabolites_hold_out/train.txt'
        test_path = '../data/me_data/train_data/embedding_benchmark/secondary_metabolites_hold_out/test.txt'
        model_saved_path = '../trained_model/embedding_benchmark/{}/'.format(config['run_name'])
    
    elif config['split'] == 'gene_hold_out_1':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/embedding_benchmark/gene_hold_out_1/train.txt'
        test_path = '../data/me_data/train_data/embedding_benchmark/gene_hold_out_1/test.txt'
        model_saved_path = '../trained_model/embedding_benchmark/{}/'.format(config['run_name'])

    elif config['split'] == 'gene_hold_out_2':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/embedding_benchmark/gene_hold_out_2/train.txt'
        test_path = '../data/me_data/train_data/embedding_benchmark/gene_hold_out_2/test.txt'
        model_saved_path = '../trained_model/embedding_benchmark/{}/'.format(config['run_name'])

    elif config['split'] == 'application1':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application1/test.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    elif config['split'] == 'application1_eco2023':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application1/ECO_2023.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application1_eco2023_head':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train_head.txt'
        test_path = '../data/me_data/application1/test1.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    elif config['split'] == 'application1_fseof':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train_eco_fseof.txt'
        test_path = '../data/me_data/application1/test1.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    elif config['split'] == 'application1_eco2023_m':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train_mix.txt'
        test_path = '../data/me_data/application1/test1.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    elif config['split'] == 'application1_unseen_met_head':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application1/representional_split/test_unseen_met_head.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    elif config['split'] == 'application1_low_met_head':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application1/representional_split/test_low_met_head.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application1_high_met_head':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application1/representional_split/test_high_met_head.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    elif config['split'] == 'application1_unseen_gene_tail':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application1/representional_split/test_unseen_gene_tail.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    elif config['split'] == 'application1_low_gene_tail':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application1/representional_split/test_low_gene_tail.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application1_high_gene_tail':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application1/representional_split/test_high_gene_tail.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    elif config['split'] == 'application1_eco2024':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application1/ECO_2024.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application1_laser':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application1/laser.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    elif config['split'] == 'application2_eco2023':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application2/ECO_2023.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application2_eco2024':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application2/ECO_2024.txt'
        model_saved_path = '../trained_model/ECO-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application3_hetero_gene':
        kg_path = '../data/KG/ECO/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/metabolic_train.txt'
        test_path = '../data/me_data/application3/test.txt'
        model_saved_path = '../trained_model/ECO-KG-hetero_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_fseof':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train_fseof.txt'
        test_path = '../data/me_data/application4/sce/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    elif config['split'] == 'application4_m':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train_m.txt'
        test_path = '../data/me_data/application4/sce/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    elif config['split'] == 'application4_sce1':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application4/sce/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_sce2':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application4/sce/non_metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_cgl1':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application4/cgl/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_cgl2':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/train_data/train.txt'
        test_path = '../data/me_data/application4/cgl/non_metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_sce_fn10':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/sce/fn_0.txt'
        test_path = '../data/me_data/application4/sce/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_sce_fn20':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/sce/fn_1.txt'
        test_path = '../data/me_data/application4/sce/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_sce_fn30':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/sce/fn_2.txt'
        test_path = '../data/me_data/application4/sce/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_sce_fn40':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/sce/fn_3.txt'
        test_path = '../data/me_data/application4/sce/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_sce_fn50':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/sce/fn_4.txt'
        test_path = '../data/me_data/application4/sce/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_sce_fs50':
        kg_path = '../data/KG/SCE/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/sce/fn_4.txt'
        test_path = '../data/me_data/application4/sce/metabolic_test.txt'
        model_saved_path = '../trained_model/SCE-KG_SCE-50/{}/'.format(config['run_name'])

    elif config['split'] == 'application4_sce_fnall':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/sce/fn_all.txt'
        test_path = '../data/me_data/application4/sce/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_cgl_fn10':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/cgl/fn_0.txt'
        test_path = '../data/me_data/application4/cgl/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_cgl_fn20':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/cgl/fn_1.txt'
        test_path = '../data/me_data/application4/cgl/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_cgl_fn30':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/cgl/fn_2.txt'
        test_path = '../data/me_data/application4/cgl/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_cgl_fn40':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/cgl/fn_3.txt'
        test_path = '../data/me_data/application4/cgl/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])
    
    elif config['split'] == 'application4_cgl_fn50':
        kg_path = '../data/KG/ALL/{}.txt'.format(config['kg_name'])
        train_path = '../data/me_data/application4/cgl/fn_4.txt'
        test_path = '../data/me_data/application4/cgl/metabolic_test.txt'
        model_saved_path = '../trained_model/UNI-KG_ECO-Pre-2023/{}/'.format(config['run_name'])

    return kg_path, train_path, test_path, model_saved_path

    

def load_tester(query_met, query_gene_prefix, model_weight):
    if model_weight == 'ECO-KG_ECO-Pre-2023':
        config = {
            'device': 'cuda:3',

            'hid_dim': 300,
            'gnn_layer': 3,
            'lr':1e-5,
            'wd': 1e-5,
            'num_neg_samples': 1,

            'model_path':'../trained_model/ECO-KG_ECO-Pre-2023/kg_enhanced'
        }

        h_prefix = query_gene_prefix

        processor = DeepMEData_test(kg_path='../data/KG/ECO/kg_enhanced.txt', train_path='../data/me_data/train_data/train.txt',
                        dict_saved_path=config['model_path'],
                        no_inverse_relations=['PPI'],)
        data = processor.process(h_prefix).to(config['device'])
        entity2idx = processor.entity2idx
        relation2idx = processor.relation2idx

        query_met_tensor = torch.LongTensor([entity2idx['Metabolite:' + x] for x in query_met]).to(config['device'])

        ensemble_model = []
        for i in range(5):
            kge_emb = torch.load(config['model_path'] + '/DistMult_emb.pkl',weights_only=True).to(config['device'])

            model = DeepME_Model(data.num_nodes,  config['hid_dim'], # for node emb
                config['hid_dim'], data.num_edge_type, num_rgcn_layers=config['gnn_layer'],
                decoder_name='kge_emb',
                dropout=0.1,
                task_rel=data.task_rel,
                kge_emb=kge_emb,
                pcm_emb=None,)
            model.load_state_dict(torch.load(config['model_path'] + '/DeepME_{}.pkl'.format(i),weights_only=True))
            model.to(config['device'])
            # print(next(model.parameters()).device)
            ensemble_model.append(model)
        
        negative_sampler = NegativeSampling(processor.entity2idx, config['num_neg_samples'])  # only for kge
        tester = Ensemble_Tester(
            ensemble_model, negative_sampler, 
        )

        return tester, data, query_met_tensor
    
    elif model_weight == 'ECO-KG-hetero_ECO-Pre-2023':
        config = {
            'device': 'cuda:3',

            'hid_dim': 300,
            'gnn_layer': 3,
            'lr':1e-5,
            'wd': 1e-5,
            'num_neg_samples': 1,

            'model_path':'../trained_model/ECO-KG-hetero_ECO-Pre-2023/kg_basic_repeat10'
        }

        h_prefix = query_gene_prefix

        processor = DeepMEData_test(kg_path='../data/KG/ECO/kg_basic_extend.txt', train_path='../data/me_data/train_data/metabolic_train.txt',
                        dict_saved_path=config['model_path'],
                        no_inverse_relations=['PPI'],)
        data = processor.process(h_prefix).to(config['device'])
        entity2idx = processor.entity2idx
        relation2idx = processor.relation2idx

        query_met_tensor = torch.LongTensor([entity2idx['Metabolite:' + x] for x in query_met]).to(config['device'])

        ensemble_model = []
        for i in range(5):
            kge_emb = torch.load(config['model_path'] + '/DistMult_emb.pkl',weights_only=True).to(config['device'])

            model = DeepME_Model(data.num_nodes,  config['hid_dim'], # for node emb
                config['hid_dim'], data.num_edge_type, num_rgcn_layers=config['gnn_layer'],
                decoder_name='kge_emb',
                dropout=0.1,
                task_rel=data.task_rel,
                kge_emb=kge_emb,
                pcm_emb=None,)
            model.load_state_dict(torch.load(config['model_path'] + '/DeepME_{}.pkl'.format(i),weights_only=True))
            model.to(config['device'])
            # print(next(model.parameters()).device)
            ensemble_model.append(model)
        
        negative_sampler = NegativeSampling(processor.entity2idx, config['num_neg_samples'])  # only for kge
        tester = Ensemble_Tester(
            ensemble_model, negative_sampler, 
        )

        return tester, data, query_met_tensor
    
    elif model_weight == 'UNI-KG_ECO-Pre-2023_fn-sce-50':
        config = {
            'device': 'cuda:3',

            'hid_dim': 300,
            'gnn_layer': 3,
            'lr':1e-5,
            'wd': 1e-5,
            'num_neg_samples': 1,

            'model_path':'../trained_model/UNI-KG_ECO-Pre-2023/kg_enhanced'
        }

        h_prefix = query_gene_prefix

        processor = DeepMEData_test(kg_path='../data/KG/ALL/kg_enhanced.txt', train_path='../data/me_data/train_data/train.txt',
                        dict_saved_path=config['model_path'],
                        no_inverse_relations=['PPI'],)
        data = processor.process(h_prefix).to(config['device'])
        entity2idx = processor.entity2idx
        relation2idx = processor.relation2idx

        query_met_tensor = torch.LongTensor([entity2idx['Metabolite:' + x] for x in query_met]).to(config['device'])

        ensemble_model = []
        for i in range(5):
            kge_emb = torch.load(config['model_path'] + '/DistMult_emb.pkl',weights_only=True).to(config['device'])

            model = DeepME_Model(data.num_nodes,  config['hid_dim'], # for node emb
                config['hid_dim'], data.num_edge_type, num_rgcn_layers=config['gnn_layer'],
                decoder_name='w_all',
                dropout=0.1,
                task_rel=data.task_rel,
                kge_emb=kge_emb,
                pcm_emb=None,)
            model.load_state_dict(torch.load(config['model_path'] + '/DeepME_sce_fn_50_{}.pkl'.format(i),weights_only=True))
            model.to(config['device'])
            # print(next(model.parameters()).device)
            ensemble_model.append(model)
        
        negative_sampler = NegativeSampling(processor.entity2idx, config['num_neg_samples'])  # only for kge
        tester = Ensemble_Tester(
            ensemble_model, negative_sampler, 
        )

        return tester, data, query_met_tensor