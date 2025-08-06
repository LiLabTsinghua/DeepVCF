import os.path as osp
import time
import sys
import pandas as pd
import os
code_path = os.path.abspath(os.path.join('../code'))
sys.path.append(code_path)
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import tqdm

from utils import load_tester


# user query → (met, species, model) 
#---------------------------------------------------------------------
run_name = 'pser__L_c'
query_met = ['pser__L_c']
query_species = 'eco'
query_gene_prefix = 'ECO_Protein:'
model_weight = 'ECO-KG_ECO-Pre-2023'
#---------------------------------------------------------------------


# load tester
#---------------------------------------------------------------------
tester, data, query_met_tensor = load_tester(query_met, query_gene_prefix, model_weight)
#---------------------------------------------------------------------


# make predict
#---------------------------------------------------------------------
candidate_gene, score, uncertainty = tester.genome_scale_predict(data, query_met_tensor)

func_df = pd.read_csv('../pred/gene_func/{}2.txt'.format(query_species),sep='\t')
id2entry = dict(zip(func_df['Entry'], func_df['Entry Name']))
id2func = dict(zip(func_df['Entry'], func_df['Function [CC]']))
id2ec = dict(zip(func_df['Entry'], func_df['EC number']))
id2db = dict(zip(func_df['Entry'], func_df['DNA binding']))
id2ca = dict(zip(func_df['Entry'], func_df['Catalytic activity']))
id2name = dict(zip(func_df['Entry'], func_df['name']))

entry = [id2entry[x.split(':')[1]] if x.split(':')[1] in id2entry.keys() else '' for x in tqdm(candidate_gene)]
name = [id2name[x.split(':')[1]] if x.split(':')[1] in id2name.keys() else '' for x in tqdm(candidate_gene)]
func = [id2func[x.split(':')[1]] if x.split(':')[1] in id2func.keys() else '' for x in tqdm(candidate_gene)]
ec = [id2ec[x.split(':')[1]] if x.split(':')[1] in id2ec.keys() else '' for x in tqdm(candidate_gene)]
db = [id2ec[x.split(':')[1]] if x.split(':')[1] in id2db.keys() else '' for x in tqdm(candidate_gene)]
ca = [id2ec[x.split(':')[1]] if x.split(':')[1] in id2ca.keys() else '' for x in tqdm(candidate_gene)]


df = pd.DataFrame({'uniprot_id':[x.split(':')[1] for x in candidate_gene]*2,
                'entry':entry*2,
                'name':name*2,
                'modification':['KO']*len(candidate_gene) + ['OE']*len(candidate_gene),
                'score':score.cpu().squeeze(dim=1).numpy(),
                'uncertainty':uncertainty.cpu().squeeze(dim=1).numpy(),
                'func':func*2,})

df = df.sort_values(by="score", ascending=False)

df.to_csv('../pred/{}.txt'.format(run_name), sep='\t', index=False)
#---------------------------------------------------------------------
