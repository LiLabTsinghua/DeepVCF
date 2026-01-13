# AI virtual cell factories for enhanced and genome-wide target prediction
This repo contains a reconstructed verison for DeepVCF, which is model proposed in our paper **"AI virtual cell factories for enhanced and genome-wide target prediction"**.


## Brief Introduction
DeepVCF is AI-driven framework that integrates comprehensive biological knowledge with experimental data to predict engineering targets at a genome-wide scale. By learning system-level relationships between genes and metabolites, DeepVCF extends the scope of traditional metabolic modelling and enables accurate identification of both metabolic and non-metabolic targets.
![](./fig/figure1.jpg)


## Requirements
Build the environment using the following commands in few minutes.
(We have tested that DeepVCF can run on the latest [PyTorch](https://pytorch.org/get-started/locally/).)
```
conda create -n deepvcf python=3.9 -y
conda activate deepvcf
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install torch-geometric
pip install scikit-learn matplotlib pandas ipykernel 
```
(optional)[Mamba](https://mamba.readthedocs.io/en/latest/) can be used for faster package installation.

## Data & Code
We provide necessary data and code for running DeepVCF in following structure:
```
.
├── code
│   └── __pycache__
├── data
│   ├── KG
│   │   ├── ALL
│   │   ├── CGL
│   │   ├── ECO
│   │   └── SCE
│   ├── me_data
│   │   ├── cross_species_transfer
│   │   │   ├── cgl
│   │   │   └── sce
│   │   ├── dataset
│   │   ├── ffa
│   │   ├── metabolic_gene
│   │   ├── non_metabolic_gene
│   │   └── train_data
│   │       └── embedding_benchmark
│   │           ├── amino_acid_hold_out
│   │           ├── carbohydrate_hold_out
│   │           ├── cofactors_and_vitamins_hold_out
│   │           ├── gene_hold_out_1
│   │           ├── gene_hold_out_2
│   │           ├── lipid_hold_out
│   │           ├── metabolite_hold_out
│   │           ├── nucleotide_hold_out
│   │           ├── random
│   │           ├── random_rev
│   │           └── secondary_metabolites_hold_out
│   └── other_data
├── fig
├── script
└── trained_model

```
see our paper for details.


## To train new DeepVCF from scarch, please run the following script
```
# Modify the config if needed.
python script/train_deepvcf.py
```

## Reproduce
For easily reproduce, we reconstruct the code. \
This version largely reproduce our paper results (see script/tutorials.ipynb).
![](./fig/reproduce.png)


## Using DeepVCF for real-world genome-scale target prediction
see script/tutorials.ipynb for more details.

🔥 We have successfully applied DeepVCF to the following cases:
```
1.FFAs overproduction → 6 new non-metabolic KO targets with 66.7% success rate (paper).
2.Taurine overproduction → 18 new OE targets with 60.0% success rate (in preparation).
```

🔔 NOTE: 
```
1.We recommend to use DeepVCF_PreFT in real-world applications.
2.DeepVCF/DeepVCF_PreFT might cause confusion in practical applications by simultaneously prioritizing KO and OE of same gene. (For example, rank one in the top 10, and rank the other in the top 50)
```

## To do list
- [ ] Add more species KG.
- [ ] Integrate automated text-mining pipeline.
- [ ] Add active learning part→“lab in the Loop”.
- [ ] Refine algorithms.


## Coopration
We welcome co-operation on cell factory design alghrithm development and real-world applications. If you have any questions or suggestions, please feel free to contact us.


## Contact
nsk25@mails.tsinghua.edu.cn.
