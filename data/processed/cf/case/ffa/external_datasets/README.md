# External FFA gene-label datasets

These are the Xu 2013 (`xu_2013nc`) and Fang 2021 (`fang_2021nc`)
gene-level FFA tables from
`/home/shengkun/project/deepvcf_iter/DeepVCF/data/CF_data/ffa`.
They are **not** CF protein–relation–metabolite triples: the source files
contain a category, an ECO gene name, and a binary label, but no single
metabolite identifier or intervention relation.

Each processed TXT is tab-delimited and headerless, with four columns in this
order: `protein_node`, `gene`, `category`, `label`. `protein_node` uses the ECO gene map from
`data/raw/ECO/eco_gene_info.txt` when the gene name matches exactly,
ignoring case. Unmapped records are retained with an empty first field; no
ambiguous aliases are guessed. Exact duplicate source rows are removed.
`manifest.json` records input and output counts.

Regenerate from the repository root:

```bash
conda run --no-capture-output -n deepvcf \
  python data/scripts/cf/process_ffa_external.py
```

Use `--input-dir` to select a different source location and `--output-dir`
to write a separate reproduction copy.
