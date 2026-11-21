# Tau target validation data

`initial_gene_target_validation.txt` 是正式绘图数据：E7c 基线和 21 个靶点，
共 22 组、66 条重复记录，其中 64 条具有 tau 浓度及 tau/OD 测量值。
表格按绘图顺序排列：基线、14 个 GEM-covered 靶点、7 个 GEM-uncovered 靶点。

`initial_gene_target_validation_full_precision.txt` 保留相同记录的完整精度值；
`rounding_p_value_comparison.csv` 保留这些靶点舍入前后的独立样本、等方差、
单侧 Student t 检验对照（目标均值大于 E7c）。上一级
`initial_gene_target.txt` 保留相同实验组的来源格式记录。

| 字段 | 含义 |
| --- | --- |
| `strain` | 原始菌株/基因标签 |
| `is_baseline` | 首组 E7c 暂按对照标为 1，其余为 0；原始表未明确标注对照身份 |
| `replicate` | 原始每组内的记录序号，1–3 |
| `od600` | 原始“收样OD”；沿用 FFA 字段名，原始表未注明测量波长 |
| `tau_mg_per_l` | 原始 tau 浓度，mg/L |
| `tau_per_od600` | 由完整精度的匹配记录计算 tau/OD，再独立四舍五入至三位小数 |
| `source_group` | 原始实验组编号，保留来源编号，不重新连续编号；与 replicate 共同唯一标识一条记录 |
| `protein_node` | ECO 蛋白标识，基线留空 |
| `locus_tag` | 基因座标识，基线留空 |
| `gem_group` | Baseline、GEM-covered 或 GEM-uncovered |

正式表中的 OD、tau 浓度及 tau/OD 使用 ROUND_HALF_UP 独立保留三位小数。
保留记录的数值没有重新计算。

rpe 第 3 条、sspB 第 2 条记录的 tau 浓度缺失，对应的 tau/OD 也留空，OD 数值保留。空值不表示零值。后续汇总时应使用各指标实际可用的记录数，并通过 source_group 保留原始分组关系。

GEM 分类沿用 ECO basic KG 的 Catalyzes-source 蛋白集合；katE 按原始
iML1515 CAT 反应的 GPR（b1732 or b3942）记录为 GEM-covered。
该反应的代谢物均被列为常见代谢物，因此 KG 中未保留其 Catalyzes 边。
分类已写入数据，绘图脚本直接读取。
