# 2025/4/15
"""
(1) 将<separated>变成>
(2) 将所有的数据去掉空格合并在一起
"""
import os
save_dir = r'D:\t5chem\data\USPTO-Schwaller-shf-s-aug5'
datadir_source = r'E:\zlx\rxn_yields-0.0.1-MLST\data\USPTO-Schwaller-Subword-shf-s-aug5-norm\train.source'
datadir_target = r'E:\zlx\rxn_yields-0.0.1-MLST\data\USPTO-Schwaller-Subword-shf-s-aug5-norm\train.target'

with open(datadir_source, 'r', encoding='utf-8') as f_source:
    reaction_list = f_source.readlines()

with open(datadir_target, 'r', encoding='utf-8') as f_target:
    product = [line.rstrip('\n') for line in f_target]

processed_reaction_list = []
processed_product_list = []

for line in reaction_list:
    line = line.replace('<separated>', '>')
    line = line.replace(' ', '')
    processed_reaction_list.append(line)

for line1 in product:
    line1 = line1.replace(' ', '')
    processed_product_list.append(line1)

with open(os.path.join(save_dir, 'src.source'), 'w', encoding='utf-8') as f:
    for src in processed_reaction_list:
        f.write('{}\n'.format(src.strip()))

with open(os.path.join(save_dir, 'tgt.target'), 'w', encoding='utf-8') as f:
    for tgt in processed_product_list:
        f.write('{}\n'.format(tgt.strip()))

print(f"已成功将处理后的内容保存到 {save_dir}")