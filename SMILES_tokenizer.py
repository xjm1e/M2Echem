from rdkit import Chem
#
# # 创建完整分子（乙醇）
# mol = Chem.MolFromSmiles('CCO')
#
# # 指定要提取的原子索引（例如，只包含前两个碳原子）
# atoms_to_use = [0, 1]  # 原子索引从0开始
#
# # 生成子结构的 SMILES
# smiles_fragment = Chem.MolFragmentToSmiles(mol, atoms_to_use)
# print(f"完整分子 SMILES: {Chem.MolToSmiles(mol)}")  # 输出: CCO
# print(f"子结构 SMILES: {smiles_fragment}")         # 输出: CC


# mol = Chem.MolFromSmiles('CCOC(=O)CC')  # 乙酸乙酯
# substructure = Chem.MolFromSmiles('C(=O)O')  # 酯基片段
#
# # 在 mol 中查找子结构的匹配
# matches = mol.GetSubstructMatches(substructure)
#
# if matches:
#     # 提取第一个匹配的原子索引
#     atoms_to_use = list(matches[0])
#     smiles_fragment = Chem.MolFragmentToSmiles(mol, atoms_to_use)
#     print(f"酯基片段 SMILES: {smiles_fragment}")  # 输出: C(=O)O
