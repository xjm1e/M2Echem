import linecache
import os
import subprocess
from typing import Dict, List, NamedTuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput


class TaskSettings(NamedTuple):
    prefix: str
    max_source_length: int
    max_target_length: int
    output_layer: str


T5ChemTasks: Dict[str, TaskSettings] = {
    'product': TaskSettings('Product:', 400, 200, 'seq2seq'),
    'reactants': TaskSettings('Reactants:', 200, 300, 'seq2seq'),
    'reagents': TaskSettings('Reagents:', 400, 200, 'seq2seq'),
    'classification': TaskSettings('Classification:', 500, 1, 'classification'),
    'regression': TaskSettings('Yield:', 500, 1, 'regression'),
    'pretrain': TaskSettings('Fill-Mask:', 400, 200, 'seq2seq'),
    'mixed': TaskSettings('', 400, 300, 'seq2seq'),
}


class LineByLineTextDataset(Dataset):
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        file_path: str, 
        block_size: int, 
        prefix: str = ''
    ) -> None:
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        
        self.prefix: str = prefix
        self._file_path: str = file_path
        self._len: int = int(subprocess.check_output("wc -l " + file_path, shell=True).split()[0])
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length: int = block_size
        
    def __getitem__(self, idx: int) -> torch.Tensor:
        line: str = linecache.getline(self._file_path, idx + 1).strip()
        sample: BatchEncoding = self.tokenizer(
                        self.prefix+line,
                        max_length=self.max_length,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )
        return sample['input_ids'].squeeze(0)
      
    def __len__(self) -> int:
        return self._len


class TaskPrefixDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_dir: str,
        prefix: str='',
        type_path: str="train",
        max_source_length: int=300,
        max_target_length: int=100,
        separate_vocab: bool=False,
    ) -> None:
        super().__init__()

        self.prefix: str = prefix
        self._source_path: str = os.path.join(data_dir, type_path + ".source")
        self._target_path: str = os.path.join(data_dir, type_path + ".target")
        # self._len_source: int = int(subprocess.check_output("wc -l " + self._source_path, shell=True).split()[0])
        # self._len_target: int = int(subprocess.check_output("wc -l " + self._target_path, shell=True).split()[0])
        try:
            with open(self._source_path, 'r', encoding='utf-8') as f:
                self._len_source = sum(1 for _ in f)
        except FileNotFoundError:
            print(f"Error: Source file {self._source_path} not found.")
            self._len_source = 0

        try:
            with open(self._target_path, 'r', encoding='utf-8') as f:
                self._len_target = sum(1 for _ in f)
        except FileNotFoundError:
            print(f"Error: Target file {self._target_path} not found.")
            self._len_target = 0

        assert self._len_source == self._len_target, "Source file and target file don't match!"
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_source_len: int = max_source_length
        self.max_target_len: int = max_target_length
        self.sep_vocab: bool = separate_vocab

    def __len__(self) -> int:
        return self._len_source

    # 2025/4/11 根对齐数据增强
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 2025 edit
        global rea_smi
        aug = 5
        separated = True

        from rdkit import Chem
        import re
        import random

        def get_root_id(mol, root_map_number):
            root = -1
            for i, atom in enumerate(mol.GetAtoms()):
                if atom.GetAtomMapNum() == root_map_number:
                    root = i
                    break
            return root

        def smi_tokenizer(smi):
            pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
            regex = re.compile(pattern)
            tokens = [token for token in regex.findall(smi)]
            assert smi == ''.join(tokens)
            return ' '.join(tokens)

        def clear_map_canonical_smiles(smi, canonical=True, root=-1):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                for atom in mol.GetAtoms():
                    if atom.HasProp('molAtomMapNumber'):
                        atom.ClearProp('molAtomMapNumber')
                return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
            else:
                return smi

        def get_cano_map_number(smi, root=-1):
            atommap_mol = Chem.MolFromSmiles(smi)
            canonical_mol = Chem.MolFromSmiles(clear_map_canonical_smiles(smi, root=root))
            cano2atommapIdx = atommap_mol.GetSubstructMatch(canonical_mol)
            correct_mapped = [
                canonical_mol.GetAtomWithIdx(i).GetSymbol() == atommap_mol.GetAtomWithIdx(index).GetSymbol() for
                i, index in enumerate(cano2atommapIdx)]
            atom_number = len(canonical_mol.GetAtoms())
            if np.sum(correct_mapped) < atom_number or len(cano2atommapIdx) < atom_number:
                cano2atommapIdx = [0] * atom_number
                atommap2canoIdx = canonical_mol.GetSubstructMatch(atommap_mol)
                if len(atommap2canoIdx) != atom_number:
                    return None
                for i, index in enumerate(atommap2canoIdx):
                    cano2atommapIdx[index] = i
            id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]

            return [id2atommap[cano2atommapIdx[i]] for i in range(atom_number)]

        def add_atom_mapping(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"无法解析 SMILES 字符串: {smiles}")
                return None
            # 为每个原子添加原子映射
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx() + 1)
            mapped_smiles = Chem.MolToSmiles(mol)
            return mapped_smiles

        source_line: str = linecache.getline(self._source_path, idx + 1).strip()
        target_line: str = linecache.getline(self._target_path, idx + 1).strip()

        parts = source_line.split('>')
        if len(parts) == 2:
            reactant = parts[0]
            reagent = parts[1]
        else:
            reactant = parts[0]
            reagent = False

        # 2025/4/14 具有原子映射的反应物和产物
        mapped_source = add_atom_mapping(reactant)
        mapped_target = add_atom_mapping(target_line)

        reactant = mapped_source.split(".")
        product = mapped_target.split(".")

        rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
        max_times = np.prod([len(map_numbers) for map_numbers in rea_atom_map_numbers])
        times = min(aug, max_times)
        reactant_roots = [[-1 for _ in reactant]]
        j = 0
        while j < times:
            reactant_roots.append([random.sample(rea_atom_map_numbers[k], 1)[0] for k in range(len(reactant))])
            if reactant_roots[-1] in reactant_roots[:-1]:
                reactant_roots.pop()
            else:
                j += 1
        if j < aug:
            reactant_roots.extend(random.choices(reactant_roots, k=aug - times))
            times = aug
        reversable = False  # no reverse
        assert times == aug
        if reversable:
            times = int(times / 2)

        result1 = []
        result2 = []
        pro_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", pro))) for pro in product]
        full_pro_atom_map_numbers = set(map(int, re.findall(r"(?<=:)\d+", ".".join(product))))
        for k in range(times):
            tmp = list(zip(reactant, reactant_roots[k], rea_atom_map_numbers))
            random.shuffle(tmp)
            reactant_k, reactant_roots_k, rea_atom_map_numbers_k = [i[0] for i in tmp], [i[1] for i in tmp], [i[2] for i
                                                                                                              in tmp]
            aligned_reactants = []
            aligned_products = []
            aligned_products_order = []
            all_atom_map = []
            for i, rea in enumerate(reactant_k):
                rea_root_atom_map = reactant_roots_k[i]
                rea_root = get_root_id(Chem.MolFromSmiles(rea), root_map_number=rea_root_atom_map)
                cano_atom_map = get_cano_map_number(rea, rea_root)
                if cano_atom_map is None:
                    print(f"Reactant Failed to find Canonical Mol with Atom MapNumber")
                    continue
                rea_smi = clear_map_canonical_smiles(rea, canonical=True, root=rea_root)
                aligned_reactants.append(rea_smi)
                all_atom_map.extend(cano_atom_map)

            for i, pro_map_number in enumerate(pro_atom_map_numbers):
                reactant_candidates = []
                selected_reactant = []
                for j, map_number in enumerate(all_atom_map):
                    if map_number in pro_map_number:
                        for rea_index, rea_atom_map_number in enumerate(rea_atom_map_numbers_k):
                            if map_number in rea_atom_map_number and rea_index not in selected_reactant:
                                selected_reactant.append(rea_index)
                                reactant_candidates.append((map_number, j, len(rea_atom_map_number)))

                # select maximal reactant
                reactant_candidates.sort(key=lambda x: x[2], reverse=True)
                map_number = reactant_candidates[0][0]
                j = reactant_candidates[0][1]
                pro_root = get_root_id(Chem.MolFromSmiles(product[i]), root_map_number=map_number)
                pro_smi = clear_map_canonical_smiles(product[i], canonical=True, root=pro_root)
                aligned_products.append(pro_smi)
                aligned_products_order.append(j)

            sorted_products = sorted(list(zip(aligned_products, aligned_products_order)), key=lambda x: x[1])
            aligned_products = [item[0] for item in sorted_products]
            pro_smi = ".".join(aligned_products)
            if separated:
                reactants = []
                for i, cano_atom_map in enumerate(rea_atom_map_numbers_k):
                    if len(set(cano_atom_map) & full_pro_atom_map_numbers) > 0:
                        reactants.append(aligned_reactants[i])
                rea_smi = ".".join(reactants)
                if reagent and len(reagent) > 0:
                    rea_smi = 'Product:' + rea_smi + ">" + reagent
                else:
                    rea_smi = 'Product:' + rea_smi
                # if len(reagent) > 0:
                #     rea_smi = 'Product:' + rea_smi + ">" + reagent
                # else:
                #     rea_smi = 'Product:' + rea_smi
            result1.append(rea_smi)
            result2.append(pro_smi)

        source_sample: BatchEncoding = self.tokenizer(
                        result1,
                        max_length=self.max_source_len,
                        padding="do_not_pad",
                        truncation=True,
                        return_tensors='pt',
                    )

        if self.sep_vocab:
            try:
                target_value: float = float(target_line)
                target_ids: torch.Tensor = torch.Tensor([target_value])
            except TypeError:
                print("The target should be a number, \
                        not {}".format(target_line))
                raise AssertionError
        else:
            target_sample: BatchEncoding = self.tokenizer(
                            result2,
                            max_length=self.max_target_len,
                            padding="do_not_pad",
                            truncation=True,
                            return_tensors='pt',
                        )
            target_ids = [tensor.squeeze(0) for tensor in target_sample["input_ids"]]
        source_ids = [tensor.squeeze(0) for tensor in source_sample["input_ids"]]
        src_mask = [tensor.squeeze(0) for tensor in source_sample["attention_mask"]]
        index1 = [f"{idx}-{i}" for i in range(len(source_ids))]
        #     target_ids = target_sample["input_ids"].squeeze(0)
        # source_ids: torch.Tensor = source_sample["input_ids"].squeeze(0)
        # src_mask: torch.Tensor = source_sample["attention_mask"].squeeze(0)

        return {"input_ids": source_ids, "attention_mask": src_mask,
                "decoder_input_ids": target_ids, "aug": index1}

    def sort_key(self, ex: BatchEncoding) -> int:
        """ Sort using length of source sentences. """
        return len(ex['input_ids'])


def data_collator(batch: List[BatchEncoding], pad_token_id: int) -> Dict[str, torch.Tensor]:
    whole_batch: Dict[str, torch.Tensor] = {}
    ex: BatchEncoding = batch[0]
    import itertools
    for key in ex.keys():
        if 'mask' in key:
            padding_value = 0
        else:
            padding_value = pad_token_id
        if key == 'aug':
            ID_values = [s["aug"] for s in batch]
            whole_batch[key] = list(itertools.chain(*ID_values))
        else:
            whole_batch[key] = pad_sequence([x[key] for x in batch],
                                            batch_first=True,
                                            padding_value=padding_value)
    return {'input_ids': whole_batch["input_ids"], 'attention_mask': whole_batch["attention_mask"],
            'labels': whole_batch["decoder_input_ids"], 'aug': whole_batch["aug"]}


def CalMSELoss(model_output: PredictionOutput) -> Dict[str, float]:
    predictions: np.ndarray = model_output.predictions # type: ignore
    label_ids: np.ndarray = model_output.label_ids.squeeze() # type: ignore
    loss: float = ((predictions - label_ids)**2).mean().item()
    return {'mse_loss': loss}

def AccuracyMetrics(model_output: PredictionOutput) -> Dict[str, float]:
    label_ids: np.ndarray = model_output.label_ids # type: ignore
    predictions: np.ndarray = model_output.predictions.reshape(-1, label_ids.shape[1]) # type: ignore
    correct: int = np.all(predictions==label_ids, 1).sum()
    return {'accuracy': correct/len(predictions)}
