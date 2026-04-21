"""t5chem - A Unified Deep Learning Model for Multi-task Reaction Predictions"""
from .__version__ import __version__
from .data_utils_origin import LineByLineTextDataset, TaskPrefixDataset, data_collator
from .model import T5ForProperty
from .mol_tokenizers import (AtomTokenizer, MolTokenizer, SelfiesTokenizer,
                             SimpleTokenizer)
from .trainer import EarlyStopTrainer

__author__ = 'Jocelyn Lu <jl8570@nyu.edu>'
__all__: list = [
    "TaskPrefixDataset",
    "data_collator",
    "LineByLineTextDataset",
    "T5ForProperty",
    "AtomTokenizer",
    "MolTokenizer",
    "SelfiesTokenizer",
    "SimpleTokenizer",
    "EarlyStopTrainer",
]
'''它明确指定了哪些类和函数可以在模块外部通过通配符导入的方式访问和使用。
这样做的好处是，它提供了一种方式来控制模块的命名空间，使得代码更加清晰和可维护。'''