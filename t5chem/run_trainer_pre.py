# import argparse
# import logging
# import os
# import random
# from functools import partial
# from typing import Dict
#
# import numpy as np
# import torch
# from transformers import (DataCollatorForLanguageModeling, T5Config,
#                           T5ForConditionalGeneration, TrainingArguments)
#
# from .data_utils import (AccuracyMetrics, CalMSELoss, LineByLineTextDataset,
#                         T5ChemTasks, TaskPrefixDataset, TaskSettings,
#                         data_collator)
# from .model import T5ForProperty
# from .mol_tokenizers import (AtomTokenizer, MolTokenizer, SelfiesTokenizer,
#                             SimpleTokenizer)
# from .trainer import EarlyStopTrainer
#
# tokenizer_map: Dict[str, MolTokenizer] = {
#     'simple': SimpleTokenizer,  # type: ignore
#     'atom': AtomTokenizer,  # type: ignore
#     'selfies': SelfiesTokenizer,    # type: ignore
# }
#
#
# # 该函数用于向命令行解析器中添加不同的命令行参数
# def add_args(parser):
#     parser.add_argument(
#         '''添加一个名为--data_dir的命令行参数，类型为字符串，必需参数。用于指定输入数据的目录路径，其中包含了训练、验证和测试数据的文件'''
#         "--data_dir",
#         type=str,
#         required=True,
#         help="The input data dir. Should contain train.source, train.target, val.source, val.target, test.source, test.target",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         required=True,
#         help="The output directory where the model predictions and checkpoints will be written.",
#         # 用于指定模型预测结果和训练检查点将被保存的输出目录路径
#     )
#     parser.add_argument(
#         "--task_type",
#         type=str,
#         required=True,
#         help="Task type to use. ('product', 'reactants', 'reagents', \
#             'regression', 'classification', 'pretrain', 'mixed')",
#         # 用于指定要执行的任务类型，可以是上述提供的几种任务类型之一
#     )
#     parser.add_argument(
#         "--pretrain",
#         default='',
#         help="Path to a pretrained model. If not given, we will train from scratch",
#         # 添加一个名为--pretrain的命令行参数，如果未提供，默认为空字符串。用于指定预训练模型的路径，如果不提供路径，则表示从头开始训练模型
#     )
#     parser.add_argument(
#         "--vocab",
#         default='',
#         help="Vocabulary file to load.",
#         # 添加一个名为--vocab的命令行参数，如果未提供，默认为空字符串。用于指定要加载的词汇文件的路径
#     )
#     parser.add_argument(
#         "--tokenizer",
#         default='',
#         help="Tokenizer to use. ('simple', 'atom', 'selfies')",
#         # 用于指定要使用的分词器类型，可以是上述提供的几种分词器之一
#     )
#
#     parser.add_argument(
#         "--random_seed",
#         default=8570,
#         type=int,
#         help="The random seed for model initialization",
#         # 添加一个名为--random_seed的命令行参数，如果未提供，默认值为8570。用于指定模型初始化时的随机种子
#     )
#
#     parser.add_argument(
#         "--num_epoch",
#         default=100,
#         type=int,
#         help="Number of epochs for training.",
#         # 训练轮数
#     )
#
#     parser.add_argument(
#         "--log_step",
#         default=5000,
#         type=int,
#         help="Logging after every log_step",
#         # 用于指定记录日志的步长
#     )
#
#     parser.add_argument(
#         "--batch_size",
#         default=32,
#         type=int,
#         help="Batch size for training and validation.",
#         # 训练和验证时的批量大小
#     )
#
#     parser.add_argument(
#         "--init_lr",
#         default=5e-4,
#         type=float,
#         help="The initial leanring rate for model training",
#         # 模型训练的初始学习率
#     )
#
#     parser.add_argument(
#         "--num_classes",
#         type=int,
#         help="The number of classes in classification task. Only used when task_type is Classification",
#         # 分类任务中的类别数量，仅在任务类型为分类时使用
#     )
#
#
# # 用于训练一个深度学习模型
# def train(args):
#     torch.cuda.manual_seed(args.random_seed)
#     # 这行代码设置PyTorch的随机种子，以确保在使用GPU进行训练时结果的可重复性
#
#     np.random.seed(args.random_seed)
#     torch.manual_seed(args.random_seed)
#     # this one is needed for torchtext random call (shuffled iterator)
#     # in multi gpu it ensures datasets are read in the same order
#     random.seed(args.random_seed)
#     # some cudnn methods can be random even after fixing the seed
#     # unless you tell it to be deterministic
#     torch.backends.cudnn.deterministic = True
#     # 设置不同的随机种子，以确保在不同的随机操作中结果的可重复性
#
#     assert args.task_type in T5ChemTasks, \
#         "only {} are currenly supported, but got {}".\
#             format(tuple(T5ChemTasks.keys()), args.task_type)
#     task: TaskSettings = T5ChemTasks[args.task_type]
#
# # 这行代码检查参数args中的任务类型是否在预定义的任务列表中（T5ChemTasks）。如果不在列表中，会抛出一个断言错误
#
# # '''接下来的代码根据不同的情况，初始化模型、分词器（tokenizer）和相关配置。如果使用预训练模型，会加载预训练模型和相应的分词器。
# # 如果从头开始训练，会根据指定的分词器类型创建新的分词器和模型'''
#
#     if args.pretrain:
#         if task.output_layer == 'seq2seq':
#             model = T5ForConditionalGeneration.from_pretrained(args.pretrain)
#         else:
#             model = T5ForProperty.from_pretrained(
#                 args.pretrain,
#                 head_type = task.output_layer,
#             )
#         if not hasattr(model.config, 'tokenizer'):
#             logging.warning("No tokenizer type detected, will use SimpleTokenizer as default")
#         tokenizer_type = getattr(model.config, "tokenizer", 'simple')
#         vocab_path = os.path.join(args.pretrain, 'vocab.pt')
#         if not os.path.isfile(vocab_path):
#             vocab_path = args.vocab
#             if not vocab_path:
#                 raise ValueError(
#                         "Can't find a vocabulary file at path '{}'.".format(args.pretrain)
#                     )
#         tokenizer = tokenizer_map[tokenizer_type](vocab_file=vocab_path)
#         model.config.tokenizer = tokenizer_type # type: ignore
#         model.config.task_type = args.task_type # type: ignore
#     else:
#         if not args.tokenizer:
#             warn_msg = "This model is trained from scratch, but no \
#                 tokenizer type is specified, will use simple tokenizer \
#                 as default for this training."
#             logging.warning(warn_msg)
#             args.tokenizer = 'simple'
#         assert args.tokenizer in ('simple', 'atom', 'selfies'), \
#             "{} tokenizer is not supported."
#         vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab/'+args.tokenizer+'.pt')
#         tokenizer = tokenizer_map[args.tokenizer](vocab_file=vocab_path)
#         config = T5Config(
#             vocab_size=len(tokenizer),
#             pad_token_id=tokenizer.pad_token_id,
#             decoder_start_token_id=tokenizer.pad_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             output_past=True,
#             num_layers=4,
#             num_heads=8,
#             d_model=256,
#             tokenizer=args.tokenizer,
#             task_type=args.task_type,
#         )
#         if task.output_layer == 'seq2seq':
#             model = T5ForConditionalGeneration(config)
#         else:
#             model = T5ForProperty(config, head_type=task.output_layer, num_classes=args.num_classes)
#
#
#
#     os.makedirs(args.output_dir, exist_ok=True)
#     # 这行代码创建一个输出目录，用于保存训练过程中的模型和日志等文件
#     tokenizer.save_vocabulary(os.path.join(args.output_dir, 'vocab.pt'))
#     if args.task_type == 'pretrain':
#         dataset = LineByLineTextDataset(
#             tokenizer=tokenizer,
#             file_path=os.path.join(args.data_dir,'train.txt'),
#             block_size=task.max_source_length,
#             prefix=task.prefix,
#         )
#         data_collator_padded = DataCollatorForLanguageModeling(
#             tokenizer=tokenizer, mlm=True, mlm_probability=0.15
#         )
#     else:
#         dataset = TaskPrefixDataset(
#             tokenizer,
#             data_dir=args.data_dir,
#             prefix=task.prefix,
#             max_source_length=task.max_source_length,
#             max_target_length=task.max_target_length,
#             separate_vocab=(task.output_layer != 'seq2seq'),
#             type_path="train",
#         )
#         data_collator_padded = partial(
#             data_collator, pad_token_id=tokenizer.pad_token_id)
#
#     if args.task_type == 'pretrain':
#         do_eval = os.path.exists(os.path.join(args.data_dir, 'val.txt'))
#         if do_eval:
#             eval_strategy = "steps"
#             eval_iter = LineByLineTextDataset(
#                 tokenizer=tokenizer,
#                 file_path=os.path.join(args.data_dir,'val.txt'),
#                 block_size=task.max_source_length,
#                 prefix=task.prefix,
#             )
#         else:
#             eval_strategy = "no"
#             eval_iter = None
#     else:
#         do_eval = os.path.exists(os.path.join(args.data_dir, 'val.source'))
#         if do_eval:
#             eval_strategy = "steps"
#             eval_iter = TaskPrefixDataset(
#                 tokenizer,
#                 data_dir=args.data_dir,
#                 prefix=task.prefix,
#                 max_source_length=task.max_source_length,
#                 max_target_length=task.max_target_length,
#                 separate_vocab=(task.output_layer != 'seq2seq'),
#                 type_path="val",
#             )
#         else:
#             eval_strategy = "no"
#             eval_iter = None
#
#     if task.output_layer == 'regression':
#         compute_metrics = CalMSELoss
#     elif args.task_type == 'pretrain':
#         compute_metrics = None
#         # We don't want any extra metrics for faster pretraining
#     else:
#         compute_metrics = AccuracyMetrics
#
# # 训练参数匹配
#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         overwrite_output_dir=True,
#         do_train=True,
#         evaluation_strategy=eval_strategy,
#         num_train_epochs=args.num_epoch,
#         per_device_train_batch_size=args.batch_size,
#         logging_steps=args.log_step,
#         per_device_eval_batch_size=args.batch_size,
#         save_steps=10000,
#         save_total_limit=5,
#         learning_rate=args.init_lr,
#         prediction_loss_only=(compute_metrics is None),
#     )
#
# # 训练的调用的函数
#     trainer = EarlyStopTrainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator_padded,
#         train_dataset=dataset,
#         eval_dataset=eval_iter,
#         compute_metrics=compute_metrics,
#     )
#
#     trainer.train()
#     print(args)
#     print("logging dir: {}".format(training_args.logging_dir))
#     trainer.save_model(args.output_dir)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     add_args(parser)
#     args = parser.parse_args()
#     train(args)
