import argparse  # 导入argparse模块，用于解析命令行参数
import linecache # 导入linecache模块，用于随机读取文本文件的指定行。
import logging  # 导入logging模块，用于实现日志记录功能。
import os  # 导入os模块，提供了丰富的方法来处理文件和目录
import random  # 导入random模块，用于生成伪随机数
import shutil  # 导入shutil模块，提供了文件操作的高级工具，包括文件复制、删除等功能
import subprocess  # 导入subprocess模块，用于创建子进程，执行外部命令。
from functools import partial  # 从functools模块导入partial函数，用于部分应用一个函数，固定部分参数，返回一个新的函数
from pathlib import Path  # 从pathlib模块导入Path类，用于处理文件路径。
from typing import Dict, List, Optional  # 从typing模块导入Dict、List、Optional等类型，用于静态类型检查

import numpy as np
import torch  # 导入PyTorch库，用于构建和训练神经网络
import torch.nn as nn  # 导入PyTorch的神经网络模块，包含了各种预定义的层和损失函数
from torch.nn.utils.rnn import pad_sequence  # 从PyTorch的神经网络模块中导入pad_sequence函数，用于将序列填充到相同的长度
from torch.utils.data import DataLoader, Dataset  # 从PyTorch的数据加载模块中导入DataLoader和Dataset类，用于加载和处理数据
from transformers import (DataCollatorForLanguageModeling, PreTrainedModel,
                          T5Config, T5ForConditionalGeneration, Trainer,
                          TrainingArguments)

from transformers.modeling_outputs import Seq2SeqLMOutput
# 从transformers库中导入Seq2SeqLMOutput类，表示Seq2Seq模型的输出

from transformers.optimization import (AdamW,
                                       get_constant_schedule_with_warmup,
                                       get_linear_schedule_with_warmup)

from transformers.trainer_pt_utils import (DistributedTensorGatherer,
                                           nested_concat)
# 从transformers库中导入DistributedTensorGatherer和nested_concat函数，用于分布式训练中的数据处理


from transformers.trainer_utils import EvalPrediction, PredictionOutput
# 从transformers库中导入EvalPrediction和PredictionOutput类，表示评估和预测的输出

from t5chem.t5chem import SimpleTokenizer, T5ForProperty, data_collator
# 从t5chem库中导入SimpleTokenizer、T5ForProperty和data_collator类或函数，用于处理化学领域的T5模型和数据


class MultiTaskTrainer(Trainer):
    """
    Save model weights based on validation error.(根据验证错误保存模型权重）
    """
        # 定义一个特殊方法 "__init__"，这是Python中的构造函数，用于创建类的实例。它接受任意数量的关键字参数(kwargs)。
        super().__init__(**kwargs)
        # 调用父类的构造函数，这里是指继承类的父类的构造函数，以确保正确地初始化对象。
        # 在这个例子中，super() 被用来调用父类的构造函数，初始化继承类的实例。
        self.min_eval_loss: float = float('inf')
        # 创建一个类属性 "min_eval_loss"，并将其初始化为正无穷大的浮点数。这是一个特殊的值，表示没有上限的数值。
        # 这个属性可以用来跟踪在后续的程序执行中的最小评估损失值。

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

'''这里定义了一个名为evaluate的方法，它接受几个参数：eval_dataset（可选的，表示评估数据集），ignore_keys（可选的，表示要忽略的键的列表），
metric_key_prefix（字符串类型，默认值为"eval"，表示指标的前缀），并且该方法返回一个字典，其中包含字符串键和浮点数值'''

        eval_dataloader: DataLoader = self.get_eval_dataloader(eval_dataset)
# 数据加载器eval_dataloader，用于加载评估数据集，数据加载器的具体实现是通过调用self.get_eval_dataloader(eval_dataset)来得到的

        output: PredictionOutput = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
'''在这里，使用prediction_loop方法进行预测。prediction_loop方法接受几个参数，包括eval_dataloader（用于加载评估数据集的数据加载器），
description（描述字符串，这里为"Evaluation"），prediction_loss_only（布尔值，如果没有指标则为True，否则为None），ignore_keys（要忽略的键的列表），
和metric_key_prefix（指标的前缀）'''

        self.log(output.metrics) # type: ignore
# 这一行代码将评估得到的指标记录下来，output.metrics是一个包含评估指标的字典

        cur_loss: float = output.metrics['eval_loss'] # type: ignore
# 这里获取了当前评估的损失值，该值保存在output.metrics字典中的'eval_loss'键下

        if self.min_eval_loss >= cur_loss:
            self.min_eval_loss = cur_loss
            for f in Path(self.args.output_dir).glob('best_cp-*'):
                shutil.rmtree(f)
            output_dir: str = os.path.join(self.args.output_dir, f"best_cp-{self.state.global_step}")

            self.save_model(output_dir)
            # self.save_model(output_dir): 将模型保存到新的输出目录中，实际函数调用会保存模型的权重、配置等信息到指定目录。
        return output.metrics  # type: ignore
        # 最后，该方法返回包含评估指标的字典

# 这段代码用于比较当前评估损失值cur_loss和之前记录的最小评估损失值self.min_eval_loss。如果当前损失值更小，就更新self.min_eval_loss的值，
# 并且在输出目录中保存当前模型的参数。这里使用了shutil.rmtree(f)来删除之前保存的最佳模型参数文件。
# 上面这个函数用于评估模型的性能

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:

# 设置预测损失标志，默认为None
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
        Works both with or without labels.(预测/求值循环）
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )
# 如果未提供，则使用self.args.prediction_loss_only的值

        model = self.model

        batch_size = dataloader.batch_size
# 获取dataloader的批次大小

        num_examples = self.num_examples(dataloader)
        losses_host = None
        preds_host = None
        labels_host = None
# 初始化损失、预测值和标签的变量

        world_size = 1

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples)

        model.eval()
# 初始化一个用于收集评估损失的DistributedTensorGatherer对象，指定世界大小、样本数量和批次大小

        if self.args.past_index >= 0:
            self._past = None
# 如果模型使用了过去的信息（例如，GPT模型的past_key_values），将_past变量设置为None

        self.callback_handler.eval_dataloader = dataloader
# 将当前数据加载器设置为评估回调处理器的数据加载器

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if loss is not None:
                losses = loss.repeat(batch_size) # type: ignore
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0) # type: ignore
            if logits is not None:
                # preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                # logits = torch.stack(logits).unsqueeze(0)
                # logits_reduced = torch.argmax(logits, dim=-1) if (len(logits.size())>1 and logits.size()[-1]>2) else logits
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    # def create_optimizer_and_scheduler(self, num_training_steps: int):
    #     """
    #     Setup the optimizer and the learning rate scheduler.
    #     We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    #     Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    #     """
    #     if self.optimizer is None:
    #         no_decay = ["bias", "LayerNorm.weight"]
    #         optimizer_grouped_parameters = [
    #             {
    #                 "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and not ('lm_head' in n)],
    #                 "weight_decay": self.args.weight_decay,
    #             },
    #             {
    #                 "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and not ('lm_head' in n)],
    #                 "weight_decay": 0.0,
    #             },
    #             {
    #                 "params": [p for n, p in self.model.named_parameters() if 'lm_head' in n],
    #                 'lr': self.args.learning_rate * 0.1,
    #             },
    #         ]
    #         self.optimizer = AdamW(
    #             optimizer_grouped_parameters,
    #             lr=self.args.learning_rate,
    #             betas=(self.args.adam_beta1, self.args.adam_beta2),
    #             eps=self.args.adam_epsilon,
    #         )
    #     if self.lr_scheduler == 'constant':
    #         self.lr_scheduler = get_constant_schedule_with_warmup(
    #             self.optimizer, num_warmup_steps=self.args.warmup_steps
    #         )

    #     elif self.lr_scheduler == 'cosine':
    #         self.lr_scheduler = get_cosine_schedule_with_warmup(
    #             self.optimizer,
    #             num_warmup_steps=self.args.warmup_steps,
    #             num_training_steps=num_training_steps,
    #         )

    #     else:
    #         self.lr_scheduler = get_linear_schedule_with_warmup(
    #             self.optimizer,
    #             num_warmup_steps=self.args.warmup_steps,
    #             num_training_steps=num_training_steps,
    #         )

'''类 MultiTaskDataset，用于处理多任务学习中的数据加载和预处理'''
class MultiTaskDataset(Dataset):

# MultiTaskDataset 的类，继承自 torch.utils.data.Dataset 类，表示这是一个 PyTorch 数据集类。
    def __init__( # 类的构造函数
        self,
        tokenizer, # 用于文本编码的分词器
        data_dir, # 数据文件所在的目录路径
        type_path: str="train",
    ) -> None:
        super().__init__() # 调用父类的构造函数
        
        self.task_types = ["Product", "Reactants", "Reagents", "Classification", "Yield"]
        self._source_path = [] # 源数据文件路径
        self._target_path = []
        for task in self.task_types:
            self._source_path.append(os.path.join(data_dir, task, type_path + ".source"))
            self._target_path.append(os.path.join(data_dir, task, type_path + ".target"))

            # 构建源数据文件和目标数据文件的完整路径，并将路径添加到相应的列表中。

            self._len_source: int = int(subprocess.check_output("wc -l " + self._source_path[-1], shell=True).split()[0])
            self._len_target: int = int(subprocess.check_output("wc -l " + self._target_path[-1], shell=True).split()[0])

            assert self._len_source == self._len_target,  # 判断源数据文件和目标数据文件函数是否相等，确保相等
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self._len_source
# 返回数据集的长度，即数据文件的行数

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    # 实现了 __getitem__ 方法，用于获取指定索引位置 idx 的数据样本
        inputs = {}
        for i,task in enumerate(self.task_types):
            source_line: str = linecache.getline(self._source_path[i], idx + 1).strip()
    # 使用 linecache.getline() 函数获取指定行索引的源数据，并去除字符串两端的空白字符。

            source_sample: BatchEncoding = self.tokenizer(
                            task+':'+source_line,
                            max_length=400,
                            padding="do_not_pad",
                            truncation=True,
                            return_tensors='pt',
                        )
    # 使用分词器对源数据进行编码，构建输入样本，包括任务类型和源数据。这里限制了输入文本的最大长度为400，并且不进行填充。

            target_line: str = linecache.getline(self._target_path[i], idx + 1).strip()
            if task in ("Classification","Yield"):
                try:
                    target_value: float = float(target_line)
                    target_ids: torch.Tensor = torch.Tensor([target_value])
                except TypeError:
                    print("The target should be a number, \
                            not {}".format(target_line))
                    raise AssertionError
    # 如果任务是 "Classification" 或 "Yield"，则将目标数据转换为浮点数，并构建一个包含该值的张量。如果目标数据不是数值类型，会引发错误。

            else:
                target_sample: BatchEncoding = self.tokenizer(
                                target_line,
                                max_length=300,
                                padding="do_not_pad",
                                truncation=True,
                                return_tensors='pt',
                            )
                target_ids = target_sample["input_ids"].squeeze(0)
            source_ids: torch.Tensor = source_sample["input_ids"].squeeze(0)
            src_mask: torch.Tensor = source_sample["attention_mask"].squeeze(0)
            inputs[task] = {"input_ids": source_ids, "attention_mask": src_mask, "decoder_input_ids": target_ids}
        return inputs
# 将编码后的源数据和目标数据张量添加到 inputs 字典中，以任务类型为键。
                

    def sort_key(self, ex) -> int:
# 定义了一个名为 sort_key 的方法，用于指定数据集中样本的排序规则
        return len(ex['Classification']['input_ids'])

def dummy_metrics(model_output: PredictionOutput) -> Dict[str, float]:
    # mask = (model_output.predictions==-100)
    # masked_preds = np.ma.masked_array(model_output.predictions, mask=mask)
    prod_acc, rct_acc, rgt_acc, cls_acc, mse_yield = model_output.predictions.mean(0)
    return {
        'product_acc': prod_acc.item(),
        'reactants_acc': rct_acc.item(),
        'reagents_acc': rgt_acc.item(),
        'classification_acc': cls_acc.item(),
        'mse_loss': mse_yield.item(),
    }

def MT_collator(batches, pad_token_id: int):
    ex = batches[0]
    # Product, Reactants, Reagents, Classification, Yield
    whole_batch = {}
    for task in ex.keys():
        task_batch = {}
        for key in ex[task].keys():
            if 'mask' in key:
                padding_value = 0
            else:
                padding_value = pad_token_id
            task_batch[key] = pad_sequence([x[task][key] for x in batches],
                                            batch_first=True,
                                            padding_value=padding_value)
        source_ids, source_mask, y = \

            task_batch["input_ids"], task_batch["attention_mask"], task_batch["decoder_input_ids"]
    # task_batch字典中获取了三个关键的张量:input_ids（输入的标识符序列），attention_mask（注意力掩码），和decoder_input_ids（解码器的输入标识符序列）

        whole_batch[task] = {'input_ids': source_ids, 'attention_mask': source_mask,
            'labels': y}
    return {"input_dict":whole_batch, 'labels':y}

class T5ForMultiTask(nn.Module):
    def __init__(self, pretrain_path, seq2seqW=1): # pretrain_path（预训练模型的路径）和可选参数seq2seqW（默认值为1，用于控制序列到序列模型的权重）
        super().__init__()
        self.Seq2seqModel = T5ForConditionalGeneration.from_pretrained(pretrain_path)
        self.ClassificationModel = T5ForProperty.from_pretrained(pretrain_path, head_type='classification')
        self.RegressionModel = T5ForProperty.from_pretrained(pretrain_path, head_type='regression')
        self.seq2seqW = seq2seqW

        # tie weights
        self.shared = self.Seq2seqModel.shared
        self.encoder = self.Seq2seqModel.encoder
        self.decoder = self.Seq2seqModel.decoder

        self.RegressionModel.shared = self.ClassificationModel.shared = self.shared
        self.RegressionModel.encoder = self.ClassificationModel.encoder = self.encoder
        self.RegressionModel.decoder = self.ClassificationModel.decoder = self.decoder

        # Assign correct task type
        self.Seq2seqModel.config.task_type = 'mixed'
        self.ClassificationModel.config.task_type = 'classification'
        self.RegressionModel.config.task_type = 'regression'

    def forward(self,input_dict,labels):
    #         pdb.set_trace()
        loss = 0
        # product
        outputs_prod = self.Seq2seqModel(**input_dict['Product'])
        preds = torch.argmax(outputs_prod['logits'], dim=-1)
        label = input_dict['Product']['labels']
        prod_acc = torch.all(preds==label,1) #.sum()
        loss += self.seq2seqW*outputs_prod["loss"] if isinstance(outputs_prod, dict) else outputs_prod[0]

        # reactants
        outputs_rct = self.Seq2seqModel(**input_dict['Reactants'])
        preds = torch.argmax(outputs_rct['logits'], dim=-1)
        label = input_dict['Reactants']['labels']
        rct_acc = torch.all(preds==label,1) #.sum()
        loss += self.seq2seqW*outputs_rct["loss"] if isinstance(outputs_rct, dict) else outputs_rct[0]
        
        # reagents
        outputs_rgt = self.Seq2seqModel(**input_dict['Reagents'])
        preds = torch.argmax(outputs_rgt['logits'], dim=-1)
        label = input_dict['Reagents']['labels']
        rgt_acc = torch.all(preds==label,1) #.sum()
        loss += self.seq2seqW*outputs_rgt["loss"] if isinstance(outputs_rgt, dict) else outputs_rgt[0]
        
        # classification
        outputs_cls = self.ClassificationModel(**input_dict['Classification'])
        preds = outputs_cls['logits']
        label = input_dict['Classification']['labels'].to(outputs_cls['logits']).squeeze()
        cls_acc = preds==label #(preds==label).sum()
        loss += outputs_cls["loss"] if isinstance(outputs_cls, dict) else outputs_cls[0]

        # regression
        outputs_yd = self.RegressionModel(**input_dict['Yield'])
        preds = outputs_yd["logits"]
        label = input_dict['Yield']['labels'].squeeze()
        mse_sum = (preds-label)**2 #.sum()
        loss += outputs_yd["loss"] if isinstance(outputs_yd, dict) else outputs_yd[0]
        return Seq2SeqLMOutput(
            loss=loss,
            logits=torch.stack([prod_acc, rct_acc, rgt_acc, cls_acc, mse_sum],1),
        )

if __name__ == "__main__":
    model = T5ForMultiTask("models/pretrain/simple/", seq2seqW=1.5)
    tokenizer = SimpleTokenizer(vocab_file="models/pretrain/simple/vocab.pt")
    os.makedirs("models/MultiTaskTrainw1.5/", exist_ok=True)
    tokenizer.save_vocabulary(os.path.join("models/MultiTaskTrainw1.5/", 'vocab.pt'))
    dataset = MultiTaskDataset(
        tokenizer, 
        data_dir="../t5chem_data/USPTO_MT/",
    )
    data_collator_padded = partial(
        MT_collator, pad_token_id=tokenizer.pad_token_id)
    eval_strategy = "steps"
    eval_iter = MultiTaskDataset(
        tokenizer, 
        data_dir="../t5chem_data/USPTO_MT/",
        type_path="val",
    )

    #     if task.output_layer == 'regression':
    #         compute_metrics = CalMSELoss
    #     elif args.task_type == 'pretrain':
    compute_metrics = dummy_metrics  
    #         # We don't want any extra metrics for faster pretraining
    #     else:
    #         compute_metrics = AccuracyMetrics

    training_args = TrainingArguments(
        output_dir="models/MultiTaskTrainw1.5/",
        overwrite_output_dir=True,
        do_train=True,
        evaluation_strategy=eval_strategy,
        num_train_epochs=100,
        per_device_train_batch_size=32,
        logging_steps=5000,
        per_device_eval_batch_size=32,
        save_steps=50000,
        save_total_limit=5,
        learning_rate=5e-4,
        prediction_loss_only=(compute_metrics is None),
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator_padded,
        train_dataset=dataset,
        eval_dataset=eval_iter,
        compute_metrics=compute_metrics,
    )
    # pdb.set_trace()
    trainer.train()
    # print(args)
    # print("logging dir: {}".format(training_args.logging_dir))
    trainer.save_model("models/MultiTaskTrainw1.5/")
