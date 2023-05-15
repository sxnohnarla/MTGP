# -*- coding: utf-8 -*-
from random import random
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, progress
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from argparse import ArgumentParser
from sacrebleu.metrics import BLEU
from torchmetrics import SacreBLEUScore
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from nlgeval import NLGEval
import json
import os
import numpy as np
from evaluate import load

class PathGenModel(pl.LightningModule):
    def __init__(self, batch_size=None, lr=None, num_workers=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.save_hyperparameters()
        self.dec_tok = GPT2Tokenizer.from_pretrained("cache_gpt")
        self.decoder = GPT2LMHeadModel.from_pretrained("cache_gpt")
        self.perplexity = load("perplexity", module_type="metric")
        

        self.dec_tok.add_special_tokens({
            'additional_special_tokens': ['<PAD>', '<SEP>', '<END>']
        })

        self.decoder.resize_token_embeddings(len(self.dec_tok))

        self.PAD = self.dec_tok.convert_tokens_to_ids('<PAD>')
        self.SEP = self.dec_tok.convert_tokens_to_ids('<SEP>')
        self.END = self.dec_tok.convert_tokens_to_ids('<END>')        

        self.relationsfound = set()

    def prepare_data(self):
        self.val_result = []
        self.predict_result = []
        self.predict_refs = []
        self.source_list = []
        self.target_list = []
        self.path_list = []

    def setup(self, stage: str = None):
        # ! 这种数据的设置形式，是global的，即训练数据与这个无关
        # TODO 可以考虑用数据集的路径训练模型，作为local
        test_data = []
        with open('data/DailyDialog/test/test_input.json', 'r', encoding='utf-8') as f:
            for row in f:
                data = json.loads(row)
                # path = data['entity_path']
                source = data['source']
                target = data['target']
                test_data.append({'source': source, 'target': target})
        self.test_dataset = test_data
        if stage == 'fit':
            with open('0_construct_cpnet/global_paths/train.txt', 'r', encoding='utf-8') as f:
                train_data = [row for row in f]
            self.train_dataset = train_data
            with open('0_construct_cpnet/global_paths/dev.txt', 'r', encoding='utf-8') as f:
                dev_data = [row for row in f]
            self.dev_dataset = dev_data
            print('train_len:', len(self.train_dataset), 'dev_len:', len(self.dev_dataset))
        
        elif stage == 'predict':
             print(f"predicting len: {len(self.test_dataset)}")
             
    def collate_fn_predict(self, batch):
        st_batch = {}
        max_context_len = 16
        predict_dec_inputs = []
        source_list = []
        target_list = []

        for i, line in enumerate(batch):
            source = line['source']
            target = line['target']
            # path = line['path']

            context = target.replace('_', ' ') + '<SEP>' + source.replace('_', ' ')
            context_ids = self.dec_tok.encode(context)[:max_context_len]
            context_ids += [self.PAD] * (max_context_len - len(context_ids))
            predict_dec_inputs.append(context_ids)
            source_list.append(source)
            target_list.append(target)
        
        st_batch['source'] = source_list
        st_batch['target'] = target_list    
        st_batch['predict_dec_inputs'] = predict_dec_inputs
        
        # ? 这里加不加torch.tensor
        return st_batch

    def collate_fn(self, batch):
        dec_labels = []
        dec_inputs = []
        dec_mask = []

        # target<sep>source 的长度
        max_context_len = 16
        # 路径的长度 （n0）e0 n1 ... nt
        max_label_len = 31

        # nt[sep]nh[pad][pad]16[pad][pad]e0 n1 e1 nt [pad][pad]31[pad][end]
        for i, line in enumerate(batch):
            line_split = line.strip().split('\t')
            source = line_split[0]
            target = line_split[-1]
            # text : e0 n1 e1 n2 e2 n2 e3 n3 e4 nt
            text = ''
            # 中间实体 n1-nt
            intermediate_ents = []
            for idx, item in enumerate(line_split[1:]):
                if idx % 2 != 0:
                    ent_words = item.replace('_', ' ')
                    text += ent_words
                    intermediate_ents.append(ent_words)
                else:
                    text += ' ' + item + ' '
            
            for item in text.split():
                # 反向关系
                if '_' in item:
                    self.relationsfound.add(item)
            
            path_ids = self.dec_tok.encode(text)[:max_label_len]
            path_ids += [self.PAD] * (max_label_len - len(path_ids)) # 31

            context = target.replace('_', ' ') + '<SEP>' + source.replace('_', ' ')
            context_ids = self.dec_tok.encode(context)[:max_context_len]
            context_ids += [self.PAD] * (max_context_len - len(context_ids))# 16

            dec_inputs.append(context_ids + path_ids + [self.END])
            dec_labels.append([-100] * len(context_ids) + path_ids + [self.END])
            dec_mask.append([1] * len(dec_inputs[-1])) 

        return{
            'dec_inputs': torch.tensor(dec_inputs),
            'dec_mask': torch.tensor(dec_mask),
            'dec_labels': torch.tensor(dec_labels),
        }        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.dev_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn_predict)

    def forward(self, dec_inputs=None, dec_mask=None, dec_labels=None):
        outputs = self.decoder(
            dec_inputs,
            attention_mask = dec_mask,
            labels = dec_labels,
        )
        return {'loss': outputs['loss'], 'logits': outputs['logits']}
    
    def training_step(self, batch: tuple, batch_idx: int):
        outputs = self(**batch)
        return outputs['loss']
    
    def validation_step(self, batch: dict, batch_idx: int):
        outputs = self(**batch)
        tokens = outputs['logits'].argmax(2).tolist()
        for row in tokens:
            self.val_result.append(self.dec_tok.decode(row))
        return {'val_loss': outputs['loss'].item()}

    def validation_epoch_end(self, val_step_outputs: list):
        print('\n\n', '-' * 100, '\n\n')
        print('validation_epoch_end')
        val_loss = [x['val_loss'] for x in val_step_outputs]
        ppl = np.exp(np.mean(val_loss))
        print('val_loss %.4f \n' %torch.tensor(val_loss).mean().item())
        self.log('val_loss', torch.tensor(val_loss).mean().item())
        print('ppl %.4f \n' %torch.tensor(ppl).item())
        self.log('ppl', torch.tensor(ppl).item())
        if len(self.val_result) > 1:
            print('sample result')
            for row in self.val_result[-5:]:
                print(row)
        self.val_result = []
        print('\n\n', '-' * 100, '\n\n')

    def predict_step(self, batch: dict, batch_idx: int):
        dec_inputs = []
        dec_mask = []
        topN = 1

        multi_predict_result = []

        for context in batch['predict_dec_inputs']:
            dec_inputs.append(context)
            dec_mask.append([1] * len(dec_inputs[-1]))
        # ?self.device
        input_ids = torch.tensor(dec_inputs).to(self.device)
        input_mask = torch.tensor(dec_mask).to(self.device)
        generated_ids = self.decoder.generate(
            input_ids=input_ids,
            attention_mask=input_mask,
            max_new_tokens=31,
            pad_token_id=self.PAD,
            eos_token_id=self.END,
            num_beams=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            # repetition_penalty=1.0,
            # length_penalty=1.0,
            early_stopping=True,
            num_return_sequences = topN,
            # no_repeat_ngram_size=2
        )
        preds = generated_ids[:, input_ids.shape[-1]-1:]
        for row in preds:
            path = self.dec_tok.decode(row, skip_special_tokens=True)
            # results = self.perplexity.compute(model_id='./gen_model', predictions=path)
            multi_predict_result.append(path)
        for i in range(0, len(multi_predict_result), topN):
            self.predict_result.append(multi_predict_result[i:i+topN])
        # self.path_list.extend(path_list)
        self.source_list.extend(batch['source'])
        self.target_list.extend(batch['target'])

    def on_predict_epoch_end(self, predict_outputs=None):
        print('\n\n', '-' * 100, '\n\n')
        print('on_predict_epoch_end')
        print('\n\n', '-' * 100, '\n\n')
        with open('1_path_generation/results_paths/predict_dd_test.json', 'w', encoding='utf-8') as f:
            for source, target, generate in zip(self.source_list, self.target_list, self.predict_result):
                if len(generate) > 0:
                    generate = [source + generate[i] for i in range(len(generate))]
                f.write(json.dumps({'source': source, 'target': target, 'generate': generate}) + '\n')
 
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)
        return optimizer
if __name__ == '__main__':
    torch.cuda.empty_cache()
    seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--acc_batch", type=int, default=1)
    parser.add_argument("--run_predict", type=str, default=None)
    parser.add_argument("--key_model", type=str, default=None)
    parser.add_argument("--output_file", type=str, default='path_gen.log')
    args = parser.parse_args()
    tb_logger = pl_loggers.TensorBoardLogger('1_path_generation/logs_path_gen', name='')
    checkpoint_callback = ModelCheckpoint(
        filename='best',
        save_weights_only=True,
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    bar_callback = progress.TQDMProgressBar(refresh_rate=25 if args.run_predict is None else 1)

    model = PathGenModel(**vars(args))

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=5,
        logger=tb_logger,
        callbacks=[checkpoint_callback, bar_callback],
        gradient_clip_val=0.5,
        log_every_n_steps=25,
        accumulate_grad_batches=args.acc_batch,
    )
    if args.run_predict is not None:
        model = model.load_from_checkpoint(args.run_predict, strict=True)
        model.batch_size = 128
        model.output_file = args.output_file
        trainer.predict(model)
    else:
        trainer.fit(model)


# nohup python -u path_gen_model.py > path_gen_3.log 2>&1
# python -u 1_path_generation/path_gen_model.py --run_predict 1_path_generation/logs_path_gen/global_my_cpnet_version_33/checkpoints/best.ckpt
# python -u 1_path_generation/path_gen_model.py --run_predict 1_path_generation/logs_path_gen/local_my_cpnet_version_46/checkpoints/best.ckpt
# cd 1_path_generatcd
# version 33 global 用我的 图谱训练的结果
# 