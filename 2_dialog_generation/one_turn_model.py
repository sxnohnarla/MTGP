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

def pad_to_max_seq_len(arr, max_seq_len=None, pad_token_id=0, max_len=None):
    """
    a = [ [1, 2, 3], [1, 3] ]
    pad_to_max_seq_len(a, 5)
    a -> [[1, 2, 3, 0, 0], [1, 3, 0, 0, 0]]
    """
    if max_seq_len is None:
        max_seq_len = 0
        for sub_a in arr:
            if len(sub_a) >= max_seq_len:
                max_seq_len = len(sub_a)
    if max_len is not None:
        if max_seq_len > max_len:
            max_seq_len = max_len
    for index, text in enumerate(arr):
        seq_len = len(text)
        if seq_len < max_seq_len:
            padded_tokens = [
                pad_token_id for _ in range(max_seq_len - seq_len)
            ]
            new_text = text + padded_tokens
            arr[index] = new_text
        elif seq_len > max_seq_len:
            new_text = text[:max_seq_len]
            arr[index] = new_text
    return max_seq_len

class OneTurnModel(pl.LightningModule):
    def __init__(self, batch_size=None, lr=None, num_workers=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.save_hyperparameters()
        self.dec_tok = GPT2Tokenizer.from_pretrained("cache_gpt")
        self.decoder = GPT2LMHeadModel.from_pretrained("cache_gpt")
        self.bleu = BLEU()
        self.dec_tok.add_special_tokens({
            'additional_special_tokens': ['<eos>', '<pth>', '<ctx>', '<tgt>', '<res>', '<b_c>']
        })

        self.decoder.resize_token_embeddings(len(self.dec_tok))
        self.EOS = self.dec_tok.convert_tokens_to_ids('<eos>')
        self.PTH = self.dec_tok.convert_tokens_to_ids('<pth>')
        self.CTX = self.dec_tok.convert_tokens_to_ids('<ctx>')
        self.TGT = self.dec_tok.convert_tokens_to_ids('<tgt>')   
        self.RES = self.dec_tok.convert_tokens_to_ids('<res>') 
        self.T_C = self.dec_tok.convert_tokens_to_ids('<b_c>')            

    def prepare_data(self):
        self.val_result = []
        self.predict_result = []
        self.predict_ref = []

    def setup(self, stage: str = None):
        if stage == 'fit':
            train_data = []
            with open('data/triple_dialog_data/ott_dd_train.json', 'r', encoding='utf-8') as f:
                for row in f:
                    data = json.loads(row)
                    train_data.append(data)
            self.train_dataset = train_data
            dev_data = []
            with open('data/triple_dialog_data/ott_dd_dev.json', 'r', encoding='utf-8') as f:
                for row in f:
                    data = json.loads(row)
                    dev_data.append(data)
            self.dev_dataset = dev_data
            print('train_len:', len(self.train_dataset), 'dev_len:', len(self.dev_dataset))
        elif stage == 'predict':
            test_data = []
            with open('2_dialog_generation/dialog_gen_data/test_ott_2.json', 'r', encoding='utf-8') as f:
                for row in f:
                    data = json.loads(row)
                    test_data.append(data)
            self.test_dataset = test_data
            print(f"predicting len: {len(self.test_dataset)}")

    def collate_fn_predict(self, batch):
        pc_batch = {}

        path_list = []
        context_list = []
        target_list = []
        for i, line in enumerate(batch):
            path =  line['path']
            context = line['context']
            target = line['target']
            path_list.append(path)
            context_list.append(context)
            target_list.append(target)
        
        pc_batch['path'] = path_list
        pc_batch['context'] = context_list
        pc_batch['target'] = target_list

        return pc_batch

    def collate_fn(self, batch):
        dec_labels = []
        dec_inputs = []
        dec_mask = []

        predict_inputs = []
        predict_mask = []
        predict_labels = []

        for item in batch:
            context = item['context']
            response = item['output']
            path = item['full_path']
            b_c = item["path"][1]
             
            input_seq = '<b_c>' + b_c + '<pth>' + path + '<ctx>' + context[0] + '<tgt>' + context[-1]
            input_ids = self.dec_tok.encode(input_seq, add_special_tokens=False)

            res_seq = '<res>' + response + '<eos>'
            res_ids = self.dec_tok.encode(res_seq, add_special_tokens=False) 

            dec_inputs.append(input_ids + res_ids)
            dec_mask.append([1] * len(dec_inputs[-1]))
            dec_labels.append([-100] * len(input_ids) + res_ids)

            predict_inputs.append(input_ids + [self.RES])
            predict_mask.append([1] * len(predict_inputs[-1]))
            predict_labels.append(response)

        pad_to_max_seq_len(dec_inputs, pad_token_id=self.EOS, max_seq_len=96)
        pad_to_max_seq_len(dec_mask, pad_token_id=0, max_seq_len=96)
        pad_to_max_seq_len(dec_labels, pad_token_id=-100, max_seq_len=96)

        pad_to_max_seq_len(predict_inputs, pad_token_id=self.RES, max_seq_len=64)
        pad_to_max_seq_len(predict_mask, pad_token_id=0, max_seq_len=64)
        
        return{
            'dec_inputs': torch.tensor(dec_inputs),
            'dec_mask': torch.tensor(dec_mask),
            'dec_labels': torch.tensor(dec_labels),
            'predict_inputs': torch.tensor(predict_inputs),
            'predict_mask': torch.tensor(predict_mask),
            'predict_labels': predict_labels
        }        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.dev_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def forward(self, batch):
        outputs = self.decoder(
            batch['dec_inputs'],
            attention_mask=batch['dec_mask'],
            labels=batch['dec_labels'],
        )
        return {'loss': outputs['loss'], 'logits': outputs['logits']}
    
    def training_step(self, batch: tuple, batch_idx: int):
        outputs = self(batch)
        return outputs['loss']
    
    def validation_step(self, batch: dict, batch_idx: int):
        outputs = self(batch)
        tokens = outputs['logits'].argmax(2).tolist()
        for row in tokens:
            self.val_result.append(self.dec_tok.decode(row))
        return {'val_loss': outputs['loss'].item()}

    def validation_epoch_end(self, val_step_outputs: list):
        print('\n\n', '-' * 100, '\n\n')
        print('validation_epoch_end')
        val_loss = [x['val_loss'] for x in val_step_outputs]
        print('val_loss %.4f \n' %torch.tensor(val_loss).mean().item())
        self.log('val_loss', torch.tensor(val_loss).mean().item())
        if len(self.val_result) > 1:
            print('sample result')
            for row in self.val_result[-5:]:
                print(row)
        self.val_result = []
        print('\n\n', '-' * 100, '\n\n')

    def predict(self, batch):
        ids = batch['predict_inputs']
        mask = batch['predict_mask']
        generated_ids = self.decoder.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=32,
            pad_token_id=self.EOS,
            bos_token_id=self.RES,
            eos_token_id=self.EOS,
            num_beams=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.0,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = generated_ids[:, ids.shape[-1]-1:]
        for pred in preds:
            self.predict_result.append(self.dec_tok.decode(pred, skip_special_tokens=True))
            print(self.dec_tok.decode(pred, skip_special_tokens=True))
            print('\n\n', '-' * 50, '\n\n')
        self.predict_ref.extend(batch['predict_labels'])
    def predict_step(self, batch: dict, batch_idx: int):
        self.predict(batch)

    def on_predict_epoch_end(self, predict_outputs=None):
        print('\n\n', '-' * 100, '\n\n')
        print('predict_epoch_end')
        print('\n\n', '-' * 100, '\n\n')
        bleu = self.bleu.corpus_score(self.predict_result, self.predict_ref)
        print('bleu', bleu)
        with open('3_baselines/result/predict_coda_conv.', 'w', encoding='utf-8') as f:
            for ref, pred in zip(self.predict_ref, self.predict_result):
                f.write('predict:' + pred + '\n')
                f.write('reference:' + ref + '\n\n')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)
        return optimizer
if __name__ == '__main__':
    torch.cuda.empty_cache()
    seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--acc_batch", type=int, default=1)
    parser.add_argument("--run_predict", type=str, default=None)
    parser.add_argument("--key_model", type=str, default=None)
    parser.add_argument("--output_file", type=str, default='path_gen.log')
    args = parser.parse_args()
    tb_logger = pl_loggers.TensorBoardLogger('2_dialog_generation/logs_one_turn/version_1', name='')
    checkpoint_callback = ModelCheckpoint(
        filename='best',
        save_weights_only=True,
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    bar_callback = progress.TQDMProgressBar(refresh_rate=25 if args.run_predict is None else 1)

    model = OneTurnModel(**vars(args))

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=20,
        logger=tb_logger,
        callbacks=[checkpoint_callback, bar_callback],
        gradient_clip_val=0.5,
        log_every_n_steps=25,
        accumulate_grad_batches=args.acc_batch,
    )
    if args.run_predict is not None:
        model = model.load_from_checkpoint(args.run_predict, strict=True)
        model.output_file = args.output_file
        model.batch_size *= 2
        trainer.predict(model)
    else:
        trainer.fit(model)


# nohup python -u 2_dialog_generation/one_turn_model.py > 2_dialog_generation/logs/one_turn_model_12.8.log 2>&1
# nohup python 2_dialog_generation/user_simulator.py --run_predict 3_baselines/logs_coda/version_5/checkpoints/best.ckpt > 3_baselines/logs/predict_coda_conv.log 2>&1