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

class DiaGenModelTarget(pl.LightningModule):
    def __init__(self, batch_size=None, lr=None, num_workers=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.save_hyperparameters()
        self.dec_tok = GPT2Tokenizer.from_pretrained("cache_gpt")
        self.decoder = GPT2LMHeadModel.from_pretrained("cache_gpt")
        self.bleu = BLEU()
        self.nlg_eval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=[]) # "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"
        
        self.dec_tok.add_special_tokens({
            'additional_special_tokens': ['<t_c>','<path>', '<ctx>', '<res>', '<eos>']
        })

        self.decoder.resize_token_embeddings(len(self.dec_tok))

        self.t_c = self.dec_tok.convert_tokens_to_ids('<t_c>')
        self.path = self.dec_tok.convert_tokens_to_ids('<path>')
        self.ctx = self.dec_tok.convert_tokens_to_ids('<ctx>')
        self.res = self.dec_tok.convert_tokens_to_ids('<res>')
        self.eos = self.dec_tok.convert_tokens_to_ids('<eos>')          

        self.relationsfound = set()
        self.relation2text = {}

    def prepare_data(self):
        self.val_result = []
        self.predict_result = []
        self.turn_count_list = []
        self.predict_bleu_result = []

    def setup(self, stage: str = None):
        # ! 这种数据的设置形式，是global的，即训练数据与这个无关
        # TODO 可以考虑用数据集的路径训练模型，作为local

        with open('utils/newrelation2text.json', 'r') as f:
            relation2text = json.load(f)
        self.relation2text = relation2text
        test_data = []
        with open('2_dialog_generation/dialog_gen_data/test_dd.json', 'r', encoding='utf-8') as f:
            for row in f:
                data = json.loads(row)
                triples = data['triples']
                context = data['context']
                dialog = data['dialog']
                test_data.append({'triples': triples, 'context': context, 'dialog': dialog})
        self.test_dataset = test_data
        if stage == 'fit':
            train_data = []
            with open('data/triple_dialog_data/dd_train.json', 'r', encoding='utf-8') as f:
                for row in f:
                    data = json.loads(row)
                    train_data.append(data)
            self.train_dataset = train_data
            dev_data = []
            with open('data/triple_dialog_data/dd_dev.json', 'r', encoding='utf-8') as f:
                for row in f:
                    data = json.loads(row)
                    dev_data.append(data)
            self.dev_dataset = dev_data
            print('train_len:', len(self.train_dataset), 'dev_len:', len(self.dev_dataset))
        
        elif stage == 'predict':
             print(f"predicting len: {len(self.test_dataset)}")
            
    def collate_fn_predict(self, batch):
        pc_batch = {}

        triples_list = []
        context_list = []
        dialog_list = []
        for i, line in enumerate(batch):
            triples =  line['triples']
            context = line['context']
            dialog = line['dialog']
            triples_list.append(triples)
            context_list.append(context)
            dialog_list.append(dialog)
        
        pc_batch['triples'] = triples_list
        pc_batch['context'] = context_list
        pc_batch['dialog'] = dialog_list

        return pc_batch

    def collate_fn(self, batch):
        dec_labels = []
        dec_inputs = []
        dec_mask = []

        # [t_c]__[path]__[ctx]__|[res]__
        for i, line in enumerate(batch):
            path = line['full_path']
            context_list = line['context']
            output = line['output']
            target = line['path'][-1]

            ctx_seq = '<t_c>' + target + '<path>' + path + '<ctx>' + '<ctx>'.join(context_list)
            res_seq = '<res>' + output + '<eos>'

            ctx_ids = self.dec_tok.encode(ctx_seq, add_special_tokens=False)
            res_ids = self.dec_tok.encode(res_seq, add_special_tokens=False)
            '''
               "<t_c> cat <path> story is the location which has cat <ctx> i have 3 and they're all named mike. what story? 
               <res> it is called people and thier cats. seeing trends and quirks of cat owners <eos>" 
            '''
            dec_inputs.append(ctx_ids + res_ids)
            dec_mask.append([1] * len(dec_inputs[-1]))
            dec_labels.append([-100] * len(ctx_ids) + res_ids)

        pad_to_max_seq_len(dec_inputs, pad_token_id=self.eos, max_seq_len=160)
        pad_to_max_seq_len(dec_mask, pad_token_id=0, max_seq_len=160)
        pad_to_max_seq_len(dec_labels, pad_token_id=-100, max_seq_len=160)

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
        if len(self.val_result) > 1:
            print('sample result')
            for row in self.val_result[-5:]:
                print(row)
        
        self.val_result = []
        print('\n\n', '-' * 100, '\n\n')

    def predict_step(self, batch: dict, batch_idx: int):
        topN = 1

        for triples, context, dialog in zip(batch['triples'], batch['context'], batch['dialog']):
            turn_count = 0
            path_input = []
            context_input = []
            output_list = []
            i = 0
            while True:
                dec_inputs = []
                dec_mask = []
                # ! 最终的target
                target = triples[-1][-1]
                turn_count += 1
                if i < len(triples):
                    temp_target = triples[i][-1]
                    triple = triples[i]
                    triple[1] = self.relation2text[triple[1]]
                    if i == 0:
                        path_input.append(triple[0]+" ")
                    i += 1
                    # ! 拼接path
                    path_input.append(" " + " ".join(triple[1:]))
                # !  训练的时候是有多个context拼接的
                context_input.append('<ctx>' + context)
                predict_ids = self.dec_tok.encode('<t_c>' + temp_target +'<path>' + ''.join(path_input) + ''.join(context_input))
                dec_inputs.append(predict_ids + [self.res])
                dec_mask.append([1] * len(dec_inputs[-1]))

                pad_to_max_seq_len(dec_inputs, pad_token_id=self.res, max_seq_len=128)
                pad_to_max_seq_len(dec_mask, pad_token_id=0, max_seq_len=128)

                input_ids = torch.tensor(dec_inputs).to(self.device)
                input_mask = torch.tensor(dec_mask).to(self.device)
                generated_ids = self.decoder.generate(
                                    input_ids=input_ids,
                                    attention_mask=input_mask,
                                    max_new_tokens=32,
                                    pad_token_id=self.eos,
                                    bos_token_id=self.res,
                                    eos_token_id=self.eos,
                                    num_beams=3,
                                    do_sample=True,
                                    top_k=50,
                                    top_p=0.95,
                                    early_stopping=True,
                                    num_return_sequences = topN,
                                    no_repeat_ngram_size=2
                                )
                preds = generated_ids[:, input_ids.shape[-1]-1:]
                # 为啥有的会有多个输出
                for pred in preds:
                    context = self.dec_tok.decode(pred, skip_special_tokens=True)
                output_list.append(context)
                self.predict_bleu_result.append((context, dialog))
                if target in context:
                    break
                if turn_count >= 6:
                    break
            self.predict_result.append(output_list)
            self.turn_count_list.append(turn_count)
            for item in output_list:
                print(item)
            print("--------------------------------")
    '''
     固定轮数
    '''
    # def predict_step(self, batch: dict, batch_idx: int):
    #     dec_inputs = []
    #     dec_mask = []
    #     topN = 1

    #     for triples, context in zip(batch['triples'], batch['context']):
    #         turn_count = len(triples) # 对话会有几轮
    #         path_input = []
    #         context_input = []
    #         output_list = []
    #         for i in range(turn_count):
    #             triple = triples[i]
    #             triple[1] = self.relation2text[triple[1]]
    #             if i == 0:
    #                 path_input.append(triple[0]+" ")
    #             path_input.append(" " + " ".join(triple[1:]))
    #             context_input.append('<SEP>' + context)

    #             dec_input = self.dec_tok.encode(''.join(path_input) + ''.join(context_input))[:128]
    #             dec_input += [self.PAD] * (128 - len(dec_input))
    #             dec_inputs.append(dec_input)
    #             dec_mask_pre = [1] * len(dec_input)
    #             dec_mask.append(dec_mask_pre)

    #             input_ids = torch.tensor(dec_inputs).to(self.device)
    #             input_mask = torch.tensor(dec_mask).to(self.device)
    #             generated_ids = self.decoder.generate(
    #                                 input_ids=input_ids,
    #                                 attention_mask=input_mask,
    #                                 max_new_tokens=31,
    #                                 pad_token_id=self.PAD,
    #                                 eos_token_id=self.END,
    #                                 num_beams=3,
    #                                 do_sample=True,
    #                                 top_k=50,
    #                                 top_p=0.95,
    #                                 early_stopping=True,
    #                                 num_return_sequences = topN,
    #                                 no_repeat_ngram_size=2
    #                             )
    #             preds = generated_ids[:, input_ids.shape[-1]-1:]
    #             # 为啥有的会有多个输出
    #             context = self.dec_tok.decode(preds[-1], skip_special_tokens=True)
    #             output_list.append(context)
    #         self.predict_result.append(output_list)
    #         for item in output_list:
    #             print(item)
    #         print("--------------------------------")
        
    def on_predict_epoch_end(self, predict_outputs=None):
        print('\n\n', '-' * 100, '\n\n')
        print('on_predict_epoch_end')
        print('\n\n', '-' * 100, '\n\n')

        pred_response = [ predict[0] for predict in self.predict_bleu_result ]
        refs = [ ref[1] for ref in self.predict_bleu_result]

        ref1, ref2, ref3, ref4, ref5, ref6 = [], [], [], [], [], []
        for ref in refs:
            ref1.append(ref[0])
            ref2.append(ref[0] if len(ref) < 2 else ref[1])
            ref3.append(ref[0] if len(ref) < 3 else ref[2])
            ref4.append(ref[0] if len(ref) < 4 else ref[3])
            ref5.append(ref[0] if len(ref) < 5 else ref[4])
            ref6.append(ref[0] if len(ref) < 6 else ref[5])
        '''
            hypotheses: Sequence[str],
            references: Optional[Sequence[Sequence[str]]
        '''
        # bleu = self.bleu.corpus_score(pred_response, refs)
        # print('bleu', bleu)
        # result = self.nlg_eval.compute_metrics(ref_list=[ref1, ref2, ref3, ref4, ref5, ref6], hyp_list=pred_response)
        # print('nlg_eval ', result)

        with open('2_dialog_generation/predict_result/predict_global_result_dd_notgt.jsonl', 'w', encoding='utf-8') as f:
            for dialog_list, turn  in zip(self.predict_result, self.turn_count_list) :
                f.write(json.dumps({'dialog_list': dialog_list, 'turn_count':turn}) + '\n')
 
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)
        return optimizer
if __name__ == '__main__':
    torch.cuda.empty_cache()
    seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--acc_batch", type=int, default=1)
    parser.add_argument("--run_predict", type=str, default=None)
    parser.add_argument("--key_model", type=str, default=None)
    parser.add_argument("--output_file", type=str, default='path_gen.log')
    args = parser.parse_args()
    tb_logger = pl_loggers.TensorBoardLogger('2_dialog_generation/logs_dialog_gen_dd_target', name='')
    checkpoint_callback = ModelCheckpoint(
        filename='best',
        save_weights_only=True,
        save_last=True,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    bar_callback = progress.TQDMProgressBar(refresh_rate=25 if args.run_predict is None else 1)

    model = DiaGenModelTarget(**vars(args))

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10,
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



# nohup python -u 2_dialog_generation/dialog_gen_model_target.py > 2_dialog_generation/logs/dialog_gen_raw_target.12.1.log 2>&1