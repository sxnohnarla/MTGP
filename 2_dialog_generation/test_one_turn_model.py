import json
import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from nlgeval import NLGEval

def collate_fn_predict(model, batch):
    pc_batch = {}
    path_list = []
    context_list = []
    target_list = []
    bc_list = []
    res_list = []
    for line in batch:
        path = line['path']
        bc = line['b_c']
        context = line['context']
        target = line['target']
        res = line['res']
        res_list.append(res)
        path_list.append(path)
        context_list.append(context)
        target_list.append(target)
        bc_list.append(bc)
    pc_batch['path'] = path_list
    pc_batch['context'] = context_list
    pc_batch['target'] = target_list
    pc_batch['bc'] = bc_list
    pc_batch['res'] = res_list

    return pc_batch
    

if __name__ == '__main__':
    from one_turn_model import OneTurnModel,pad_to_max_seq_len
    import argparse

    parser_model = argparse.ArgumentParser()
    parser_model.add_argument("--ckpt", default='2_dialog_generation/logs_one_turn/version_1/checkpoints/best.ckpt')

    args = parser_model.parse_args()
    model = OneTurnModel.load_from_checkpoint(checkpoint_path = args.ckpt)
    model.freeze()
    model.to('cuda')


    test_dataset = []
    with open("2_dialog_generation/dialog_gen_data/test_ott_2.json", "r") as f:
        for line in f:
            test_dataset.append(json.loads(line))
    
    with torch.no_grad():
        nlg_eval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=[]) # "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"
        predict_result = []
        predict_ref = []
        for batch_data in tqdm([test_dataset[i:i+64] for i in range(0, len(test_dataset), 64)]):
            batch = collate_fn_predict(model, batch_data)
            dec_inputs = []
            dec_mask = []

            for path, context, target, b_c, res in zip(batch['path'], batch['context'], batch['target'], batch['bc'],batch['res']):
                input_seq = ''
                if len(b_c) != 0:
                    for b in b_c:
                        input_seq = '<b_c>' + b
                input_seq += '<pth>' + path + '<ctx>' + context + '<tgt>' + target
                input_ids = model.dec_tok.encode(input_seq, add_special_tokens=False)

                dec_inputs.append(input_ids + [model.RES])
                dec_mask.append([1] * len(dec_inputs[-1]))
                predict_ref.append(res)
            pad_to_max_seq_len(dec_inputs, pad_token_id=model.RES, max_seq_len=64)
            pad_to_max_seq_len(dec_mask, pad_token_id=0, max_seq_len=64)

            ids = torch.tensor(dec_inputs).to('cuda')
            mask = torch.tensor(dec_mask).to('cuda')

            generated_ids = model.decoder.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=32,
                pad_token_id=model.EOS,
                bos_token_id=model.RES,
                eos_token_id=model.EOS,
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
                predict_result.append(model.dec_tok.decode(pred, skip_special_tokens=True))
            
        with open('2_dialog_generation/predict_result/one_turn_result.jsonl', 'w', encoding='utf-8') as f:
            for predict, ref  in zip(predict_result, predict_ref) :
                f.write(json.dumps({'generate': predict, 'ref':ref}) + '\n')
        bleu = BLEU().corpus_score(predict_result, predict_ref)
        print('bleu', bleu)
        ref1 = []
        for ref in predict_ref:
            ref1.append(ref)
        result = nlg_eval.compute_metrics(ref_list=[ref1], hyp_list = predict_result)
        print('nlg_eval ', result)
'''
bleu BLEU = 5.90 68.3/18.4/1.7/0.8 (BP = 0.929 ratio = 0.932 hyp_len = 259 ref_len = 278)
'''


                




