import json
import torch
from tqdm import tqdm

if __name__ == '__main__':
    from train_CODA import CODA, pad_to_max_seq_len
    import argparse

    parser_model = argparse.ArgumentParser()
    parser_model.add_argument("--ckpt", default='3_baselines/logs_coda/version_6/checkpoints/best.ckpt')

    args = parser_model.parse_args()
    model = CODA.load_from_checkpoint(checkpoint_path = args.ckpt)
    model.freeze()
    model.to('cuda')

    with open('utils/newrelation2text.json', 'r') as f:
        relation2text = json.load(f)

    test_data = []
    with open('2_dialog_generation/dialog_gen_data/test_coda.json', 'r', encoding='utf-8') as f:
        for row in f:
            data = json.loads(row)
            path = data['path']
            context = data['context']
            target = data['target']
            test_data.append({'path': path, 'context': context, 'target': target})
    test_dataset = test_data

    with torch.no_grad():
        predict_result = []
        turn_count_list = []
        for batch_data in tqdm([test_dataset[i:i+64] for i in range(0, len(test_dataset), 64)]):
            batch = model.collate_fn_predict(batch_data)
            for path, context, target in zip(batch['path'], batch['context'], batch['target']):
                t_c = path.split(" ")[-1]
                turn_count = 0
                output_list = []
                i = 0
                while True:
                    predict_inputs = []
                    predict_mask = []
                    turn_count += 1

                    input_seq = '<pth>' + path + '<ctx>' + context + '<tgt>' + target
                    input_ids = model.dec_tok.encode(input_seq, add_special_tokens=False)

                    predict_inputs.append(input_ids + [model.RES])
                    predict_mask.append([1] * len(predict_inputs[-1]))

                    pad_to_max_seq_len(predict_inputs, pad_token_id=model.RES, max_seq_len=64)
                    pad_to_max_seq_len(predict_mask, pad_token_id=0, max_seq_len=64)

                    ids = torch.tensor(predict_inputs).to('cuda')
                    mask = torch.tensor(predict_mask).to('cuda')

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
                        early_stopping=True,
                    )

                    preds = generated_ids[:, ids.shape[-1]-1:]
                    for pred in preds:
                        context = model.dec_tok.decode(pred, skip_special_tokens=True)
                    output_list.append(context)
                    if t_c in context:
                        break
                    if turn_count >= 6:
                        break
                predict_result.append(output_list)
                turn_count_list.append(turn_count)
                print(output_list)
                print('\n\n', '-' * 100, '\n\n')
        with open('3_baselines/result/predict_coda_conv_nouser.json', 'w', encoding='utf-8') as f:
            for dialog_list, turn  in zip(predict_result, turn_count_list) :
                f.write(json.dumps({'dialog_list': dialog_list, 'turn_count':turn}) + '\n')








