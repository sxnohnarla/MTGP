import json
import torch
from tqdm import tqdm



if __name__ == '__main__':
    from dialog_gen_model_target import DiaGenModelTarget
    from user_simulator import UserModel, pad_to_max_seq_len
    import argparse

    parser_model = argparse.ArgumentParser()
    parser_model.add_argument("--ckpt", default='2_dialog_generation/logs_dialog_gen_target/version_1/checkpoints/best.ckpt')

    args = parser_model.parse_args()
    model = DiaGenModelTarget.load_from_checkpoint(checkpoint_path = args.ckpt)
    model.freeze()
    model.to('cuda')

    parser_user = argparse.ArgumentParser()
    parser_user.add_argument("--ckpt", default='2_dialog_generation/logs_user_simulator/version_31/checkpoints/best.ckpt')

    args = parser_user.parse_args()
    user_model = UserModel.load_from_checkpoint(checkpoint_path = args.ckpt)
    user_model.freeze()
    user_model.to('cuda')

    with open('utils/newrelation2text.json', 'r') as f:
        relation2text = json.load(f)

    test_data = []
    with open('2_dialog_generation/dialog_gen_data/test_dd.json', 'r', encoding='utf-8') as f:
        for row in f:
            data = json.loads(row)
            triples = data['triples']
            context = data['context']
            dialog = data['dialog']
            test_data.append({'triples': triples, 'context': context, 'dialog': dialog})
    test_dataset = test_data

    with torch.no_grad():
        predict_result = []
        turn_count_list = []
        for batch_data in tqdm([test_dataset[i:i+64] for i in range(0, len(test_dataset), 64)]):
            batch = model.collate_fn_predict(batch_data)
            for triples, context, dialog in zip(batch['triples'], batch['context'], batch['dialog']):
                turn_count = 0
                output_list = []
                path_input = []
                i = 0
                while True:
                    dec_inputs = []
                    dec_mask = []
                    model_input_context = []
                    predict_inputs = []
                    predict_mask = []
                    target = triples[-1][-1]
                    turn_count += 1
                    '''
                       path_input 准备路径
                    '''
                    if i < len(triples):
                        temp_target = triples[i][-1]
                        triple = triples[i]
                        triple[1] = relation2text[triple[1]]
                        if i == 0:
                            path_input.append(triple[0]+" ")
                        i += 1
                        path_input.append(" ".join(triple[1:]))
                    '''
                        context_input 输入model中的context
                    '''
                    model_input_context.append('<ctx>' + context)
                    user_input_context = '[ctx]' + context

                    ctx_ids = user_model.dec_tok.encode(user_input_context, add_special_tokens=False)
                    predict_inputs.append(ctx_ids + [user_model.RES]) 
                    predict_mask.append([1] * len(predict_inputs[-1]))
                    pad_to_max_seq_len(predict_inputs, pad_token_id=user_model.EOS, max_seq_len=32)
                    pad_to_max_seq_len(predict_mask, pad_token_id=0, max_seq_len=32)
                    ids = torch.tensor(predict_inputs).to('cuda')
                    mask = torch.tensor(predict_mask).to('cuda')
                    generated_ids = user_model.decoder.generate(
                                        input_ids=ids,
                                        attention_mask=mask,
                                        max_new_tokens=32,
                                        pad_token_id=user_model.EOS,
                                        bos_token_id=user_model.RES,
                                        eos_token_id=user_model.EOS,
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
                        user_context = user_model.dec_tok.decode(pred, skip_special_tokens=True)
                        model_input_context.append('<ctx>' + user_context)
                    output_list.append(user_context)
                    if target in user_context:
                        break

                    predict_ids = model.dec_tok.encode('<t_c>' + temp_target +'<path>' + ''.join(path_input) + ''.join(model_input_context))
                    dec_inputs.append(predict_ids + [model.res])
                    dec_mask.append([1] * len(dec_inputs[-1]))

                    pad_to_max_seq_len(dec_inputs, pad_token_id=model.res, max_seq_len=128)
                    pad_to_max_seq_len(dec_mask, pad_token_id=0, max_seq_len=128)

                    input_ids = torch.tensor(dec_inputs).to('cuda')
                    input_mask = torch.tensor(dec_mask).to('cuda')
                    generated_ids = model.decoder.generate(
                                    input_ids=input_ids,
                                    attention_mask=input_mask,
                                    max_new_tokens=32,
                                    pad_token_id=model.eos,
                                    bos_token_id=model.res,
                                    eos_token_id=model.eos,
                                    num_beams=3,
                                    do_sample=True,
                                    top_k=50,
                                    top_p=0.95,
                                    early_stopping=True,
                                    num_return_sequences = 1,
                                    no_repeat_ngram_size=2
                                )
                    preds = generated_ids[:, input_ids.shape[-1]-1:]
                    for pred in preds:
                        context = model.dec_tok.decode(pred, skip_special_tokens=True)
                    
                    output_list.append(context)
                    if target in context:
                        break
                    if turn_count >= 6:
                        break
                predict_result.append(output_list)
                turn_count_list.append(turn_count)
                print(output_list)
                print('\n\n', '-' * 100, '\n\n')
        with open('2_dialog_generation/predict_result/predict_global_result_user_target_dd_1.14.jsonl', 'w', encoding='utf-8') as f:
            for dialog_list, turn  in zip(predict_result, turn_count_list) :
                f.write(json.dumps({'dialog_list': dialog_list, 'turn_count':turn}) + '\n')       

# nohup python -u 2_dialog_generation/dialog_gen_user_target.py > 2_dialog_generation/logs/user_logs/dialog_gen_user_target_dd_1.14.log 2>&1
