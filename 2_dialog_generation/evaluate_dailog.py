import json
from evaluate import load
from numpy import mean
from dialog_gen_model import DiaGenModel
from transformers import AutoTokenizer, AutoModel
import torch


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def pad_to_max_len(model, lista, listb):
    # lista 和 listb的最大长度
    max_len = max(len(lista), len(listb))
    # 用'<PAD>'填充
    lista.extend([0] * (max_len - len(lista)))
    listb.extend([0] * (max_len - len(listb)))
    return lista, listb

def read_predict_result(path):
    all_gen_dialog = []
    turn_count_list = []
    with open(path, "r") as f:
        for row in f:
            data = json.loads(row)
            dialog_list = data["dialog_list"]
            turn_count = data["turn_count"]
            all_gen_dialog.append(dialog_list)
            turn_count_list.append(turn_count)

    return all_gen_dialog, turn_count_list

def read_context(path):
    all_context = []
    all_triples = []
    with open(path, "r") as f:
        for row in f:
            data = json.loads(row)
            context = data["context"]
            triples = data["triples"]
            all_context.append(context)
            all_triples.append(triples)

    return all_context,all_triples

if __name__ == "__main__":
    # gen_dialog_path = "2_dialog_generation/predict_result/predict_global_result_dd_notgt.jsonl"
    gen_dialog_path = "2_dialog_generation/predict_result/predict_global_result_raw_target_12.1.jsonl"
    gen_dialog_list, turn_count_list = read_predict_result(gen_dialog_path)
    # context_path = "2_dialog_generation/dialog_gen_data/test_dd.json"
    context_path = "2_dialog_generation/dialog_gen_data/test_bleu.json"
    context_list, triples_list = read_context(context_path)
    bertscore = load("bertscore")

    sent_tok = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    sent_enc = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to('cuda')

    turn_count = 0
    # turn_count_list = []
    achieve_count = 0
    achieve_list = []
    along_path_count = 0
    along_path_list = []
    responses = []
    contexts = []
    dataset_len = 0
    i = 1
    for triples, context, dialogs, turn in zip(triples_list, context_list, gen_dialog_list, turn_count_list):
        contexts.append(context)
        dataset_len += 1
        # turn_count += len(dialogs) + 1
        turn_count += turn
        # turn_count_list.append(len(dialogs) + 1)
        source = triples[0][0]
        target = triples[-1][-1].replace("_"," ")
        last_sentece = dialogs[-1]
        if target in last_sentece:
            achieve_count += 1
            achieve_list.append(i)
        else:
            print(i,context)
        temp_along_path_count = 0
        # for triple, dialog in zip(triples, dialogs):
        #     if triple[-1] in dialog:
        #         temp_along_path_count += 1
        # if temp_along_path_count == len(dialogs):
        #     along_path_count += 1
        #     along_path_list.append(i)
        responses.extend(dialogs)
        contexts.extend(dialogs[:-1])
        i += 1
    print(achieve_count, dataset_len)
    print("#Turns: ", '{:.2f}'.format(turn_count / dataset_len))
    print("Achievement: ", '{:.2f}'.format(achieve_count / dataset_len))
    # results = bertscore.compute(predictions=responses, references=contexts, model_type="bert-base-uncased")
    # results['precision'] = mean(results['precision'])
    # results['recall'] = mean(results['recall'])
    # results['f1'] = mean(results['f1'])
    # print(results)
    # re, co = pad_to_max_len(model, model.dec_tok.encode(responses[0]), model.dec_tok.encode(contexts[0]))
    # responses_embed = torch.tensor(re,dtype=torch.float)
    # contexts_embed = torch.tensor(co,dtype=torch.float)
    # cosine_sim = torch.cosine_similarity(responses_embed, contexts_embed, -1)
    # print(cosine_sim)
    # print(responses[0], contexts[0])
    cosine_sim_list = []
    if len(responses) == len(contexts):
        for j in range(len(responses)):
            encode_context = sent_tok(contexts[j], padding=True, truncation=True, return_tensors="pt").to('cuda')
            encode_response = sent_tok(responses[j], padding=True, truncation=True, return_tensors="pt").to('cuda')
            context_output = sent_enc(**encode_context, output_hidden_states=True, return_dict=True)
            response_output = sent_enc(**encode_response, output_hidden_states=True, return_dict=True)
            context_sentence_embeddings  = mean_pooling(context_output, encode_context['attention_mask'])
            context_sentence_embeddings = torch.functional.F.normalize(context_sentence_embeddings, p=2, dim=1)
            response_sentence_embeddings  = mean_pooling(response_output, encode_response['attention_mask'])
            response_sentence_embeddings = torch.functional.F.normalize(response_sentence_embeddings, p=2, dim=1)
            cosine_sim = torch.cosine_similarity(response_sentence_embeddings, context_sentence_embeddings).tolist()
            cosine_sim_list.append(cosine_sim)
        # tensor算平均值
        print("Cosine Similarity: ", mean(cosine_sim_list))
    else:
        print("calc sematic semilarity error")
    # print("Along Path: ", '{:.2f}'.format(along_path_count / dataset_len))
    # print(along_path_list)
    # print(turn_count_list)
    # print(achieve_list)
    # '{:.2f}'.format(sum_triple_sum_score / len(one_path_triple_list)
