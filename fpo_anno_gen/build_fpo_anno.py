# ------------------------------------------------------------------------
# D2VLM
# Copyright (c) 2025 Wenzheng Zeng. Licensed under the BSD-3-Clause license.
# ------------------------------------------------------------------------


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoTokenizer, HfArgumentParser
from d2vlm.train.train import ModelArguments, DataArguments, TrainingArguments, MultimodalDataset, MultimodalDataCollator
from d2vlm.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_MATCH_TOKEN, IGNORE_INDEX, DEFAULT_EVIDENCE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_EVIDENCE_END_TOKEN
import copy
import re
import random
import nncore
from d2vlm.utils.tokenization import tokenize
from tqdm import tqdm


# import debugpy


# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9507))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
    
    
def task_filter(anno, task):
    all_samples = [] 
    for i in range(len(anno)):
        sample = copy.deepcopy(anno[i])
        if sample.get('task') == task:
            all_samples.append(sample)
    return all_samples





def update_mask_and_label(full_text_token_ids, last_context_token_pos, wrong_text_token_ids, insert_point, old_label, old_special_mask_list):
    """
    创建自定义的attention mask
    
    参数:
    - full_text_token_ids: 完整的token序列
    - last_context_token_pos: 最后一个context token，是evi后面的逗号
    - text_before_insert_len: text_before_insert的长度
    - wrong_text_len: wrong_text的长度
    - correct_text_len: correct_text的长度
    """
    seq_length = len(full_text_token_ids)
    
    # mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
    
    wrong_text_start = insert_point   
    assert len(wrong_text_token_ids[:-1])>=1    
    wrong_text_end = wrong_text_start + len(wrong_text_token_ids[:-1]) -1  
    
    correct_text_start = last_context_token_pos + 1 
    correct_text_end = insert_point -1 

    new_special_mask_list = []
    
    for i in range(wrong_text_start, wrong_text_end+1):
        # mask[i, correct_text_start:correct_text_end+1] = False
        if len(old_special_mask_list):  
            previous_wrong_info = old_special_mask_list[-1]
            special_mask_pos_list=previous_wrong_info[1]
            new_special_mask_list.append([i, special_mask_pos_list + [[correct_text_start, correct_text_end]]])
        else:
            new_special_mask_list.append([i, [[correct_text_start, correct_text_end]]])


    for i in range(wrong_text_end+1, seq_length):
        if len(old_special_mask_list):
            previous_wrong_info = old_special_mask_list[-1]
            special_mask_pos_list=previous_wrong_info[1]
            new_special_mask_list.append([i, special_mask_pos_list + [[wrong_text_start, wrong_text_end]]])
        else:
            new_special_mask_list.append([i, [[wrong_text_start, wrong_text_end]]])
    
    new_label = full_text_token_ids.clone()
    old_label_start = old_label[0]
    old_label = old_label[1:]
    new_label = new_label[1:]   
    new_label[wrong_text_start-1] = full_text_token_ids[wrong_text_end+1]  
    new_label[wrong_text_start:wrong_text_end+1] = IGNORE_INDEX

    new_label = torch.cat((old_label_start.unsqueeze(0), old_label[:insert_point], new_label[insert_point:]))
    if len(old_special_mask_list):
        insert_index_in_old = [i for i, item in enumerate(old_special_mask_list) if item[0] == insert_point]
        assert len(insert_index_in_old) == 1
        new_special_mask_list = old_special_mask_list[:insert_index_in_old[0]] + new_special_mask_list
    
    return new_special_mask_list, new_label





if __name__ == "__main__":

    random.seed(42)
    orginal_anno_path = "/storage/wenzheng/dataset/D2VLM-Dataset/ET-Instruct/FPO/fpo_annotation.json"
    out_path = "/storage/wenzheng/dataset/D2VLM-Dataset/ET-Instruct/FPO/tokenized_fpo_annotation.pt"
    task = "all"
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    if len(sys.argv) > 1:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args = ModelArguments()
        data_args = DataArguments()
        training_args = TrainingArguments()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        add_prefix_space=training_args.add_prefix_space,
        legacy=training_args.legacy_tokenizer,
        use_fast=training_args.fast_tokenizer
    )

    if tokenizer.pad_token is None:
        print(f'PAD token not found, using EOS token instead: {tokenizer.eos_token} ({tokenizer.eos_token_id})')
        tokenizer.pad_token = tokenizer.eos_token

    evidence_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EVIDENCE_TOKEN)
    evidence_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EVIDENCE_END_TOKEN)

    ##############################################################################################
    # args = parse_args()

    anno = nncore.load(orginal_anno_path)
    
    print(f"step1: 数据加载, 原始数据集共{len(anno)}个样本")

    if task != "all":
        task_samples = task_filter(anno, task)
    else:
        task_samples = anno
    print(f"step2: 任务筛选, {task}任务子集共{len(task_samples)}个样本")


    for sample in tqdm(task_samples):
        
        class_list = list(sample['conversations'].keys())
        class_list.remove('chosen')
        sample_processed_response = {}
        for key in class_list:
            
            cur_chosen_branch = sample['conversations']['chosen']
            cur_reject_branch = sample['conversations'][key]
            conflict_events = []
            
            correct_text_whole = cur_chosen_branch[1]['value']
            reject_text_whole = cur_reject_branch[1]['value']
            full_text_token_ids = tokenize(correct_text_whole, tokenizer, is_multimodal=True)
            full_text_token_ids_original = full_text_token_ids.clone()
            label = full_text_token_ids.clone()
            
            pos_id = list(range(0, len(full_text_token_ids)))
            special_mask_list = []
            evi_pos_list = torch.where(full_text_token_ids==32011)[0]
            evi_half_num = evi_pos_list.shape[0]//2

            if key.split('_')[2] == 'f' or key.split('_')[2] == 'repeat' or key.split('_')[2] == 'qwen':
                assert sample['task'] in ['dvc', 'slc', 'tvc', 'rvc']

                if key.split('_')[2] == 'qwen':   
                    assert sample['task'] == 'rvc'  
                    insert_point = len(full_text_token_ids) - 1 
                    insert_point_dynamic = insert_point
                    last_context_token_pos = 0
                    last_context_token_pos_dynamic = 0
                    wrong_text_token_ids = tokenize(reject_text_whole, tokenizer, is_multimodal=True)[1:]

                    correct_text_token_ids = full_text_token_ids[last_context_token_pos_dynamic+1: insert_point_dynamic+1]  
                    pos_id = pos_id[0:insert_point_dynamic] + list(range(int(last_context_token_pos)+1, int(last_context_token_pos)+1+len(wrong_text_token_ids[:-1]))) + list(range(insert_point, insert_point + len(full_text_token_ids[insert_point_dynamic:])))
                    full_text_token_ids = torch.cat([full_text_token_ids[:insert_point_dynamic], wrong_text_token_ids[:-1], full_text_token_ids[insert_point_dynamic:]], dim=0)
                    # label = full_text_token_ids.clone()
                    special_mask_list, label = update_mask_and_label(full_text_token_ids, last_context_token_pos_dynamic, wrong_text_token_ids, insert_point_dynamic, label, special_mask_list)

                    
                    chosen = {'text':{"input_index":list(range(last_context_token_pos_dynamic, insert_point_dynamic)), "logit_token_id":correct_text_token_ids.tolist()}}
                    reject = {'text':{"input_index":[int(last_context_token_pos_dynamic)] + list(range(insert_point_dynamic, insert_point_dynamic + len(wrong_text_token_ids[:-1]))), "logit_token_id":wrong_text_token_ids.tolist()}}
                    conflict_events.append({'chosen': chosen, 'reject': reject})

                    chosen_branch = {"full_token_ids": full_text_token_ids, "label":label, "pos_id": pos_id, "conflict_events":conflict_events, "special_mask_list": special_mask_list}    

                    sample_processed_response[key] = chosen_branch

                    continue


            pos_id = pos_id + [pos_id[-1]+1] 

            if len(special_mask_list)>0:
                previous_wrong_info = special_mask_list[-1]
                special_mask_pos_list=previous_wrong_info[1]
                special_mask_list.append([len(pos_id)-1, special_mask_pos_list])


            end_token_ids = torch.tensor([32007])  
            full_text_token_ids = torch.cat([full_text_token_ids, end_token_ids], dim=0)
            label = torch.cat([label, torch.tensor([32007])], dim = 0)   


            reject_text_tokens = tokenize(reject_text_whole, tokenizer, is_multimodal=True)

            pos_id = pos_id + list(range(0, len(reject_text_tokens)))   


            reject_start_pos = full_text_token_ids.shape[0]
            
            reject_special_mask_list = []
            for i in range(reject_start_pos, reject_start_pos + len(reject_text_tokens)):
                # mask[i, correct_text_start:correct_text_end+1] = False
                reject_special_mask_list.append([i, [[0, reject_start_pos-1]]])

            special_mask_list = special_mask_list + reject_special_mask_list    
            full_text_token_ids = torch.cat([full_text_token_ids, reject_text_tokens], dim=0)
            padding_label = torch.full((reject_text_tokens.shape[0],), -100)
            label = torch.cat([label, padding_label], dim = 0)
            
            # print(1)
            indices = torch.where(label != -100)
            chosen_objective = label[indices]
            chosen_input = full_text_token_ids[indices[0]]
            real_chosen_input_index = indices[0]-1
            
            chosen_tgt = sample['tgt']['chosen']
            pairs = [[chosen_tgt[i], chosen_tgt [i+1]] for i in range(0, len(chosen_tgt), 2)]
            chosen_result = pairs + pairs
            chosen_evi_pos_list = torch.where(label==32011)[0]

            reject_input_pos = list(range(reject_start_pos, reject_start_pos + len(reject_text_tokens)))

            reject_tgt = sample['tgt'][key]
            pairs = [[reject_tgt[i], reject_tgt [i+1]] for i in range(0, len(reject_tgt), 2)]
            reject_result = pairs + pairs
            reject_evi_pos_list = torch.where(reject_text_tokens==32011)[0] + reject_start_pos

            assert reject_start_pos==reject_evi_pos_list[0].item()
            chosen = {'grounding':{"input_index":[x.item()-1 for x in chosen_evi_pos_list], "tgt":chosen_result}, 'text':{"input_index":[x.item() for x in real_chosen_input_index], "logit_token_id":chosen_objective}} 
            reject = {'grounding':{"input_index":[-1] + [x.item()-1 for x in reject_evi_pos_list[1:]], "tgt":reject_result}, 'text':{"input_index":[-1] + reject_input_pos, "logit_token_id":reject_text_tokens.tolist() + [32007]}}
            conflict_events.append({'chosen': chosen, 'reject': reject})

            only_dpo_conflict_events = []
            only_dpo_conflict_events.append({'chosen': chosen, 'reject': reject})

            chosen_branch = {"full_token_ids": full_text_token_ids, "label":label, "pos_id": pos_id, "conflict_events":only_dpo_conflict_events, "special_mask_list": special_mask_list}    

            sample_processed_response[key] = chosen_branch
            
        sample['processed_response'] = sample_processed_response

    torch.save(task_samples, out_path)  

    # begin a minor format alignment
    per_sample_num = 1  
    new_samples = []
    for sample in tqdm(task_samples):
        all_responses = list(sample['processed_response'].items())
        
        num_samples = min(per_sample_num , len(all_responses))
        
        selected_responses = random.sample(all_responses, num_samples)  
        
        for random_key, random_value in selected_responses:
            new_sample = sample.copy()  
            if new_sample.get('tgt'):
                new_sample['tgt_reject'] = new_sample['tgt'][random_key]    
                new_sample['tgt'] = new_sample['tgt']['chosen']           
            else:
                new_sample['src'] = new_sample['src']['chosen']
            new_sample['conversations'] = new_sample['conversations']['chosen']
            new_sample['conversations'][1]['value'] = ''
            
            random_value['class'] = random_key
            new_sample['processed_response'] = random_value
            
            new_samples.append(new_sample)

    torch.save(new_samples, out_path)


