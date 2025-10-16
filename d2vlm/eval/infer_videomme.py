# ------------------------------------------------------------------------
# D2VLM
# Copyright (c) 2025 Wenzheng Zeng. Licensed under the BSD-3-Clause license.
# ------------------------------------------------------------------------
# Modified from E.T. Chat (https://github.com/PolyU-ChenLab/ETBench)
# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.
# ------------------------------------------------------------------------

import sys
# sys.path.append('/home/dancer/ETBench/')

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
import copy
import nncore
import torch
from d2vlm.constants import DEFAULT_IMAGE_TOKEN
from d2vlm.conversation import get_conv
from d2vlm.model.builder import build_model
from d2vlm.utils.inference import KeywordsStoppingCriteria, RepetitionPenaltyLogitsProcessor
from d2vlm.utils.io import load_video
from d2vlm.utils.tokenization import detokenize, tokenize
import pandas as pd
import numpy as np
import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9503))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def convert_ndarray_to_list(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(element) for element in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj




def gen_question(domain, question, options):
    options_text = "\n".join([f"{option}" for option in options])
    full_question = question + '\n' + options_text
    mme_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n{{Question}}\nThe best answer is:"

    full_prompt = mme_prompt.replace('{{Question}}', full_question)

    return full_prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_path', default="/storage/wenzheng/dataset/D2VLM_other_benchmark/Video-MME/videomme/test-00000-of-00001.parquet")
    parser.add_argument('--data_path', default='/storage/wenzheng/dataset/D2VLM_other_benchmark/Video-MME/data')
    parser.add_argument('--pred_path', default='output')    
    parser.add_argument('--model_path')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--arch_module', type=str, default='d2vlm_arch')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'etbench_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'etbench3.json')
    print(f'Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    df = pd.read_parquet(args.anno_path)
    anno = df.to_dict(orient='records')
    anno = [anno[i::args.chunk] for i in range(args.chunk)][args.index]
    
    # model, tokenizer, transform = build_model(args.model_path, device=args.device)
    model, tokenizer, transform = build_model(args.model_path, device=args.device, arch_module=args.arch_module)

    all_samples = [] 
    for i in nncore.ProgressBar(range(len(anno))):
        sample = copy.deepcopy(anno[i])

        video = nncore.join(args.data_path, sample['videoID']+'.mp4')
        video, tag = load_video(video, num_threads=1)
        video = transform(video).half().to(args.device)

        query = gen_question(sample['domain'], sample['question'], sample['options'])

        conv = get_conv(model.config.conv_type)
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        logits_processor = None
        input_ids = tokenize(prompt, tokenizer).unsqueeze(0).to(args.device)
        stop_str = conv.seps[-1]
        stopping_criteria = [KeywordsStoppingCriteria(tokenizer, stop_str)]

        model.eval()
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                image=[video],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=2048,
                cache_implementation=None,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                query=[[query]],
                src=[None],
                tag=[tag])

        tokens = out[0, input_ids.size(1):]
        response = detokenize(tokens, model, tokenizer)
        if response.endswith(stop_str):
            response = response[:-len(stop_str)].strip()

        sample['a'] = response
        all_samples.append(sample)

        if args.verbose:
            print()
            print(response)
            print(f"Question: {sample.get('question')}")
            print(f"Answer: {sample.get('a')}")
            print(f"GT: {sample.get('answer')}")

    converted_samples = [convert_ndarray_to_list(sample) for sample in all_samples]
    nncore.dump(converted_samples, pred_path)
