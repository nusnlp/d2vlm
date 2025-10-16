# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import torch

from d2vlm.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_MATCH_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_EVIDENCE_TOKEN

def format_intervals(intervals):
    return [f"{subinterval[0][0]} - {subinterval[0][1]} " for subinterval in intervals] 

def tokenize(text, tokenizer, is_multimodal=True):
    if not is_multimodal:
        return tokenizer(text, return_tensors='pt').input_ids[0]

    chunks = [tokenizer(c).input_ids for c in text.split(DEFAULT_IMAGE_TOKEN)]

    input_ids, offset = [], 0
    if len(chunks) > 0 and len(chunks[0]) > 0 and chunks[0][0] == tokenizer.bos_token_id:
        input_ids.append(chunks[0][0])
        offset = 1

    img_token = [IMAGE_TOKEN_INDEX] * (offset + 1)
    chunks = [e[offset:] for c in zip(chunks, [img_token] * len(chunks)) for e in c][:-1]

    for chunk_ids in chunks:
        input_ids.extend(chunk_ids)

    input_ids = torch.LongTensor(input_ids)
    return input_ids


def detokenize(tokens, model, tokenizer, template=None):
    text = tokenizer.decode(tokens, skip_special_tokens=False).strip()

    tgt = getattr(model, 'tgt', None)   
    sim = getattr(model, 'sim', None)   # for debug
    if tgt is not None:
        assert len(tokens) == len(tgt), (tokens, tgt)

        if not torch.is_tensor(tokens):
            tokens = torch.LongTensor(tokens)

        if model.config.use_matching:
            model.match_inds = torch.where(tokens == model.config.match_token_id)[0].tolist()
        else:
            model.match_inds = torch.where(tokens == model.config.evidence_token_id)[0].tolist()
        tgt = [tgt[i] for i in model.match_inds]    
        sim = [sim[i] for i in model.match_inds]    # for debug

        if model.config.use_matching:
            assert text.count(DEFAULT_MATCH_TOKEN) == len(tgt), (text, tgt)

        elif all(not isinstance(x, list) for x in tgt):
            assert text.count(DEFAULT_EVIDENCE_TOKEN) == len(tgt), (text, tgt)
        else:
            tgt = format_intervals(tgt)
            assert text.count(DEFAULT_EVIDENCE_TOKEN) == len(tgt), (text, tgt)
        text = text.replace('{', '{{').replace('}', '}}')
        if model.config.use_matching:
            text = text.replace(DEFAULT_MATCH_TOKEN, '{}' if template is None else template)
        else:
            text = text.replace(DEFAULT_EVIDENCE_TOKEN, '{}' if template is None else template)
        text = text.format(*tgt)
    else:
        model.match_inds = None

    return text
