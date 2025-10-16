import json
import os
import nncore

timechat_json_path = "/storage/wenzheng/dataset/TimeChat/TimeIT/data/dense_video_captioning/youcook2/val.caption_coco_format.json"  
# etbench_json_path = "/storage/wenzheng/dataset/etbench/annotations/evi/dvc_youcook2.json" 
output_path = "/storage/wenzheng/dataset/D2VLM_other_benchmark/youcook2/inference_input.json"

timechat_info = json.load(open(timechat_json_path, 'r'))
# etbench_info = json.load(open(etbench_json_path, 'r'))


new_etbench_info = []

for index, sample in enumerate(timechat_info['annotations']):

    cur_new_info = {}
    cur_new_info['version'] = 1.0

    cur_new_info['idx'] = index
    cur_new_info['task'] = 'dvc'
    cur_new_info['source'] = 'youcook2'
    relative_path = 'youcook2/compressed_videos'
    path = relative_path + '/' + sample['image_id']
    cur_new_info['video'] = path
    cur_new_info['duration'] = sample['duration']
    cur_new_info['tgt'] = sample['segments']
    cur_new_info['g'] = sample['pure_cap'].split('. ') 
    assert len(cur_new_info['g']) == len(cur_new_info['tgt'])

    cur_new_info['q'] = "You are given a video about 'making fattoush'. Watch the video carefully and densly describe all the cooking steps. For each step, you need to determine the start and ends times and provide a concise description. The format of your response should be: '<start time> - <end time>, <description>'."

    new_etbench_info.append(cur_new_info)

# print(1)

# nncore.load(args.llm_path)

nncore.dump(new_etbench_info, output_path)