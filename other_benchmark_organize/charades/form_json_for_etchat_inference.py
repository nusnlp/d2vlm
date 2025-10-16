import json
import os
import nncore

timechat_json_path = "/storage/wenzheng/dataset/TimeChat/TimeIT/data/temporal_video_grounding/charades/test.caption_coco_format.json"   
# etbench_json_path = "/storage/wenzheng/dataset/etbench/annotations/evi/tvg_charades_sta.json"
output_path = "/storage/wenzheng/dataset/D2VLM_other_benchmark/charades/inference_input.json"

timechat_info = json.load(open(timechat_json_path, 'r'))
# etbench_info = json.load(open(etbench_json_path, 'r'))


new_etbench_info = []

for index, sample in enumerate(timechat_info['annotations']):

    cur_new_info = {}
    cur_new_info['version'] = 1.0

    cur_new_info['idx'] = index
    cur_new_info['task'] = 'tvg'
    cur_new_info['source'] = 'charades_sta'
    relative_path = 'charades/compressed_videos'
    path = relative_path + '/' + sample['image_id']
    cur_new_info['video'] = path
    cur_new_info['tgt'] = sample['timestamp']

    cur_new_info['q'] = f"You are given a video about indoor activities. Watch the video carefully and find a visual event described by the sentence: '{sample["caption"].strip(".")}'. The format of your response should be: 'The event happens in <start time> - <end time>'."

    new_etbench_info.append(cur_new_info)

# print(1)

# nncore.load(args.llm_path)

nncore.dump(new_etbench_info, output_path)