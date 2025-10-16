import os
import json
import argparse

def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_folder", type=str, default='output/mvbench')
    parser.add_argument("--output_file", type=str, default='output/mvbench_metric.txt')

    args = parser.parse_args()
    folder_path = args.results_folder
    data_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data_dict[os.path.splitext(filename)[0]] = data

    correct = 0
    total = 0
    acc_dict = {}
    with open(args.output_file, 'w') as file:
        for task_type, value_list in data_dict.items():
            acc_dict[task_type] = [0, 0] # correct, total
            for item in value_list:
                acc_dict[task_type][1] += 1
                total += 1
                if check_ans(pred=item['pred'].replace('</evi>', ''), gt=item['A']):
                    acc_dict[task_type][0] += 1
                    correct += 1
            file.write(f"{task_type} Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%\n")
        file.write(f"Total Acc: {correct / total * 100 :.2f}%\n")
