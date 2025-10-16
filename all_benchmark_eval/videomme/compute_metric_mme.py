import os
import argparse
import json
from typing import List, Dict, Optional, Union
import re

from collections import defaultdict
import nncore


CATEGORIES = [
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual"
]

SUB_CATEGORIES = [
    "Humanity & History",
    "Literature & Art",
    "Biology & Medicine",
    "Finance & Commerce",
    "Astronomy",
    "Geography",
    "Law",
    "Life Tip",
    "Technology",
    "Animation",
    "Movie & TV Show",
    "Documentary",
    "News Report",
    "Esports",
    "Basketball",
    "Football",
    "Athletics",
    "Other Sports",
    "Stage Play",
    "Magic Show",
    "Variety Show",
    "Acrobatics",
    "Handicraft",
    "Food",
    "Fashion",
    "Daily Life",
    "Travel",
    "Pet & Animal",
    "Exercise",
    "Multilingual"
]

TASK_CATEGORIES = [
    "Temporal Perception",
    "Spatial Perception",
    "Attribute Perception",
    "Action Recognition",
    "Object Recognition",
    "OCR Problems",
    "Counting Problem",
    "Temporal Reasoning",
    "Spatial Reasoning",
    "Action Reasoning",
    "Object Reasoning",
    "Information Synopsis",
]

def group_dicts(A):
    grouped = defaultdict(list)

    for item in A:
        group_key = (item['video_id'], item['duration'], item['domain'], item['sub_category'])

        sub_dict = {k: item[k] for k in ['question_id', 'task_type', 'question', 'options', 'answer', 'a']}

        grouped[group_key].append(sub_dict)

    result = []
    for group_key, sub_dicts in grouped.items():
        new_entry = {
            'video_id': group_key[0],
            'duration': group_key[1],
            'domain': group_key[2],
            'sub_category': group_key[3],
            'questions': sub_dicts
        }
        result.append(new_entry)

    return result

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:",
        "Best option: ",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    s = s.replace('</evi>', "")
    s = s.split('.')[0]
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ""
    return matches[0]


def eval_your_results(
        your_results_path: str, 
        output_file: str,
        video_types: Optional[Union[List[str], str]] = None,
        skip_missing: Optional[bool] = False,
        return_categories_accuracy: Optional[bool] = True,
        return_sub_categories_accuracy: Optional[bool] = False,
        return_task_types_accuracy: Optional[bool] = False,
        gt_answer_key: Optional[str] = "answer",
        your_answer_key: Optional[str] = "a"

    ):
    """
    Evaluate your results against the ground truth

    Args:
    - your_results_path (str): Path to your results file
    - video_types (Optional[List[str], str]): List of video types to evaluate. 
    - skip_missing (Optional[bool]): If True, missing files will be skipped. If False, an error will be raised if there are missing files.
    - return_categories_accuracy (Optional[bool]): If True, the accuracy for each video category will be returned.
    - return_sub_categories_accuracy (Optional[bool]): If True, the accuracy for each video sub category will be returned.
    - return_task_types_accuracy (Optional[bool]): If True, the accuracy for each task category will be returned.
    - gt_answer_key (Optional[str]): Key to access the ground truth answer in the results file.
    - your_answer_key (Optional[str]): Key to access your answer in the results file.
    """

    # Load your results
    # with open(your_results_path, 'r') as f:
    #     your_results = json.load(f)


    if your_results_path.endswith('.json') or your_results_path.endswith('.jsonl'):
        results_path = [your_results_path]
        # dir_name = nncore.dir_name(your_results_path)
    else:
        results_path = nncore.ls(your_results_path, ext=['json', 'jsonl'], join_path=True)
        results_path = [path for path in results_path if path != 'metrics.json']
        # dir_name = your_results_path

    all_samples = []
    for path in results_path:
        nncore.log(f'Loading {path}...')
        all_samples += nncore.load(path)
    
    your_results = all_samples
    your_results = group_dicts(your_results)

    if isinstance(video_types, str):
        video_types = video_types.split(",")

    q_type_dict = {}
    v_type_dict = {}
    v_sub_type_dict = {}


    for video_type in video_types:

        # Filter your results based on video types
        your_results_video_type = [item for item in your_results if item["duration"] == video_type]

        # Task Categories
        q_type_dict[video_type] = {}
        for q_type in TASK_CATEGORIES:
            q_type_dict[video_type][q_type] = {"correct": 0, "answered": 0}

        # Video categories
        v_type_dict[video_type] = {}
        for v_type in CATEGORIES:
            v_type_dict[video_type][v_type] = {"correct": 0, "answered": 0}
        
        v_sub_type_dict[video_type] = {}
        for v_sub_type in SUB_CATEGORIES:
            v_sub_type_dict[video_type][v_sub_type] = {"correct": 0, "answered": 0}

        if not skip_missing:
            # Check if the number of files in your results and ground truth are the same
            assert len(your_results_video_type) == 300, f"Number of files in {video_type} is not 300. Check if there are missing files."

        for item in your_results_video_type:

            if skip_missing and item["missing"]:
                continue

            # Get the video category, sub category and question category
            video_category = item["domain"]
            video_sub_category = item["sub_category"]
            
            questions = item["questions"]

            for question in questions:
                q_type = question["task_type"]

                # Get the ground truth and your response
                gt_answer = question[gt_answer_key]
                response = question[your_answer_key]

                # Extract the answer from the response
                extration = extract_characters_regex(response)
    
                if extration != "":
                    q_type_dict[video_type][q_type]["answered"] += 1
                    q_type_dict[video_type][q_type]["correct"] += extration == gt_answer

                    v_type_dict[video_type][video_category]["answered"] += 1
                    v_type_dict[video_type][video_category]["correct"] += extration == gt_answer

                    v_sub_type_dict[video_type][video_sub_category]["answered"] += 1
                    v_sub_type_dict[video_type][video_sub_category]["correct"] += extration == gt_answer


    # Print the results for each video type
    with open(output_file, 'w', encoding='utf-8') as f:
        for video_type in video_types:
            f.write("=====================================\n")
            f.write(f"Evaluation on video Type: {video_type}\n")
            f.write("=====================================\n")
            
            if return_categories_accuracy:
                f.write("-------------------------------------\n")
                f.write("Video Categories\n")
                f.write("-------------------------------------\n")
                for v_type in v_type_dict[video_type]:
                    accuracy = 100 * v_type_dict[video_type][v_type]['correct'] / v_type_dict[video_type][v_type]['answered'] if v_type_dict[video_type][v_type]['answered'] > 0 else 0
                    f.write(f"{v_type}: {accuracy:.1f}%\n")

            if return_sub_categories_accuracy:
                f.write("-------------------------------------\n")
                f.write("Video Sub Categories\n")
                f.write("-------------------------------------\n")
                for v_sub_type in v_sub_type_dict[video_type]:
                    accuracy = 100 * v_sub_type_dict[video_type][v_sub_type]['correct'] / v_sub_type_dict[video_type][v_sub_type]['answered'] if v_sub_type_dict[video_type][v_sub_type]['answered'] > 0 else 0
                    f.write(f"{v_sub_type}: {accuracy:.1f}%\n")

            if return_task_types_accuracy:
                f.write("-------------------------------------\n")
                f.write("Task Categories\n")
                f.write("-------------------------------------\n")
                for q_type in q_type_dict[video_type]:
                    accuracy = 100 * q_type_dict[video_type][q_type]['correct'] / q_type_dict[video_type][q_type]['answered'] if q_type_dict[video_type][q_type]['answered'] > 0 else 0
                    f.write(f"{q_type}: {accuracy:.1f}%\n")

            total_correct = sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES])
            total_answered = sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES])
            overall_accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
            f.write("-------------------------------------\n")
            f.write("Overall Performance\n")
            f.write("-------------------------------------\n")
            f.write(f"Overall: {overall_accuracy:.1f}%\n")
            f.write("\n")

        f.write("=====================================\n")
        f.write("Evaluation on the entire dataset\n")
        f.write("=====================================\n")

        if return_categories_accuracy:
            f.write("-------------------------------------\n")
            f.write("Video Domains\n")
            f.write("-------------------------------------\n")
            for v_type in CATEGORIES:
                total_correct = sum([v_type_dict[video_type][v_type]["correct"] for video_type in video_types])
                total_answered = sum([v_type_dict[video_type][v_type]["answered"] for video_type in video_types])
                accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
                f.write(f"{v_type}: {accuracy:.1f}%\n")

        if return_sub_categories_accuracy:
            f.write("-------------------------------------\n")
            f.write("Video Sub Categories\n")
            f.write("-------------------------------------\n")
            for v_sub_type in SUB_CATEGORIES:
                total_correct = sum([v_sub_type_dict[video_type][v_sub_type]["correct"] for video_type in video_types])
                total_answered = sum([v_sub_type_dict[video_type][v_sub_type]["answered"] for video_type in video_types])
                accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
                f.write(f"{v_sub_type}: {accuracy:.1f}%\n")

        if return_task_types_accuracy:
            f.write("-------------------------------------\n")
            f.write("Task Categories\n")
            f.write("-------------------------------------\n")
            for q_type in TASK_CATEGORIES:
                total_correct = sum([q_type_dict[video_type][q_type]["correct"] for video_type in video_types])
                total_answered = sum([q_type_dict[video_type][q_type]["answered"] for video_type in video_types])
                accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
                f.write(f"{q_type}: {accuracy:.1f}%\n")

        f.write("-------------------------------------\n")
        f.write("Overall Performance\n")
        f.write("-------------------------------------\n")
        total_correct = sum([sum([q_type_dict[video_type][q_type]["correct"] for q_type in TASK_CATEGORIES]) for video_type in video_types])
        total_answered = sum([sum([q_type_dict[video_type][q_type]["answered"] for q_type in TASK_CATEGORIES]) for video_type in video_types])
        overall_accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        f.write(f"Overall: {overall_accuracy:.1f}%\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="/storage/wenzheng/model_zoo/important_checkpoint_collection_HF/202505_mcqa/with_mcqa_dpo_only_letter/videomme")  
    parser.add_argument("--output_file", type=str, default='/storage/wenzheng/model_zoo/important_checkpoint_collection_HF/202505_mcqa/with_mcqa_dpo_only_letter/videomme/mme_metric.json')    
    parser.add_argument("--video_duration_type", type=str, default=["short","medium","long"])
    parser.add_argument("--return_categories_accuracy", default=True)
    parser.add_argument("--return_sub_categories_accuracy", default=True)
    parser.add_argument("--return_task_types_accuracy", default=True)

    args = parser.parse_args()

    eval_your_results(
        args.results_file, 
        args.output_file, 
        video_types=args.video_duration_type,
        return_categories_accuracy=args.return_categories_accuracy,
        return_sub_categories_accuracy=args.return_sub_categories_accuracy,
        return_task_types_accuracy=args.return_task_types_accuracy,
    )


