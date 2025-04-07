import json
from collections import defaultdict
import numpy as np
import cv2

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clear_polys(rec_split):
    for annotation in rec_split['annotations']:
        annotation['polys'] = []

def replace_polys(rec_split, text_results):
    text_results_dict = defaultdict(list)
    
    # 将 text_results 中的 polys 存储为字典，每个 key 对应一个 rec，value 是一个包含所有 polys 的列表
    for ann in text_results:
        key = (ann['image_id'], tuple(ann['rec']))
        text_results_dict[key].append(ann['polys'])
    
    for annotation in rec_split['annotations']:
        key = (annotation['image_id'], tuple(annotation['rec']))
        if key in text_results_dict:
            # 将所有找到的 polys 列表保留为列表中的列表
            annotation['polys'] = text_results_dict[key]
            # annotation['polys'] = [poly for sublist in polys_lists for poly in sublist]
    
    return rec_split['annotations']

def process_and_merge_annotations(rec_split_file, text_results_file, output_file):
    rec_split = load_json(rec_split_file)
    text_results = load_json(text_results_file)
    
    clear_polys(rec_split)
    
    replaced_annotations = replace_polys(rec_split, text_results)
    rec_split['annotations'] = replaced_annotations
    
    save_json(rec_split, output_file)

rec_split_file = 'datasets/CTW1500/process_json/rec_split.json'
text_results_file = 'datasets/CTW1500/process_json/unique_text_results_0.2.json'
output_file = 'datasets/CTW1500/process_json/rec_split_polys_0.2.json'

process_and_merge_annotations(rec_split_file, text_results_file, output_file)
