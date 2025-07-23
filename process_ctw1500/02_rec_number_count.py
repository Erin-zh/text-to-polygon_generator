import json
from collections import defaultdict

CTLABELS = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4',
            '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^',
            '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', u'口']

ind_to_chr = {k: v for k, v in enumerate(CTLABELS)}
chr_to_ind = {v: k for k, v in enumerate(CTLABELS)}

def indices_to_text(indices):
    return "".join(ind_to_chr[index] if index != 96 else "" for index in indices)

def add_rec_number_to_coco(coco_json_path, output_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    rec_count = defaultdict(lambda: defaultdict(int))
    for annotation in coco_data['annotations']:
        rec_text = indices_to_text(annotation['rec'])
        rec_count[annotation['image_id']][rec_text] += 1
    
    for annotation in coco_data['annotations']:
        rec_text = indices_to_text(annotation['rec'])
        annotation['rec_number'] = rec_count[annotation['image_id']][rec_text]

    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

coco_json_path = 'datasets/CTW1500/process_json/rec_split.json'
output_json_path = 'datasets/CTW1500/process_json/rec_split_count.json'
add_rec_number_to_coco(coco_json_path, output_json_path)
