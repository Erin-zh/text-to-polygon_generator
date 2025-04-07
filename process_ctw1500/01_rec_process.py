import json

CTLABELS = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4',
            '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^',
            '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', u'口']

ind_to_chr = {k: v for k, v in enumerate(CTLABELS)}
chr_to_ind = {v: k for k, v in enumerate(CTLABELS)}

def indices_to_text(indices):
    return "".join(ind_to_chr[index] if index != 96 else "" for index in indices)

def text_to_indices(text, pad=25):
    return [chr_to_ind[c] for c in text] + [96 for _ in range(pad - len(text))]

def split_and_pad_text(rec):
    text_line = indices_to_text(rec)
    words = text_line.split(' ')
    split_words = []
    
    for word in words:
        if len(word) > 0:
            split_words.append(word)
    
    padded_words = []
    for word in split_words:
        if len(word) > 25: # 如果单词长度超过25，则截断
          word = word[:25]
        padded_words.append(text_to_indices(word))
    
    return padded_words

def process_annotations(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_annotations = []
    annotation_id = 1

    for annotation in data['annotations']:
        padded_words = split_and_pad_text(annotation['rec'])
        word_start_idx = 0

        for word in padded_words:
            word_text = indices_to_text(word).rstrip(u'口')
            word_end_idx = word_start_idx + len(word_text) - 1
            new_annotation = {
                'image_id': annotation['image_id'],
                'bbox': annotation['bbox'],
                'area': annotation['area'],
                'rec': text_to_indices(word_text),
                'category_id': annotation['category_id'],
                'iscrowd': annotation['iscrowd'],
                'id': annotation_id,
                'polys': annotation['polys'],
                'iou': annotation['iou'],
                'original_id': annotation['id'],
                'rec_range': (word_start_idx, word_end_idx)
            }
            new_annotations.append(new_annotation)
            annotation_id += 1
            word_start_idx = word_end_idx + 1

    data['annotations'] = new_annotations

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

input_file = 'datasets/CTW1500/annotations/train_poly.json'
output_file = 'datasets/CTW1500/process_json/rec_split.json'
process_annotations(input_file, output_file)
