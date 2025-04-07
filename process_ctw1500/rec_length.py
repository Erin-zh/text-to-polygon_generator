import json

with open('datasets/CTW1500/process_json/rec_split.json', 'r') as f:
    data = json.load(f)

length_distribution = {
    '<=25': 0,
    '>25': 0
}

for annotation in data['annotations']:
    rec = annotation.get('rec', [])
    first_96_index = rec.index(96) if 96 in rec else len(rec)
    char_length = first_96_index
    
    if char_length <= 25:
        length_distribution['<=25'] += 1
    else:
        length_distribution['>25'] += 1

print(length_distribution)
