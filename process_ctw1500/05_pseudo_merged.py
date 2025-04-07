import json
from shapely.geometry import Polygon

real_json_file = "datasets/CTW1500/annotations/train_poly.json"
pseudo_annotation_file = "datasets/CTW1500/process_json/rec_split_merged_polys_0.2.json"
process_train_file = "datasets/pseudo_annotation/process_ctw_train_poly_maxlen100_v2_0.2.json"

with open(real_json_file, 'r') as f:
    real_data = json.load(f)

with open(pseudo_annotation_file, 'r') as f:
    pseudo_annotations = json.load(f)

def calculate_bbox_and_area(polys):
    polygon = Polygon(polys)
    bbox = polygon.bounds
    area = polygon.area
    bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    return bbox, area

# 将 real_data 中的 annotations 按 id 索引
real_data_dict = {ann["id"]: ann for ann in real_data["annotations"]}

new_annotations = []

for pseudo_ann in pseudo_annotations:
    polys = pseudo_ann['polys']
    
    # 跳过 polys 为空的情况
    if not polys:
        continue
    
    original_id = pseudo_ann["original_id"]
    
    if original_id in real_data_dict:
        real_ann = real_data_dict[original_id]
        
        bbox, area = calculate_bbox_and_area(polys)
        
        new_ann = {
            "image_id": pseudo_ann["image_id"],
            "bbox": bbox,
            "area": area,
            "rec": real_ann["rec"],  # 从 real_data 中获取 rec
            "category_id": pseudo_ann.get("category_id", 1),
            "iscrowd": 0,
            "id": pseudo_ann.get("id", len(new_annotations) + 1),
            "polys": [coord for point in polys for coord in point],
            "iou": pseudo_ann.get("iou", 0)
        }
        
        new_annotations.append(new_ann)

# 更新 real_data 的 annotations
real_data['annotations'] = new_annotations

with open(process_train_file, 'w') as f:
    json.dump(real_data, f, indent=4)

print(f"Processed annotations have been saved to {process_train_file}")
