import json
from shapely.geometry import Polygon

real_json_file = "datasets/totaltext/train_poly.json"
pseudo_annotation_file = "output/r_50_poly/debug/inference/text_results.json"
process_train_file= "datasets/pseudo_annotation/process_totaltext_train_poly.json"

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


new_annotations = []

for pseudo_ann in pseudo_annotations:
    polys = pseudo_ann['polys']
    bbox, area = calculate_bbox_and_area(polys)
    
    new_ann = {
        "image_id": pseudo_ann["image_id"],
        "bbox": bbox,
        "area": area,
        "rec": pseudo_ann["rec"],
        "category_id": pseudo_ann["category_id"],
        "iscrowd": 0,
        "id": pseudo_ann.get("id", len(new_annotations) + 1), 
        "polys": [coord for point in polys for coord in point], 
        "iou": pseudo_ann.get("iou", 0)  
    }
    new_annotations.append(new_ann)


real_data['annotations'] = new_annotations

with open(process_train_file, 'w') as f:
    json.dump(real_data, f, indent=4)
