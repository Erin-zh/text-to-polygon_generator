import json
from shapely.geometry import Polygon
from tqdm import tqdm

def calculate_iou(poly1, poly2):
    polygon1 = Polygon(poly1)
    polygon2 = Polygon(poly2)
    if not polygon1.is_valid or not polygon2.is_valid:
        return 0
    intersection = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersection / union
    return iou

def remove_duplicates_within_image(text_results, iou_threshold=0.7):
    results_by_image = {}
    
    for result in text_results:
        image_id = result['image_id']
        if image_id not in results_by_image:
            results_by_image[image_id] = []
        results_by_image[image_id].append(result)
    
    unique_results = []
    
    for image_id, results in tqdm(results_by_image.items(), desc="Processing images"):
        non_duplicate_results = []
        
        for result in results:
            is_duplicate = False
            for unique_result in non_duplicate_results:
                if unique_result['category_id'] == result['category_id']:
                    iou = calculate_iou(result['polys'], unique_result['polys'])
                    if iou > iou_threshold:
                        is_duplicate = True
                        if result['score'] > unique_result['score']:
                            unique_result.update(result)
                        break
            
            if not is_duplicate:
                non_duplicate_results.append(result)
        
        unique_results.extend(non_duplicate_results)
    
    return unique_results

with open('output/r_50_poly/debug/inference/text_results.json', 'r') as file:
    text_results = json.load(file)

unique_text_results = remove_duplicates_within_image(text_results)

with open('output/r_50_poly/debug/inference/unique_text_results_0.2.json', 'w') as file:
    json.dump(unique_text_results, file)
