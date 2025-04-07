import json
import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
import tqdm
from shapely.geometry import Polygon

def calculate_center(poly):
    if not poly:
        return (0, 0)
    x_coords = [p[0] for p in poly]
    y_coords = [p[1] for p in poly]
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    return (center_x, center_y)

def calculate_height(poly):
    if not poly:
        return 0
    y_coords = [p[1] for p in poly]
    return max(y_coords) - min(y_coords)

def is_smooth_curve(points, threshold=0.3):
    if len(points) < 2:
        return True
    distances = [np.linalg.norm(np.array(points[i+1]) - np.array(points[i])) for i in range(len(points)-1)]
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    return (max_distance - mean_distance) / mean_distance < threshold if mean_distance != 0 else False

def sample_points(points, num_samples):
    if len(points) <= num_samples:
        return points
    distances = [np.linalg.norm(np.array(points[i]) - np.array(points[i+1])) for i in range(len(points)-1)]
    total_distance = sum(distances)
    distance_per_sample = total_distance / (num_samples - 1)
    sampled_points = [points[0]]
    accumulated_distance = 0
    for i in range(len(points) - 1):
        segment_distance = distances[i]
        while accumulated_distance + segment_distance >= distance_per_sample:
            t = (distance_per_sample - accumulated_distance) / segment_distance
            new_point = [
                points[i][0] + t * (points[i+1][0] - points[i][0]),
                points[i][1] + t * (points[i+1][1] - points[i][1])
            ]
            sampled_points.append(new_point)
            segment_distance -= (distance_per_sample - accumulated_distance)
            accumulated_distance = 0
        accumulated_distance += segment_distance
    if len(sampled_points) < num_samples:
        sampled_points.append(points[-1])
    return sampled_points

def merge_polys(polys_list):
    polys_list = [poly for poly in polys_list if poly]
    if not polys_list:
        return []
    if len(polys_list) == 1 and len(polys_list[0]) == 16:
        return polys_list[0]
    top_boundary = []
    bottom_boundary = []
    for poly in polys_list:
        if len(poly) == 16:
            top_boundary.extend(poly[:8])
            bottom_boundary.extend(poly[8:][::-1])
        else:
            top_boundary.extend(poly[:8])
            bottom_boundary.extend(poly[8:][::-1])

    num_samples_half = 8
    top_sampled_points = sample_points(top_boundary, num_samples_half)
    bottom_sampled_points = sample_points(bottom_boundary, num_samples_half)

    final_polys = top_sampled_points + bottom_sampled_points[::-1]
    if len(final_polys) != 16 :
        print("error")
    return final_polys

def filter_combinations_by_y_coord(polys, relative_threshold=0.6):
    all_combinations = list(product(*polys))
    if len(all_combinations) > 20:
        all_combinations = all_combinations[:20]
    filtered_combinations = []
    
    for combination in all_combinations:
        comb_centers = [calculate_center(poly) for poly in combination]
        y_coords = [center[1] for center in comb_centers]
        mean_y = np.mean(y_coords)
        heights = [calculate_height(poly) for poly in combination]
        max_height = max(heights) if heights else 1
        dynamic_threshold = relative_threshold * max_height

        combination = [poly for poly, y in zip(combination, y_coords) if abs(y - mean_y) < dynamic_threshold]
        
        if combination:
            filtered_combinations.append(combination)
    
    return filtered_combinations

def generate_combinations(polys, relative_threshold=0.9):
    filtered_combinations = filter_combinations_by_y_coord(polys, relative_threshold)
    
    best_combination = None
    best_score = float('inf')
    
    for combination in filtered_combinations:
        centers = [calculate_center(poly) for poly in combination]
        if len(centers) < 2:
            continue
        """ if is_smooth_curve(centers):
            score = np.sum([np.linalg.norm(np.array(centers[i+1]) - np.array(centers[i])) for i in range(len(centers)-1)])
            if score < best_score:
                best_score = score
                best_combination = combination 

        if best_combination is None:
            for combination in filtered_combinations:"""
        # centers = [calculate_center(poly) for poly in combination]
        score = np.sum([np.linalg.norm(np.array(centers[i+1]) - np.array(centers[i])) for i in range(len(centers)-1)])
        if score < best_score:
            best_score = score
            best_combination = combination

    return best_combination if best_combination else filtered_combinations[0] if filtered_combinations else []

def is_self_intersecting_polygon(points):
    """使用 shapely 检查多边形是否自交"""
    if len(points) < 3:
        return True  # 小于三个点无法形成有效多边形
    polygon = Polygon(points)
    return polygon.is_simple  # 如果 is_simple 为 False 则表示多边形自交

def process_annotation(original_id, polys_list, annotations):
    polys_list = [p for p in polys_list if p]
    if len(polys_list) > 10:
        polys_list = [p[:20] for p in polys_list if len(p)>10]
    if len(polys_list) == 0:
        return None

    if len(polys_list) == 1:
        polys_list = [poly for sublist in polys_list for poly in sublist]
        merged_poly = polys_list[0]
    else:
        best_combination = generate_combinations(polys_list)
        merged_poly = merge_polys(best_combination)
    
    if not is_self_intersecting_polygon(merged_poly):
        return None
    first_annotation = next(ann for ann in annotations["annotations"] if ann['original_id'] == original_id)
    return {
        'image_id': first_annotation['image_id'],
        'polys': merged_poly,
        'original_id': original_id,
    }

def process_annotations(annotations, batch_size=100):
    grouped_polys = {}
    for annotation in annotations["annotations"]:
        original_id = annotation['original_id']
        if original_id not in grouped_polys:
            grouped_polys[original_id] = []
        grouped_polys[original_id].append(annotation['polys'])

    merged_annotations = []
    keys = list(grouped_polys.keys())
    total = len(keys)
    
    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = min(start + batch_size, total)
        batch_keys = keys[start:end]
        for original_id in batch_keys:
            result = process_annotation(original_id, grouped_polys[original_id], annotations)
            if result:
                merged_annotations.append(result)
    
    return merged_annotations

with open('datasets/CTW1500/process_json/rec_split_polys_0.2.json', 'r') as f:
    annotations = json.load(f)

merged_annotations = process_annotations(annotations)

output_file = 'datasets/CTW1500/process_json/rec_split_merged_polys_0.2.json'
with open(output_file, 'w') as f:
    json.dump(merged_annotations, f, indent=2)

print(f"Merged annotations have been saved to {output_file}")
