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

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def filter_unique_polys(polys_list):
    unique_polys = []
    remaining_polys = []
    
    for idx, polys in enumerate(polys_list):
        if len(polys) == 1:
            unique_polys.append((polys[0], idx))
        else:
            remaining_polys.append((polys, idx))
    
    return unique_polys, remaining_polys

def select_closest_polys(unique_polys, remaining_polys):
    selected_polys = []
    
    for polys, polys_idx in remaining_polys:
        best_poly = None
        best_distance = float('inf')
        
        for poly in polys:
            center = calculate_center(poly)
            distances = [calculate_distance(center, calculate_center(up[0])) for up in unique_polys]
            avg_distance = np.mean(distances)
            
            if avg_distance < best_distance:
                best_distance = avg_distance
                best_poly = poly
        
        if best_poly is not None:
            selected_polys.append((best_poly, polys_idx))
    
    return selected_polys

def merge_polys(all_polys):
    all_polys.sort(key=lambda x: x[1])
    
    polys_list = [poly[0] for poly in all_polys if poly[0]]
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
    return final_polys

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

def is_self_intersecting_polygon(points):
    if len(points) < 3:
        return True
    polygon = Polygon(points)
    return not polygon.is_simple

def process_annotation(args):
    original_id, polys_list, annotations = args
    polys_list = [p for p in polys_list if p]
    
    if len(polys_list) == 0:
        return None
    if len(polys_list) == 1:
        merged_poly = polys_list[0]
    else:
        unique_polys, remaining_polys = filter_unique_polys(polys_list)
        if len(remaining_polys) == 0:
            merged_poly = merge_polys(unique_polys)
        else:
            selected_polys = select_closest_polys(unique_polys, remaining_polys)
            merged_poly = merge_polys(unique_polys + selected_polys)
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
        batch_results = []

        with Pool(cpu_count() // 2) as pool:
            for original_id in batch_keys:
                polys_list = grouped_polys[original_id]
                result = pool.apply_async(process_annotation, args=((original_id, polys_list, annotations),))
                batch_results.append(result)
            
            for result in batch_results:
                try:
                    merged_annotation = result.get()
                    if merged_annotation:
                        merged_annotations.append(merged_annotation)
                except Exception as e:
                    print(f"Error processing annotation: {e}")
    
    return merged_annotations

with open('datasets/CTW1500/process_json/rec_split_polys.json', 'r') as f:
    annotations = json.load(f)

merged_annotations = process_annotations(annotations)

output_file = 'datasets/CTW1500/process_json/rec_split_merged_polys.json'
with open(output_file, 'w') as f:
    json.dump(merged_annotations, f, indent=2)

print(f"Merged annotations have been saved to {output_file}")
