import argparse
import json
import os
from PIL import Image
import numpy as np
import cv2
import tqdm
from detectron2.data.detection_utils import read_image

def overlay_predictions(img, polygons_list=None, radius=3):
    overlayed = img.copy()
    if polygons_list is not None:
        for polygons in polygons_list:
            overlayed = plot_polygons(overlayed, polygons, radius=radius)
    return overlayed

def plot_polygons(img, polygons, radius=3):
    polylines = np.array(polygons, dtype=np.int32)
    polylines = polylines.reshape((-1, 1, 2))
    cv2.polylines(img, [polylines], isClosed=True,  color=(0,255,0), thickness=1)
    return img


def display_poly(img_path, pred_path, output_path):

    image_dir = img_path
    annot_file = pred_path

    save_dir = output_path
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # Load COCO-style annotations
    with open(annot_file, 'r') as file:
        coco_annotations = json.load(file)
    
    # Group polygons by image_id
    polygons_by_image = {}
    scores_by_image = {}
    for annotation in coco_annotations:
        image_id = annotation['image_id']
        if image_id not in polygons_by_image:
            polygons_by_image[image_id] = []
        polygons_by_image[image_id].append(annotation['polys'])               

    # Iterate over image_id
    for image_id, polygons_list in tqdm.tqdm(polygons_by_image.items()):
        if "totaltext" in img_path: 
            fn = f"{image_id:07d}.jpg"
        elif "icdar2015" in img_path:
            fn = f"img_{image_id+1}.jpg"
        image_file = os.path.join(image_dir, fn)
        img = read_image(image_file, format="BGR")
        img = img[:, :, ::-1]
        # Display polygons on image using overlay_predictions
        img_with_polys = overlay_predictions(np.array(img), polygons_list)

        # Save image with polys
        vis_file = os.path.join(vis_dir, f"{image_id}_vis.jpg")
        Image.fromarray(img_with_polys.astype(np.uint8)).save(vis_file)

def display_gt_poly(img_path, gt_path, output_path):

    image_dir = img_path
    annot_file = gt_path

    save_dir = output_path
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, 'gt_vis')
    os.makedirs(vis_dir, exist_ok=True)

    # Load COCO-style annotations
    with open(annot_file, 'r') as file:
        coco_annotations = json.load(file)

    # Group polygons by image_id
    polygons_by_image = {}
    for annotation in coco_annotations["annotations"]:
        image_id = annotation['image_id']
        if image_id not in polygons_by_image:
            polygons_by_image[image_id] = []
        polygons_by_image[image_id].append(annotation['polys'])       

    # Iterate over image_id
    for image_id, polygons_list in tqdm.tqdm(polygons_by_image.items()):
        # Load image
        if "totaltext" in img_path: 
            fn = f"{image_id:07d}.jpg"
        elif "icdar2015" in img_path:
            fn = f"img_{image_id+1}.jpg"
        elif "syntext2" in img_path:
            fn = f"{image_id:07d}.jpg"
        elif "CTW1500" in img_path:
            fn = f"{image_id:04d}.jpg"
        img = Image.open(os.path.join(image_dir, fn)).convert("RGB")
        # Display polygons on image using overlay_predictions
        img_with_polys = overlay_predictions(np.array(img), polygons_list)

        # Save image with polys
        vis_file = os.path.join(vis_dir, f"{image_id}_vis.jpg")
        Image.fromarray(img_with_polys.astype(np.uint8)).save(vis_file)


if __name__ == "__main__":
    gt_path = 'datasets/pseudo_annotation/process_ctw_train_poly_maxlen100_v2_0.2.json' # 'datasets/totaltext/train_poly.json'
    pred_path = 'datasets/pseudo_annotation/process_totaltext_train_poly_v2.json'
    img_path = 'datasets/CTW1500/ctwtrain_text_image' # 'datasets/totaltext/train_images'
    output_path = 'output/r_50_poly/train_eval/ctw1500'
    gt_output = 'output/r_50_poly/train_eval/syntext2' #'output/r_50_poly/train_eval/totaltext'
    # display_poly(img_path, pred_path, output_path)
    display_gt_poly(img_path, gt_path, output_path)