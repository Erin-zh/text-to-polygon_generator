# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import multiprocessing as mp
import time
import tqdm
import json
import glob
import re

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import torch
import numpy as np
import cv2
import editdistance

from predictor import VisualizationDemo
from adet.config import get_cfg
from adet.utils.queries import generate_query, text_to_query_t2
from adet.utils.queries import ind_to_chr, indices_to_text

import contextlib
import io
import logging
import os
from fvcore.common.timer import Timer
from fvcore.common.file_io import PathManager

from detectron2.structures import BoxMode

from detectron2.data import DatasetCatalog, MetadataCatalog

# constants
WINDOW_NAME = "COCO detections"

opj = os.path.join

def nms(bounding_boxes, confidence_score, recogs, threshold):
    if len(bounding_boxes) == 0:
        return [], []

    boxes = np.array(bounding_boxes)
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    score = np.array(confidence_score)

    picked_boxes = []
    picked_score = []
    picked_recogs = []

    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    order = np.argsort(score)

    while order.size > 0:
        index = order[-1]

        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_recogs.append(recogs[index])

        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_recogs


def bb_intersection_over_union(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def load_text_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with totaltext annotation format.
    Currently supports text detection and recognition.

    Args:
        json_file (str): full path to the json file in totaltext annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'rec': [84, 72, ... 96],
    #   'bezier_pts': [169.0, 425.0, ..., ]
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "rec", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if not isinstance(segm, dict):
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            bezierpts = anno.get("bezier_pts", None)
            # Bezier Points are the control points for BezierAlign Text recognition (BAText)
            if bezierpts:  # list[float]
                obj["beziers"] = bezierpts

            polypts = anno.get("polys", None)
            if polypts:
                obj["polygons"] = polypts

            text = anno.get("rec", None)
            if text:
                obj["text"] = text

            cor = anno.get("cor", None)
            obj["cor"] = cor

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. "
            "There might be issues in your dataset generation process.".format(
                num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts



def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def filter_preds(boxes, recs, confs):
    matched_boxes = []
    matched_scores = []
    matched_recs = []
    for pred_box, pred_rec, rec_conf in zip(boxes, recs, confs):
        if rec_conf > args.rc:
            matched_boxes.append(pred_box)
            matched_scores.append(rec_conf)
            matched_recs.append(pred_rec)

    if matched_boxes:
        matched_boxes, matched_scores, matched_recs = nms(matched_boxes, matched_scores, matched_recs, 0.5)
    matches = []
    for box, rec, rec_score in zip(matched_boxes, matched_recs, matched_scores):
        matches.append((box, rec, rec_score))

    return matches


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "-rc",
        type=float,
        default=0.0,
        help="Recognition min. confidence",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--vis",
        help="visualization",
        action='store_true'
    )

    parser.add_argument(
        "--det",
        help="eval detection",
        action="store_true"
    )

    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    image_root = 'datasets/totaltext/test_images'
    json_file = "datasets/totaltext/test_poly.json"

    TPs = 0
    FPs = 0
    FNs = 0
    global_metrics = dict()

    gts = load_text_json(json_file, image_root, dataset_name='totaltext_poly_val', extra_annotation_keys=None)
    for gt in tqdm.tqdm(gts):
        img = read_image(gt["file_name"], format="BGR")

        all_gts = []
        all_matches = []
        
        pred_boxes = []
        pred_recs = []
        rec_confs = []
        img_metrics = {"TP": 0, "FP": 0, "FN": 0}

        for anno in gt['annotations']:
            query =  text_to_query_t2(anno["text"], [1 for _ in range(len(anno["text"]))])
            query_input = [query.unsqueeze(dim=0).type(torch.FloatTensor).cuda()]

            predictions, visualized_output = demo.run_on_image(img, query_input)

            polygons = predictions["instances"].get("polygons")
            for pol_num in range(polygons.shape[0]):
                xs = polygons[pol_num, ::2]
                ys = polygons[pol_num, 1::2]
                bbox = [float(min(xs)),
                        float(min(ys)),
                        float(max(xs)),
                        float(max(ys))]
                pred_boxes.append(bbox)

            recs = predictions["instances"].get("recs")
            pred_recs.extend([list(map(lambda x: int(x), list(recs[i]))) for i in range(recs.shape[0])])

            if len(recs):
                rec_confs.extend([float(v.item()) for v in predictions["instances"].get("rec_scores").max(dim=2)[0].min(dim=1)[0]])

            matches = filter_preds(pred_boxes, pred_recs, rec_confs)
            all_matches.extend(matches)

            target_points = anno['polygons']
            xs = target_points[::2]
            ys = target_points[1::2]
            gt_bbox = [float(min(xs)),
                        float(min(ys)),
                        float(max(xs)),
                        float(max(ys))]

            for pred_box, pred_rec, rec_conf in matches:
                iou = bb_intersection_over_union(gt_bbox, pred_box)
                if iou > 0.3:
                    img_metrics['TP'] += 1
                    break


            if args.vis:
                image = cv2.imread(gt['file_name'])

                target_points = np.array(target_points, dtype=np.int32).reshape(-1,2)
                target_points = target_points.reshape((-1, 1, 2))
                cv2.polylines(image, [target_points], isClosed=True, color=(0, 255, 0), thickness=2)

                # for point in target_points:
                #     cv2.circle(image, tuple(point[0]), radius=6, color=(0, 0, 255), thickness=-1)

                for pred_contour in polygons:
                    pred_contour = pred_contour.cpu().numpy().reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(image, [pred_contour], isClosed=True, color=(0, 255, 255), thickness=2)
                    # for point in pred_contour:
                    #     cv2.circle(image, tuple(point[0]), radius=6, color=(0, 255, 255), thickness=-1)

                cv2.imwrite(f"{args.output}/{os.path.basename(gt['file_name']).split('.')[0]}_{''.join(ind_to_chr[c] if c != 96 else '' for c in anno['text'])}.jpg", image)
                # import matplotlib.pyplot as plt
                # import matplotlib.patches as pat

                # fig, ax = plt.subplots()

                # # for gt_bbox, gt_trans in all_gts:
                # rect = pat.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1],
                #                         linewidth=3, edgecolor='g', facecolor='none')
                # # plt.text(gt_bbox[2], gt_bbox[1], gt_trans, color="g")
                # ax.add_patch(rect)

                # for pred_box, pred_rec, rec_conf in matches:  # zip preds to see all the preds
                #     rect = pat.Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0],
                #                         pred_box[3] - pred_box[1], linewidth=1,
                #                         edgecolor='r', facecolor='none')
                #     plt.text(pred_box[0], pred_box[1],
                #             "{:.2f} - ".format(rec_conf) + "".join(ind_to_chr[c] if c != 96 else "" for c in pred_rec), color="r")
                #     ax.add_patch(rect)

                # ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # plt.savefig(f"{args.output}/{os.path.basename(gt['file_name']).split('.')[0]}_{''.join(ind_to_chr[c] if c != 96 else '' for c in anno['text'])}.jpg")


        img_metrics['FP'] = len(all_matches) - img_metrics['TP']
        img_metrics['FN'] = len(gt['annotations']) - img_metrics['TP']
        global_metrics[os.path.basename(gt['file_name'])] = img_metrics


        TPs += img_metrics['TP']
        FPs += img_metrics['FP']
        FNs += img_metrics['FN']

    p = TPs / (TPs + FPs)
    r = TPs / (TPs + FNs)
    print("prec:", p, "rec:", r, " Fsc:", 2 * ((p * r) / (p + r)))

