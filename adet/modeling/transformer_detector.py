from typing import List
import numpy as np
import torch
from torch import nn

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import build_backbone
from detectron2.structures import ImageList, Instances

from adet.layers.label_encoder import max_query_types
from adet.layers.pos_encoding import PositionalEncoding2D
from adet.modeling.dptext_detr.losses import SetCriterion
from adet.modeling.dptext_detr.matcher import build_matcher
from adet.modeling.dptext_detr.models import DPText_DETR
from adet.utils.misc import NestedTensor, box_xyxy_to_cxcywh
from shapely.geometry import Polygon


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


def detector_postprocess(results, output_height, output_width):
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])

    if results.has("beziers"):
        beziers = results.beziers
        # scale and clip in place
        h, w = results.image_size
        beziers[:, 0].clamp_(min=0, max=w)
        beziers[:, 1].clamp_(min=0, max=h)
        beziers[:, 6].clamp_(min=0, max=w)
        beziers[:, 7].clamp_(min=0, max=h)
        beziers[:, 8].clamp_(min=0, max=w)
        beziers[:, 9].clamp_(min=0, max=h)
        beziers[:, 14].clamp_(min=0, max=w)
        beziers[:, 15].clamp_(min=0, max=h)
        beziers[:, 0::2] *= scale_x
        beziers[:, 1::2] *= scale_y

    # scale point coordinates
    if results.has("polygons"):
        polygons = results.polygons
        polygons[:, 0::2] *= scale_x
        polygons[:, 1::2] *= scale_y

    return results


@META_ARCH_REGISTRY.register()
class TransformerPureDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        d2_backbone = MaskedBackbone(cfg)
        N_steps = cfg.MODEL.TRANSFORMER.HIDDEN_DIM // 2
        self.test_score_threshold = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
        self.use_polygon = cfg.MODEL.TRANSFORMER.USE_POLYGON
        self.num_ctrl_points = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        assert self.use_polygon and self.num_ctrl_points == 16  # only the polygon version is released now
        backbone = Joiner(d2_backbone, PositionalEncoding2D(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels
        self.dptext_detr = DPText_DETR(cfg, backbone)

        box_matcher, point_matcher = build_matcher(cfg)

        loss_cfg = cfg.MODEL.TRANSFORMER.LOSS
        weight_dict = {
                    'loss_ce': loss_cfg.POINT_CLASS_WEIGHT,  
                    'loss_ctrl_points': loss_cfg.POINT_COORD_WEIGHT,
                    }
        enc_weight_dict = {
            'loss_bbox': loss_cfg.BOX_COORD_WEIGHT,
            'loss_giou': loss_cfg.BOX_GIOU_WEIGHT,
            'loss_ce': loss_cfg.BOX_CLASS_WEIGHT
        }
        if loss_cfg.AUX_LOSS:
            aux_weight_dict = {}
            # decoder aux loss
            for i in range(cfg.MODEL.TRANSFORMER.DEC_LAYERS - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v for k, v in weight_dict.items()})
            # encoder aux loss
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in enc_weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        enc_losses = ['labels', 'boxes']
        dec_losses = ['labels', 'ctrl_points']

        self.criterion = SetCriterion(
            self.dptext_detr.num_classes,
            self.dptext_detr.points_num_classes,
            box_matcher,
            point_matcher,
            weight_dict,
            enc_losses,
            dec_losses,
            self.dptext_detr.num_ctrl_points,
            cfg.MODEL.TRANSFORMER.VOC_SIZE,
            focal_alpha=loss_cfg.FOCAL_ALPHA,
            focal_gamma=loss_cfg.FOCAL_GAMMA,
        )

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images
    def preprocess_texts(self, batched_inputs):
        # x["instances"][0].queries
        #texts = [x["instances"][0].text.to(self.device) if len(x["instances"])
        #            else torch.zeros(1, self.step.max_text_len, max_query_types()).to(self.device) for x in batched_inputs]
        """ texts = []
        for x in batched_inputs:
            if "instances" in x and len(x["instances"]) > 0:
                num_instances = len(x["instances"])
                text_list = []
                for i in range(num_instances):
                    text_list.append(x["instances"][i].text.to(self.device))
                if len(text_list) > 70:
                    text_list = text_list[:70]
                texts.append(torch.stack(text_list))
            else:
                texts.append(torch.zeros(1, 1, self.dptext_detr.max_text_len).to(self.device))
        return texts """
        texts = []
        for x in batched_inputs:
            if "current_instance" in x and len(x["current_instance"]) > 0:
                num_instances = len(x["current_instance"])
                text_list = []
                for i in range(num_instances):
                    text_list.append(x["current_instance"][i].text.to(self.device))
                if len(text_list) > 70:
                    text_list = text_list[:70]
                texts.append(torch.stack(text_list))
            else:
                texts.append(torch.zeros(1, 1, self.dptext_detr.max_text_len).to(self.device))
        return texts 

    def preprocess_texts_number(self, batched_inputs):
        texts_number = []
        for x in batched_inputs:
            if "current_instance" in x and len(x["current_instance"]) > 0:
                num_instances = len(x["current_instance"])
                text_number_list = []
                for i in range(num_instances):
                    text_number_list.append(x["current_instance"][i].text_number.to(self.device))
                texts_number.append(torch.stack(text_number_list))
        return texts_number
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "scores", "pred_classes", "polygons"
        """
        images = self.preprocess_image(batched_inputs)
        texts = self.preprocess_texts(batched_inputs)
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_current_instance = [x['current_instance'].to(self.device) for x in batched_inputs]
            targets, current_target = self.prepare_targets(gt_instances, gt_current_instance)
            output = self.dptext_detr((images, texts))
            # compute the loss
            loss_dict = self.criterion(output, targets, current_target)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            texts_number = self.preprocess_texts_number(batched_inputs)
            output = self.dptext_detr((images, texts))
            ctrl_point_cls = output["pred_logits"]
            ctrl_point_coord = output["pred_ctrl_points"]
            # ctrl_point_text = output["pred_text_logits"]
            results = self.inference(ctrl_point_cls, ctrl_point_coord, images.image_sizes, texts_number)  # ctrl_point_text,
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        
    def forward_query(self, batched_inputs, texts, texts_number):
        """
            TODO: rewrite
        """
        images = self.preprocess_image(batched_inputs)
        
        texts_input=[]
        processed_text=[]
        numbers_input=[]
        processed_number=[]
        for t in texts:
            processed_text.append(t.to(self.device))
        texts_input.append(torch.stack(processed_text))
        for n in texts_number:
            processed_number.append(n.to(self.device))
        numbers_input.append(torch.stack(processed_number))       
        output = self.dptext_detr((images, texts_input))
        ctrl_point_cls = output["pred_logits"]
        ctrl_point_coord = output["pred_ctrl_points"]
        results = self.inference(ctrl_point_cls, ctrl_point_coord, images.image_sizes, numbers_input)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def prepare_targets(self, targets, current_target):
        new_targets = []
        new_current_target = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            raw_ctrl_points = targets_per_image.polygons if self.use_polygon else targets_per_image.beziers
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.dptext_detr.num_ctrl_points, 2) / \
                             torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            gt_ctrl_points = torch.clamp(gt_ctrl_points[:,:,:2], 0, 1)
            new_targets.append(
                {"labels": gt_classes, "boxes": gt_boxes, "ctrl_points": gt_ctrl_points}
            )
        for targets_per_image in current_target:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            gt_texts = targets_per_image.text
            raw_ctrl_points = targets_per_image.polygons if self.use_polygon else targets_per_image.beziers
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.dptext_detr.num_ctrl_points, 2) / \
                             torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            gt_ctrl_points = torch.clamp(gt_ctrl_points[:,:,:2], 0, 1)
            new_current_target.append(
                {"labels": gt_classes, "boxes": gt_boxes, "texts": gt_texts, "ctrl_points": gt_ctrl_points}
            )
        return new_targets, new_current_target

    """     
    def inference(self, ctrl_point_cls, ctrl_point_coord, image_sizes): # ctrl_point_text, 
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []
        
        # cls shape: (b, nq, n_pts, voc_size)
        # ctrl_point_text = torch.softmax(ctrl_point_text, dim=-1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        for scores_per_image, labels_per_image, ctrl_point_per_image, image_size in zip(
                scores, labels, ctrl_point_coord, image_sizes          # ctrl_point_text, ctrl_point_text_per_image,
        ):
            selector = scores_per_image >= self.test_score_threshold
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]
            # ctrl_point_text_per_image = ctrl_point_text_per_image[selector]

            result = Instances(image_size)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            # result.rec_scores = ctrl_point_text_per_image
            ctrl_point_per_image[..., 0] *= image_size[1]
            ctrl_point_per_image[..., 1] *= image_size[0]
            if self.use_polygon:
                result.polygons = ctrl_point_per_image.flatten(1)
            else:
                result.beziers = ctrl_point_per_image.flatten(1)
            #_, text_pred = ctrl_point_text_per_image.topk(1)
            #result.recs = text_pred.squeeze(-1)    
            results.append(result)

        return results """
    def inference(self, ctrl_point_cls, ctrl_point_coord, image_sizes,  texts_number):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []
        def intersection_over_union(poly1, poly2):
            """
            计算两个多边形的 IOU (Intersection Over Union).
            poly1 和 poly2 都是形状为 (n, 2) 的点列表。
            """
            polygon1 = Polygon(poly1)
            polygon2 = Polygon(poly2)
            
            if not polygon1.is_valid or not polygon2.is_valid:
                return 0.0
            
            inter_area = polygon1.intersection(polygon2).area
            union_area = polygon1.union(polygon2).area
            return inter_area / union_area
        # cls shape: (b, nq, n_pts, voc_size)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)

        for scores_per_image, labels_per_image, ctrl_point_per_image, image_size, text_number in zip(
                scores, labels, ctrl_point_coord, image_sizes, texts_number):
            
            selector = scores_per_image >= self.test_score_threshold
            scores_per_image = scores_per_image[selector]
            labels_per_image = labels_per_image[selector]
            ctrl_point_per_image = ctrl_point_per_image[selector]

            sorted_indices = scores_per_image.argsort(descending=True)
            scores_per_image = scores_per_image[sorted_indices]
            labels_per_image = labels_per_image[sorted_indices]
            ctrl_point_per_image = ctrl_point_per_image[sorted_indices]

            result = Instances(image_size)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            ctrl_point_per_image[..., 0] *= image_size[1]
            ctrl_point_per_image[..., 1] *= image_size[0]
            if self.use_polygon:
                result.polygons = ctrl_point_per_image.view(-1, ctrl_point_per_image.shape[-2], 2)
            else:
                result.beziers = ctrl_point_per_image.view(-1, ctrl_point_per_image.shape[-2], 2)
            
            results.append(result)
        
        final_results = []
        for result in results:
            selected_indices = []
            polygons = result.polygons.tolist()  # Assuming polygons are provided

            i = 0
            while len(selected_indices) < text_number and i < len(result.scores):
                poly_i = polygons[i]

                keep = True
                for idx in selected_indices:
                    poly_j = polygons[idx]
                    iou = intersection_over_union(poly_i, poly_j)
                    if iou >= 0.7:
                        keep = False
                        break

                if keep:
                    selected_indices.append(i)
                
                i += 1

            while len(selected_indices) < text_number and i < len(result.scores):
                poly_i = polygons[i]

                max_iou = 0
                for idx in selected_indices:
                    poly_j = polygons[idx]
                    iou = intersection_over_union(poly_i, poly_j)
                    max_iou = max(max_iou, iou)

                if max_iou < 0.7:
                    selected_indices.append(i)

                i += 1

            final_result = Instances(result.image_size)
            final_result.scores = result.scores[selected_indices]
            final_result.pred_classes = result.pred_classes[selected_indices]
            if self.use_polygon:
                final_result.polygons = result.polygons[selected_indices]
            else:
                final_result.beziers = result.beziers[selected_indices]
            
            final_results.append(final_result)
        
        return final_results
