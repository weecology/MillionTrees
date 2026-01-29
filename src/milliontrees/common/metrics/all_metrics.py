import copy
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_iou
from torchvision.ops import masks_to_boxes
from torchvision.models.detection._utils import Matcher
from milliontrees.common.metrics.metric import Metric, ElementwiseMetric, MultiTaskMetric
from milliontrees.common.metrics.loss import ElementwiseLoss
from milliontrees.common.utils import minimum, maximum, get_counts
import sklearn.metrics
from scipy.stats import pearsonr


def binary_logits_to_score(logits):
    assert logits.dim() in (1, 2)
    if logits.dim() == 2:  #multi-class logits
        assert logits.size(1) == 2, "Only binary classification"
        score = F.softmax(logits, dim=1)[:, 1]
    else:
        score = logits
    return score


def multiclass_logits_to_pred(logits):
    """Converts multi-class logits into predictions.

    This function takes a tensor of logits with shape (batch_size, ..., n_classes)
    and computes predictions by applying `argmax` along the last dimension.

    Args:
        logits (Tensor): A tensor of shape (batch_size, ..., n_classes) representing
                         multi-class logits.

    Returns:
        Tensor: A tensor containing predicted class indices.
    """
    assert logits.dim() > 1
    return logits.argmax(-1)


def binary_logits_to_pred(logits):
    return (logits > 0).long()


def pseudolabel_binary_logits(logits, confidence_threshold):
    """Applies a confidence threshold to binary logits and generates pseudo-
    labels.

    Args:
        logits (Tensor): A tensor of shape (batch_size, n_tasks) representing binary logits.
                         A positive value (>0) indicates a positive prediction for the corresponding (example, task).
        confidence_threshold (float): A threshold in the range [0,1] used to filter predictions.

    Returns:
        tuple:
            - unlabeled_y_pred (Tensor): A filtered version of `logits`, discarding rows (examples)
                                         where no predictions exceed the confidence threshold.
            - unlabeled_y_pseudo (Tensor): A hard pseudo-labeled version of `logits`, where entries
                                           below the confidence threshold are set to NaN. Rows with no
                                           confident predictions are discarded.
            - pseudolabels_kept_frac (float): The fraction of (example, task) pairs that are not set
                                              to NaN or discarded.
            - mask (Tensor): A mask indicating which predictions meet the confidence threshold.
    """

    if len(logits.shape) != 2:
        raise ValueError('Logits must be 2-dimensional.')
    probs = 1 / (1 + torch.exp(-logits))
    mask = (torch.max(probs, 1 - probs) >= confidence_threshold)
    unlabeled_y_pseudo = (logits > 0).float()
    unlabeled_y_pseudo[~mask] = float('nan')
    # mask is bool, so no .mean()
    pseudolabels_kept_frac = mask.sum() / mask.numel()
    example_mask = torch.any(~torch.isnan(unlabeled_y_pseudo), dim=1)
    unlabeled_y_pseudo = unlabeled_y_pseudo[example_mask]
    unlabeled_y_pred = logits[example_mask]
    return (unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac,
            example_mask)


def pseudolabel_multiclass_logits(logits, confidence_threshold):
    """Applies a confidence threshold to multi-class logits and generates
    pseudo-labels.

    Args:
        logits (Tensor): A tensor of shape (batch_size, ..., n_classes) representing multi-class logits.
        confidence_threshold (float): A threshold in the range [0,1] used to filter predictions.

    Returns:
        tuple:
            - unlabeled_y_pred (Tensor): A filtered version of `logits`, discarding rows (examples)
                                         where no predictions exceed the confidence threshold.
            - unlabeled_y_pseudo (Tensor): A hard pseudo-labeled version of `logits`, where examples
                                           with confidence below the threshold are discarded.
            - pseudolabels_kept_frac (float): The fraction of examples retained after filtering.
            - mask (Tensor): A mask indicating which predictions meet the confidence threshold.
    """

    mask = torch.max(F.softmax(logits, -1), -1)[0] >= confidence_threshold
    unlabeled_y_pseudo = multiclass_logits_to_pred(logits)
    unlabeled_y_pseudo = unlabeled_y_pseudo[mask]
    unlabeled_y_pred = logits[mask]
    # mask is bool, so no .mean()
    pseudolabels_kept_frac = mask.sum() / mask.numel()
    return unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, mask


def pseudolabel_identity(logits, confidence_threshold):
    return logits, logits, 1, None


def pseudolabel_detection(preds, confidence_threshold):
    """Filters detection predictions based on a confidence threshold.

    Args:
        preds (List[dict]): A list of length `batch_size`, where each entry is a dictionary
                            containing the following keys:
                            - 'boxes' (Tensor): Bounding box coordinates.
                            - 'labels' (Tensor): Class labels for detected objects.
                            - 'scores' (Tensor): Confidence scores for each detection.
                            - 'losses' (dict): An empty dictionary (not used).
        confidence_threshold (float): A threshold in the range [0,1] used to filter predictions.

    Returns:
        List[dict]: A filtered version of `preds`, where detections with confidence scores
                    below `confidence_threshold` are removed.
    """

    preds, pseudolabels_kept_frac = _mask_pseudolabels_detection(
        preds, confidence_threshold)
    unlabeled_y_pred = [{
        'boxes': pred['boxes'],
        'labels': pred['labels'],
        'scores': pred['scores'],
        'losses': pred['losses'],
    } for pred in preds]
    unlabeled_y_pseudo = [{
        'boxes': pred['boxes'],
        'labels': pred['labels'],
    } for pred in preds]

    # Keep all examples even if they don't have any confident-enough predictions
    # They will be treated as empty images
    example_mask = torch.ones(len(preds), dtype=torch.bool)
    return (unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac,
            example_mask)


def pseudolabel_detection_discard_empty(preds, confidence_threshold):
    """Filters detection predictions based on a confidence threshold and
    discards empty entries.

    Args:
        preds (List[dict]): A list of length `batch_size`, where each entry is a dictionary
                            containing the following keys:
                            - 'boxes' (Tensor): Bounding box coordinates.
                            - 'labels' (Tensor): Class labels for detected objects.
                            - 'scores' (Tensor): Confidence scores for each detection.
                            - 'losses' (dict): An empty dictionary (not used).
        confidence_threshold (float): A threshold in the range [0,1] used to filter predictions.

    Returns:
        List[dict]: A filtered version of `preds`, where detections with confidence scores
                    below `confidence_threshold` are removed. Entries with no remaining detections
                    are discarded from the list.
    """

    preds, pseudolabels_kept_frac = _mask_pseudolabels_detection(
        preds, confidence_threshold)
    unlabeled_y_pred = [{
        'boxes': pred['boxes'],
        'labels': pred['labels'],
        'scores': pred['scores'],
        'losses': pred['losses'],
    } for pred in preds if len(pred['labels']) > 0]
    unlabeled_y_pseudo = [{
        'boxes': pred['boxes'],
        'labels': pred['labels'],
    } for pred in preds if len(pred['labels']) > 0]
    example_mask = torch.tensor([len(pred['labels']) > 0 for pred in preds])
    return unlabeled_y_pred, unlabeled_y_pseudo, pseudolabels_kept_frac, example_mask


def _mask_pseudolabels_detection(preds, confidence_threshold):
    total_boxes = 0.0
    kept_boxes = 0.0
    preds = copy.deepcopy(preds)
    for pred in preds:
        mask = (pred['scores'] >= confidence_threshold)
        pred['boxes'] = pred['boxes'][mask]
        pred['labels'] = pred['labels'][mask]
        pred['scores'] = pred['scores'][mask]
        total_boxes += len(mask)
        kept_boxes += mask.sum()
    pseudolabels_kept_frac = kept_boxes / total_boxes
    return preds, pseudolabels_kept_frac


class Accuracy(ElementwiseMetric):

    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'acc'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        return torch.round((y_pred == y_true).float(), decimals=3)

    def worst(self, metrics):
        return torch.round(minimum(metrics), decimals=3)


class MultiTaskAccuracy(MultiTaskMetric):

    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn  # should work on flattened inputs
        if name is None:
            name = 'acc'
        super().__init__(name=name)

    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        if self.prediction_fn is not None:
            flattened_y_pred = self.prediction_fn(flattened_y_pred)
        return (flattened_y_pred == flattened_y_true).float()

    def worst(self, metrics):
        return minimum(metrics)


class MultiTaskAveragePrecision(MultiTaskMetric):

    def __init__(self, prediction_fn=None, name=None, average='macro'):
        self.prediction_fn = prediction_fn
        if name is None:
            name = f'avgprec'
            if average is not None:
                name += f'-{average}'
        self.average = average
        super().__init__(name=name)

    def _compute_flattened(self, flattened_y_pred, flattened_y_true):
        if self.prediction_fn is not None:
            flattened_y_pred = self.prediction_fn(flattened_y_pred)
        ytr = np.array(flattened_y_true.squeeze().detach().cpu().numpy() > 0)
        ypr = flattened_y_pred.squeeze().detach().cpu().numpy()
        score = sklearn.metrics.average_precision_score(ytr,
                                                        ypr,
                                                        average=self.average)
        to_ret = torch.tensor(score).to(flattened_y_pred.device)
        return to_ret

    def _compute_group_wise(self, y_pred, y_true, g, n_groups):
        group_metrics = []
        group_counts = get_counts(g, n_groups)
        for group_idx in range(n_groups):
            if group_counts[group_idx] == 0:
                group_metrics.append(torch.tensor(0., device=g.device))
            else:
                flattened_metrics, _ = self.compute_flattened(
                    y_pred[g == group_idx],
                    y_true[g == group_idx],
                    return_dict=False)
                group_metrics.append(flattened_metrics)
        group_metrics = torch.stack(group_metrics)
        worst_group_metric = self.worst(group_metrics[group_counts > 0])
        return group_metrics, group_counts, worst_group_metric

    def worst(self, metrics):
        return minimum(metrics)


class Recall(Metric):

    def __init__(self, prediction_fn=None, name=None, average='binary'):
        self.prediction_fn = prediction_fn
        if name is None:
            name = f'recall'
            if average is not None:
                name += f'-{average}'
        self.average = average
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        recall = sklearn.metrics.recall_score(y_true,
                                              y_pred,
                                              average=self.average,
                                              labels=torch.unique(y_true))
        return torch.tensor(recall)

    def worst(self, metrics):
        return minimum(metrics)


class F1(Metric):

    def __init__(self, prediction_fn=None, name=None, average='binary'):
        self.prediction_fn = prediction_fn
        if name is None:
            name = f'F1'
            if average is not None:
                name += f'-{average}'
        self.average = average
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        score = sklearn.metrics.f1_score(y_true,
                                         y_pred,
                                         average=self.average,
                                         labels=torch.unique(y_true))
        return torch.tensor(score)

    def worst(self, metrics):
        return minimum(metrics)


class PearsonCorrelation(Metric):

    def __init__(self, name=None):
        if name is None:
            name = 'r'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        r = pearsonr(y_pred.squeeze().detach().cpu().numpy(),
                     y_true.squeeze().detach().cpu().numpy())[0]
        return torch.tensor(r)

    def worst(self, metrics):
        return minimum(metrics)


def mse_loss(out, targets):
    assert out.size() == targets.size()
    if out.numel() == 0:
        return torch.Tensor()
    else:
        assert out.dim(
        ) > 1, 'MSE loss currently supports Tensors of dimensions > 1'
        losses = (out - targets)**2
        reduce_dims = tuple(list(range(1, len(targets.shape))))
        losses = torch.mean(losses, dim=reduce_dims)
        return losses


class MSE(ElementwiseLoss):

    def __init__(self, name=None):
        if name is None:
            name = 'mse'
        super().__init__(name=name, loss_fn=mse_loss)


class PrecisionAtRecall(Metric):
    """Given a specific model threshold, determine the precision score
    achieved."""

    def __init__(self, threshold, score_fn=None, name=None):
        self.score_fn = score_fn
        self.threshold = threshold
        if name is None:
            name = "precision_at_global_recall"
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        score = self.score_fn(y_pred)
        predictions = (score > self.threshold)
        return torch.tensor(sklearn.metrics.precision_score(
            y_true, predictions))

    def worst(self, metrics):
        return minimum(metrics)


class DummyMetric(Metric):
    """For testing purposes.

    This Metric always returns -1.
    """

    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'dummy'
        super().__init__(name=name)

    def _compute(self, y_pred, y_true):
        return torch.tensor(-1)

    def _compute_group_wise(self, y_pred, y_true, g, n_groups):
        group_metrics = torch.ones(n_groups, device=g.device) * -1
        group_counts = get_counts(g, n_groups)
        worst_group_metric = self.worst(group_metrics)
        return group_metrics, group_counts, worst_group_metric

    def worst(self, metrics):
        return minimum(metrics)


class DetectionAccuracy(ElementwiseMetric):
    """Given a specific Intersection over union threshold, determine the
    accuracy achieved for a one-class detector."""

    def __init__(self,
                 iou_threshold=0.3,
                 score_threshold=0.1,
                 name=None,
                 geometry_name="boxes",
                 metric="accuracy"):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.geometry_name = geometry_name
        self.metric = metric
        if name is None:
            name = "detection_{}".format(metric)
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        batch_results = []
        for gt, target in zip(y_true, y_pred):
            target_boxes = target[self.geometry_name]
            target_scores = target["scores"]
            gt_boxes = gt[self.geometry_name]
            if target_boxes.dim() == 1:
                target_boxes = target_boxes.view(-1, 4)
            pred_boxes = target_boxes[target_scores > self.score_threshold]
            if self.metric == "accuracy":
                det_accuracy = self._accuracy(gt_boxes, pred_boxes,
                                              self.iou_threshold)
            elif self.metric == "recall":
                det_accuracy = self._recall(gt_boxes, pred_boxes,
                                            self.iou_threshold)
            batch_results.append(det_accuracy)
        return torch.tensor(batch_results)

    def _recall(self, src_boxes, pred_boxes, iou_threshold):
        total_gt = len(src_boxes)
        total_pred = len(pred_boxes)
        if total_gt > 0 and total_pred > 0:
            # Define the matcher and distance matrix based on iou
            matcher = Matcher(iou_threshold,
                              iou_threshold,
                              allow_low_quality_matches=False)
            match_quality_matrix = box_iou(src_boxes, pred_boxes)
            results = matcher(match_quality_matrix)
            true_positive = torch.count_nonzero(results.unique() != -1)
            return true_positive / total_gt
        elif total_gt == 0:
            if total_pred > 0:
                return torch.tensor(0.)
            else:
                return torch.tensor(1.)
        elif total_gt > 0 and total_pred == 0:
            return torch.tensor(0.)

    def _accuracy(self, src_boxes, pred_boxes, iou_threshold):
        total_gt = len(src_boxes)
        total_pred = len(pred_boxes)
        if total_gt > 0 and total_pred > 0:
            # Define the matcher and distance matrix based on iou
            matcher = Matcher(iou_threshold,
                              iou_threshold,
                              allow_low_quality_matches=False)
            match_quality_matrix = box_iou(src_boxes, pred_boxes)
            results = matcher(match_quality_matrix)
            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results > -1]
            # in Matcher, a pred element can be matched only twice
            false_positive = (
                torch.count_nonzero(results == -1) +
                (len(matched_elements) - len(matched_elements.unique())))
            false_negative = total_gt - true_positive
            acc = true_positive / (true_positive + false_positive +
                                   false_negative)
            return acc
        elif total_gt == 0:
            if total_pred > 0:
                return torch.tensor(0.)
            else:
                return torch.tensor(1.)
        elif total_gt > 0 and total_pred == 0:
            return torch.tensor(0.)

    def worst(self, metrics):
        return minimum(metrics)


class KeypointAccuracy(ElementwiseMetric):
    """Given a specific Intersection over union threshold, determine the
    accuracy achieved for a one-class detector."""

    def __init__(self,
                 distance_threshold=0.1,
                 score_threshold=0.1,
                 name=None,
                 geometry_name="y"):
        self.distance_threshold = distance_threshold
        self.score_threshold = score_threshold
        self.geometry_name = geometry_name
        if name is None:
            name = "keypoint_acc"
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        batch_results = []
        for gt, target in zip(y_true, y_pred):
            target_boxes = target[self.geometry_name]
            target_scores = target["scores"]

            gt_boxes = gt[self.geometry_name]
            pred_boxes = target_boxes[target_scores > self.score_threshold]
            det_accuracy = self._accuracy(gt_boxes, pred_boxes,
                                          self.distance_threshold)
            batch_results.append(det_accuracy)
        return torch.tensor(batch_results)

    def _point_nearness(self, src_keypoints, pred_keypoints):
        distance = torch.cdist(src_keypoints.float(),
                               pred_keypoints.float(),
                               p=2)
        return distance

    def _accuracy(self, src_keypoints, pred_keypoints, distance_threshold):
        total_gt = len(src_keypoints)
        total_pred = len(pred_keypoints)
        if total_gt > 0 and total_pred > 0:
            # Define the matcher and distance matrix based on iou
            # Convert distances to a similarity score where higher is better
            # and threshold accordingly so that matches within the distance_threshold are accepted
            distance_matrix = self._point_nearness(src_keypoints,
                                                   pred_keypoints)
            # Similarity in [0, 1], higher is better (0 distance -> 1 similarity)
            similarity_matrix = 1.0 / (1.0 + distance_matrix)
            sim_threshold = 1.0 / (1.0 + distance_threshold)

            matcher = Matcher(sim_threshold,
                              sim_threshold,
                              allow_low_quality_matches=False)
            match_quality_matrix = similarity_matrix
            results = matcher(match_quality_matrix)
            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results > -1]
            # in Matcher, a pred element can be matched only twice
            false_positive = (
                torch.count_nonzero(results == -1) +
                (len(matched_elements) - len(matched_elements.unique())))
            false_negative = total_gt - true_positive
            acc = true_positive / (true_positive + false_positive +
                                   false_negative)
            return acc
        elif total_gt == 0:
            if total_pred > 0:
                return torch.round(torch.tensor(0.), decimals=3)
            else:
                return torch.round(torch.tensor(1.), decimals=3)
        elif total_gt > 0 and total_pred == 0:
            return torch.round(torch.tensor(0.), decimals=3)

    def worst(self, metrics):
        return torch.round(minimum(metrics), decimals=3)


class MaskAccuracy(ElementwiseMetric):
    """Given a specific Intersection over union threshold, determine the
    accuracy achieved for a Mask R-CNN detector."""

    def __init__(self,
                 iou_threshold=0.5,
                 score_threshold=0.1,
                 name=None,
                 geometry_name="masks",
                 metric="accuracy"):
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.geometry_name = geometry_name
        if name is None:
            name = "mask_acc"
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        batch_results = []
        for gt, target in zip(y_true, y_pred):
            target_masks = target[self.geometry_name]
            target_scores = target["scores"]

            gt_masks = gt[self.geometry_name]
            # Convert to tensors if needed
            if not isinstance(target_scores, torch.Tensor):
                target_scores = torch.as_tensor(target_scores, dtype=torch.float32)
            
            pred_masks = target_masks[target_scores > self.score_threshold]
            det_accuracy = self._accuracy(gt_masks, pred_masks,
                                          self.iou_threshold)
            batch_results.append(det_accuracy)
        return torch.tensor(batch_results)

    def _boxes_to_masks(self, boxes, height, width):
        """Convert bounding boxes [N, 4] (xyxy format) to masks [N, H, W]."""
        if len(boxes) == 0:
            device = boxes.device if isinstance(boxes, torch.Tensor) else 'cpu'
            return torch.zeros((0, height, width), dtype=torch.bool, device=device)
        
        # Convert to tensor if needed
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        boxes = boxes.clone()
        # Clamp boxes to image bounds
        boxes[:, 0] = torch.clamp(boxes[:, 0], 0, width)
        boxes[:, 1] = torch.clamp(boxes[:, 1], 0, height)
        boxes[:, 2] = torch.clamp(boxes[:, 2], 0, width)
        boxes[:, 3] = torch.clamp(boxes[:, 3], 0, height)
        
        device = boxes.device
        masks = torch.zeros((len(boxes), height, width), dtype=torch.bool, device=device)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int()
            # Ensure valid box
            if x2 > x1 and y2 > y1:
                masks[i, y1:y2, x1:x2] = True
        return masks

    def _mask_iou(self, src_masks, pred_masks):
        # Convert to tensors if needed (preserve original dtype for shape detection)
        src_is_tensor = isinstance(src_masks, torch.Tensor)
        pred_is_tensor = isinstance(pred_masks, torch.Tensor)
        
        if not src_is_tensor:
            src_masks = torch.as_tensor(src_masks)
        if not pred_is_tensor:
            pred_masks = torch.as_tensor(pred_masks)
        
        # Handle case where pred_masks are actually bounding boxes [M, 4]
        # Check if pred_masks are boxes (shape [M, 4]) instead of masks [M, H, W]
        # For empty tensors, check the shape tuple
        is_pred_boxes = (pred_masks.dim() == 2 and 
                        (len(pred_masks) == 0 or (pred_masks.shape[1] == 4 and pred_masks.dim() == 2)))
        if is_pred_boxes:
            # Get image dimensions from src_masks
            if len(src_masks) > 0 and src_masks.dim() == 3:
                height, width = src_masks.shape[1], src_masks.shape[2]
                # Convert boxes to masks
                pred_masks = self._boxes_to_masks(pred_masks, height, width)
            else:
                # If no ground truth masks, return zero IoU
                device = pred_masks.device if isinstance(pred_masks, torch.Tensor) else 'cpu'
                return torch.zeros((0, len(pred_masks)), dtype=torch.float32, device=device)
        
        # Handle case where src_masks are boxes (shouldn't happen, but handle gracefully)
        is_src_boxes = (src_masks.dim() == 2 and 
                       (len(src_masks) == 0 or (src_masks.shape[1] == 4 and src_masks.dim() == 2)))
        if is_src_boxes:
            if len(pred_masks) > 0 and pred_masks.dim() == 3:
                height, width = pred_masks.shape[1], pred_masks.shape[2]
                src_masks = self._boxes_to_masks(src_masks, height, width)
            else:
                device = src_masks.device if isinstance(src_masks, torch.Tensor) else 'cpu'
                return torch.zeros((len(src_masks), 0), dtype=torch.float32, device=device)
        
        # Ensure masks are bool type for bitwise operations
        if src_masks.dtype != torch.bool:
            src_masks = src_masks.bool()
        if pred_masks.dtype != torch.bool:
            pred_masks = pred_masks.bool()
        
        # Memory optimization: Use bbox IoU to pre-filter before computing expensive mask IoU
        # This reduces memory usage from O(N*M*H*W) to O(N*M) for filtering, then only
        # compute mask IoU for candidate pairs
        device = src_masks.device
        N, M = len(src_masks), len(pred_masks)
        
        if N == 0 or M == 0:
            return torch.zeros((N, M), dtype=torch.float32, device=device)
        
        # Compute bboxes from masks for pre-filtering.
        # Torchvision's masks_to_boxes errors if any individual mask is empty (all zeros),
        # so compute boxes only for non-empty masks and zero-fill the rest.
        src_nonempty = src_masks.flatten(1).any(dim=1)
        pred_nonempty = pred_masks.flatten(1).any(dim=1)

        src_boxes = torch.zeros((N, 4), dtype=torch.float32, device=device)
        pred_boxes = torch.zeros((M, 4), dtype=torch.float32, device=device)

        if src_nonempty.any():
            src_boxes[src_nonempty] = masks_to_boxes(src_masks[src_nonempty])
        if pred_nonempty.any():
            pred_boxes[pred_nonempty] = masks_to_boxes(pred_masks[pred_nonempty])
        
        # Compute bbox IoU for all pairs (cheap: O(N*M))
        bbox_iou = box_iou(src_boxes, pred_boxes)  # [N, M]
        bbox_iou[~src_nonempty, :] = 0.0
        bbox_iou[:, ~pred_nonempty] = 0.0
        
        # Initialize IoU matrix with bbox IoU values (will be refined for ambiguous cases)
        iou = bbox_iou.clone()
        
        # Option 3: Hybrid bbox/mask IoU - use bbox IoU as approximation for obvious cases
        # For very low bbox IoU (< 0.1), no mask overlap is possible
        iou[bbox_iou < 0.1] = 0.0
        
        # For very high bbox IoU (> 0.9), bbox IoU is a good approximation of mask IoU
        # Only compute expensive mask IoU for ambiguous cases (0.1 <= bbox_iou <= 0.9)
        ambiguous_mask = (bbox_iou >= 0.1) & (bbox_iou <= 0.9)
        
        if ambiguous_mask.any():
            # Get indices of ambiguous pairs that need mask IoU computation
            ambiguous_gt_indices, ambiguous_pred_indices = torch.where(ambiguous_mask)
            num_ambiguous = len(ambiguous_gt_indices)
            
            # Process ambiguous pairs in chunks to avoid creating huge tensors
            # Even with downsampling, U_gt * U_pred * H * W can be massive
            chunk_size = 500  # Process 500 ambiguous pairs at a time
            target_size = 224  # Downsample to 224x224 for memory efficiency
            
            for chunk_start in range(0, num_ambiguous, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_ambiguous)
                chunk_gt_idx = ambiguous_gt_indices[chunk_start:chunk_end]
                chunk_pred_idx = ambiguous_pred_indices[chunk_start:chunk_end]
                
                # Get unique indices for this chunk to avoid redundant mask loading
                unique_gt_idx = torch.unique(chunk_gt_idx)
                unique_pred_idx = torch.unique(chunk_pred_idx)
                
                # Load masks for this chunk
                chunk_src_masks = src_masks[unique_gt_idx]  # [U_gt, H, W]
                chunk_pred_masks = pred_masks[unique_pred_idx]  # [U_pred, H, W]
                
                # Option 1: Downsample masks for IoU computation to reduce memory
                if chunk_src_masks.shape[1] > target_size:
                    # Downsample using nearest neighbor to preserve binary nature
                    chunk_src_masks = F.interpolate(
                        chunk_src_masks.unsqueeze(1).float(),
                        size=(target_size, target_size),
                        mode='nearest'
                    ).squeeze(1).bool()
                    chunk_pred_masks = F.interpolate(
                        chunk_pred_masks.unsqueeze(1).float(),
                        size=(target_size, target_size),
                        mode='nearest'
                    ).squeeze(1).bool()
                
                # Create mapping from original indices to chunk indices
                gt_idx_map = {int(idx): i for i, idx in enumerate(unique_gt_idx)}
                pred_idx_map = {int(idx): i for i, idx in enumerate(unique_pred_idx)}
                
                # Compute mask IoU for chunk pairs using vectorized operations
                src_expanded = chunk_src_masks.unsqueeze(1)  # [U_gt, 1, H, W]
                pred_expanded = chunk_pred_masks.unsqueeze(0)  # [1, U_pred, H, W]
                intersection = (src_expanded & pred_expanded).float().sum((2, 3))  # [U_gt, U_pred]
                union = (src_expanded | pred_expanded).float().sum((2, 3))  # [U_gt, U_pred]
                chunk_mask_iou = intersection / union.clamp(min=1e-6)
                chunk_mask_iou[union == 0] = 0.0
                
                # Map chunk results back to original indices
                for i, j in zip(chunk_gt_idx, chunk_pred_idx):
                    orig_gt_idx = int(i)
                    orig_pred_idx = int(j)
                    chunk_gt_pos = gt_idx_map[orig_gt_idx]
                    chunk_pred_pos = pred_idx_map[orig_pred_idx]
                    iou[orig_gt_idx, orig_pred_idx] = chunk_mask_iou[chunk_gt_pos, chunk_pred_pos]
        
        return iou  # Returns [N, M] matrix

    def _recall(self, src_masks, pred_masks, iou_threshold):
        total_gt = len(src_masks)
        total_pred = len(pred_masks)
        if total_gt > 0 and total_pred > 0:
            # Define the matcher and distance matrix based on iou
            matcher = Matcher(iou_threshold,
                              iou_threshold,
                              allow_low_quality_matches=False)
            match_quality_matrix = self._mask_iou(src_masks, pred_masks)
            results = matcher(match_quality_matrix)
            true_positive = torch.count_nonzero(results.unique() != -1)
            return true_positive / total_gt
        elif total_gt == 0:
            if total_pred > 0:
                return torch.tensor(0.)
            else:
                return torch.tensor(1.)
        elif total_gt > 0 and total_pred == 0:
            return torch.tensor(0.)

    def _accuracy(self, src_masks, pred_masks, iou_threshold):
        total_gt = len(src_masks)
        total_pred = len(pred_masks)
        if total_gt > 0 and total_pred > 0:
            # Define the matcher and distance matrix based on iou
            matcher = Matcher(iou_threshold,
                              iou_threshold,
                              allow_low_quality_matches=False)
            match_quality_matrix = self._mask_iou(src_masks, pred_masks)
            results = matcher(match_quality_matrix)
            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results > -1]
            # in Matcher, a pred element can be matched only twice
            false_positive = (
                torch.count_nonzero(results == -1) +
                (len(matched_elements) - len(matched_elements.unique())))
            false_negative = total_gt - true_positive
            acc = true_positive / (true_positive + false_positive +
                                   false_negative)
            return acc
        elif total_gt == 0:
            if total_pred > 0:
                return torch.tensor(0.)
            else:
                return torch.tensor(1.)
        elif total_gt > 0 and total_pred == 0:
            return torch.tensor(0.)

    def worst(self, metrics):
        return minimum(metrics)


class CountingError(ElementwiseMetric):
    """Mean Absolute Error between ground truth and predicted detection counts.

    Calculates MAE between the number of detections in ground truth vs
    predictions for each sample in the batch.
    """

    def __init__(self, score_threshold=0.1, name=None, geometry_name="y"):
        self.score_threshold = score_threshold
        self.geometry_name = geometry_name
        if name is None:
            name = "counting_mae"
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        batch_results = []
        for gt, target in zip(y_true, y_pred):
            # Count ground truth detections
            gt_count = len(gt[self.geometry_name])

            # Count predicted detections above threshold
            target_scores = target["scores"]
            pred_count = (target_scores > self.score_threshold).sum().item()

            # Calculate absolute error
            mae = abs(gt_count - pred_count)
            batch_results.append(mae)

        return torch.tensor(batch_results, dtype=torch.float)

    def worst(self, metrics):
        return maximum(metrics)
