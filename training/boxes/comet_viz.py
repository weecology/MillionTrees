"""Log detection overlays to Comet's native annotation viewer.

Why this exists: stage-1 pretraining was judged entirely on scalar curves, and those
curves are ambiguous. A jagged ``train_loss`` on 7.1M weak boxes is consistent with
"learning slowly", "fitting noise", and "the pseudo-labels disagree with each other" --
the scalars cannot tell those apart. Comet renders boxes as *interactive, toggleable
layers* on the image (``Experiment.log_image(annotations=[Layer(...)])``), so logging
the same fixed tiles every epoch turns the question into one you can answer by
looking: are the predicted boxes converging on the targets, or spraying?

Two panels, because they answer different questions:

* ``train-weak``  -- unsupervised tiles with their Weinstein et al. 2018 pseudo-labels.
  Can the model fit its own training targets at all, and are those targets any good?
* ``val-supervised`` -- held-out human-labelled tiles. Does any of it transfer?

Prediction post-processing is deliberately *not* DeepForest's inference default. That
default is ``nms_thresh=0.05``, which deletes any box overlapping a higher-scoring one
by >5% IoU -- crippling on tiles averaging 143 heavily-overlapping crowns, and the
reason an earlier audit under-measured stage-1 recall as 0.217 when the network was
actually recalling 0.42-0.49 of its own labels. Visualising through that gate would
show a near-empty image and misattribute a postprocessing artifact to the model, so
the thresholds are explicit arguments here.
"""

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset


def sample_fixed_batch(subset, collate_fn, n_images, seed):
    """Materialise a fixed, seeded slice of ``subset`` as one (images, targets, paths).

    Fixed and seeded on purpose: the panel is only readable if it shows the *same*
    tiles at every epoch, so you are watching one model change rather than a new
    sample each time. Loaded once, held on CPU, and reused for the whole run.
    """
    n_images = min(n_images, len(subset))
    if n_images == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(subset), size=n_images, replace=False).tolist())
    loader = DataLoader(
        Subset(subset, idx),
        batch_size=n_images,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    images, targets, paths = next(iter(loader))
    return images.cpu(), [{k: v.cpu() for k, v in t.items()} for t in targets], list(paths)


def _to_uint8_hwc(image):
    """MillionTrees hands out CHW float32 in 0-1; Comet wants HWC uint8."""
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def _threshold_holder(net):
    """Return the module whose post-processing thresholds actually govern inference.

    RetinaNet keeps ``score_thresh`` / ``nms_thresh`` / ``detections_per_img`` on
    the model itself. Mask R-CNN keeps the *effective* ones on ``roi_heads`` and
    mirrors only the first two onto the model as inert bookkeeping attributes --
    it never mirrors ``detections_per_img`` at all. Writing to the model would
    therefore raise on one attribute and silently change nothing on the other
    two, which is the worst possible failure for a visualisation whose entire
    job is to show what the model really predicts.
    """
    roi_heads = getattr(net, "roi_heads", None)
    if roi_heads is not None and hasattr(roi_heads, "detections_per_img"):
        return roi_heads
    return net


class CometDetectionViz(pl.Callback):
    """Log fixed tiles with ground-truth and predicted box layers, once per epoch.

    Attach one instance per run; call :meth:`log_panels` directly (with
    ``step=0``) before ``trainer.fit`` to capture the pre-training state, so the
    epoch sequence in Comet starts from the initialisation rather than from
    whatever epoch 0 already produced.
    """

    def __init__(self, experiment, panels, score_thresh=0.05, nms_thresh=0.4,
                 detections_per_img=600, max_boxes=300, label="Tree"):
        super().__init__()
        self.experiment = experiment
        self.panels = panels  # {tag: (images, targets, paths)}
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.max_boxes = max_boxes
        self.label = label

    # -- prediction ---------------------------------------------------------

    def _predict(self, pl_module, images):
        """Forward pass with the viz post-processing thresholds swapped in.

        The overrides are restored in a ``finally`` so a visualisation can never
        change what the surrounding training/eval run measures.
        """
        net = pl_module.model
        holder = _threshold_holder(net)
        saved = (holder.score_thresh, holder.nms_thresh, holder.detections_per_img)
        was_training = net.training
        device = next(pl_module.parameters()).device
        try:
            holder.score_thresh = self.score_thresh
            holder.nms_thresh = self.nms_thresh
            holder.detections_per_img = self.detections_per_img
            net.eval()
            with torch.no_grad():
                return net(images.to(device))
        finally:
            (holder.score_thresh, holder.nms_thresh,
             holder.detections_per_img) = saved
            net.train(was_training)

    # -- logging ------------------------------------------------------------

    def _layers(self, gt_boxes, pred):
        from comet_ml.annotations import Box, Layer

        gt = [Box(_xyxy_to_xywh(b), self.label) for b in gt_boxes[: self.max_boxes]]

        boxes = pred.get("boxes", torch.zeros((0, 4)))
        scores = pred.get("scores", torch.zeros((0,)))
        keep = scores >= self.score_thresh
        boxes, scores = boxes[keep], scores[keep]
        # Highest-confidence first, so the max_boxes cap drops the tail rather
        # than an arbitrary slice -- an undertrained model emits hundreds of
        # near-threshold boxes and the cap would otherwise hide its best ones.
        order = torch.argsort(scores, descending=True)[: self.max_boxes]
        preds = [
            Box(_xyxy_to_xywh(boxes[i]), self.label, score=float(scores[i]))
            for i in order
        ]
        # Empty layers must be dropped, not passed through. Layer(boxes=[])
        # *constructs* fine, but Comet's upload validator requires len(data) > 0
        # for EVERY layer (validation/image/annotation_validator_helpers.py:
        # has_data_fields) and rejects the whole image if any layer is empty --
        # so a tile the model found nothing on would silently not upload at all,
        # which is precisely the tile worth looking at. Dropping the layer keeps
        # the image; n_predicted in the metadata records that it was zero.
        layers = []
        if gt:
            layers.append(Layer(boxes=gt, name="Ground truth"))
        if preds:
            layers.append(Layer(boxes=preds, name="Predictions"))
        return (
            layers or None,
            len(gt_boxes),
            int(keep.sum()),
            float(scores.max()) if len(scores) else 0.0,
        )

    def log_panels(self, pl_module, step):
        for tag, (images, targets, paths) in self.panels.items():
            predictions = self._predict(pl_module, images)
            n_gt, n_pred, max_scores = [], [], []
            for i, (image, target, pred) in enumerate(zip(images, targets, predictions)):
                layers, gt_count, pred_count, top_score = self._layers(
                    target["boxes"], pred)
                n_gt.append(gt_count)
                n_pred.append(pred_count)
                max_scores.append(top_score)
                self.experiment.log_image(
                    _to_uint8_hwc(image),
                    # The epoch goes in the NAME, not just the step. Reusing one
                    # name per tile and relying on `step` looks right but is not:
                    # Comet disambiguates repeated filenames by appending " (N)"
                    # in upload-arrival order, so the same tile came back as
                    # "train-weak/00", "train-weak/00 (1)" and "train-weak/00 (2)"
                    # with the suffixes in no particular epoch order (uploads are
                    # async). Naming each one explicitly keeps them unique,
                    # correctly ordered, and grouped per tile by Comet's "/"
                    # directory handling: one folder per tile, epochs sorted
                    # inside it.
                    name=f"{tag}/{i:02d}/epoch-{step:03d}",
                    image_channels="last",
                    annotations=layers,
                    step=step,
                    metadata={
                        "source_tile": str(paths[i]),
                        "n_ground_truth": gt_count,
                        "n_predicted": pred_count,
                        "max_score": round(top_score, 4),
                    },
                )
            # Scalar companions to the pictures: "sprays boxes" vs "predicts
            # nothing" vs "converging" is visible in these three curves alone.
            self.experiment.log_metrics(
                {
                    f"viz_{tag}_mean_dets_per_img": float(np.mean(n_pred)),
                    f"viz_{tag}_mean_gt_per_img": float(np.mean(n_gt)),
                    f"viz_{tag}_mean_max_score": float(np.mean(max_scores)),
                },
                step=step,
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        # Sanity-check passes run before any training and would overwrite the
        # step-0 baseline with an identical image.
        if trainer.sanity_checking:
            return
        self.log_panels(pl_module, step=trainer.current_epoch + 1)
