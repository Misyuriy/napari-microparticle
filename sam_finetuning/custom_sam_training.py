import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import vigra
from skimage import measure

import torch
from torch.optim import Optimizer
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.utils import make_grid

import torch_em
import torch_em.loss as loss
from torch_em.trainer.logger_base import TorchEmLogger
from torch_em.trainer.wandb_logger import WandbLogger
from torch_em.trainer.tensorboard_logger import normalize_im
import torch_em.metric.instance_segmentation_metric as metric

try:
    from qtpy.QtCore import QObject
except Exception:
    QObject = Any

from micro_sam.training import sam_trainer as trainers
from micro_sam.training import joint_sam_trainer as joint_trainers
from micro_sam.training.training import _check_loader, _filter_warnings, _get_optimizer_and_scheduler, _get_trainer_fit_params
from micro_sam.instance_segmentation import get_unetr, watershed_from_center_and_boundary_distances
from micro_sam.models.peft_sam import ClassicalSurgery
from micro_sam.util import get_device, get_model_names, export_custom_sam_model, get_sam_model
from micro_sam.training.util import get_trainable_sam_model, ConvertToSamInputs, require_8bit, get_raw_transform

import wandb


def _label_array_to_binary_masks(labels):
    unique_labels = np.unique(labels)
    masks = [labels == label for label in unique_labels]
    return masks


def _feret_diameter(mask):
    mask = mask.astype(np.uint8)  # mask may have multiple connected components, that's still the same particle
    regions = measure.regionprops(mask)
    return regions[0].feret_diameter_max


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


class CustomSamTrainer(joint_trainers.JointSamTrainer):
    def __init__(self, diameter_val_loader: DataLoader, **kwargs):
        super().__init__(**kwargs)

        assert diameter_val_loader.batch_size == 1
        self.diameter_val_loader = diameter_val_loader

    def fit(self, **kwargs):
        print('CUSTOM: Running initial test validation')
        metric = self._validate()
        print('metric: ', metric)
        print()

        super().fit(**kwargs)


    def _validate_impl(self, forward_context):
        rmse, matching_results = self.compute_particle_size_metric()

        if self.logger is not None:
            self.logger.log_metric(self._iteration, rmse, matching_results)

        super()._validate_impl(forward_context)

    def compute_particle_size_metric(self,
                                     center_distance_threshold: float = 0.5,
                                     boundary_distance_threshold: float = 0.5,
                                     foreground_threshold: float = 0.5,
                                     foreground_smoothing: float = 1.0,
                                     distance_smoothing: float = 1.6,
                                     ):
        self.model.eval()
        device = self.device

        all_true_diams = []  # true diameters for all GT masks (including unmatched -> 0)
        all_pred_diams = []  # predicted diameters for all predictions (including unmatched -> 0)

        with torch.no_grad():
            loader_iter = iter(self.diameter_val_loader)
            x, y = next(loader_iter)  # this loader always has exactly 1 sample (full image) and batch_size=1
            x = x.to(device)

            # Ground truth instance masks
            gt_masks = y[:, 0, ...].cpu().numpy()
            # Run automatic head
            unetr_outputs = self.unetr(x).cpu()  # raw outputs (B, C, H, W)

            foreground = unetr_outputs[0, 0, ...].numpy()
            center_distances = unetr_outputs[0, 1, ...].numpy()
            boundary_distances = unetr_outputs[0, 2, ...].numpy()

            if foreground_smoothing > 0:
                foreground = vigra.filters.gaussianSmoothing(foreground, foreground_smoothing)

            pred = watershed_from_center_and_boundary_distances(
                center_distances=center_distances,
                boundary_distances=boundary_distances,
                foreground_map=foreground,
                center_distance_threshold=center_distance_threshold,
                boundary_distance_threshold=boundary_distance_threshold,
                foreground_threshold=foreground_threshold,
                distance_smoothing=distance_smoothing,
                min_size=0,
            )
            gt = gt_masks[0]  # 2D array with instance IDs (0 = background)

            gt_instances = _label_array_to_binary_masks(gt)  # list of binary masks
            pred_instances = _label_array_to_binary_masks(pred)  # list of binary masks

            # Match predictions to ground truth using IoU
            matched_pairs, unmatched_gt, unmatched_pred = self.match_masks_gpu(
                gt_instances, pred_instances, iou_threshold=0.5
            )
            matching_results: dict = {
                'gt': len(gt_instances),
                'matched': len(matched_pairs),
                'unmatched_gt': len(unmatched_gt),
                'unmatched_pred': len(unmatched_pred)
            }

            # For each ground truth, collect true diameter and the matched predicted diameter (or 0)
            for gt_idx in range(len(gt_instances)):
                diam_true = _feret_diameter(gt_instances[gt_idx])
                if gt_idx in matched_pairs:
                    pred_idx = matched_pairs[gt_idx]
                    diam_pred = _feret_diameter(pred_instances[pred_idx])
                else:
                    diam_pred = 0.0
                all_true_diams.append(diam_true)
                all_pred_diams.append(diam_pred)

            # For each unmatched prediction, add a pair with true diameter = 0
            for pred_idx in unmatched_pred:
                diam_pred = _feret_diameter(pred_instances[pred_idx])
                all_true_diams.append(0.0)
                all_pred_diams.append(diam_pred)

        # Compute RMSE over all collected pairs
        true = np.array(all_true_diams)
        pred = np.array(all_pred_diams)
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        return rmse, matching_results

    def match_masks_gpu(self, gt_masks, pred_masks, iou_threshold=0.5):
        '''
        gt_masks : list of 2D binary numpy arrays (H,W) for one image
        pred_masks: list of 2D binary numpy arrays (H,W) for one image
        Returns: matched dict, unmatched_gt list, unmatched_pred list
        '''
        if len(gt_masks) == 0 or len(pred_masks) == 0:
            return {}, list(range(len(gt_masks))), list(range(len(pred_masks)))

        device = self.device

        # Convert masks to flattened float32 tensors on GPU
        def masks_to_tensor(masks):
            tensors = []
            for m in masks:
                # Ensure binary and float32
                t = torch.from_numpy(m.astype(np.float32)).to(device).flatten()
                tensors.append(t)
            return torch.stack(tensors)  # (N, L)

        gt_mat = masks_to_tensor(gt_masks)  # (N, L)
        pred_mat = masks_to_tensor(pred_masks)  # (M, L)

        # Compute IoU matrix: (N, M)
        intersection = torch.mm(gt_mat, pred_mat.t())  # (N, M)
        gt_areas = gt_mat.sum(dim=1)  # (N,)
        pred_areas = pred_mat.sum(dim=1)  # (M,)
        union = gt_areas[:, None] + pred_areas[None, :] - intersection
        iou = intersection / (union + 1e-6)  # (N, M)

        # Greedy matching on CPU
        iou_cpu = iou.cpu().numpy()
        matched = {}
        used_pred = set()
        for i in range(len(gt_masks)):
            if iou_cpu[i].max() >= iou_threshold:
                j = iou_cpu[i].argmax()
                if j not in used_pred:
                    matched[i] = j
                    used_pred.add(j)

        unmatched_gt = [i for i in range(len(gt_masks)) if i not in matched]
        unmatched_pred = [j for j in range(len(pred_masks)) if j not in used_pred]

        # Cleanup GPU memory
        del gt_mat, pred_mat, intersection, iou, gt_areas, pred_areas, union
        torch.cuda.empty_cache()

        return matched, unmatched_gt, unmatched_pred


class CustomSamLogger(WandbLogger):
    def __init__(self, trainer, save_root, **kwargs):
        # Forward all kwargs to the parent WandbLogger.
        # This allows passing project_name, log_model, mode, etc. either directly
        # or via the trainer / unused arguments.
        super().__init__(trainer, save_root, **kwargs)
        # The parent already sets self.log_image_interval = trainer.log_image_interval

    def _log_joint_images(self, step, x, y, samples, name):
        """Log input, target, and a grid of samples as separate wandb images."""
        # Selection slice (same as original add_image)
        selection = np.s_[0] if x.ndim == 4 else np.s_[0, :, x.shape[2] // 2]

        # ---- Input image ----
        img = normalize_im(x[selection].cpu())
        img_np = img.numpy().transpose((1, 2, 0))          # (H, W, C)
        wandb.log({f'images_{name}/input': wandb.Image(img_np, caption='Input')}, step=step)

        # ---- Target image ----
        target = y[selection].cpu()
        # Convert to numpy with sensible shape for wandb
        if target.ndim == 3 and target.shape[0] in (1, 3):
            target_np = target.numpy().transpose((1, 2, 0))
        else:
            target_np = target.numpy()
            # Squeeze single channel if needed (wandb accepts 2D as grayscale)
            if target_np.ndim == 3 and target_np.shape[0] == 1:
                target_np = target_np.squeeze(0)
        wandb.log({f'images_{name}/target': wandb.Image(target_np, caption='Target')}, step=step)

        # ---- Samples grid ----
        # Original code: make_grid([sample[0] for sample in samples], nrow=4, padding=4)
        sample_imgs = [sample[0].cpu() for sample in samples]   # first element of each batch
        grid = make_grid(sample_imgs, nrow=4, padding=4)
        grid_np = grid.numpy().transpose((1, 2, 0))
        wandb.log({f'images_{name}/samples': wandb.Image(grid_np, caption='Samples')}, step=step)

    def log_train(self, step, loss, lr, x, y, samples,
                  mask_loss, iou_regression_loss, model_iou, instance_loss):
        # Scalars
        wandb.log({
            'train/loss': loss,
            'train/mask_loss': mask_loss,
            'train/iou_loss': iou_regression_loss,
            'train/model_iou': model_iou,
            'train/instance_loss': instance_loss,
            'train/learning_rate': lr,
        }, step=step)

        # Update best loss in run summary
        if loss < self.wand_run.summary.get('train/loss', np.inf):
            self.wand_run.summary['train/loss'] = loss

        # Images (at specified interval)
        if step % self.log_image_interval == 0:
            self._log_joint_images(step, x, y, samples, 'train')

    def log_validation(self, step, metric, loss, x, y, samples,
                       mask_loss, iou_regression_loss, model_iou, instance_loss):
        # JointSamTrainer uses loss as metric
        # Actual metric (that is logged in log_metric) is calculated separately by CustomSamTrainer._validate_impl
        wandb.log({
            'validation/loss': loss,
            'validation/mask_loss': mask_loss,
            'validation/iou_loss': iou_regression_loss,
            'validation/model_iou': model_iou,
            'validation/instance_loss': instance_loss,
            #'validation/metric': metric,
        }, step=step)

        # Update best validation loss
        if loss < self.wand_run.summary.get('validation/loss', np.inf):
            self.wand_run.summary['validation/loss'] = loss
        #if metric < self.wand_run.summary.get('validation/metric', np.inf):
        #    self.wand_run.summary['validation/metric'] = metric

        self._log_joint_images(step, x, y, samples, 'validation')

    def log_metric(self, step, rmse: float, matching_results: dict):
        matched_gt_ratio = matching_results['matched'] / matching_results['gt']  # best possible ratio: 1.0

        wandb.log({
            'validation/matched_gt_ratio': matched_gt_ratio,
            'validation/unmatched_gt': matching_results['unmatched_gt'],
            'validation/unmatched_pred': matching_results['unmatched_pred'],
            'validation/feret_diameter_rmse': rmse,
        }, step=step)
'''
    def log_validation_extra(
            self, step, metric, loss, x, y, samples, mask_loss, iou_regression_loss, model_iou
    ):
        wandb.log({
            'validation_extra/loss': loss,
            'validation_extra/mask_loss': mask_loss,
            'validation_extra/iou_loss': iou_regression_loss,
            'validation_extra/model_iou': model_iou,
            'validation_extra/metric': metric,
        }, step=step)

        self._log_joint_images(step, x, y, samples, 'validation_extra')
'''


def custom_train_sam(
    name: str,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    diameter_val_loader: DataLoader,
    n_epochs: int = 100,
    early_stopping: Optional[int] = 10,
    n_objects_per_batch: Optional[int] = 25,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    freeze: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    lr: float = 1e-5,
    n_sub_iteration: int = 8,
    save_root: Optional[Union[str, os.PathLike]] = None,
    mask_prob: float = 0.5,
    n_iterations: Optional[int] = None,
    scheduler_class: Optional[_LRScheduler] = torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    save_every_kth_epoch: Optional[int] = None,
    pbar_signals: Optional[QObject] = None,
    optimizer_class: Optional[Optimizer] = torch.optim.AdamW,
    peft_kwargs: Optional[Dict] = None,
    ignore_warnings: bool = True,
    verify_n_labels_in_loader: Optional[int] = 50,
    box_distortion_factor: Optional[float] = 0.025,
    overwrite_training: bool = True,
    instance_seg_loss: torch.nn.modules.module.Module = loss.DiceBasedDistanceLoss(mask_distances_in_bg=True),
    instance_seg_metric: metric.BaseInstanceSegmentationMetric = None,
    **model_kwargs,
):
    with _filter_warnings(ignore_warnings):

        t_start = time.time()
        if verify_n_labels_in_loader is not None:
            _check_loader(train_loader, True, 'train', verify_n_labels_in_loader)
            _check_loader(val_loader, True, 'val', verify_n_labels_in_loader)

        device = get_device(device)
        # Get the trainable segment anything model.
        model, state = get_trainable_sam_model(
            model_type=model_type,
            device=device,
            freeze=freeze,
            checkpoint_path=checkpoint_path,
            return_state=True,
            peft_kwargs=peft_kwargs,
            **model_kwargs
        )

        # This class creates all the training data for a batch (inputs, prompts and labels).
        convert_inputs = ConvertToSamInputs(transform=model.transform, box_distortion_factor=box_distortion_factor)

        # Create the UNETR decoder (if train with it) and the optimizer.
        # Get the UNETR.
        unetr = get_unetr(
            image_encoder=model.sam.image_encoder, decoder_state=state.get('decoder_state', None), device=device,
        )

        # Get the parameters for SAM and the decoder from UNETR.
        joint_model_params = [params for params in model.parameters()]  # sam parameters
        for param_name, params in unetr.named_parameters():  # unetr's decoder parameters
            if not param_name.startswith('encoder'):
                joint_model_params.append(params)

        model_params = joint_model_params

        optimizer, scheduler = _get_optimizer_and_scheduler(
            model_params, lr, optimizer_class, scheduler_class, scheduler_kwargs
        )

        # The trainer which performs training and validation.
        if instance_seg_metric is None:
            instance_seg_metric = instance_seg_loss
        trainer = CustomSamTrainer(
            name=name,
            save_root=save_root,
            train_loader=train_loader,
            val_loader=val_loader,
            diameter_val_loader=diameter_val_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            lr_scheduler=scheduler,
            logger=CustomSamLogger,
            log_image_interval=100,
            mixed_precision=True,
            convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch,
            n_sub_iteration=n_sub_iteration,
            compile_model=False,
            unetr=unetr,
            instance_loss=instance_seg_loss,
            instance_metric=instance_seg_metric,
            early_stopping=early_stopping,
            mask_prob=mask_prob,
        )

        trainer_fit_params = _get_trainer_fit_params(
            n_epochs, n_iterations, save_every_kth_epoch, pbar_signals, overwrite_training
        )
        trainer.fit(**trainer_fit_params)

        t_run = time.time() - t_start
        hours = int(t_run // 3600)
        minutes = int(t_run // 60)
        seconds = int(round(t_run % 60, 0))
        print('Training took', t_run, f'seconds (= {hours:02}:{minutes:02}:{seconds:02} hours)')

