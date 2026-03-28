import os

import torch

from torch_em.data import MinForegroundSampler
from torch_em.transform.augmentation import get_augmentations

import micro_sam.training as sam_training
import custom_sam_training

import wandb


root_dir = "your root dir"
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")
diam_val_dir = os.path.join(root_dir, "diam")

raw_key = "*.bmp"
label_key = "*.tif"

full_image_shape = (1, 960, 1280)
patch_shape = (1, 512, 512)  # (channels, height, width)
batch_size = 2

train_instance_segmentation = True

sampler = MinForegroundSampler(min_fraction=0.4)

train_loader = sam_training.default_sam_loader(
    raw_paths=os.path.join(train_dir, "images"),
    raw_key=raw_key,
    label_paths=os.path.join(train_dir, "labels"),
    label_key=label_key,
    with_segmentation_decoder=train_instance_segmentation,
    patch_shape=patch_shape,
    batch_size=batch_size,
    is_seg_dataset=True,
    shuffle=True,
    raw_transform=sam_training.identity,
    transform=get_augmentations(ndim=2),
    sampler=sampler,
)

val_loader = sam_training.default_sam_loader(
    raw_paths=os.path.join(val_dir, "images"),
    raw_key=raw_key,
    label_paths=os.path.join(val_dir, "labels"),
    label_key=label_key,
    with_segmentation_decoder=train_instance_segmentation,
    patch_shape=patch_shape,
    batch_size=batch_size,
    is_seg_dataset=True,
    shuffle=False,
    raw_transform=sam_training.identity,
    sampler=sampler,
)

diameter_val_loader = sam_training.default_sam_loader(
    raw_paths=os.path.join(diam_val_dir, "images"),
    raw_key=raw_key,
    label_paths=os.path.join(diam_val_dir, "labels"),
    label_key=label_key,
    with_segmentation_decoder=train_instance_segmentation,
    patch_shape=full_image_shape,
    batch_size=1,
    n_samples=1,
    is_seg_dataset=True,
    shuffle=False,
    raw_transform=sam_training.identity,
    sampler=MinForegroundSampler(min_fraction=0.0),  # we don't care, it's the full image all the time
)

output_dir = os.path.join(root_dir, "models")
os.makedirs(output_dir, exist_ok=True)

#n_objects_per_batch = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 10

model_type = "vit_l"
freeze = ['image_encoder', 'prompt_encoder']

checkpoint_name = "vit_l_mp_2"

wandb.init(
    project="micro-sam",

    config={
        "model_type": model_type,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        #"n_objects_per_batch": n_objects_per_batch,
        "patch_shape": patch_shape,
        "freeze": freeze
    }
)

custom_sam_training.custom_train_sam(
    name=checkpoint_name,
    save_root=output_dir,
    model_type=model_type,
    train_loader=train_loader,
    val_loader=val_loader,
    diameter_val_loader=diameter_val_loader,
    n_epochs=n_epochs,
    #n_objects_per_batch=n_objects_per_batch,
    freeze=freeze,
    device=device
)
