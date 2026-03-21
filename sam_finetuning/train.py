import os

import torch

from torch_em.data import MinInstanceSampler

import micro_sam.training as sam_training


root_dir = "your root dir"
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")

raw_key = "*.bmp"
label_key = "*.tif"

patch_shape = (1, 256, 256)  # (channels, height, width)
batch_size = 2

train_instance_segmentation = True

sampler = MinInstanceSampler(min_size=25)

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
    shuffle=True,
    raw_transform=sam_training.identity,
    sampler=sampler,
)

output_dir = os.path.join(root_dir, "models")
os.makedirs(output_dir, exist_ok=True)

n_objects_per_batch = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
n_epochs = 10

model_type = "vit_l"

checkpoint_name = "sam_model"

sam_training.train_sam(
    name=checkpoint_name,
    save_root=output_dir,
    model_type=model_type,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=n_epochs,
    n_objects_per_batch=n_objects_per_batch,
    with_segmentation_decoder=train_instance_segmentation,
    freeze=['image_encoder', 'prompt_encoder'],
    device=device,
)
