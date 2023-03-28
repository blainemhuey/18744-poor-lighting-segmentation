"""
https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
"""

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader

from augmentations import RandomLight, RandomShadow
from datasets import FusedDataset
from mfnet_spec import MFNetModified

from tqdm import tqdm
from datetime import datetime


def train(model, loader, criterion, optimizer, scheduler, scaler):
    model.train()

    for it, (images, labels, names) in enumerate(loader):
        optimizer.zero_grad()
        images = images.cuda()
        labels = labels.cuda()

        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()  # This is something added just for FP16

    scheduler.step()


def validate(model, loader, criterion):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_elements = 0
    with torch.no_grad():
        for it, (images, labels, names) in tqdm(enumerate(loader)):
            images = images.cuda()
            labels = labels.cuda()

            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += float(loss)
            total_correct += int(((logits.argmax(1) == labels)*(labels != -1)).sum())
            total_elements += int(labels.numel() - (labels == -1).sum())

    val_loss = float(total_loss / len(loader))
    val_acc = 100 * total_correct / total_elements
    return val_loss, val_acc


def main():
    epochs = 100
    batch_size = 8
    data_dir = "/home/blaine/Desktop/18744/Project/ir_seg_dataset"

    train_albumentations = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25, 0.25)),
        ToTensorV2(),
    ])
    train_transforms = [
        RandomLight(0.5),
        RandomShadow(0.5),
        lambda x, y: tuple(map(train_albumentations(image=x, mask=y).get, ["image", "mask"]))
    ]

    val_albumentations = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25, 0.25)),
        ToTensorV2()
    ])
    val_transforms = [
        lambda x, y: tuple(map(val_albumentations(image=x, mask=y).get, ["image", "mask"]))
    ]

    train_dataset = FusedDataset(data_dir, 'train', have_label=True, transform=train_transforms)
    val_dataset = FusedDataset(data_dir, 'val', have_label=True, transform=val_transforms)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2
    )

    model = MFNetModified(
        rgb_ch=MFNetModified.DEFAULT_RGB_CH_SIZE,
        inf_ch=MFNetModified.DEFAULT_INF_CH_SIZE,
        n_class=9
    ).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train(model, train_loader, criterion, optimizer, scheduler, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Loss {val_loss}, Accuracy {val_acc}")
        torch.save(model.state_dict(),
                   datetime.now().strftime(f"./weights/model_%y_%m_%d_%H_%M_%S_epoch{epoch + 1}.pt"))


if __name__ == '__main__':
    main()
