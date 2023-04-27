"""
https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
"""

import albumentations as A
import albumentations.augmentations.functional as F
import numpy as np
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchmetrics import JaccardIndex

from augmentations import RandomLight, RandomShadow, RandomFlare
from datasets import MFNetDataset, HeatNetDataset, CustomDataset
from mfnet_spec import MFNetModified
from mfnet.util.augmentation import RandomCrop
from helpers import calculate_result

from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import cv2


def train(model, loader, criterion, optimizer, scheduler, scaler):
    model.train()

    for it, (images, labels) in tqdm(enumerate(loader)):
        optimizer.zero_grad()
        # for i in range(images.shape[0]):
        #     plt.imshow(images.cpu()[i, :3, :, :].permute((1, 2, 0))/255)
        #     plt.show()
        #     plt.imshow(images.cpu()[i, 3:, :, :].permute((1, 2, 0))/255)
        #     plt.show()
        images = images.cuda()
        labels = labels.cuda()

        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()  # This is something added just for FP16

    scheduler.step()


def validate(model, loader, criterion, n_class):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_elements = 0
    all_predictions = []
    all_labels = []
    cf = np.zeros((n_class, n_class))
    with torch.no_grad():
        for it, (images, labels) in tqdm(enumerate(loader)):
            images = images.cuda()[:, :3, :, :]
            labels = labels.cuda()

            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += float(loss)
            total_correct += int(((logits.argmax(1) == labels) * (labels != -1)).sum())
            total_elements += int(labels.numel() - (labels == -1).sum())

            predictions = logits.argmax(1)
            all_predictions.append(predictions)
            all_labels.append(labels)

            for gtcid in range(n_class):
                for pcid in range(n_class):
                    gt_mask = labels == gtcid
                    pred_mask = predictions == pcid
                    intersection = gt_mask * pred_mask
                    cf[gtcid, pcid] += int(intersection.sum())

    jac = JaccardIndex("multiclass", num_classes=5, average=None)
    IoU = jac(torch.cat(all_predictions, dim=0).cpu(), torch.cat(all_labels, dim=0).cpu()).numpy()

    overall_acc, acc, _ = calculate_result(cf)
    val_loss = float(total_loss / len(loader))
    val_acc = 100 * total_correct / total_elements
    return val_loss, val_acc, acc, IoU


def main(epochs=100, batch_size=8, n_class=5, pipeline_scalars=(1.0, 1.0)):
    mfnet_data_dir = "./datasets/ir_seg_dataset"
    heatnet_data_dir = "./datasets/heatnet_data/train"
    custom_data_dir = "./datasets/custom_data"

    train_visual_only_albumentations = A.Compose([
        A.CLAHE(),
        A.FancyPCA(),
        A.ISONoise(),
        A.RGBShift(),
    ])

    train_albumentations = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.HorizontalFlip(p=0.5),
        # A.OpticalDistortion(),
        A.MotionBlur(),
        # A.GridDistortion(),
        # A.Blur(blur_limit=3),
        # A.GaussNoise(9, 0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Resize(480, 640),
        # A.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25, 0.25)),
        ToTensorV2(),
    ])
    train_transforms = [
        RandomLight(0.05),
        RandomShadow(0.05),
        RandomFlare(0.025, label_num=4),
        lambda x, y: (np.concatenate((train_visual_only_albumentations(image=x[:, :, :3])["image"], x[:, :, 3:]), axis=2), y),
        RandomCrop(),
        lambda x, y: tuple(map(train_albumentations(image=x, mask=y).get, ["image", "mask"]))
    ]

    val_albumentations = A.Compose([
        # A.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25, 0.25)),
        ToTensorV2()
    ])
    val_transforms = [
        lambda x, y: tuple(map(val_albumentations(image=x, mask=y).get, ["image", "mask"]))
    ]

    # unlabelled, car, person, bike, curve, car_stop, guardrail, color_cone, bump
    mfnet_label_map = [0, 1, 2, 3, 0, 0, 0, 0, 0]  # Exclude all except car, person, bike
    train_dataset_mfnet = MFNetDataset(mfnet_data_dir, 'train', have_label=True, transform=train_transforms,
                                       label_map=mfnet_label_map)
    val_dataset_mfnet = MFNetDataset(mfnet_data_dir, 'val', have_label=True, transform=val_transforms,
                                     label_map=mfnet_label_map)

    # unlabelled, road, sidewalk, building, curb, fence, pole, vegetation, terrain, sky, person, car, bicycle
    heatnet_label_map = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 3]  # Exclude all except car, person, bike
    # train_dataset_heatnet = HeatNetDataset(heatnet_data_dir, 'train', have_label=True, transform=train_transforms,
    #                                        label_map=heatnet_label_map)
    # val_dataset_heatnet = HeatNetDataset(heatnet_data_dir, 'val', have_label=True, transform=val_transforms,
    #                                      label_map=heatnet_label_map)

    # unlabelled, car, person, lights, bikes
    custom_label_map = [0, 1, 2, 4, 3]  # Exclude all except car, person, bike
    train_dataset_custom = CustomDataset(custom_data_dir, 'train', have_label=True, transform=train_transforms,
                                         label_map=custom_label_map)
    val_dataset_custom = CustomDataset(custom_data_dir, 'val', have_label=True, transform=val_transforms,
                                       label_map=custom_label_map)

    # train_dataset_heatnet = torch.utils.data.Subset(train_dataset_heatnet, np.arange(1000))
    train_dataset = ConcatDataset((train_dataset_mfnet, train_dataset_custom))  #
    val_dataset = ConcatDataset((val_dataset_custom,))  # val_dataset_mfnet
    test_dataset = ConcatDataset((val_dataset_custom,))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=7,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=7,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=7,
    )

    model = MFNetModified(
        rgb_ch=MFNetModified.DEFAULT_RGB_CH_SIZE,
        inf_ch=np.rint(np.array(MFNetModified.DEFAULT_INF_CH_SIZE) // 2 * pipeline_scalars[1]).astype(int) * 2,
        n_class=n_class
    ).cuda()
    # model.load_state_dict(torch.load('./weights/model_23_03_28_11_06_59_epoch48.pt'))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train(model, train_loader, criterion, optimizer, scheduler, scaler)
        val_loss, val_acc, acc, iou = validate(model, val_loader, criterion, n_class)
        print(f"Epoch {epoch + 1}: Loss {val_loss}, Accuracy {val_acc}, Class Acc {acc}, IoU {iou}")
        torch.save(model.state_dict(),
                   datetime.now().strftime(f"./weights/model_%y_%m_%d_%H_%M_%S_epoch{epoch + 1}.pt"))
        with open('validation.csv', 'a') as fd:
            fd.write(f'{epoch},{val_loss},{val_acc}\n')

    # Final Testing Loop
    test_loss, test_acc, test_acs, test_iou = validate(model, test_loader, criterion, n_class)
    print(f"Final: Loss {test_loss}, Accuracy {test_acc}, Class Acc {test_acs}, IoU {test_iou}")


if __name__ == '__main__':
    # for s in np.linspace(0, 1, 5)[1:]:
    #     scalar = (1.0, 1+s)
    #     main(pipeline_scalars=scalar)
    main()
