import albumentations as A
import albumentations.augmentations.functional as F
import numpy as np
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from augmentations import RandomLight, RandomShadow, RandomFlare
from datasets import MFNetDataset, HeatNetDataset, CustomDataset
from mfnet_spec import MFNetModified
from mfnet.util.augmentation import RandomCrop
from helpers import calculate_result

from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import cv2


def validate(model, loader, criterion, n_class):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_elements = 0
    cf = np.zeros((n_class, n_class))
    with torch.no_grad():
        for it, (images, labels) in tqdm(enumerate(loader)):
            images = images.cuda()
            labels = labels.cuda()

            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += float(loss)
            total_correct += int(((logits.argmax(1) == labels)*(labels != -1)).sum())
            total_elements += int(labels.numel() - (labels == -1).sum())

            predictions = logits.argmax(1)
            for gtcid in range(n_class):
                for pcid in range(n_class):
                    gt_mask      = labels == gtcid
                    pred_mask    = predictions == pcid
                    intersection = gt_mask * pred_mask
                    cf[gtcid, pcid] += int(intersection.sum())

    overall_acc, acc, IoU = calculate_result(cf)
    val_loss = float(total_loss / len(loader))
    val_acc = 100 * total_correct / total_elements
    return val_loss, val_acc, acc, IoU


def main(batch_size=8, n_class=5, pipeline_scalars=(1.0, 1.0)):
    custom_data_dir = "./datasets/custom_data"
    weights_path = "./weights/model_23_04_25_12_50_58_epoch100.pt"

    val_albumentations = A.Compose([
        # A.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25, 0.25)),
        ToTensorV2()
    ])
    val_transforms = [
        lambda x, y: tuple(map(val_albumentations(image=x, mask=y).get, ["image", "mask"]))
    ]

    # unlabelled, car, person, lights, bikes
    custom_label_map = [0, 1, 2, 4, 3]  # Exclude all except car, person, bike
    val_dataset_custom = CustomDataset(custom_data_dir, 'val', have_label=True, transform=val_transforms,
                                       label_map=custom_label_map)

    test_dataset = ConcatDataset((val_dataset_custom,))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=7,
    )

    model = MFNetModified(
        rgb_ch=MFNetModified.DEFAULT_RGB_CH_SIZE,
        inf_ch=np.rint(np.array(MFNetModified.DEFAULT_INF_CH_SIZE)//2*pipeline_scalars[1]).astype(int)*2,
        n_class=n_class
    )
    model.load_state_dict(torch.load(weights_path))
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_acs, test_iou = validate(model, test_loader, criterion, n_class)
    print(f"Final: Loss {test_loss}, Accuracy {test_acc}, Class Acc {test_acs}, IoU {test_iou}")


if __name__ == '__main__':
    main()
