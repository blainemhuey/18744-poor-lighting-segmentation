import torch
import torch.nn as nn
import numpy as np

import cv2
import PIL.Image
from PIL import ImageEnhance, ImageOps
import os


class RandomShadow(object):

    def __init__(self, masking_prob, masks_path='./augmented_shadows/'):
        self.masks_path = masks_path
        self.masks_list = [mask for mask in sorted(os.listdir(masks_path)) if ".jpg" in mask]
        self.masks = np.array([cv2.imread(os.path.join(masks_path, mask), cv2.IMREAD_UNCHANGED) for mask in self.masks_list])
        self.masking_prob = masking_prob

    def __call__(self, image, label=None, **kwargs):
        if np.random.rand(1) > self.masking_prob:
            choice_index = np.random.choice(np.arange(self.masks.shape[0]), 1, replace=False)[0]
            mask = self.masks[choice_index]

            # downsize mask
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
            mask_gray_full = np.zeros_like(mask_resized)
            mask_gray_full[:, :, 0] = mask_gray
            mask_gray_full[:, :, 1] = mask_gray
            mask_gray_full[:, :, 2] = mask_gray

            # increase mask contrast
            mask_gray_full = cv2.cvtColor(mask_gray_full, cv2.COLOR_BGR2RGB)
            pil_mask = PIL.Image.fromarray(mask_gray_full)
            contrast_mask = ImageEnhance.Contrast(pil_mask).enhance(2)

            ############# for dark spots only #############
            contrast_mask = ImageOps.invert(contrast_mask)
            contrast_mask = cv2.cvtColor(np.array(contrast_mask), cv2.COLOR_RGB2GRAY)
            ret, mask_thresh = cv2.threshold(contrast_mask, 120, 255, cv2.THRESH_TRUNC)

            contrast_mask = cv2.cvtColor(np.array(mask_thresh), cv2.COLOR_GRAY2BGR)

            # # overlay dark spots
            res = cv2.addWeighted(image[:, :, :3], 0.5, contrast_mask, 0.5, -60)
            return np.concatenate((res, image[:, :, 3:]), axis=2), label
        else:
            return image, label


class RandomLight(object):

    def __init__(self, masking_prob, masks_path='./masks/light-masks/'):
        self.masks_path = masks_path
        self.masks_list = [mask for mask in sorted(os.listdir(masks_path)) if ".jpg" in mask]
        self.masks = np.array([cv2.imread(os.path.join(masks_path, mask), cv2.IMREAD_UNCHANGED) for mask in self.masks_list])
        self.masking_prob = masking_prob

    def __call__(self, image, label=None, **kwargs):
        if np.random.rand(1) < self.masking_prob:
            choice_index = np.random.choice(np.arange(self.masks.shape[0]), 1, replace=False)[0]
            mask = self.masks[choice_index]

            # downsize mask
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
            mask_gray_full = np.zeros_like(mask_resized)
            mask_gray_full[:, :, 0] = mask_gray
            mask_gray_full[:, :, 1] = mask_gray
            mask_gray_full[:, :, 2] = mask_gray

            # increase mask contrast
            mask_gray_full = cv2.cvtColor(mask_gray_full, cv2.COLOR_BGR2RGB)
            pil_mask = PIL.Image.fromarray(mask_gray_full)
            contrast_mask = ImageEnhance.Contrast(pil_mask).enhance(2)
            contrast_mask = cv2.cvtColor(np.array(contrast_mask), cv2.COLOR_RGB2BGR)

            ############# for bright spots only #############
            res = cv2.addWeighted(image[:, :, :3], 1, contrast_mask, 0.7, 0)
            return np.concatenate((res, image[:, :, 3:]), axis=2), label
        else:
            return image, label


class RandomFlare(object):

    def __init__(self, flare_prob, masks_path='./masks/lens-flare/', label_num=-1):
        self.masks_path = masks_path
        self.masks_list = [mask for mask in sorted(os.listdir(masks_path)) if ".png" in mask]
        self.masks = np.array(
            [cv2.imread(os.path.join(masks_path, mask), cv2.IMREAD_UNCHANGED) for mask in self.masks_list])
        self.flare_prob = flare_prob
        self.label_num = label_num

    @staticmethod
    def calculate_flare_mask(flare):
        (T, thresh) = cv2.threshold(flare, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        T2, thresh2 = cv2.threshold(flare, T + 130, 255, cv2.THRESH_BINARY)
        return thresh2

    def __call__(self, image, label=None, **kwargs):
        if np.random.rand(1) < self.flare_prob:
            choice_index = np.random.choice(np.arange(self.masks.shape[0]), 1, replace=False)[0]
            mask = self.masks[choice_index]

            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)

            # add flare by brightening all rgb channels based on grayscale flare brightness
            flare = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            r, g, b = cv2.split(image[:, :, :3].astype(np.uint16))
            r = np.minimum(r + flare, 255)
            g = np.minimum(g + flare, 255)
            b = np.minimum(b + flare, 255)
            if self.label_num >= 0:
                mask = self.calculate_flare_mask(flare)
                label[mask] = self.label_num
            return np.concatenate((np.stack((r, g, b), axis=2).astype(np.uint8), image[:, :, 3:]), axis=2), label
        else:
            return image, label


if __name__ == '__main__':
    test_image = cv2.imread("augmented_shadows/000000_augshadow.jpg", cv2.IMREAD_UNCHANGED)
    # cv2.imshow("test_image", test_image)
    # cv2.waitKey(0)
    augmentation = RandomFlare(1.0)
    res = augmentation(test_image, None)
    cv2.imshow("res", res[0])
    cv2.waitKey(0)
