import torch
import torch.nn as nn
import numpy as np

import cv2
import PIL.Image
from PIL import ImageEnhance, ImageOps
import os


class RandomShadow(object):

    def __init__(self, masking_multiplier, masks_path='./masks/'):
        self.masks_path = masks_path
        self.masks_list = [mask for mask in sorted(os.listdir(masks_path)) if ".jpg" in mask]
        self.masks = np.array([cv2.imread(os.path.join(masks_path, mask), cv2.IMREAD_UNCHANGED) for mask in self.masks_list])
        self.masking_multiplier = masking_multiplier

    def __call__(self, images, **kwargs):
        updated_shape = list(images.shape)
        updated_shape[0] *= self.masking_multiplier
        output_images = np.zeros(updated_shape, dtype=images[0].dtype)

        for i, image in enumerate(images):
            choice_indices = np.random.choice(np.arange(self.masks.shape[0]), self.masking_multiplier, replace=False)
            chosen_masks = self.masks[choice_indices]

            for j, mask in enumerate(chosen_masks):
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
                res = cv2.addWeighted(image, 0.5, contrast_mask, 0.5, -60)
                output_images[i*self.masking_multiplier+j] = res
        return output_images


class RandomLight(object):

    def __init__(self, masking_multiplier, masks_path='./masks/'):
        self.masks_path = masks_path
        self.masks_list = [mask for mask in sorted(os.listdir(masks_path)) if ".jpg" in mask]
        self.masks = np.array([cv2.imread(os.path.join(masks_path, mask), cv2.IMREAD_UNCHANGED) for mask in self.masks_list])
        self.masking_multiplier = masking_multiplier

    def __call__(self, images, **kwargs):
        updated_shape = list(images.shape)
        updated_shape[0] *= self.masking_multiplier
        output_images = np.zeros(updated_shape, dtype=images[0].dtype)

        for i, image in enumerate(images):
            choice_indices = np.random.choice(np.arange(self.masks.shape[0]), 2, replace=False)
            chosen_masks = self.masks[choice_indices]

            for j, mask in enumerate(chosen_masks):
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
                res = cv2.addWeighted(image, 1, contrast_mask, 0.7, 0)
                output_images[i*self.masking_multiplier+j] = res
        return output_images


if __name__ == '__main__':
    images_path = '../customdata/rgb/'
    test_image = cv2.imread("augmented_shadows/000000_augshadow.jpg", cv2.IMREAD_UNCHANGED)
    cv2.imshow("test_image", test_image)
    cv2.waitKey(0)
    augmentation = RandomLight(2)
    res = augmentation(np.array([test_image]))
    cv2.imshow("res0", res[0])
    cv2.imshow("res1", res[1])
    cv2.waitKey(0)
