import numpy as np
import cv2
import os
import PIL.Image
from PIL import ImageEnhance, ImageOps


images_path = 'customdata/rgb/'
images_list = [image for image in sorted(os.listdir(images_path))]
images_len = len(images_list)


def aug_bright(): 
    for image_ind, img in enumerate(images_list):
        if (image_ind == 0):
            print("augmenting image ", image_ind + 1, "out of ", images_len)
            res_name = img[:-4] + '_aug.jpg'
            # paths
            masks_path ='customdata/masks/'
            masks_list = [mask for mask in sorted(os.listdir(masks_path)) if ".jpg" in mask]
            msk = masks_path + masks_list[image_ind % len(masks_list)]
            print(msk)
            img = images_path + img

            image = cv2.imread(img, cv2.IMREAD_UNCHANGED)

            mask = cv2.imread(msk, cv2.IMREAD_UNCHANGED)

            # downsize mask
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_AREA)
            mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
            mask_gray_full = np.zeros_like(mask_resized)
            mask_gray_full[:,:,0] = mask_gray
            mask_gray_full[:,:,1] = mask_gray
            mask_gray_full[:,:,2] = mask_gray

            # increase mask contrast 
            mask_gray_full = cv2.cvtColor(mask_gray_full, cv2.COLOR_BGR2RGB)
            pil_mask = PIL.Image.fromarray(mask_gray_full)
            contrast_mask = ImageEnhance.Contrast(pil_mask).enhance(2)

            ############# for bright spots only #############
            contrast_mask = cv2.cvtColor(np.array(contrast_mask), cv2.COLOR_RGB2BGR)    
            res = cv2.addWeighted(image, 1, contrast_mask, 0.7, 0)
            ###############################################

            cv2.imshow("res", res)
            cv2.waitKey(0)
            # print(res_name)
            res_path = 'customdata/augmented/'
            cv2.imwrite(os.path.join(res_path,res_name), res)

            print("done!")

def aug_shadow():
    for image_ind, img in enumerate(images_list):
        if (image_ind == 0):
            print("augmenting image ", image_ind + 1, "out of ", images_len)
            res_name = img[:-4] + '_augshadow.jpg'
            # paths
            masks_path ='customdata/masks/'
            masks_list = [mask for mask in sorted(os.listdir(masks_path)) if ".jpg" in mask]
            msk = masks_path + masks_list[image_ind % len(masks_list)]
            print(msk)
            img = images_path + img

            image = cv2.imread(img, cv2.IMREAD_UNCHANGED)

            mask = cv2.imread(msk, cv2.IMREAD_UNCHANGED)

            # downsize mask
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_AREA)
            mask_gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
            mask_gray_full = np.zeros_like(mask_resized)
            mask_gray_full[:,:,0] = mask_gray
            mask_gray_full[:,:,1] = mask_gray
            mask_gray_full[:,:,2] = mask_gray

            # increase mask contrast 
            mask_gray_full = cv2.cvtColor(mask_gray_full, cv2.COLOR_BGR2RGB)
            pil_mask = PIL.Image.fromarray(mask_gray_full)
            contrast_mask = ImageEnhance.Contrast(pil_mask).enhance(2)

            ############ for dark spots only #############
            contrast_mask = ImageOps.invert(contrast_mask)
            contrast_mask = cv2.cvtColor(np.array(contrast_mask), cv2.COLOR_RGB2GRAY)    
            ret,mask_thresh = cv2.threshold(contrast_mask,120,255,cv2.THRESH_TRUNC)

            contrast_mask = cv2.cvtColor(np.array(mask_thresh), cv2.COLOR_GRAY2BGR)

            # overlay dark spots
            res = cv2.addWeighted(image, 0.5, contrast_mask, 0.5, -60)
            ##############################################

            cv2.imshow("res", res)
            cv2.waitKey(0)
            # print(res_name)
            res_path = 'customdata/augmented_shadows/'
            cv2.imwrite(os.path.join(res_path,res_name), res)

            print("done!")


def add_flare():
    for image_ind, img in enumerate(images_list):
        if (image_ind < 20):
            print("augmenting image ", image_ind + 1, "out of ", images_len)
            res_name = img[:-4] + '_augflare.jpg'
            # paths
            masks_path ='./masks/lens-flare/simulated/'
            masks_list = [mask for mask in sorted(os.listdir(masks_path)) if ".png" in mask]
            msk = masks_path + masks_list[image_ind % len(masks_list)]
            img = images_path + img

            # read images and resize
            image = cv2.imread(img, cv2.IMREAD_UNCHANGED)

            mask = cv2.imread(msk, cv2.IMREAD_UNCHANGED)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_AREA)

            # add flare by brightening all rgb channels based on grayscale flare brightness
            flare = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            b, g, r = cv2.split(image.astype(np.uint16))
            b = np.minimum(b + flare, 255)
            g = np.minimum(g + flare, 255)
            r = np.minimum(r + flare, 255)

            res = cv2.merge((b, g, r)).astype(np.uint8)

            res_path = 'customdata/augmented_flares/'
            cv2.imwrite(os.path.join(res_path,res_name), res)

    print("done!")



add_flare()