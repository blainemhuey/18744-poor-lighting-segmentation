import os
from glob import glob
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset

import numpy as np
from PIL import Image
import cv2

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MFNetDataset(Dataset):
    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640, transform=None, label_map=None):
        super(MFNetDataset, self).__init__()

        if transform is None:
            transform = []
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'

        with open(os.path.join(data_dir, split + '.txt'), 'r') as f:
            self.names = sorted([name.strip() for name in f.readlines()])

        self.data_dir = data_dir
        self.split = split
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.is_train = have_label
        self.n_data = len(self.names)
        self.label_map = label_map

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image_pil = Image.open(file_path)
        #image_pil.load()
        image = np.asarray(image_pil).copy()  # (w,h,c)
        image_pil.close()
        return image

    def get_train_item(self, index):
        name = self.names[index]
        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')

        if self.label_map is not None:
            label = np.array(self.label_map)[label]

        for func in self.transform:
            image, label = func(image, label)

        return image.float(), label.long()

    def get_test_item(self, index):
        name = self.names[index]
        image = self.read_image(name, 'images')
        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255

        return torch.tensor(image)

    def __getitem__(self, index):
        if self.is_train is True:
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def __len__(self):
        return self.n_data


class HeatNetDataset(Dataset):
    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640, transform=None, label_map=None):
        super(HeatNetDataset, self).__init__()

        if transform is None:
            transform = []

        fl_rgb_files = glob(os.path.join(data_dir, '*/*/fl_rgb/*.png'))
        fl_label_files = glob(os.path.join(data_dir, '*/*/fl_rgb_labels/*.png'))
        fl_ir_files = glob(os.path.join(data_dir, '*/*/fl_ir_aligned/*.png'))

        fl_rgb_files.sort()
        fl_ir_files.sort()
        fl_label_files.sort()

        if have_label:
            files = list(fl_label_files)
            if split == 'train':
                names = filter(lambda x: "seq_04" not in x, files)
            else:
                names = filter(lambda x: "seq_04" in x, files)
            self.names = [(name.replace('fl_rgb_labels', 'fl_rgb'), name.replace('fl_rgb_labels', 'fl_ir_aligned'), name) for name in names
                          if os.path.exists(name.replace('fl_rgb_labels', 'fl_rgb')) and os.path.exists(name.replace('fl_rgb_labels', 'fl_ir_aligned'))]
        else:
            self.names = [(file, file.replace('fl_rgb', 'fl_ir_aligned')) for file in fl_rgb_files
                          if os.path.exists(file.replace('fl_rgb', 'fl_ir_aligned'))]

        self.data_dir = data_dir
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.is_train = have_label
        self.n_data = len(self.names)
        self.label_map = label_map

    def rescale_ir(self, ir):
        minval = 21800
        maxval = 25000
        ir[ir < minval] = minval
        ir[ir > maxval] = maxval
        return (ir - minval) / (maxval - minval)

    def read_image(self, path):
        image_pil = Image.open(path)
        image_pil.load()
        image = np.asarray(image_pil).copy()  # (w,h,c)
        image_pil.close()
        return image

    def get_train_item(self, rgb, ir, label):
        rgb_image = self.read_image(rgb)
        ir_image = self.read_image(ir)
        label_image = self.read_image(label)
        label_image = cv2.resize(label_image, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        ir_array = np.asarray(ir_image)[:480, 640:1280]
        rgb_array = np.asarray(rgb_image)[:480, 640:1280, :]
        #print(label_image.shape)
        label_array = np.asarray(label_image)[:480, 640:1280]
        flir_array = (self.rescale_ir(ir_array) * 255).astype('uint8')
        merged_array = np.stack((rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2], flir_array[:, :]), axis=2)
        return merged_array, label_array

    def get_test_item(self, rgb, ir):
        rgb_image = self.read_image(rgb)
        ir_image = self.read_image(ir)
        ir_array = np.asarray(ir_image)[:480, 640:1280]
        rgb_array = np.asarray(rgb_image)[:480, 640:1280, :]
        flir_array = (self.rescale_ir(ir_array) * 255).astype('uint8')
        merged_array = np.stack((rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2], flir_array[:, :]), axis=2)
        return merged_array

    def __getitem__(self, index):
        if self.is_train:
            image, label = self.get_train_item(*self.names[index])
            if self.label_map is not None:
                label = np.array(self.label_map)[label]
            for func in self.transform:
                image, label = func(image, label)
            return image.float(), label.long()
        else:
            image = self.get_train_item(*self.names[index])
            for func in self.transform:
                image = func(image)
            return image

    def __len__(self):
        return self.n_data


class CustomDataset(Dataset):
    def __init__(self, data_dir, split, have_label, transform=None, label_map=None):
        super().__init__()

        if transform is None:
            transform = []

        data_path = Path(data_dir)
        if have_label:
            raw_data = [(name, data_path / "labels" / "Default" / name.name)
                        for name in sorted((data_path / "rgba").glob("*")) if (data_path / "labels" / "Default" / name.name).exists()]
            if split == 'train':
                self.data = raw_data[:int(len(raw_data)*0.8)]
            else:
                self.data = raw_data[int(len(raw_data)*0.8):]
            self.n_data = len(self.data)
        else:
            self.data = list((data_path / "rgba").glob("*"))
            self.n_data = len(self.data)

        self.data_dir = data_dir
        self.transform = transform
        self.is_train = have_label
        self.label_map = label_map

    def read_image(self, path):
        image_pil = Image.open(path)
        image_pil.load()
        image = np.asarray(image_pil).copy()  # (w,h,c)
        image_pil.close()
        return image

    def get_train_item(self, rgba, label):
        image = self.read_image(rgba)
        label = self.read_image(label)

        if self.label_map is not None:
            label = np.array(self.label_map)[label]

        for func in self.transform:
            image, label = func(image, label)

        return image.float(), label.long()

    def get_test_item(self, rgba):
        image = self.read_image(rgba)
        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255

        return torch.tensor(image)

    def __getitem__(self, index):
        if self.is_train:
            return self.get_train_item(*self.data[index])
        else:
            return self.get_test_item(*self.data[index])

    def __len__(self):
        return self.n_data