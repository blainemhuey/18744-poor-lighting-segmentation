import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


class FusedDataset(Dataset):
    def __init__(self, data_dir, split, have_label, input_h=480, input_w=640, transform=None):
        super(FusedDataset, self).__init__()

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

    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s.png' % (folder, name))
        image_pil = Image.open(file_path)
        image_pil.load()
        image = np.asarray(image_pil).copy()  # (w,h,c)
        image_pil.close()
        return image

    def get_train_item(self, index):
        name = self.names[index]
        image = self.read_image(name, 'images')
        label = self.read_image(name, 'labels')

        for func in self.transform:
            image, label = func(image, label)

        # pil_image = Image.fromarray(image).resize((self.input_w, self.input_h))
        # image = np.asarray(pil_image, dtype=np.float32).copy()
        # image = image.permute((2, 0, 1))

        # pil_label = Image.fromarray(label).resize((self.input_w, self.input_h))
        # label = np.asarray(pil_label, dtype=np.int64).copy()

        return image, label.long(), name

    def get_test_item(self, index):
        name = self.names[index]
        image = self.read_image(name, 'images')
        image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose(
            (2, 0, 1)) / 255

        return torch.tensor(image), name

    def __getitem__(self, index):
        if self.is_train is True:
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def __len__(self):
        return self.n_data
