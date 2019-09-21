from __future__ import print_function, division
import os
import collections
import sys

import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import lmdb
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
# from warpctc_pytorch import CTCLoss

from create_dataset import *


class LmdbDataset(Dataset):
    def __init__(self, db_path, column, transform=None, target_transform=None):
        self.env = lmdb.open(
            db_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (db_path))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.count = int.from_bytes(txn.get(b'num-samples'), byteorder='big')

        self.column = column
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        assert index < len(self), 'index range error'
        column = self.column
        with self.env.begin(write=False) as txn:
            label_key = f'label-{column}-{index:06}'
            label = txn.get(label_key.encode()).decode()

            image_key = f'image-{column}-{index:06}'
            img_byte = txn.get(image_key.encode())
            shape = shape_map[column]
            if column == 'address':
                shape = (address_y_set[1]-address_y_set[0], int(ADDRESS_CHAR_WIDTH * len(label)))
            elif column == 'psb':
                shape = (psb_y_set[1]-psb_y_set[0], int(PSB_CHAR_WIDTH * len(label)))
            img = np.frombuffer(img_byte, dtype=np.dtype('uint8')).reshape(shape)
            if self.transform is not None:
                img = self.transform(img)

            label_key = f'label-{column}-{index:06}'
            label = txn.get(label_key.encode()).decode()

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, height):
        assert isinstance(height, int)
        self.height = height

    def __call__(self, image):
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = self.height * h / w, self.height
        else:
            new_h, new_w = self.height, self.height * w / h

        new_h, new_w = int(new_h), int(new_w)

        return cv2.resize(image, (new_w, new_h))


class Normalize(object):
    def __call__(self, image):
        return image.sub_(0.5).div_(0.5)


class LabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
