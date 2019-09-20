from __future__ import print_function, division
import os
import string
import random
import json
import datetime

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss

from dataset import LmdbDataset, LabelConverter, Rescale, Normalize
from model import CRNN
import utils

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


IMAGE_HEIGHT = 32
BATCH_SIZE = 32
db_path = './ocr-db/'
model_path = './ocr-model/'
number_hidden = 256
nc = 1
lr = 0.0001  # learning rate


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(field):
    alphabet = ''.join(json.load(open('./cn-alphabet.json', 'rb')))
    nclass = len(alphabet) + 1  # add the dash -
    batch_size = BATCH_SIZE
    if field == 'address' or field == 'psb':
        batch_size = 1  # image length varies

    converter = LabelConverter(alphabet)
    criterion = CTCLoss(zero_infinity=True)

    crnn = CRNN(IMAGE_HEIGHT, nc, nclass, number_hidden)
    crnn.apply(weights_init)

    image_transform = transforms.Compose([
        Rescale(IMAGE_HEIGHT),
        transforms.ToTensor(),
        Normalize()
    ])

    dataset = LmdbDataset(db_path, field, image_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)

    image = torch.FloatTensor(batch_size, 3, IMAGE_HEIGHT, IMAGE_HEIGHT)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    loss_avg = utils.averager()
    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)

    if torch.cuda.is_available():
        crnn.cuda()
        crnn = nn.DataParallel(crnn)
        image = image.cuda()
        criterion = criterion.cuda()

    def train_batch(net, iteration):
        data = iteration.next()
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.load_data(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.load_data(text, t)
        utils.load_data(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        crnn.zero_grad()
        cost.backward()
        optimizer.step()
        return cost

    nepoch = 25
    for epoch in range(nepoch):
        train_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            cost = train_batch(crnn, train_iter)
            loss_avg.add(cost)
            i += 1

            if i % 500 == 0:
                print('%s [%d/%d][%d/%d] Loss: %f' %
                        (datetime.datetime.now(), epoch, nepoch, i, len(dataloader), loss_avg.val()))
                loss_avg.reset()

            # do checkpointing
            if i % 500 == 0:
                torch.save(
                    crnn.state_dict(), f'{model_path}crnn_{field}_{epoch}_{i}.pth')


if __name__ == '__main__':
    train('address')
