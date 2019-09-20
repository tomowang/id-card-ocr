import json

import cv2
from torchvision import transforms
import torch
from torch.autograd import Variable
from dataset import LabelConverter, Rescale, Normalize

from model import CRNN

IMAGE_HEIGHT = 32

model_path = './ocr-model/crnn_address.pth'
img_path = './ocr_address.jpg'
# alphabet = '0123456789X'
alphabet = alphabet = ''.join(json.load(open('./cn-alphabet.json', 'rb')))

model = CRNN(IMAGE_HEIGHT, 1, len(alphabet) + 1, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = LabelConverter(alphabet)

image_transform = transforms.Compose([
    Rescale(IMAGE_HEIGHT),
    transforms.ToTensor(),
    Normalize()
])
image = cv2.imread(img_path, 0)
image = image_transform(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
