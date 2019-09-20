from __future__ import print_function, division
import os
import math
import datetime

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import lmdb
import diagonal_crop
from PIL import Image

columns = ['uuid', 'name', 'nation', 'gender', 'year', 'month', 'day', 'address', 'id', 'psb', 'validity']

CARD_WIDTH = 445
CARD_HEIGHT = 280
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1000
DOTS_THRESHOLD = 5
PADDING = 5

ADDRESS_LINE = 4
ADDRESS_CHAR_WIDTH = 17
ADDRESS_CHAR_PER_LINE = 12
ADDRESS_CHAR_SUM_THRESHOLD = 10000

PSB_LINE = 2
PSB_CHAR_WIDTH = 17.2
PSB_CHAR_PER_LINE = 12
PSB_CHAR_SUM_THRESHOLD = 10000

shape_map = {}

name_x = (88, 176)
name_y = (46, 68)
shape_map['name'] = (name_y[1]-name_y[0], name_x[1]-name_x[0])

nation_x = (188, 276)
nation_y = (78, 100)
shape_map['nation'] = (nation_y[1]-nation_y[0], nation_x[1]-nation_x[0])

gender_x = (88, 132)
gender_y = (78, 100)
shape_map['gender'] = (gender_y[1]-gender_y[0], gender_x[1]-gender_x[0])

year_x = (88, 132)
year_y = (110, 132)
shape_map['year'] = (year_y[1]-year_y[0], year_x[1]-year_x[0])

month_x = (150, 175)
month_y = (110, 132)
shape_map['month'] = (month_y[1]-month_y[0], month_x[1]-month_x[0])

day_x = (200, 225)
day_y = (110, 132)
shape_map['day'] = (day_y[1]-day_y[0], day_x[1]-day_x[0])

address_x = (88, 292)
address_y_set = [142, 164, 186, 208, 230]
shape_map['address'] = (address_y_set[1]-address_y_set[0], (address_x[1]-address_x[0])*4)

id_x = (132, 380)
id_y = (226, 252)
shape_map['id'] = (id_y[1]-id_y[0], id_x[1]-id_x[0])

psb_x = (172, 378)
psb_y_set = [196, 218, 240]
shape_map['psb'] = (psb_y_set[1]-psb_y_set[0], (psb_x[1]-psb_x[0])*2)

validity_x = (170, 362)
validity_y = (232, 252)
shape_map['validity'] = (validity_y[1]-validity_y[0], validity_x[1]-validity_x[0])


def _boundy(revert_img):
    dots_per_row = np.apply_along_axis(lambda col: np.sum(col) // 255, 1, revert_img)
    y = PADDING
    was_blank = True
    y1, y2, y3, y4 = 0, 0, 0, 0
    while y < IMAGE_HEIGHT - PADDING:
        if dots_per_row[y] >= DOTS_THRESHOLD:
            if y1 == 0:
                y1 = y
                y += CARD_HEIGHT
            elif y1 != 0 and y2 != 0 and y3 == 0:
                y3 = y
                y += CARD_HEIGHT
            was_blank = False
        elif not was_blank and dots_per_row[y] < DOTS_THRESHOLD:
            if y2 == 0:
                y2 = y
                y += PADDING
            elif y2 != 0 and y4 == 0:
                y4 = y
                break
            was_blank = True
        y += 1

    img_top = revert_img[y1:y2, :]
    if y3 == 0:
        y4 = y2
        y2 = y1 + img_top.shape[0] // 2
        y3 = y2

#     img_bottom = revert_img[y3:y4, :]

    return y1, y2, y3, y4


def _boundx(img):
    dots_per_col = np.apply_along_axis(lambda row: np.sum(row) // 255, 0, img)
    x1, x2 = 0, 0
    x = PADDING
    was_blank = True
    while x < IMAGE_WIDTH - PADDING:
        if dots_per_col[x] > DOTS_THRESHOLD:
            if x1 == 0:
                x1 = x
                x += CARD_WIDTH
            was_blank = False
        elif not was_blank and dots_per_col[x] <= DOTS_THRESHOLD:
            if x2 == 0:
                x2 = x
                break
        x += 1
    return x1, x2


def bound(img):
    y1, y2, y3, y4 = _boundy(img)
    img_top = img[y1:y2, :]
    img_bottom = img[y3:y4, :]
    x1, x2 = _boundx(img_top)
    x3, x4 = _boundx(img_bottom)
    return x1, x2, x3, x4, y1, y2, y3, y4


def _crop(img, a, b, x, y, theta):
    if theta < 0:
        x_s, y_s = x + b * math.sin(-theta), y  # clockwise
    else:
        x_s, y_s = x, y + a * math.sin(theta)  # counterclockwise
    return np.asarray(diagonal_crop.crop(Image.fromarray(img), (x_s, y_s), theta, b, a))


def crop(img, sub_img, top_left):
    a, b = CARD_WIDTH, CARD_HEIGHT
    b2, a2 = sub_img.shape
    x, y = top_left
    cos_theta = (2*b*b2 + math.sqrt(4*b*b*b2*b2 - 4*(a*a+b*b)*(b2*b2-a*a))) / (2*a*a+2*b*b)
    theta = math.acos(cos_theta)

    ccw_sub_img = _crop(sub_img, a, b, 0, 0, theta)
    cw_sub_img = _crop(sub_img, a, b, 0, 0, -theta)

    if np.sum(ccw_sub_img) < np.sum(cw_sub_img):
        return _crop(img, a, b, x, y, theta)
    else:
        return _crop(img, a, b, x, y, -theta)


def top_rotate(img):
    # we use inversed to detect whether we should rotate image
    img_inv = cv2.bitwise_not(img)
    h, w = img.shape
    half_h, half_w = h//2, w//2
    address_1 = img_inv[address_y_set[0]:address_y_set[1], address_x[0]:address_x[1]]
    address_2 = img_inv[h-address_y_set[1]:h-address_y_set[0], w-address_x[1]:w-address_x[0]]
    id_1 = img_inv[id_y[0]:id_y[1], id_x[0]:id_x[1]]
    id_2 = img_inv[h-id_y[1]:h-id_y[0], w-id_x[1]:w-id_x[0]]
    if (np.sum(address_2) + np.sum(id_2)) > (np.sum(address_1) + np.sum(id_1)):
        matrix = cv2.getRotationMatrix2D((half_w, half_h), 180, 1.0)
        return cv2.warpAffine(img, matrix, (w, h))
    return img


def bottom_rotate(img):
    img_inv = cv2.bitwise_not(img)
    h, w = img.shape
    half_h, half_w = h//2, w//2
    # we use inversed to detect which
    top = img_inv[0:half_h, :]
    bottom = img_inv[half_h: h, :]
    if np.sum(bottom) > np.sum(top):
        matrix = cv2.getRotationMatrix2D((half_w, half_h), 180, 1.0)
        return cv2.warpAffine(img, matrix, (w, h))
    return img


def crop_address(address_img):
    address_img_inv = cv2.bitwise_not(address_img)
    for i in range(ADDRESS_CHAR_PER_LINE * ADDRESS_LINE):
        s = np.sum(address_img_inv[:,i*ADDRESS_CHAR_WIDTH:(i+1)*ADDRESS_CHAR_WIDTH])
        if s < ADDRESS_CHAR_SUM_THRESHOLD:
            break
    return address_img[:, 0:i*ADDRESS_CHAR_WIDTH]


def crop_psb(psb_img):
    psb_img_inv = cv2.bitwise_not(psb_img)
    for i in range(PSB_CHAR_PER_LINE * PSB_LINE):
        s = np.sum(psb_img_inv[:,int(i*PSB_CHAR_WIDTH):int((i+1)*PSB_CHAR_WIDTH)])
        if s < PSB_CHAR_SUM_THRESHOLD:
            break
    return psb_img[:, 0:int(i*PSB_CHAR_WIDTH)]


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if isinstance(v, str):
                txn.put(k.encode(), v.encode())
            elif isinstance(v, np.ndarray):
                txn.put(k.encode(), v.tobytes())


def create_dataset(db_path, image_folder, image_result_folder, label_path):
    for f in columns:
        path = os.path.join(image_result_folder, f)
        if not os.path.exists(path):
            os.makedirs(path)

    if not os.path.exists(db_path):
        os.makedirs(db_path)

    env = lmdb.open(db_path, map_size=1099511627776)
    label_df = pd.read_csv(label_path, header=None, names=columns, dtype='str')
    idx = label_df.address.str.len().sort_values().index
    label_df = label_df.reindex(idx).reset_index(drop=True)
    count = label_df.shape[0]
    cache = {}
    for i, row in label_df.iterrows():
        image_id = row['uuid']
        img = cv2.imread(f'{image_folder}{image_id}.jpg', 0)
        thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1.0)
        thr2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 3.0)
        revert_img = cv2.bitwise_not(thr)

        x1, x2, x3, x4, y1, y2, y3, y4 = bound(revert_img)
#         print(x1, x2, x3, x4, y1, y2, y3, y4)
        img_top = thr[y1:y2, x1:x2]
        img_bottom = thr[y3:y4, x3:x4]

        img_top = crop(thr2, img_top, (x1, y1))
        img_top = top_rotate(img_top)

        img_bottom = crop(thr2, img_bottom, (x3, y3))
        img_bottom = bottom_rotate(img_bottom)

        images = {}
        images['name'] = img_top[name_y[0]:name_y[1], name_x[0]:name_x[1]]
        images['nation'] = img_top[nation_y[0]:nation_y[1], nation_x[0]:nation_x[1]]
        images['gender'] = img_top[gender_y[0]:gender_y[1], gender_x[0]:gender_x[1]]
        images['year'] = img_top[year_y[0]:year_y[1], year_x[0]:year_x[1]]
        images['month'] = img_top[month_y[0]:month_y[1], month_x[0]:month_x[1]]
        images['day'] = img_top[day_y[0]:day_y[1], day_x[0]:day_x[1]]

        address_img = np.concatenate([img_top[address_y[0]:address_y[1], address_x[0]:address_x[1]]
                                      for address_y in zip(address_y_set[:-1], address_y_set[1:])], axis=1)
        # address_img = crop_address(address_img)
        images['address'] = address_img

        images['id'] = img_top[id_y[0]:id_y[1], id_x[0]:id_x[1]]

        psb_img = np.concatenate([img_bottom[psb_y[0]:psb_y[1], psb_x[0]:psb_x[1]]
                                    for psb_y in zip(psb_y_set[:-1], psb_y_set[1:])], axis=1)
        # psb_img = crop_psb(psb_img)
        images['psb'] = psb_img

        images['validity'] = img_bottom[validity_y[0]:validity_y[1], validity_x[0]:validity_x[1]]

        for k, img in images.items():
            label = row[k]
            if k == 'address':
                img = img[:, 0:int(len(label) * ADDRESS_CHAR_WIDTH)]
            elif k == 'psb':
                label = label[:PSB_CHAR_PER_LINE*PSB_LINE]  # sometimes we have three lines!
                img = img[:, 0:int(len(label) * PSB_CHAR_WIDTH)]
            cache[f'image-{k}-{i:06}'] = img
            cache[f'label-{k}-{i:06}'] = label
            path = os.path.join(image_result_folder, f'{k}/{image_id}.jpg')
            cv2.imwrite(path, img)

        if (i+1) % 100 == 0:
            write_cache(env, cache)
            cache = {}
            print(f'{datetime.datetime.now()} processed {i+1} / {count}')
    with env.begin(write=True) as txn:
        txn.put(b'num-samples', count.to_bytes((count.bit_length() + 7) // 8, byteorder='big'))


if __name__ == "__main__":
    db_path = './ocr-db/'
    image_folder = './data/Train_DataSet/'
    image_result_folder = './ocr-images/'
    label_path = './data/Train_Labels.csv'
    create_dataset(db_path, image_folder, image_result_folder, label_path)
