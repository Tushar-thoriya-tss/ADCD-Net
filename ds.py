import albumentations as A
import albumentations.augmentations.crops.functional as F
import cv2
import numpy as np
import os.path as op
import pickle
import tempfile
import torch
import torchvision.transforms as T
from PIL import Image
from albumentations import CropNonEmptyMaskIfExists
from albumentations.pytorch import ToTensorV2
from copy import deepcopy
from jpeg2dct.numpy import load
from random import randint, random
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import cfg


def load_qt(qt_path):
    with open(qt_path, 'rb') as fpk:
        pks_ = pickle.load(fpk)
    pks = {}
    for k, v in pks_.items():
        pks[k] = torch.LongTensor(v)
    return pks


def multi_jpeg(img, num_jpeg, min_qf, upper_bound, jpeg_record=None):
    with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp:
        img = img.convert("L")
        im_ori = img.copy()
        qf_record = []
        if jpeg_record is not None:
            num_jpeg = len(jpeg_record)
        for each_jpeg in range(num_jpeg):
            if jpeg_record is not None:
                qf = jpeg_record[each_jpeg]
            else:
                qf = randint(min_qf, upper_bound)
            qf_record.append(qf)
            img.save(tmp.name, "JPEG", quality=int(qf))
            img.close()
            img = Image.open(tmp.name)

        img = Image.open(tmp.name)
        img = img.convert('RGB')
        try:
            dct_y, _, _ = load(tmp.name, normalized=False)
        except:
            with tempfile.NamedTemporaryFile(delete=True) as tmp1:
                qf = 100
                qf_record = [100]
                im_ori.save(tmp1.name, "JPEG", quality=qf)
                img = Image.open(tmp1.name)
                img = img.convert('RGB')
                dct_y, _, _ = load(tmp1.name, normalized=False)

    rows, cols, _ = dct_y.shape
    dct = np.empty(shape=(8 * rows, 8 * cols))
    for j in range(rows):
        for i in range(cols):
            dct[8 * j: 8 * (j + 1), 8 * i: 8 * (i + 1)] = dct_y[j, i].reshape(8, 8)
    dct = np.int32(dct)
    return dct, img, qf_record


class AlignCrop(CropNonEmptyMaskIfExists):
    """Crop window snapped to the nearest 8x8 DCT grid (multiples of 8)."""
    def apply(self, img, crop_coords, **params):
        x_min, y_min, x_max, y_max = crop_coords
        x_diff = x_min % 8
        x_min, x_max = x_min - x_diff, x_max - x_diff
        y_diff = y_min % 8
        y_min, y_max = y_min - y_diff, y_max - y_diff
        return F.crop(img, x_min, y_min, x_max, y_max)


class NonAlignCrop(CropNonEmptyMaskIfExists):
    """Crop window deliberately NOT aligned to the 8x8 DCT grid."""
    def apply(self, img, crop_coords, **params):
        x_min, y_min, x_max, y_max = crop_coords
        h, w = img.shape[:2]
        x_diff = x_min % 8
        y_diff = y_min % 8

        if x_diff == 0 and y_diff == 0:
            if x_max < w:
                x_min += 1
                x_max += 1
            elif x_min > 0:
                x_min -= 1
                x_max -= 1
            if y_max < h:
                y_min += 1
                y_max += 1
            elif y_min > 0:
                y_min -= 1
                y_max -= 1

        return F.crop(img, x_min, y_min, x_max, y_max)


def get_align_aug():
    return A.Compose([
        AlignCrop(cfg.img_size, cfg.img_size, p=1),
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ], p=1),
        A.OneOf([
            A.Downscale(scale_range=(0.5, 0.99), p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.RandomToneCurve(p=1),
                A.Sharpen(p=1),
            ], p=1),
        ], p=0.5)
    ], p=1, additional_targets={'ocr_mask': 'mask'})


def get_non_align_aug():
    return A.Compose([
        # scale_limit=(0, 0.5): only upscale (factor 1.0..1.5).
        # Downscaling is disabled to guarantee the post-scale image is never
        # smaller than img_size on any side.
        A.RandomScale(scale_limit=(0, 0.5), p=0.5),
        NonAlignCrop(cfg.img_size, cfg.img_size, p=1),
        A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ], p=1),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), sigma_limit=(0.5, 0.9), p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.ISONoise(p=1),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.RandomToneCurve(p=1),
                A.Sharpen(p=1),
            ], p=0.5),
        ], p=0.5)
    ], p=1, additional_targets={'ocr_mask': 'mask'})


img_totsr = T.Compose([T.ToTensor(),
                       T.Normalize(mean=(0.485, 0.455, 0.406),
                                   std=(0.229, 0.224, 0.225))])

mask_totsr = ToTensorV2()


class TrainDs(Dataset):
    """Reads (img, mask, ocr) triplets from <pkl_dir>/<train_ds_name>.pkl."""
    def __init__(self):
        pkl_path = op.join(cfg.pkl_dir, f'{cfg.train_ds_name}.pkl')
        with open(pkl_path, 'rb') as f:
            self.path_list = pickle.load(f)
        if len(self.path_list) == 0:
            raise RuntimeError(f'Empty training pkl: {pkl_path}')

        self.qts = load_qt(cfg.qt_path)

        self.S = cfg.init_S
        self.T = cfg.step_per_epoch
        self.ds_len = cfg.ds_len

        self.align_aug = get_align_aug()
        self.non_align_aug = get_non_align_aug()
        self.mask_totsr = mask_totsr
        self.img_totsr = img_totsr

    def __len__(self):
        return self.ds_len

    def __getitem__(self, _):
        idx = randint(0, len(self.path_list) - 1)
        img_path, mask_path, ocr_path = self.path_list[idx]
        img_name = op.basename(img_path).split('.')[0]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(mask_path, 0) != 0).astype(np.uint8)
        ocr_mask = (cv2.imread(ocr_path, 0) != 0).astype(np.uint8)

        if random() > 0.5:
            aug_func = self.align_aug
            is_align = True
        else:
            aug_func = self.non_align_aug
            is_align = False

        aug_out = aug_func(image=img, mask=mask, ocr_mask=ocr_mask)
        img, mask, ocr_mask = aug_out['image'], aug_out['mask'], aug_out['ocr_mask']
        img = Image.fromarray(img)

        min_qf = max(int(round(100 - (self.S / self.T))), 75)
        num_jpeg = randint(1, 3)

        dct, img, qfs = multi_jpeg(deepcopy(img),
                                   num_jpeg=num_jpeg,
                                   min_qf=min_qf,
                                   upper_bound=100)

        qf = qfs[-1]
        qt = self.qts[qf]
        img = self.img_totsr(img)
        mask = self.mask_totsr(image=mask.copy())['image']
        ocr_mask = self.mask_totsr(image=ocr_mask.copy())['image']

        return {
            'img': img,
            'dct': np.clip(np.abs(dct), 0, 20),
            'qt': qt,
            'mask': mask.long(),
            'ocr_mask': ocr_mask.long(),
            'img_name': img_name,
            'min_qf': min_qf,
            'is_align': is_align
        }


class ValDs(Dataset):
    """Reads (img, mask, ocr) triplets from <pkl_dir>/<ds_name>.pkl for evaluation."""
    def __init__(self, ds_name, is_sample=False):
        pkl_path = op.join(cfg.pkl_dir, f'{ds_name}.pkl')
        with open(pkl_path, 'rb') as f:
            self.path_list = pickle.load(f)

        self.sample_n = len(self.path_list)
        if is_sample:
            self.sample_n = cfg.val_sample_n

        self.qts = load_qt(cfg.qt_path)
        self.mask_totsr = mask_totsr
        self.img_totsr = img_totsr

        self.resize_func = A.Compose(
            [A.LongestMaxSize(cfg.val_max_size, p=1.0)],
            additional_targets={'mask2': 'mask'}
        )

    def __len__(self):
        return self.sample_n

    def __getitem__(self, index):
        img_path, mask_path, ocr_path = self.path_list[index]
        img_name = op.basename(img_path).split('.')[0]

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        mask = (cv2.imread(mask_path, 0) != 0).astype(np.uint8)
        ocr_mask = (cv2.imread(ocr_path, 0) != 0).astype(np.uint8)

        if h > cfg.val_max_size or w > cfg.val_max_size:
            aug = self.resize_func(image=img, mask=mask, mask2=ocr_mask)
            img, mask, ocr_mask = aug['image'], aug['mask'], aug['mask2']

        img = Image.fromarray(img)

        dct, img, qfs = multi_jpeg(deepcopy(img),
                                   num_jpeg=-1,
                                   min_qf=-1,
                                   upper_bound=-1,
                                   jpeg_record=[100])
        qt = self.qts[qfs[-1]]
        img = self.img_totsr(img)
        ori_img = np.array(img)
        mask = self.mask_totsr(image=mask.copy())['image']
        ocr_mask = self.mask_totsr(image=ocr_mask.copy())['image']

        return {
            'img': img,
            'dct': np.clip(np.abs(dct), 0, 20),
            'qt': qt.clamp(0, 63),
            'mask': mask.long(),
            'ocr_mask': ocr_mask.long(),
            'img_name': img_name,
            'ori_img': ori_img,
        }


def get_train_dl(world_size, rank, dp=False):
    ds = TrainDs()
    sampler = DistributedSampler(dataset=ds, num_replicas=world_size, rank=rank, shuffle=True) if not dp else None
    dl = DataLoader(dataset=ds, batch_size=cfg.train_bs, num_workers=cfg.dl_workers, sampler=sampler)
    return dl


def pad_collate(batch, pad_value=0.0, mask_ignore_index=-1):
    imgs = [item['img'] for item in batch]
    masks = [item['mask'] for item in batch]
    ocr_masks = [item['ocr_mask'] for item in batch]
    img_names = [item['img_name'] for item in batch]
    dcts = [item['dct'] for item in batch]
    qts = [item['qt'] for item in batch]

    sizes = torch.tensor([[im.shape[-2], im.shape[-1]] for im in imgs], dtype=torch.long)

    H_max = int(sizes[:, 0].max())
    W_max = int(sizes[:, 1].max())

    divide_by = 16
    if H_max % divide_by != 0:
        H_max = (H_max // divide_by + 1) * divide_by
    if W_max % divide_by != 0:
        W_max = (W_max // divide_by + 1) * divide_by
    H_max = W_max = max(H_max, W_max)

    padded_imgs, padded_masks, padded_ocr_masks, padded_dcts = [], [], [], []
    for im, m, ocr_m, dct, qt in zip(imgs, masks, ocr_masks, dcts, qts):
        C, H, W = im.shape
        im_p = torch.nn.functional.pad(im, (0, W_max - W, 0, H_max - H), value=pad_value)
        padded_imgs.append(im_p)

        C, H, W = m.shape
        m_p = torch.nn.functional.pad(m, (0, W_max - W, 0, H_max - H), value=mask_ignore_index)
        padded_masks.append(m_p)

        C, H, W = ocr_m.shape
        ocr_m_p = torch.nn.functional.pad(ocr_m, (0, W_max - W, 0, H_max - H), value=mask_ignore_index)
        padded_ocr_masks.append(ocr_m_p)

        H, W = dct.shape
        dct_p = torch.nn.functional.pad(torch.tensor(dct), (0, W_max - W, 0, H_max - H), value=pad_value)
        padded_dcts.append(dct_p)

    b = 1
    if len(padded_imgs) < b:
        b_diff = b - len(padded_imgs)
        for _ in range(b_diff):
            padded_imgs.append(torch.full((3, H_max, W_max), fill_value=pad_value))
            padded_masks.append(torch.full((1, H_max, W_max), fill_value=0, dtype=torch.long))
            padded_ocr_masks.append(torch.full((1, H_max, W_max), fill_value=0, dtype=torch.long))
            padded_dcts.append(torch.full((H_max, W_max), fill_value=0, dtype=torch.long))
            qts.append(torch.full((8, 8), fill_value=1, dtype=torch.long))
            img_names.append('padding')
            sizes = torch.cat([sizes, torch.tensor([[H_max, W_max]], dtype=torch.long)], dim=0)

    batch_imgs = torch.stack(padded_imgs)
    batch_masks = torch.stack(padded_masks)
    batch_ocr_masks = torch.stack(padded_ocr_masks)
    batch_dcts = torch.stack(padded_dcts)
    batch_qts = torch.stack(qts)

    return batch_imgs, batch_dcts, batch_qts, batch_masks, batch_ocr_masks, list(img_names), sizes


def get_val_dl(world_size, rank, dp=False):
    dl_list = {}
    for val_name in cfg.val_name_list:
        is_sample = False
        if val_name.endswith('_sample'):
            val_name = val_name[:-len('_sample')]
            is_sample = True

        ds = ValDs(val_name, is_sample)
        sampler = DistributedSampler(dataset=ds, num_replicas=world_size, rank=rank, shuffle=False) if not dp else None
        dl = DataLoader(dataset=ds, batch_size=cfg.val_bs, num_workers=cfg.dl_workers,
                        sampler=sampler, collate_fn=pad_collate)
        dl_list[val_name] = dl

    return dl_list
