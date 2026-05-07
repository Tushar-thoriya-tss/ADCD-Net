import os
import os.path as op
import pickle
import cv2
from glob import glob
from tqdm import tqdm

import cfg


def main():

    root = '/sharedrive/Tushar Thoriya/GitHub techniques/DATAAAAASET'

    ds_name_list = ['aadhar_tamper']
    splits = ['train', 'val']
    mask_folder = 'mask'

    # Training crops at cfg.img_size x cfg.img_size at original resolution.
    # Skip any image smaller than this on either side so the crop always fits.
    # Val pipeline does not crop, so val keeps every image regardless of size.
    min_train_side = cfg.img_size

    pkl_dir = op.join(root, 'path_pkl')
    os.makedirs(pkl_dir, exist_ok=True)

    for ds_name in ds_name_list:
        for split in splits:
            split_dir = op.join(root, ds_name, split)
            img_dir = op.join(split_dir, 'images')
            mask_dir = op.join(split_dir, mask_folder)
            ocr_dir = op.join(split_dir, 'ocr')

            if not op.isdir(img_dir):
                print(f'[skip] images dir not found: {img_dir}')
                continue

            path_list = []
            n_too_small = 0
            n_missing = 0
            img_list = sorted(glob(op.join(img_dir, '*')))
            print(f'Building pkl for {ds_name}/{split}: {len(img_list)} images')

            for img_path in tqdm(img_list):
                img_name = op.basename(img_path)
                stem, _ = op.splitext(img_name)

                ocr_path = op.join(ocr_dir, img_name)
                mask_path = op.join(mask_dir, stem + '.png')

                if not op.exists(ocr_path) or not op.exists(mask_path):
                    n_missing += 1
                    continue

                if split == 'train':
                    img = cv2.imread(img_path)
                    if img is None:
                        n_missing += 1
                        continue
                    h, w = img.shape[:2]
                    if h < min_train_side or w < min_train_side:
                        n_too_small += 1
                        continue

                path_list.append((img_path, mask_path, ocr_path))

            save_path = op.join(pkl_dir, f'{ds_name}_{split}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(path_list, f)
            print(f'  wrote {len(path_list)} entries -> {save_path}')
            if n_too_small:
                print(f'  skipped {n_too_small} images smaller than {min_train_side}px')
            if n_missing:
                print(f'  skipped {n_missing} images with missing mask/ocr')


if __name__ == '__main__':
    main()
