import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
from paddleocr import TextDetection
from glob import glob
from tqdm import tqdm

class TextDetector:
    def __init__(self):
        self.model = TextDetection(model_name="PP-OCRv5_server_det")

    def get_mask(self, img_path, save_path):
        output = self.model.predict(input=img_path, batch_size=1)

        # Extract detection results (assuming single image input)
        res = output[0]
        polys = res['dt_polys']
        scores = res['dt_scores']  # Optional: can filter based on scores if needed

        # Load image to get dimensions
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Create binary mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for i, poly in enumerate(polys):
            if scores[i] > 0.5:  # Optional threshold for confidence
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

        # Save the mask as PNG (multiply by 255 for visibility: white text regions on black background)
        cv2.imwrite(save_path, mask * 255)


if __name__ == '__main__':
    detector = TextDetector()

    root = '/sharedrive/Tushar Thoriya/GitHub techniques/DATAAAAASET'

    ds_name_list = ['aadhar_tamper']
    splits = ['train', 'val']

    for ds_name in ds_name_list:
        for split in splits:
            print(f'Processing dataset: {ds_name} | split: {split}')
            split_dir = os.path.join(root, ds_name, split)
            img_dir = os.path.join(split_dir, 'images')
            ocr_dir = os.path.join(split_dir, 'ocr')

            if not os.path.isdir(img_dir):
                print(f'  [skip] images dir not found: {img_dir}')
                continue

            os.makedirs(ocr_dir, exist_ok=True)
            img_list = sorted(glob(os.path.join(img_dir, '*')))
            for img_path in tqdm(img_list):
                img_name = os.path.basename(img_path)
                ocr_path = os.path.join(ocr_dir, img_name)
                detector.get_mask(img_path=img_path, save_path=ocr_path)
