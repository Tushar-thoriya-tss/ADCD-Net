"""
Batch inference for ADCD-Net.
Edit INPUT_PATH and OUTPUT_DIRNAME below, then run:  python inference.py

Outputs (saved to OUTPUT_DIRNAME):
    <name>_mask.png    — binary mask  (white = tampered, black = pristine)
    <name>_overlay.png — input image with tampered region highlighted in red
"""

# ── Configure these paths ───────────────────────────────────────────────────
INPUT_PATH   = '/sharedrive/Tushar Thoriya/GitHub techniques/ADCD-Net/input'  # image file OR folder of images
CKPT_PATH    = '/sharedrive/Tushar Thoriya/GitHub techniques/DATAAAAASET/exp_out/ADCDNet/Log_v05071531/ckpt/Step9_Score0.0139.pth'
OUTPUT_DIRNAME = 'output'
MAX_SIZE     = 1024   # longest-side cap; set very large (e.g. 99999) to disable resize
# ─────────────────────────────────────────────────────────────────────────────

import os
import os.path as op
import sys

sys.path.insert(0, op.dirname(op.abspath(__file__)))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from PIL import Image

import cfg
from ds import multi_jpeg, load_qt, img_totsr
from model.model import ADCDNet


def load_model(device):
    model = ADCDNet().to(device)
    state = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
    state = state['model'] if 'model' in state else state
    state = {k.replace('module.', ''): v for k, v in state.items()}
    miss, _ = model.load_state_dict(state, strict=False)
    if miss:
        print(f"Missing keys: {miss}")
    model.eval()
    return model


def preprocess(img_path, qts):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w = img_bgr.shape[:2]

    # Resize if too large (mirrors val pipeline)
    if h > cfg.val_max_size or w > cfg.val_max_size:
        scale = cfg.val_max_size / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # JPEG encode/decode to extract DCT coefficients (qf=100 = minimal compression)
    dct, img_out, qf_record = multi_jpeg(
        deepcopy(img_pil), num_jpeg=-1, min_qf=-1, upper_bound=-1, jpeg_record=[100]
    )
    qt = qts[qf_record[-1]].clamp(0, 63)   # [8, 8] LongTensor

    img_t = img_totsr(img_out)                              # [3, H, W]
    dct_t = torch.tensor(np.clip(np.abs(dct), 0, 20))      # [H, W]

    orig_h, orig_w = img_t.shape[1], img_t.shape[2]

    # Pad to square, divisible by 16 (mirrors pad_collate).
    # NOTE: dct grid is JPEG-aligned (multiple of 8) so its shape can differ
    # from the image — pad each tensor based on its own current shape.
    size = max(orig_h, orig_w, dct_t.shape[0], dct_t.shape[1])
    size = size if size % 16 == 0 else (size // 16 + 1) * 16
    img_t = F.pad(img_t, (0, size - img_t.shape[2], 0, size - img_t.shape[1]))
    dct_t = F.pad(dct_t, (0, size - dct_t.shape[1], 0, size - dct_t.shape[0]))

    # Dummy zero masks — not used during is_train=False forward pass
    mask     = torch.zeros(1, size, size, dtype=torch.long)
    ocr_mask = torch.zeros(1, size, size, dtype=torch.long)

    return img_t, dct_t, qt, mask, ocr_mask, orig_h, orig_w, img_bgr


@torch.no_grad()
def run_inference(model, img_t, dct_t, qt_t, mask, ocr_mask, device):
    img_t    = img_t.unsqueeze(0).to(device)
    dct_t    = dct_t.unsqueeze(0).to(device)
    qt_t     = qt_t.unsqueeze(0).to(device)
    mask     = mask.unsqueeze(0).to(device)
    ocr_mask = ocr_mask.unsqueeze(0).to(device)

    use_amp = device.type == 'cuda'
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits = model(img_t, dct_t, qt_t, mask, ocr_mask, is_train=False)[0]

    # logits: [1, 2, H, W]  →  pred: [H, W]  (0=pristine, 1=tampered)
    pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def save_outputs(pred, img_bgr, orig_h, orig_w, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)

    # Crop padding back to original size
    pred = pred[:orig_h, :orig_w]

    # Binary mask: tampered=255, pristine=0
    binary_mask = (pred * 255).astype(np.uint8)

    # Convert binary mask to 3-channel (for side-by-side concatenation)
    binary_mask_3ch = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Overlay: blend red on tampered pixels
    overlay = img_bgr.copy().astype(np.float32)
    tampered = pred == 1
    overlay[tampered] = overlay[tampered] * 0.35 + np.array([0, 0, 255]) * 0.65
    overlay = overlay.astype(np.uint8)

    # Combine: original | overlay | mask (side-by-side)
    combined = np.hstack([img_bgr, overlay, binary_mask_3ch])

    combined_path = op.join(out_dir, f'{name}_combined.png')
    cv2.imwrite(combined_path, combined)

    pct = 100.0 * tampered.mean()
    print(f"Tampered: {tampered.sum()} px  ({pct:.1f}% of image)")
    print(f"  combined → {combined_path}")


def get_image_files(path):
    """Get all image files from a path (file or folder)."""
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    if op.isfile(path):
        return [path]
    elif op.isdir(path):
        images = []
        for fname in sorted(os.listdir(path)):
            fpath = op.join(path, fname)
            if op.isfile(fpath) and op.splitext(fname)[1].lower() in image_exts:
                images.append(fpath)
        return images
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine output folder (parallel to input, at same parent level)
    if op.isfile(INPUT_PATH):
        parent_dir = op.dirname(INPUT_PATH)
    else:
        parent_dir = op.dirname(INPUT_PATH.rstrip('/'))
    output_dir = op.join(parent_dir, OUTPUT_DIRNAME)

    print(f"Device    : {device}")
    print(f"Input     : {INPUT_PATH}")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Output    : {output_dir}\n")

    img_paths = get_image_files(INPUT_PATH)
    if not img_paths:
        print(f"No images found in {INPUT_PATH}")
        return

    print(f"Found {len(img_paths)} image(s)")

    qts   = load_qt(cfg.qt_path)
    model = load_model(device)

    for i, img_path in enumerate(img_paths, 1):
        print(f"\n[{i}/{len(img_paths)}] {op.basename(img_path)}")
        try:
            img_t, dct_t, qt_t, mask, ocr_mask, orig_h, orig_w, img_bgr = preprocess(img_path, qts)
            pred = run_inference(model, img_t, dct_t, qt_t, mask, ocr_mask, device)
            name = op.splitext(op.basename(img_path))[0]
            save_outputs(pred, img_bgr, orig_h, orig_w, output_dir, name)
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == '__main__':
    main()
