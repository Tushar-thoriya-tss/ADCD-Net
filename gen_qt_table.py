"""Generate qt_table.pk required by ADCD-Net.

Maps JPEG quality factor (1-100) -> (8, 8) luma quantization table in
**row-major order** (matches PIL/libjpeg's native layout).

NOTE: We deliberately do NOT use get_qt.py's zigzag-to-row-major conversion,
because PIL's `image.quantization` already returns the table in row-major
order. Applying the zigzag remap would scramble the values.

Output file: cfg.qt_path
"""
import io
import os
import pickle
import numpy as np
from PIL import Image

import cfg


def main():
    out_path = cfg.qt_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    dummy = Image.fromarray(np.full((64, 64), 128, dtype=np.uint8), mode='L')

    qt_dict = {}
    for qf in range(1, 101):
        buf = io.BytesIO()
        dummy.save(buf, 'JPEG', quality=qf)
        buf.seek(0)
        with Image.open(buf) as im:
            q = im.quantization.get(0)
            if q is None or len(q) != 64:
                raise RuntimeError(f'Failed to read luma QT for QF={qf}')
            qt_dict[qf] = np.array(q, dtype=np.int32).reshape(8, 8)

    with open(out_path, 'wb') as f:
        pickle.dump(qt_dict, f)

    print(f'Wrote {len(qt_dict)} QF entries -> {out_path}')
    print('QF=75 sample (row-major):')
    print(qt_dict[75])


if __name__ == '__main__':
    main()
