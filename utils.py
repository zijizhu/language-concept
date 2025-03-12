import cv2
import numpy as np
from PIL import Image


def overlay_attn_map(attn_map: np.ndarray, im: Image.Image):
    """
    attn_map: np.ndarray of shape `(H, W,)`, same as im, values can be unormalized
    im: a PIL image of width H and height W
    """
    im = np.array(im.resize((224, 224), Image.BILINEAR))

    max_val, min_val = np.max(attn_map), np.min(attn_map)
    attn_map = (attn_map - min_val) / (max_val - min_val)
    heatmap = cv2.applyColorMap((255 * attn_map).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return (0.5 * heatmap + 0.5 * im).astype(np.uint8)
