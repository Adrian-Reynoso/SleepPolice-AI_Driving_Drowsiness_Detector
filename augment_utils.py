import cv2
import numpy as np
import random

def augment_image(img):
    """
    Apply a series of data augmentation transformations to a single image.
    Returns a list of augmented versions including the original image.
    """

    augmented = [img]  # Start with the original

    # === 1. Horizontal Flip ===
    flipped = cv2.flip(img, 1)  # 1 = horizontal
    augmented.append(flipped)

    # === 2. Rotation (±10°) ===
    h, w = img.shape[:2]
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented.append(rotated)

    # === 3. Zoom (scale ±10%) ===
    for scale in [0.9, 1.1]:
        zoomed = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        # Center-crop to original size
        zh, zw = zoomed.shape[:2]
        start_x = (zw - w) // 2
        start_y = (zh - h) // 2
        zoomed_crop = zoomed[start_y:start_y + h, start_x:start_x + w]
        if zoomed_crop.shape == img.shape:
            augmented.append(zoomed_crop)

    # === 4. Brightness Shift (±30%) ===
    for brightness in [0.7, 1.3]:
        bright = np.clip(img.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
        augmented.append(bright)

    # === 5. Gaussian Blur ===
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    augmented.append(blurred)

    # === 6. Gaussian Noise ===
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    augmented.append(noisy)

    return augmented
