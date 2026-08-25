from __future__ import division, print_function, unicode_literals

import numpy as np
import cv2


def normalise_depth(depth):
    """Convert integer or floating-point depth to float32 in the range [0, 1]."""
    if np.issubdtype(depth.dtype, np.floating):
        return np.clip(depth, 0.0, 1.0).astype(np.float32)
    if np.issubdtype(depth.dtype, np.integer):
        return depth.astype(np.float32) / np.iinfo(depth.dtype).max
    raise ValueError(f"unsupported depth dtype: {depth.dtype}")


def apply(image, crop=None, dims=None, channel_mode="gray", stdiz=False,
          normlz=False, binary_threshold=None, circle_mask_radius=None,
          blur_ksize=1, morphology_ksize=1, circle_mask_offset=None, **kwargs
          ) -> np.ndarray:
    """ Transform image in various ways. """

    if channel_mode:
        if channel_mode == "gray":
            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
            elif image.ndim == 2:
                image = image[..., np.newaxis]
        elif channel_mode in ["red", "green", "blue"]:
            channel_map = {"blue": 0, "green": 1, "red": 2}
            channel_idx = channel_map[channel_mode]
            image = image[:, :, channel_idx][..., np.newaxis]

    if crop:
        h, w = image.shape[:2]
        if len(crop) == 4:  # rectangular crop
            x0, y0, x1, y1 = crop
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            image = image[y0:y1, x0:x1]

        elif len(crop) == 3:  # square crop with padding
            cx, cy, r = crop
            x0, x1 = cx - r, cx + r
            y0, y1 = cy - r, cy + r
            size = x1 - x0  # = 2 * r

            src_x0, src_x1 = max(0, x0), min(w, x1)
            src_y0, src_y1 = max(0, y0), min(h, y1)

            dst_x0 = src_x0 - x0
            dst_y0 = src_y0 - y0

            canvas = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
            canvas[dst_y0:dst_y0 + (src_y1 - src_y0),
                dst_x0:dst_x0 + (src_x1 - src_x0)] = image[src_y0:src_y1, src_x0:src_x1]
            image = canvas

    if dims:
        image = cv2.resize(image, tuple(dims), interpolation=cv2.INTER_AREA)
        if image.ndim < 3:
            image = image[..., np.newaxis]

    if blur_ksize:
        if blur_ksize > 1:
            image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)

    if binary_threshold:
        image = threshold_image(image, *binary_threshold)[..., np.newaxis]

    if morphology_ksize:
        if morphology_ksize > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_ksize, morphology_ksize))
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    if stdiz:
        image = per_image_standardisation(image.astype(np.float32))

    if normlz:
        image = image.astype(np.float32) / 255.0

    if circle_mask_radius:
        if not circle_mask_offset:
            circle_mask_offset = (0, 0)
        image = apply_circle_mask(
            image, circle_mask_radius, circle_mask_offset)

    return image


def threshold_image(image, block_size=11, offset=-5):
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block_size, offset
    )


def apply_circle_mask(image, radius=110, circle_mask_offset=(0, 0)):
    hh, ww = image.shape[:2]
    hc, wc = hh // 2 + circle_mask_offset[0], ww // 2 + circle_mask_offset[1]
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.circle(mask, (hc, wc), radius, 255, thickness=-1)
    return cv2.bitwise_and(image, image, mask=mask)


def per_image_standardisation(image):
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.sqrt(((image - mean)**2).mean(axis=(0, 1), keepdims=True))
    return (image - mean) / (std+1e-6)


def camera_loop(camera,
                image_processing_kwargs, display_name='processed_image', display_size=(640, 480)
                ):
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(display_name, *display_size)

    while True:
        image = camera.process()
        processed_image = apply(image, **image_processing_kwargs)
        cv2.imshow(display_name, processed_image)
        if cv2.waitKey(10) == 27:  # Esc key to stop
            break


if __name__ == '__main__':

    from tactile_bench.data.utils.embodiment import RealSensor

    sensor_params = {'source': 1}

    camera = RealSensor(sensor_params)

    image_processing_params = {
        'channel_mode': 'gray', 'bbox': None, 'dims': None, 'stdiz': False,
        'normlz': False, 'thresh': [11, -30], 'circle_mask_radius': None}

    camera_loop(camera, image_processing_params)
