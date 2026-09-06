# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


# Define augmentation functions
def add_noise(image, mean=0, std=0.05):
    """Add Gaussian noise to the image."""
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise * 255  # Scale noise to image range
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def time_shift(image, shift):
    """Shift the image horizontally."""
    return np.roll(image, shift, axis=1)


def scale_image(image, zoom_factor):
    """
    Zooms into the image while maintaining the original dimensions.

    Args:
        image (ndarray): Input grayscale image.
        zoom_factor (float): Zoom factor greater than 1 for zoom in.

    Returns:
        ndarray: Zoomed image with the same size as the input.
    """
    height, width = image.shape
    new_height, new_width = int(height / zoom_factor), int(width / zoom_factor)

    # Crop the center of the image
    start_y = (height - new_height) // 2
    start_x = (width - new_width) // 2
    cropped = image[start_y:start_y + new_height, start_x:start_x + new_width]

    # Resize back to original dimensions
    zoomed_image = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    return zoomed_image

def rotate_image(image, angle):
    """Rotate the image by a specific angle."""
    height, width = image.shape
    center = (width // 2, height // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated_image


def flip_image(image, mode):
    """Flip the image: mode=0 (vertical), 1 (horizontal), -1 (both)."""
    return cv2.flip(image, mode)


def elastic_transform(image, alpha, sigma):
    """Apply elastic deformation to the image."""
    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).astype(np.float32), (x + dx).astype(np.float32)
    return cv2.remap(image, indices[1], indices[0], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def spectral_shift(image, shift):
    """Shift the image in the frequency domain."""
    f_image = np.fft.fft2(image)
    f_image = np.fft.fftshift(f_image)
    shifted_image = np.roll(f_image, shift, axis=1)
    shifted_image = np.fft.ifftshift(shifted_image)
    return np.abs(np.fft.ifft2(shifted_image)).astype(np.uint8)


# Augmentation Registry and Suffix Mappings
AUG_PARAM_REGISTRY = {
    "noise": (add_noise, {"mean": 0, "std": 0.15}),
    "time_shift": (time_shift, {"shift": 20}),
    "rotation": (rotate_image, {"angle": 15}),
    "flip": (flip_image, {"mode": 1}),
    "elastic": (elastic_transform, {"alpha": 34, "sigma": 4}),
    "spectral": (spectral_shift, {"shift": 100}),
    "scale": (scale_image, {"zoom_factor": 1.2}),
}

# Ground truth suffix mappings in the GPR dataset
AUG_SUFFIX_MAP = {
    "noise": {"util": "aug_1", "cav": "aug_7", "name": "Gaussian Noise"},
    "time_shift": {"util": "aug_2", "cav": "aug_8", "name": "Time Shift"},
    "rotation": {"util": "aug_3", "cav": "aug_9", "name": "Rotation"},
    "flip": {"util": "aug_4", "cav": "aug_11", "name": "Horizontal Flip"},
    "elastic": {"util": "aug_5", "cav": "aug_12", "name": "Elastic Deformation"},
    "spectral": {"util": "aug_6", "cav": "aug_13", "name": "Spectral Shift"},
    "scale": {"util": None, "cav": "aug_10", "name": "Scale / Zoom"},
}


def get_active_augmentations(
    noise=True,
    time_shift=True,
    rotation=True,
    flip=True,
    elastic=True,
    spectral=True,
    scale=False,
):
    """
    Returns a list of (function, kwargs) tuples based on active flags.
    Allows enabling/disabling any technique for ablation studies.
    """
    selected = []
    flags = [
        ("noise", noise),
        ("time_shift", time_shift),
        ("rotation", rotation),
        ("flip", flip),
        ("elastic", elastic),
        ("spectral", spectral),
        ("scale", scale),
    ]
    for name, enabled in flags:
        if enabled and name in AUG_PARAM_REGISTRY:
            selected.append(AUG_PARAM_REGISTRY[name])
    return selected


# Main function to apply augmentations
def augment_gpr_data(input_folder, output_folder, augmentations=None, **kwargs):
    """
    Apply augmentations to GPR data in the input folder.
    If augmentations is None, uses get_active_augmentations(**kwargs).
    """
    if augmentations is None:
        augmentations = get_active_augmentations(**kwargs)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Applying {len(augmentations)} active augmentation(s) to {len(image_files)} images from {input_folder}...")

    for image_file in tqdm(image_files, desc="Augmenting images"):
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (224, 224))

        base_name = os.path.splitext(image_file)[0]
        for i, (func, params) in enumerate(augmentations):
            augmented_image = func(image, **params)
            aug_file = f"{base_name}_aug_{i + 1}.jpg"
            aug_path = os.path.join(output_folder, aug_file)
            cv2.imwrite(aug_path, augmented_image)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Apply toggleable domain-specific GPR data augmentations")
    parser.add_argument("--input", type=str, default="GPR_data/Utilities", help="Input folder with GPR profiles")
    parser.add_argument("--output", type=str, default="GPR_data/Utilities/augmented_images", help="Output folder")
    parser.add_argument("--no-noise", action="store_true", help="Disable Gaussian noise")
    parser.add_argument("--no-time-shift", action="store_true", help="Disable horizontal time shift")
    parser.add_argument("--no-rotation", action="store_true", help="Disable rotation")
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal flip")
    parser.add_argument("--no-elastic", action="store_true", help="Disable elastic deformation")
    parser.add_argument("--no-spectral", action="store_true", help="Disable FFT spectral shift")
    parser.add_argument("--scale", action="store_true", help="Enable scale / zoom")
    parser.add_argument("--disable", type=str, default=None, help="Comma-separated techniques to disable (e.g. noise,rotation)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    disabled_set = set()
    if args.disable:
        disabled_set = {s.strip().lower() for s in args.disable.split(",")}

    active_augs = get_active_augmentations(
        noise=not args.no_noise and "noise" not in disabled_set,
        time_shift=not args.no_time_shift and "time_shift" not in disabled_set,
        rotation=not args.no_rotation and "rotation" not in disabled_set,
        flip=not args.no_flip and "flip" not in disabled_set,
        elastic=not args.no_elastic and "elastic" not in disabled_set,
        spectral=not args.no_spectral and "spectral" not in disabled_set,
        scale=args.scale and "scale" not in disabled_set,
    )

    augment_gpr_data(args.input, args.output, active_augs)

